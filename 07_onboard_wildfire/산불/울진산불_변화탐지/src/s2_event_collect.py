#!/usr/bin/env python3
"""사건 중심 센티넬-2 시계열 수집기 — 젯슨(저장) 또는 B200 에서 실행.

화재 74 건(산림청 MTfire 61 건 = ko_mt/events.json, 최근 대형 13 건 = ko_build_recent 의 목록을 지오코딩) 마다
화재 전 15 개월 ~ 후 6 개월의 모든 센티넬-2 L2A 장면을 사건 범위(ute, UTM 52N) 만 잘라 10 m 격자로 저장한다.
밴드: B02, B03, B04, B8A, B11, B12 + SCL (7 밴드, uint16, DEFLATE). 20 m 원본(B8A, B11, B12) 은 bilinear, SCL 은 nearest.
값은 원본 DN 그대로 두고 boa_offset_applied 를 파일 메타데이터에 적는다 (2022-01-25 이후, offset 미적용이면 사용할 때 1000 을 뺀다).

출력: $S2_EVENTS/<화재>/<장면id>.tif, <화재>/scenes.json (장면 목록과 상태, 재개 가능), $S2_EVENTS/collect.log
사용:
  python3 src/s2_event_collect.py --dry-run           # STAC 만 조회해 장면 수와 예상 용량 표를 낸다 (내려받지 않음)
  S2_EVENTS=~/Desktop/s2_events python3 src/s2_event_collect.py --run [화재이름 ...]
  --recent-only, --mtfire-only 로 대상 제한. 젯슨에서는 ~/wildfire-seg/.venv/bin/python 으로 실행 (requests, gdal 포함).
"""
import os, sys, json, math, time, datetime as dt, logging, argparse, urllib.parse
import requests
from osgeo import gdal, osr
gdal.UseExceptions()
for k,v in {"GDAL_HTTP_MAX_RETRY":"4","GDAL_HTTP_RETRY_DELAY":"5","GDAL_HTTP_TIMEOUT":"180","GDAL_HTTP_CONNECTTIMEOUT":"20",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS":".tif","VSI_CACHE":"TRUE","GDAL_DISABLE_READDIR_ON_OPEN":"EMPTY_DIR","CPL_VSIL_CURL_USE_HEAD":"NO"}.items():
    gdal.SetConfigOption(k,v)
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
EVENTS_MT=os.path.join(PROJ,"ko_mt","events.json"); EVENTS_RECENT=os.path.join(PROJ,"ko_mt","events_recent.json")
STAC="https://earth-search.aws.element84.com/v1/search"; UA={"User-Agent":"wildfire-research/1.0 (syntax-01)"}
UTM=32652; RES=10; PRE_DAYS=456; POST_DAYS=183; MAX_CLOUD=90
MAX_HALF_M=float(os.environ.get("S2_MAX_HALF_M","12000"))   # 사건 범위 반폭 상한(m). 의성 2025 처럼 큰 화재를 24 km 상자로 제한 (기본 41 km -> 28 GB 를 9 GB 로)
CAP_GB=float(os.environ.get("S2_CAP_GB","42"))               # 출력 폴더 총량 상한
MIN_FREE_GB=float(os.environ.get("S2_MIN_FREE_GB","100"))    # 디스크 여유 하한
LAST=["rf_20250322_uiseong_ang"]                             # 큰 사건은 맨 뒤에
BANDS=[("B02","blue","average"),("B03","green","average"),("B04","red","average"),("B8A","nir08","bilinear"),
       ("B11","swir16","bilinear"),("B12","swir22","bilinear"),("SCL","scl","near")]
RECENT=[("의성군 안계면","2025-03-22",52707,"rf_20250322_uiseong_ang"),("의성군 안평면","2025-03-22",46575,"rf_20250322_uiseong_anp"),
        ("울진군 북면","2022-03-04",16302,"rf_20220304_uljin"),("강릉시 옥계면","2022-03-05",4190,"rf_20220305_gangneung"),
        ("산청군 시천면","2025-03-21",3398,"rf_20250321_sancheong"),("홍성군 서부면","2023-04-02",1337,"rf_20230402_hongseong"),
        ("울주군 온양읍","2025-03-22",1117,"rf_20250322_ulju"),("금산군 복수면","2023-04-02",889,"rf_20230402_geumsan"),
        ("합천군 율곡면","2022-02-28",814,"rf_20220228_hapcheon"),("양구군 양구읍","2022-04-10",716,"rf_20220410_yanggu"),
        ("함평군 대동면","2023-04-03",682,"rf_20230403_hampyeong"),("밀양시 부북면","2022-05-31",661,"rf_20220531_miryang"),
        ("영덕군 지품면","2022-02-15",406,"rf_20220215_yeongdeok")]

def srs(epsg):
    s=osr.SpatialReference(); s.ImportFromEPSG(epsg); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER); return s
def geocode(q):
    url="https://nominatim.openstreetmap.org/search?"+urllib.parse.urlencode({"q":q,"format":"json","limit":1,"countrycodes":"kr"})
    for _ in range(3):
        try:
            d=requests.get(url,headers=UA,timeout=25).json()
            if d: return float(d[0]["lat"]),float(d[0]["lon"])
        except Exception as e: logging.warning(f"geocode 재시도 {q}: {e}")
        time.sleep(2)
    return None
def recent_events():
    """ko_build_recent.build_one 과 같은 규칙(원등가 반경 1.6 배, 최소 5.3 km). 지오코딩 결과는 파일에 저장해 재사용."""
    if os.path.exists(EVENTS_RECENT): return json.load(open(EVENTS_RECENT))
    tr=osr.CoordinateTransformation(srs(4326),srs(UTM)); tr_ll=osr.CoordinateTransformation(srs(UTM),srs(4326)); ev={}
    for q,date_s,ha,name in RECENT:
        ll=geocode(q); time.sleep(1.1)
        if not ll: logging.warning(f"[{name}] 지오코딩 실패 {q}"); continue
        lat,lon=ll; cx,cy,_=tr.TransformPoint(lon,lat); half=max(math.sqrt(ha*1e4/math.pi)*1.6,2650)
        if half>MAX_HALF_M: logging.info(f"[{name}] 반폭 {half:.0f} m -> 상한 {MAX_HALF_M:.0f} m"); half=MAX_HALF_M
        ute=[math.floor(cx-half),math.floor(cy-half),math.ceil(cx+half),math.ceil(cy+half)]
        lo1,la1,_=tr_ll.TransformPoint(ute[0],ute[1]); lo2,la2,_=tr_ll.TransformPoint(ute[2],ute[3])
        ev[name]={"name":name,"fdate":date_s,"ha":ha,"ute":ute,"bbox_ll":[round(v,5) for v in (min(lo1,lo2)-0.02,min(la1,la2)-0.02,max(lo1,lo2)+0.02,max(la1,la2)+0.02)],
                  "center":[lat,lon],"query":q,"source":"산불통계 지오코딩"}
    os.makedirs(os.path.dirname(EVENTS_RECENT),exist_ok=True); json.dump(ev,open(EVENTS_RECENT,"w"),ensure_ascii=False,indent=1); return ev
SKIP={"rf_20250322_uiseong_anp":"의성 안평면은 안계면 사건 범위(반경 20.7 km) 안에 들어가 한 사건으로 받는다"}
def all_events(args):
    ev={}
    if not args.recent_only:
        for n,e in json.load(open(EVENTS_MT))["events"].items(): ev[n]={k:e[k] for k in ("name","fdate","ha","ute","bbox_ll")}|{"source":"MTfire"}
    if not args.mtfire_only: ev.update(recent_events())
    for k,why in SKIP.items():
        if k in ev: logging.info(f"[{k}] 건너뜀: {why}"); ev.pop(k)
    return ev

def stac_items(bbox, t0, t1):
    body={"collections":["sentinel-2-l2a"],"bbox":bbox,"datetime":f"{t0}T00:00:00Z/{t1}T23:59:59Z","limit":500,
          "query":{"eo:cloud_cover":{"lt":MAX_CLOUD}}}
    out=[]; url=STAC; n=0
    while url and n<20:
        for k,wait in enumerate((5,15,30,60)):                      # earth-search 가 가끔 502 를 준다. 물러섰다 다시.
            try: r=requests.post(url,json=body,headers=UA,timeout=180); r.raise_for_status(); break
            except Exception as ex:
                if k==3: raise
                logging.warning(f"STAC 재시도 {k+1} ({ex}) {wait}s 뒤"); time.sleep(wait)
        j=r.json(); out+=j.get("features",[]); n+=1
        nxt=[l for l in j.get("links",[]) if l.get("rel")=="next"]
        if not nxt: break
        url=nxt[0]["href"]; body=nxt[0].get("body",body)
    return out
def window(fdate):
    f=dt.date.fromisoformat(fdate); return (f-dt.timedelta(PRE_DAYS)).isoformat(),(f+dt.timedelta(POST_DAYS)).isoformat()
def est_bytes(ute,n): px=((ute[2]-ute[0])/RES)*((ute[3]-ute[1])/RES); return px*7*2*0.6*n

def fetch(feat, ute, dst):
    a=feat["assets"]; tmp=dst+".part"; parts=[]
    for name,key,alg in BANDS:
        if key not in a: raise RuntimeError(f"자산 없음 {key}")
        p=f"/vsimem/{os.path.basename(dst)}_{name}.tif"
        gdal.Warp(p,f"/vsicurl/{a[key]['href']}",dstSRS=f"EPSG:{UTM}",xRes=RES,yRes=RES,outputBounds=ute,resampleAlg=alg,
                  format="GTiff",srcNodata=0,dstNodata=0,outputType=gdal.GDT_UInt16)
        parts.append(p)
    vrt=gdal.BuildVRT("",parts,separate=True)
    ds=gdal.Translate(tmp,vrt,format="GTiff",creationOptions=["COMPRESS=DEFLATE","PREDICTOR=2","TILED=YES"])
    pr=feat["properties"]
    ds.SetMetadata({"SCENE_ID":feat["id"],"DATETIME":pr.get("datetime",""),"CLOUD":str(pr.get("eo:cloud_cover","")),
                    "PLATFORM":pr.get("platform",""),"BOA_OFFSET_APPLIED":str(pr.get("earthsearch:boa_offset_applied","")),
                    "BANDS":",".join(b for b,_,_ in BANDS),"SOURCE":"Sentinel-2 L2A (AWS COG, earth-search)"})
    for i,(b,_,_) in enumerate(BANDS,1): ds.GetRasterBand(i).SetDescription(b)
    ds=None; vrt=None
    for p in parts: gdal.Unlink(p)
    os.replace(tmp,dst)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("names",nargs="*"); ap.add_argument("--dry-run",action="store_true"); ap.add_argument("--run",action="store_true")
    ap.add_argument("--recent-only",action="store_true"); ap.add_argument("--mtfire-only",action="store_true"); args=ap.parse_args()
    out=os.path.expanduser(os.environ.get("S2_EVENTS",os.path.join(PROJ,"ko_mt","s2_events"))); os.makedirs(out,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.FileHandler(os.path.join(out,"collect.log")),logging.StreamHandler()])
    ev=all_events(args); names=args.names or sorted(ev,key=lambda n:(n in LAST,ev[n]["fdate"]))
    import shutil
    def folder_gb(): return sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(out) for f in fs)/1e9
    def free_gb(): return shutil.disk_usage(out).free/1e9
    tot_n=tot_b=0; rows=[]
    for n in names:
        e=ev[n]; t0,t1=window(e["fdate"])
        try: items=stac_items(e["bbox_ll"],t0,t1)
        except Exception as ex: logging.warning(f"[{n}] STAC 실패 {ex}"); continue
        b=est_bytes(e["ute"],len(items)); tot_n+=len(items); tot_b+=b
        w=int((e['ute'][2]-e['ute'][0])/RES); h=int((e['ute'][3]-e['ute'][1])/RES)
        rows.append((n,e["fdate"],t0,t1,len(items),w,h,b))
        if args.dry_run: print(f"{n:<28} {e['fdate']} 창 {t0}~{t1} 장면 {len(items):>4} 격자 {w}x{h} 예상 {b/1e9:.2f} GB",flush=True); continue
        if not args.run: continue
        od=os.path.join(out,n); os.makedirs(od,exist_ok=True); ledger_p=os.path.join(od,"scenes.json")
        led=json.load(open(ledger_p)) if os.path.exists(ledger_p) else {}
        stop=False
        for it in sorted(items,key=lambda f:f["properties"].get("datetime","")):
            sid=it["id"]; dst=os.path.join(od,sid+".tif")
            if led.get(sid,{}).get("status")=="완료" and os.path.exists(dst): continue
            if folder_gb()>=CAP_GB or free_gb()<MIN_FREE_GB:
                logging.warning(f"멈춤: 폴더 {folder_gb():.1f} GB (상한 {CAP_GB}) 여유 {free_gb():.0f} GB (하한 {MIN_FREE_GB}). 다시 실행하면 이어서 받는다."); stop=True; break
            for k in range(3):
                try: fetch(it,e["ute"],dst); st="완료"; break
                except Exception as ex: st=f"실패 {type(ex).__name__}: {str(ex)[:80]}"; time.sleep(5)
            pr=it["properties"]; led[sid]={"datetime":pr.get("datetime"),"cloud":pr.get("eo:cloud_cover"),"platform":pr.get("platform"),
                                         "boa_offset_applied":pr.get("earthsearch:boa_offset_applied"),"status":st,"bytes":os.path.getsize(dst) if st=="완료" else 0}
            json.dump(led,open(ledger_p,"w"),ensure_ascii=False,indent=1); logging.info(f"[{n}] {sid} {st}")
        logging.info(f"[{n}] 끝: 완료 {sum(1 for v in led.values() if v['status']=='완료')}/{len(items)} 장면, {sum(v['bytes'] for v in led.values())/1e9:.2f} GB | 폴더 {folder_gb():.1f} GB")
        if stop: break
    if args.run: logging.info(f"전부 끝 | 폴더 {folder_gb():.1f} GB")
    if args.dry_run:                                                 # 이전 dry_run.json 과 합쳐 저장 (실패분만 다시 돌릴 수 있게)
        prev_p=os.path.join(out,"dry_run.json"); prev={r["name"]:r for r in json.load(open(prev_p))} if os.path.exists(prev_p) else {}
        for r in rows: prev[r[0]]={"name":r[0],"fdate":r[1],"t0":r[2],"t1":r[3],"n_scenes":r[4],"w":r[5],"h":r[6],"est_bytes":int(r[7])}
        prev.pop("rf_20250322_uiseong_anp",None); rows_all=sorted(prev.values(),key=lambda r:r["fdate"])
        tot_n=sum(r["n_scenes"] for r in rows_all); tot_b=sum(r["est_bytes"] for r in rows_all); rows=rows_all
        print(f"\n합계: 화재 {len(rows)} 건, 장면 {tot_n}, 예상 {tot_b/1e9:.1f} GB (10 m, 7 밴드, DEFLATE 0.6 가정). 20 m 로 받으면 약 {tot_b/4e9:.1f} GB.")
        json.dump(rows_all,open(prev_p,"w"),ensure_ascii=False,indent=1)
if __name__=="__main__": main()
