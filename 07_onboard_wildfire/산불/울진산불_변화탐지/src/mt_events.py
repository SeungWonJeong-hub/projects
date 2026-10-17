"""다중시기(젯슨 보관 자료) 실험 1단계 — 화재 사건 좌표 목록 만들기.

ko_refl 의 화재 폴더 이름(kf_YYYYMMDD_NNNNha) 을 산림청 MTfire 폴리곤과 (발생일, 정수 ha) 로 맞추고,
ko_build_refl.build_one 과 똑같은 규칙으로 UTM 52N 자르기 범위(ute) 와 경위도 bbox 를 계산한다.
젯슨 장면 색인(ko_mt/jetson_scenes_index.json) 과 대조해 화재별로 pre, post 창 안의 장면 수를 센다.

출력: ko_mt/events.json  (name, fdate, ha, ute, bbox_ll, jetson_pre, jetson_post)
사용: env -u LD_LIBRARY_PATH python src/mt_events.py
"""
import os, glob, json, math, datetime as dt, re
from osgeo import ogr, osr
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
SHP=os.path.join(PROJ,"..","산불데이터","korea","mtfire","MTfire.shp")
IDX=os.path.join(PROJ,"ko_mt","jetson_scenes_index.json"); OUT=os.path.join(PROJ,"ko_mt","events.json")
UTM=32652; CROP_M=2650.0
PRE_DAYS=(120,8)      # 화재 전 120일 ~ 8일
POST_DAYS=(8,130)     # 화재 후 8일 ~ 130일

def srs(epsg):
    s=osr.SpatialReference(); s.ImportFromEPSG(epsg); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER); return s

def main():
    names=sorted(os.path.basename(d) for d in glob.glob(os.path.join(PROJ,"ko_refl","kf_*")) if glob.glob(os.path.join(d,"*.npz")))
    want={}
    for n in names:
        m=re.match(r"kf_(\d{8})_(\d+)ha$",n)
        if m: want[(m.group(1),int(m.group(2)))]=n
    ds=ogr.Open(SHP); ly=ds.GetLayer(); tr=osr.CoordinateTransformation(srs(5186),srs(UTM)); tr_ll=osr.CoordinateTransformation(srs(UTM),srs(4326))
    ev={}
    for f in ly:
        od=f.GetField("OCCU_DATE") or ""
        g=f.GetGeometryRef()
        if not g or len(od)<8: continue
        gu=g.Clone(); gu.Transform(tr); ha=gu.GetArea()/1e4
        key=(od[:8],int(ha))
        if key not in want or want[key] in ev: continue
        e=gu.GetEnvelope(); cx=(e[0]+e[1])/2; cy=(e[2]+e[3])/2
        hx=max((e[1]-e[0])/2+500,CROP_M); hy=max((e[3]-e[2])/2+500,CROP_M)
        ute=[math.floor(cx-hx),math.floor(cy-hy),math.ceil(cx+hx),math.ceil(cy+hy)]
        lo1,la1,_=tr_ll.TransformPoint(e[0],e[2]); lo2,la2,_=tr_ll.TransformPoint(e[1],e[3])
        bbox_ll=[min(lo1,lo2)-0.02,min(la1,la2)-0.02,max(lo1,lo2)+0.02,max(la1,la2)+0.02]
        fdate=dt.date(int(od[:4]),int(od[4:6]),int(od[6:8]))
        ev[want[key]]={"name":want[key],"fdate":fdate.isoformat(),"ha":round(ha,1),"ute":ute,"bbox_ll":[round(v,5) for v in bbox_ll],
                       "n_tiles":len(glob.glob(os.path.join(PROJ,"ko_refl",want[key],"*.npz")))}
    missing=[n for n in names if n not in ev]
    # 젯슨 장면 대조
    idx=json.load(open(IDX))
    def inter(a,b): return not(a[2]<b[0] or b[2]<a[0] or a[3]<b[1] or b[3]<a[1])
    for n,e in ev.items():
        fd=dt.date.fromisoformat(e["fdate"]); pre=[]; post=[]
        for s in idx:
            if not s["bbox"] or not inter(s["bbox"],e["bbox_ll"]): continue
            d=dt.date.fromisoformat(s["datetime"][:10]); dd=(d-fd).days
            if -PRE_DAYS[0]<=dd<=-PRE_DAYS[1]: pre.append(s["id"])
            elif POST_DAYS[0]<=dd<=POST_DAYS[1]: post.append(s["id"])
        e["jetson_pre"]=sorted(pre); e["jetson_post"]=sorted(post)
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    json.dump({"pre_days":PRE_DAYS,"post_days":POST_DAYS,"utm":UTM,"crop_m":CROP_M,"events":ev,"unmatched":missing},open(OUT,"w"),ensure_ascii=False,indent=1)
    cov=[e for e in ev.values() if e["jetson_pre"] and e["jetson_post"]]
    print(f"ko_refl 화재 {len(names)} | MTfire 대응 {len(ev)} | 미대응 {missing}")
    print(f"젯슨 pre+post 모두 있는 화재 {len(cov)}:")
    for e in sorted(cov,key=lambda e:e['fdate']):
        print(f"  {e['name']:<24} pre {len(e['jetson_pre']):>3} post {len(e['jetson_post']):>3} 타일 {e['n_tiles']:>2} ute {e['ute']}")
    part=[e for e in ev.values() if (e["jetson_pre"] or e["jetson_post"]) and e not in cov]
    for e in sorted(part,key=lambda e:e['fdate']): print(f"  (한쪽만) {e['name']} pre {len(e['jetson_pre'])} post {len(e['jetson_post'])}")
if __name__=="__main__": main()
