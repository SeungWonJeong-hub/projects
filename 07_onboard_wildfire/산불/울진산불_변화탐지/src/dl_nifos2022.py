"""2022 NIFoS severity 화재(봉화·군위·밀양·양구) pre/post S2 다운로드(배경작업).
로컬 아카이브 2021-24 공백 우회 → STAC(Earth Search, 공개)서 B04/B08/B11/B12 20m,
화재 AOI로 클립. 재개(존재 skip)·per-scene try/except·로그. 완료 후 asap_curve가 소비.
불명확한 화재일 대비: post 후보 여러개 받아 curve서 burn 보이는 것 자동선택."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import rasterio
from rasterio.warp import transform_bounds
from osgeo import gdal
gdal.UseExceptions()
for k,v in [("GDAL_HTTP_RETRY_DELAY","5"),("GDAL_HTTP_TIMEOUT","120"),
            ("CPL_VSIL_CURL_ALLOWED_EXTENSIONS",".tif"),("VSI_CACHE","TRUE"),
            ("GDAL_HTTP_MAX_RETRY","3")]:
    gdal.SetConfigOption(k,v)
from stac import stac_search

NIF=os.path.expanduser("~/Steven/nifos_burn_gt/ext")
DL =os.path.expanduser("~/syntax_inference/Steven/산불/울진산불_변화탐지/runs/asap/dl")
LOG=os.path.expanduser("~/.asap_dl.log")
os.makedirs(DL, exist_ok=True)
BANDS={"red":"B04","nir":"B08","swir16":"B11","swir22":"B12"}
FIRES={"bonghwa":"봉화_severity_20m.tif","gunwi":"군위_severity_20m.tif",
       "miryang":"밀양_severity_20m.tif","yanggu":"양구_severity_20m.tif"}
WINDOWS=[("pre_w1","2020-12-01/2021-02-28"),("pre_sp1","2021-03-01/2021-05-15"),
         ("pre_su1","2021-06-01/2021-08-31"),("pre_w0","2019-12-05/2020-02-28"),
         ("pre_sp0","2020-03-01/2020-05-15"),("pre_su0","2020-06-01/2020-08-31"),
         ("post_su22","2022-06-15/2022-09-15"),("post_fa22","2022-10-01/2022-12-15"),
         ("post_sp23","2023-03-15/2023-05-31")]

def log(m):
    with open(LOG,"a") as f: f.write(m+"\n")
    print(m, flush=True)

def bbox_of(tif):
    g=rasterio.open(tif); b=transform_bounds(g.crs,"EPSG:4326",*g.bounds)
    return [round(b[0]-0.02,3),round(b[1]-0.02,3),round(b[2]+0.02,3),round(b[3]+0.02,3)]

def dl_band(href,dest,bbox):
    opts=gdal.TranslateOptions(xRes=20,yRes=20,resampleAlg="average",
        projWin=[bbox[0],bbox[3],bbox[2],bbox[1]],projWinSRS="EPSG:4326",format="GTiff",
        creationOptions=["COMPRESS=DEFLATE","TILED=YES"])
    for a in range(3):
        try:
            d=gdal.Translate(dest+".part","/vsicurl/"+href,options=opts); d=None
            os.replace(dest+".part",dest); return os.path.getsize(dest)
        except Exception as e:
            log(f"    band retry{a+1} {os.path.basename(dest)}: {str(e)[:80]}")
            if os.path.exists(dest+".part"):
                try: os.remove(dest+".part")
                except OSError: pass
            time.sleep(4*(a+1))
    return 0

def fetch(feat,bbox,out):
    sid=feat["id"]; sd=os.path.join(out,sid); os.makedirs(sd,exist_ok=True)
    ok=0
    for key,name in BANDS.items():
        dest=os.path.join(sd,name+".tif")
        if os.path.exists(dest) and os.path.getsize(dest)>0: ok+=1; continue
        a=feat.get("assets",{}).get(key)
        if not a or not a.get("href"): log(f"    !missing {key} {sid}"); continue
        if dl_band(a["href"],dest,bbox)>0: ok+=1
    json.dump({"id":sid,"dt":feat["properties"].get("datetime"),
               "cloud":feat["properties"].get("eo:cloud_cover")},
              open(os.path.join(sd,"meta.json"),"w"))
    return ok

def main():
    log(f"\n===== dl_nifos2022 시작 =====")
    for fire,tif in FIRES.items():
        p=os.path.join(NIF,tif)
        if not os.path.exists(p): log(f"[{fire}] severity 없음 skip"); continue
        bbox=bbox_of(p); out=os.path.join(DL,fire); os.makedirs(out,exist_ok=True)
        log(f"[{fire}] bbox={bbox}")
        picks={}
        for label,win in WINDOWS:
            s,e=win.split("/"); win=f"{s}T00:00:00Z/{e}T23:59:59Z"  # RFC3339 필수
            try: feats=stac_search(bbox,win,max_cloud=35,limit=30)
            except Exception as e: log(f"  {label} search err {str(e)[:60]}"); continue
            if not feats: log(f"  {label}: 0씬"); continue
            f=feats[0]  # 이미 cloud asc 정렬
            picks[label]={"id":f["id"],"dt":f["properties"]["datetime"][:10],
                          "cloud":round(f["properties"].get("eo:cloud_cover",-1),1)}
            n=fetch(f,bbox,out)
            log(f"  {label}: {f['id'][:30]} {picks[label]['dt']} cloud{picks[label]['cloud']} → {n}/4밴드")
        json.dump(picks,open(os.path.join(out,"picks.json"),"w"),ensure_ascii=False,indent=1)
    log(f"===== dl_nifos2022 완료 =====")

if __name__=="__main__": main()
