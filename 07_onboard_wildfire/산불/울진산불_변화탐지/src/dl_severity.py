"""dNBR severity 곡선용 SWIR pre/post 다운로드(배경).
severity GT 5화재(울진·안동·봉화·군위·밀양) 다계절 pre + post 후보를 STAC서 취득
(B04/B08/B11/B12, RFC3339 datetime 필수). 재개(존재 skip)·per-scene 예외격리·로그.
울진/산청은 로컬 6밴드도 있으나 다계절 범위 위해 다운로드로 통일."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import rasterio; from rasterio.warp import transform_bounds
from osgeo import gdal; gdal.UseExceptions()
for k,v in [("GDAL_HTTP_RETRY_DELAY","5"),("GDAL_HTTP_TIMEOUT","120"),
            ("CPL_VSIL_CURL_ALLOWED_EXTENSIONS",".tif"),("VSI_CACHE","TRUE"),("GDAL_HTTP_MAX_RETRY","3")]:
    gdal.SetConfigOption(k,v)
from stac import stac_search

NIF=os.path.expanduser("~/Steven/nifos_burn_gt")
DL =os.path.expanduser("~/syntax_inference/Steven/산불/울진산불_변화탐지/runs/asap/dl")
LOG=os.path.expanduser("~/.asap_dl.log"); os.makedirs(DL,exist_ok=True)
BANDS={"red":"B04","nir":"B08","swir16":"B11","swir22":"B12"}

def seas(y): return [(f"{y}w",f"{y}-01-05/{y}-02-28"),(f"{y}sp",f"{y}-03-01/{y}-05-15"),(f"{y}su",f"{y}-06-01/{y}-08-31")]
# 화재별 pre(계절, 화재前)/post(화재後) 윈도
FIRES={
 "uljin":   dict(gt=f"{NIF}/ext/Ulgin_severity.tif", pre=seas(2019)+seas(2020)+seas(2021),
                 post=[("po_su22","2022-06-15/2022-09-15"),("po_fa22","2022-10-01/2022-12-10"),("po_sp23","2023-03-15/2023-05-31")]),
 "andong":  dict(gt=f"{NIF}/andong/andong_severity_20m.tif", pre=seas(2019)+[("2020w","2020-01-05/2020-02-28"),("2020sp0","2020-03-01/2020-04-10")],
                 post=[("po_su20","2020-05-25/2020-08-31"),("po_fa20","2020-09-15/2020-11-15")]),
 "bonghwa": dict(gt=f"{NIF}/ext/봉화_severity_20m.tif", pre=seas(2019)+seas(2020)+seas(2021),
                 post=[("po_su22","2022-06-15/2022-09-15"),("po_fa22","2022-10-01/2022-12-10"),("po_sp23","2023-03-15/2023-05-31")]),
 "gunwi":   dict(gt=f"{NIF}/ext/군위_severity_20m.tif", pre=seas(2019)+seas(2020)+seas(2021),
                 post=[("po_su22","2022-06-15/2022-09-15"),("po_fa22","2022-10-01/2022-12-10"),("po_sp23","2023-03-15/2023-05-31")]),
 "miryang": dict(gt=f"{NIF}/ext/밀양_severity_20m.tif", pre=seas(2019)+seas(2020)+seas(2021),
                 post=[("po_su22","2022-06-15/2022-09-15"),("po_fa22","2022-10-01/2022-12-10"),("po_sp23","2023-03-15/2023-05-31")]),
}

def log(m):
    with open(LOG,"a") as f: f.write(m+"\n")
    print(m,flush=True)
def bbox_of(t):
    g=rasterio.open(t); b=transform_bounds(g.crs,"EPSG:4326",*g.bounds)
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
            log(f"    retry{a+1} {os.path.basename(dest)}: {str(e)[:70]}")
            if os.path.exists(dest+".part"):
                try: os.remove(dest+".part")
                except OSError: pass
            time.sleep(3*(a+1))
    return 0
def fetch(feat,bbox,out):
    sd=os.path.join(out,feat["id"]); os.makedirs(sd,exist_ok=True); ok=0
    for key,name in BANDS.items():
        dest=os.path.join(sd,name+".tif")
        if os.path.exists(dest) and os.path.getsize(dest)>0: ok+=1; continue
        a=feat.get("assets",{}).get(key)
        if a and a.get("href") and dl_band(a["href"],dest,bbox)>0: ok+=1
    json.dump({"id":feat["id"],"dt":feat["properties"].get("datetime"),
               "cloud":feat["properties"].get("eo:cloud_cover")},open(os.path.join(sd,"meta.json"),"w"))
    return ok

def main():
    log("\n===== dl_severity 시작 =====")
    for fire,cfg in FIRES.items():
        if not os.path.exists(cfg["gt"]): log(f"[{fire}] GT없음 skip"); continue
        bbox=bbox_of(cfg["gt"]); out=os.path.join(DL,fire+"_nbr"); os.makedirs(out,exist_ok=True)
        log(f"[{fire}] bbox={bbox}"); picks={}
        for label,win in cfg["pre"]+cfg["post"]:
            s,e=win.split("/"); win2=f"{s}T00:00:00Z/{e}T23:59:59Z"
            try: feats=stac_search(bbox,win2,max_cloud=30,limit=20)
            except Exception as ex: log(f"  {label} err {str(ex)[:50]}"); continue
            if not feats: log(f"  {label}: 0씬"); continue
            f=feats[0]; picks[label]={"id":f["id"],"dt":f["properties"]["datetime"][:10],
                                      "cloud":round(f["properties"].get("eo:cloud_cover",-1),1),
                                      "role":"post" if label.startswith("po") else "pre"}
            n=fetch(f,bbox,out); log(f"  {label}: {f['id'][:28]} {picks[label]['dt']} c{picks[label]['cloud']} →{n}/4")
        json.dump(picks,open(os.path.join(out,"picks.json"),"w"),ensure_ascii=False,indent=1)
    log("===== dl_severity 완료 =====")

if __name__=="__main__": main()
