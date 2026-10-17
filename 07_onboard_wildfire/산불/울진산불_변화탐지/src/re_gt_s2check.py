"""RapidEye GT 7사이트 — Sentinel-2 pre/post 취득 가능성 검사.

왜: US 수집에서 "L2A 는 2018년부터"가 드러났다([[wildfire-us-collection]]). 이 GT 는 2016~2019
화재라 **씬이 없으면 채점 자체가 불가능**하다. 채점 배관을 만들기 전에 이것부터 본다.

사용: conda run -n geo python re_gt_s2check.py
"""
import os, sys, json, datetime as dt
if not os.environ.get("PROJ_DATA"):
    os.environ["PROJ_DATA"]=os.path.join(sys.prefix,"share","proj")
    os.environ.setdefault("GDAL_DATA",os.path.join(sys.prefix,"share","gdal"))
from osgeo import osr
from pystac_client import Client

HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
SITES=json.load(open(os.path.join(PROJ,"data","re_gt_sites.json")))
CAT=Client.open("https://earth-search.aws.element84.com/v1")
COLLS=["sentinel-2-l2a","sentinel-2-c1-l2a"]
PRE_D=int(os.environ.get("PRE_D","75")); POST_D=int(os.environ.get("POST_D","75"))
CLOUD=float(os.environ.get("CLOUD","40"))

utm=osr.SpatialReference(); utm.ImportFromEPSG(32652)
wgs=osr.SpatialReference(); wgs.ImportFromEPSG(4326)
for s in (utm,wgs): s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
ct=osr.CoordinateTransformation(utm,wgs)

def search(bbox,d0,d1,coll):
    it=CAT.search(collections=[coll],bbox=bbox,
                  datetime=f"{d0.isoformat()}/{d1.isoformat()}",
                  query={"eo:cloud_cover":{"lt":CLOUD}},limit=100).item_collection()
    return sorted(((i.datetime.date(),i.properties.get("eo:cloud_cover",99)) for i in it))

for s in SITES:
    cands=[c for c in s["candidates"]]
    if not cands:
        print(f"■ {s['code']}: 화재일 후보 없음"); continue
    date=cands[0]["date"]; fd=dt.date(int(date[:4]),int(date[4:6]),int(date[6:]))
    e=s["bbox_utm52n"]
    x0,y0,_=ct.TransformPoint(e[0],e[2]); x1,y1,_=ct.TransformPoint(e[1],e[3])
    bbox=[min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)]
    print(f"■ {s['code']}  화재 {fd}  GT {s['gt_ha']}ha  bbox {bbox[0]:.3f},{bbox[1]:.3f}")
    for coll in COLLS:
        try:
            pre=search(bbox,fd-dt.timedelta(days=PRE_D),fd-dt.timedelta(days=1),coll)
            post=search(bbox,fd+dt.timedelta(days=1),fd+dt.timedelta(days=POST_D),coll)
        except Exception as ex:
            print(f"   {coll:18} 검색 실패: {ex}"); continue
        def fmt(v): return ", ".join(f"{d}({c:.0f}%)" for d,c in v[:4]) or "없음"
        print(f"   {coll:18} pre {len(pre):2d}건 [{fmt(pre)}]")
        print(f"   {'':18} post {len(post):2d}건 [{fmt(post)}]")
