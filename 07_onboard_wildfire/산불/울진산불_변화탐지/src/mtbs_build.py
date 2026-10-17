"""MTBS 실측 경계 + STAC S2 -> 다지역 실측라벨 CD 타일.
MTBS 폴리곤(기관 실측 화재경계, dNBR 무관)을 래스터화 = 진짜 라벨 -> 순환 탈출.
각 화재: 경계 추출 -> UTM 20m 그리드 -> pre/post S2 -> 13채널[pre6|post6|dNBR] + 실측마스크 타일.
사용: mtbs_build.py <name>
"""
from __future__ import annotations
import os, sys, math, datetime as dt
import numpy as np, cv2
from osgeo import gdal, ogr, osr
from pystac_client import Client
gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
SHP = os.path.join(HERE, "..", "mtbs", "mtbs_perims_DD.shp")
OUT = os.path.join(HERE, "..", "mtbs_data")
# name: (event_id, ig_date)
FIRES = {
    "bootleg_or2021":   ("OR4261612142120210706","2021-07-06"),
    "hermitspeak_nm2022":("NM3571810539920220406","2022-04-06"),
    "pearlhill_wa2020": ("WA4805611948420200907","2020-09-07"),
    "bush_az2020":      ("AZ3362611156220200613","2020-06-13"),
    "cameronpeak_co2020":("CO4060910587920200813","2020-08-13"),
    "richardspring_mt2021":("MT4580910676420210808","2021-08-08"),
    "moose_id2022":     ("ID4537511409120220717","2022-07-17"),
}
BAND_ASSET = {"B02":"blue","B03":"green","B04":"red","B08":"nir","B11":"swir16","B12":"swir22","SCL":"scl"}

def utm_epsg(lon,lat): return (32600 if lat>=0 else 32700)+int((lon+180)//6)+1

def get_poly(event_id):
    ds=ogr.Open(SHP); ly=ds.GetLayer(); ly.SetAttributeFilter(f"event_id='{event_id}'")
    f=ly.GetNextFeature(); g=f.GetGeometryRef().Clone()
    src=osr.SpatialReference(); src.ImportFromEPSG(4269)
    lo,la=f.GetField("burnbndlon"),f.GetField("burnbndlat")
    return g, float(lo), float(la)

def poly_to_utm_grid(g, epsg):
    src=osr.SpatialReference(); src.ImportFromEPSG(4269); src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst=osr.SpatialReference(); dst.ImportFromEPSG(epsg); dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    g=g.Clone(); g.Transform(osr.CoordinateTransformation(src,dst))
    x1,x2,y1,y2=g.GetEnvelope()
    b=800  # 800m 버퍼
    return g, [math.floor(x1-b), math.floor(y1-b), math.ceil(x2+b), math.ceil(y2+b)]

CAT=Client.open("https://earth-search.aws.element84.com/v1")
def pick(bbox_ll, dt_range):
    it=list(CAT.search(collections=["sentinel-2-l2a"], bbox=bbox_ll, datetime=dt_range,
            query={"eo:cloud_cover":{"lt":20}}, sortby=[{"field":"properties.eo:cloud_cover","direction":"asc"}], max_items=1).items())
    return it[0] if it else None

def same_date_items(bbox_ll, item):
    d=item.datetime.date(); rng=f"{d-dt.timedelta(days=2)}/{d+dt.timedelta(days=2)}"
    its=list(CAT.search(collections=["sentinel-2-l2a"], bbox=bbox_ll, datetime=rng,
             query={"eo:cloud_cover":{"lt":30}}, max_items=12).items())
    return its or [item]

def warp_mosaic(hrefs, ute, epsg, out):
    gdal.Warp(out, [f"/vsicurl/{h}" for h in hrefs], dstSRS=f"EPSG:{epsg}", xRes=20, yRes=20,
              outputBounds=ute, resampleAlg="average", format="GTiff", srcNodata=0)
    return gdal.Open(out).ReadAsArray().astype(np.float32)

def load(items, ute, epsg, tag):
    b={}
    for k,a in BAND_ASSET.items():
        hrefs=[(it.assets[a].href if a in it.assets else it.assets[k].href) for it in items]
        b[k]=warp_mosaic(hrefs, ute, epsg, f"/tmp/_mt_{tag}_{k}.tif")
    return b

def stretch(ch,valid,lo=2,hi=98):
    v=ch[valid]; a,b=np.percentile(v,[lo,hi]) if v.size else (0,1)
    return np.clip((ch-a)/max(b-a,1e-6),0,1).astype(np.float32)
def ratio(x,y): return (x-y)/(x+y+1e-6)
def idx(x): return ((np.clip(x,-1,1)+1)/2).astype(np.float32)
def stack6(bd,valid):
    return np.stack([stretch(bd["B12"],valid),stretch(bd["B08"],valid),stretch(bd["B02"],valid),
                     idx(ratio(bd["B08"],bd["B12"])),idx(ratio(bd["B08"],bd["B04"])),idx(ratio(bd["B03"],bd["B08"]))],0)

def main():
    name=sys.argv[1]; eid,ig=FIRES[name]
    igd=dt.date.fromisoformat(ig)
    g,lo,la=get_poly(eid); epsg=utm_epsg(lo,la)
    gu,ute=poly_to_utm_grid(g,epsg)
    # bbox in lonlat for STAC
    ring=g.GetEnvelope(); bbox_ll=[ring[0]-0.02,ring[2]-0.02,ring[1]+0.02,ring[3]+0.02]
    post_r=f"{igd+dt.timedelta(days=20)}/{igd+dt.timedelta(days=110)}"
    pre_r =f"{igd.replace(year=igd.year-1)-dt.timedelta(days=25)}/{igd.replace(year=igd.year-1)+dt.timedelta(days=45)}"
    pi=pick(bbox_ll,pre_r); qi=pick(bbox_ll,post_r)
    if not pi or not qi: print(f"[{name}] 장면없음 pre={bool(pi)} post={bool(qi)}"); return
    pis=same_date_items(bbox_ll,pi); qis=same_date_items(bbox_ll,qi)
    print(f"[{name}] pre {pi.datetime.date()} x{len(pis)}타일 post {qi.datetime.date()} x{len(qis)}타일",flush=True)
    pb=load(pis,ute,epsg,"pre"); qb=load(qis,ute,epsg,"post")
    H=min(pb["B02"].shape[0],qb["B02"].shape[0]); W=min(pb["B02"].shape[1],qb["B02"].shape[1])
    for d in (pb,qb):
        for k in d: d[k]=d[k][:H,:W]
    # 실측 마스크: MTBS 폴리곤 래스터화 (같은 UTM 20m 그리드)
    mem=ogr.GetDriverByName("Memory").CreateDataSource("m"); srs=osr.SpatialReference(); srs.ImportFromEPSG(epsg)
    lyr=mem.CreateLayer("f",srs,ogr.wkbPolygon); ft=ogr.Feature(lyr.GetLayerDefn()); ft.SetGeometry(gu); lyr.CreateFeature(ft)
    rast=gdal.GetDriverByName("MEM").Create("",W,H,1,gdal.GDT_Byte)
    rast.SetGeoTransform((ute[0],20,0,ute[3],0,-20)); rast.SetProjection(srs.ExportToWkt())
    gdal.RasterizeLayer(rast,[1],lyr,burn_values=[1]); mask=rast.ReadAsArray()[:H,:W].astype(np.uint8)
    valid=(pb["B02"]>0)&(pb["B08"]>0)&(pb["B12"]>0)&(qb["B02"]>0)&(qb["B08"]>0)&(qb["B12"]>0)
    valid&=~np.isin(qb["SCL"],[0,1,3,8,9,10,11])&~np.isin(pb["SCL"],[0,1,3,8,9,10,11])
    pre6=stack6(pb,valid); post6=stack6(qb,valid)
    dnbr=np.clip(ratio(pb["B08"],pb["B12"])-ratio(qb["B08"],qb["B12"]),-1,1)
    x=np.concatenate([pre6,post6,idx(dnbr)[None]],0).astype(np.float32)
    od=os.path.join(OUT,name); os.makedirs(od,exist_ok=True); T=256; n=0
    for y in range(0,H-T+1,T):
        for xx in range(0,W-T+1,T):
            v=valid[y:y+T,xx:xx+T]
            if v.mean()<0.5: continue
            np.savez_compressed(os.path.join(od,f"{name}_{y}_{xx}.npz"),
                                x=x[:,y:y+T,xx:xx+T], y=(mask[y:y+T,xx:xx+T]&v).astype(np.uint8)); n+=1
    print(f"[{name}] 타일 {n} | 실측 불탄픽셀 {mask[valid].mean()*100:.1f}% | 저장 {od}",flush=True)

if __name__=="__main__": main()
