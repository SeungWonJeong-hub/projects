"""호주 Black Summer(NSW NPWS 실측 경계) → 먼 대륙(남반구·유칼립투스) 실측 타일.
ArcGIS REST로 화재 경계(geojson) 취득 → 래스터화 + 제 STAC pre/post S2 → 13채널.
라벨 = 실측 경계(raw, dNBR 독립).  사용: au_build.py <name>
"""
from __future__ import annotations
import os, sys, math, json, datetime as dt, urllib.request
import numpy as np, rasterio
from osgeo import gdal, ogr, osr
from pystac_client import Client
gdal.UseExceptions()

HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","au_data")
REST="https://mapprod3.environment.nsw.gov.au/arcgis/rest/services/Fire/NPWS_Fire_History/MapServer/0/query"
CAT=Client.open("https://earth-search.aws.element84.com/v1")
# name: (OBJECTID, fire_date)
FIRES={
 "au_goodgood2019":("119626","2019-12-28"), "au_coombes2019":("90782","2019-10-23"),
 "au_kowmung2019":("335409","2019-11-27"), "au_northblackrange2019":("94020","2019-11-25"),
 "au_kingsgate2019":("78807","2019-09-08"), "au_bordertrail2019":("109982","2019-11-07"),
}
def utm_epsg(lo,la): return (32600 if la>=0 else 32700)+int((lo+180)//6)+1
def stretch(ch,v,lo=2,hi=98):
    x=ch[v]; a,b=np.percentile(x,[lo,hi]) if x.size else (0,1); return np.clip((ch-a)/max(b-a,1e-6),0,1).astype(np.float32)
def ratio(x,y): return (x-y)/(x+y+1e-6)
def idx(x): return ((np.clip(x,-1,1)+1)/2).astype(np.float32)
def s6(B02,B03,B04,B08,B11,B12,v):
    return np.stack([stretch(B12,v),stretch(B08,v),stretch(B02,v),idx(ratio(B08,B12)),idx(ratio(B08,B04)),idx(ratio(B03,B08))],0)

def get_geom(oid):
    u=f"{REST}?where=OBJECTID%3D{oid}&outFields=OBJECTID&returnGeometry=true&outSR=4326&f=geojson"
    g=json.loads(urllib.request.urlopen(u,timeout=60).read())
    feat=g["features"][0]
    return ogr.CreateGeometryFromJson(json.dumps(feat["geometry"]))

def stac_bands(bbox_ll, dt_range, ute, epsg):
    its=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=dt_range,
             query={"eo:cloud_cover":{"lt":25}},sortby=[{"field":"properties.eo:cloud_cover","direction":"asc"}],max_items=1).items())
    if not its: return None
    d=its[0].datetime.date()
    sib=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=f"{d-dt.timedelta(days=2)}/{d+dt.timedelta(days=2)}",query={"eo:cloud_cover":{"lt":35}},max_items=10).items()) or its
    b={}
    for k,a in [("B02","blue"),("B03","green"),("B04","red"),("B08","nir"),("B11","swir16"),("B12","swir22"),("SCL","scl")]:
        hrefs=[(x.assets[a].href if a in x.assets else x.assets[k].href) for x in sib]
        gdal.Warp(f"/tmp/_au_{k}.tif",[f"/vsicurl/{h}" for h in hrefs],dstSRS=f"EPSG:{epsg}",xRes=20,yRes=20,outputBounds=ute,resampleAlg="average",format="GTiff",srcNodata=0)
        b[k]=gdal.Open(f"/tmp/_au_{k}.tif").ReadAsArray().astype(np.float32)
    return b

def main():
    name=sys.argv[1]; oid,fd=FIRES[name]; fdate=dt.date.fromisoformat(fd)
    g=get_geom(oid); env=g.GetEnvelope()  # (minlon,maxlon,minlat,maxlat)
    lo,la=(env[0]+env[1])/2,(env[2]+env[3])/2; epsg=utm_epsg(lo,la)
    src=osr.SpatialReference(); src.ImportFromEPSG(4326); src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst=osr.SpatialReference(); dst.ImportFromEPSG(epsg); dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    gu=g.Clone(); gu.Transform(osr.CoordinateTransformation(src,dst)); e=gu.GetEnvelope()
    ute=[math.floor(e[0]-500),math.floor(e[2]-500),math.ceil(e[1]+500),math.ceil(e[3]+500)]
    bbox_ll=[env[0]-0.02,env[2]-0.02,env[1]+0.02,env[3]+0.02]
    pre=stac_bands(bbox_ll,f"{fdate-dt.timedelta(days=120)}/{fdate-dt.timedelta(days=30)}",ute,epsg)
    post=stac_bands(bbox_ll,f"{fdate+dt.timedelta(days=20)}/{fdate+dt.timedelta(days=110)}",ute,epsg)
    if pre is None or post is None: print(f"[{name}] 장면없음 pre={bool(pre)} post={bool(post)}"); return
    H=min(pre['B02'].shape[0],post['B02'].shape[0]); W=min(pre['B02'].shape[1],post['B02'].shape[1])
    for d in (pre,post):
        for k in d: d[k]=d[k][:H,:W]
    # 경계 래스터화
    mem=ogr.GetDriverByName("Memory").CreateDataSource("m"); srs=osr.SpatialReference(); srs.ImportFromEPSG(epsg)
    lyr=mem.CreateLayer("f",srs,ogr.wkbPolygon); ft=ogr.Feature(lyr.GetLayerDefn()); ft.SetGeometry(gu); lyr.CreateFeature(ft)
    rast=gdal.GetDriverByName("MEM").Create("",W,H,1,gdal.GDT_Byte); rast.SetGeoTransform((ute[0],20,0,ute[3],0,-20)); rast.SetProjection(srs.ExportToWkt())
    gdal.RasterizeLayer(rast,[1],lyr,burn_values=[1]); mask=rast.ReadAsArray()[:H,:W].astype(np.uint8)
    valid=(pre['B02']>0)&(pre['B08']>0)&(pre['B12']>0)&(post['B02']>0)&(post['B08']>0)&(post['B12']>0)
    valid&=~np.isin(pre['SCL'],[0,1,3,8,9,10,11])&~np.isin(post['SCL'],[0,1,3,8,9,10,11])
    pre6=s6(pre['B02'],pre['B03'],pre['B04'],pre['B08'],pre['B11'],pre['B12'],valid)
    post6=s6(post['B02'],post['B03'],post['B04'],post['B08'],post['B11'],post['B12'],valid)
    dnbr=np.clip(ratio(pre['B08'],pre['B12'])-ratio(post['B08'],post['B12']),-1,1)
    x=np.concatenate([pre6,post6,idx(dnbr)[None]],0).astype(np.float32); y=(mask>0).astype(np.uint8)
    od=os.path.join(OUT,name); os.makedirs(od,exist_ok=True); T=256; n=0
    for yy in range(0,H-T+1,T):
        for xx in range(0,W-T+1,T):
            v=valid[yy:yy+T,xx:xx+T]
            if v.mean()<0.5: continue
            np.savez_compressed(os.path.join(od,f"{name}_{yy}_{xx}.npz"),x=x[:,yy:yy+T,xx:xx+T],y=(y[yy:yy+T,xx:xx+T]&v).astype(np.uint8)); n+=1
    print(f"[{name}] pre{fdate-dt.timedelta(days=60)}~ post{fdate+dt.timedelta(days=40)}~ | 타일 {n} | 실측 불탄 {y[valid].mean()*100:.1f}%",flush=True)

if __name__=="__main__": main()
