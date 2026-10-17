"""한국 산불(산림청 MTfire.shp 실측 피해구역) → 동아시아 실측 타일.
피해구역 폴리곤(EPSG:5186) 래스터화 + 제 STAC pre/post S2 → 13채널. 라벨=실측 경계(raw).
"""
from __future__ import annotations
import os, math, datetime as dt
import numpy as np, rasterio
from osgeo import gdal, ogr, osr
from pystac_client import Client
gdal.UseExceptions()
HERE=os.path.dirname(os.path.abspath(__file__))
SHP=os.path.join(HERE,"..","..","산불 데이터","korea","mtfire","MTfire.shp")
OUT=os.path.join(HERE,"..","ko_data2")
CAT=Client.open("https://earth-search.aws.element84.com/v1")
UTM=32652  # Korea UTM 52N
N_FIRES=8; MIN_HA=1500; MAX_HA=6000

def stretch(ch,v,lo=2,hi=98):
    x=ch[v]; a,b=np.percentile(x,[lo,hi]) if x.size else (0,1); return np.clip((ch-a)/max(b-a,1e-6),0,1).astype(np.float32)
def ratio(x,y): return (x-y)/(x+y+1e-6)
def idx(x): return ((np.clip(x,-1,1)+1)/2).astype(np.float32)
def s6(B02,B03,B04,B08,B11,B12,v):
    return np.stack([stretch(B12,v),stretch(B08,v),stretch(B02,v),idx(ratio(B08,B12)),idx(ratio(B08,B04)),idx(ratio(B03,B08))],0)

def stac_bands(bbox_ll, dt_range, ute):
    its=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=dt_range,
             query={"eo:cloud_cover":{"lt":25}},sortby=[{"field":"properties.eo:cloud_cover","direction":"asc"}],max_items=1).items())
    if not its: return None
    d=its[0].datetime.date()
    sib=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=f"{d-dt.timedelta(days=2)}/{d+dt.timedelta(days=2)}",query={"eo:cloud_cover":{"lt":35}},max_items=10).items()) or its
    b={}
    for k,a in [("B02","blue"),("B03","green"),("B04","red"),("B08","nir"),("B11","swir16"),("B12","swir22"),("SCL","scl")]:
        hrefs=[(x.assets[a].href if a in x.assets else x.assets[k].href) for x in sib]
        gdal.Warp(f"/tmp/_ko_{k}.tif",[f"/vsicurl/{h}" for h in hrefs],dstSRS=f"EPSG:{UTM}",xRes=20,yRes=20,outputBounds=ute,resampleAlg="average",format="GTiff",srcNodata=0)
        b[k]=gdal.Open(f"/tmp/_ko_{k}.tif").ReadAsArray().astype(np.float32)
    return b

def build_one(name, gu, fdate):
    e=gu.GetEnvelope(); ute=[math.floor(e[0]-600),math.floor(e[2]-600),math.ceil(e[1]+600),math.ceil(e[3]+600)]
    # lonlat bbox
    dst=osr.SpatialReference(); dst.ImportFromEPSG(4326); dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src=osr.SpatialReference(); src.ImportFromEPSG(UTM); src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(src,dst)
    lo1,la1,_=ct.TransformPoint(e[0],e[2]); lo2,la2,_=ct.TransformPoint(e[1],e[3])
    bbox_ll=[min(lo1,lo2)-0.02,min(la1,la2)-0.02,max(lo1,lo2)+0.02,max(la1,la2)+0.02]
    pre_center=fdate.replace(year=fdate.year-1)  # 전년도 같은 계절(개엽기 매칭) — 겨울 낙엽기 pre 회피
    pre=stac_bands(bbox_ll,f"{pre_center-dt.timedelta(days=30)}/{pre_center+dt.timedelta(days=40)}",ute)
    post=stac_bands(bbox_ll,f"{fdate+dt.timedelta(days=15)}/{fdate+dt.timedelta(days=90)}",ute)
    if pre is None or post is None: print(f"[{name}] 장면없음 pre={bool(pre)} post={bool(post)}",flush=True); return
    H=min(pre['B02'].shape[0],post['B02'].shape[0]); W=min(pre['B02'].shape[1],post['B02'].shape[1])
    for d in (pre,post):
        for k in d: d[k]=d[k][:H,:W]
    mem=ogr.GetDriverByName("Memory").CreateDataSource("m"); srs=osr.SpatialReference(); srs.ImportFromEPSG(UTM)
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
    print(f"[{name}] {fdate} | 타일 {n} | 실측 불탄 {y[valid].mean()*100:.1f}%",flush=True)

def main():
    ds=ogr.Open(SHP); ly=ds.GetLayer()
    cand=[]
    for f in ly:
        yr=f.GetField("OCCU_YEAR")
        if yr not in ("2019","2020","2021"): continue
        g=f.GetGeometryRef()
        if not g: continue
        ha=g.GetArea()/1e4
        if not(MIN_HA<=ha<=MAX_HA): continue
        od=f.GetField("OCCU_DATE")
        try: fdate=dt.date(int(od[:4]),int(od[4:6]),int(od[6:8]))
        except: continue
        # UTM 52N 로 변환
        gu=g.Clone(); s=osr.SpatialReference(); s.ImportFromEPSG(5186); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        d=osr.SpatialReference(); d.ImportFromEPSG(UTM); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        gu.Transform(osr.CoordinateTransformation(s,d))
        cand.append((ha,fdate,gu.Clone(),f.GetField("ADRES")))
    cand.sort(reverse=True)
    print(f"후보 {len(cand)}개, 상위 {N_FIRES}개 빌드",flush=True)
    for i,(ha,fdate,gu,ad) in enumerate(cand[:N_FIRES]):
        name=f"ko_{fdate.isoformat().replace('-','')}_{int(ha)}ha"
        try: build_one(name,gu,fdate)
        except Exception as ex: print(f"[{name}] 실패 {ex}",flush=True)
    print(f"완성 {len(__import__('glob').glob(os.path.join(OUT,'*','*.npz')))}타일",flush=True)

if __name__=="__main__": main()
