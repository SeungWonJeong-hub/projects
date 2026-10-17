"""산불 재빌드 +적색경계(RE) — 봄 dNBR반전 우회 실험용.
ko_build_refl와 동일 파이프라인/화재/홀드아웃, 밴드만 9개로 확장:
  order=[B02,B03,B04,B05,B06,B07,B8A,B11,B12] → x=(18,256,256) [pre9|post9] int16 반사율×1e4.
RE(B05/06/07)는 20m 네이티브라 리샘플 없이 우리 20m 그대로. base(12ch)는 학습시 RE채널 마스크로 만듦.
사용: KO_OUT=../ko_refl_re python ko_build_re.py [N_FIRES]
"""
from __future__ import annotations
import os, sys, math, datetime as dt, glob
import numpy as np
from osgeo import gdal, ogr, osr
from pystac_client import Client
import socket
gdal.UseExceptions()
socket.setdefaulttimeout(90)
for _k,_v in {"GDAL_HTTP_TIMEOUT":"40","GDAL_HTTP_CONNECTTIMEOUT":"15","GDAL_HTTP_MAX_RETRY":"3",
              "GDAL_HTTP_RETRY_DELAY":"3","CPL_VSIL_CURL_USE_HEAD":"NO",
              "GDAL_DISABLE_READDIR_ON_OPEN":"EMPTY_DIR"}.items():
    gdal.SetConfigOption(_k,_v)
HERE=os.path.dirname(os.path.abspath(__file__))
SHP=os.path.join(HERE,"..","..","산불데이터","korea","mtfire","MTfire.shp")
OUT=os.environ.get("KO_OUT",os.path.join(HERE,"..","ko_refl_re"))
CAT=Client.open("https://earth-search.aws.element84.com/v1")
UTM=32652
N_FIRES=int(sys.argv[1]) if len(sys.argv)>1 else 50
MIN_HA=float(os.environ.get("MIN_HA","600")); MAX_HA=float(os.environ.get("MAX_HA","8000"))
YEARS=set(os.environ.get("YEARS","2016,2017,2018,2019,2020,2021").split(","))
CROP_M=float(os.environ.get("CROP_M","2650")); STRIDE=int(os.environ.get("STRIDE","256"))
# 9밴드: RGB(10m) + RE1/2/3(20m) + NIR08/SWIR16/SWIR22(20m). 전부 20m로 warp.
BANDS=[("B02","blue"),("B03","green"),("B04","red"),
       ("B05","rededge1"),("B06","rededge2"),("B07","rededge3"),
       ("B8A","nir08"),("B11","swir16"),("B12","swir22")]
ORDER=["B02","B03","B04","B05","B06","B07","B8A","B11","B12"]

def rfc(a,b): return f"{a}T00:00:00Z/{b}T23:59:59Z"

def stac_refl(bbox_ll, dt_range, ute):
    its=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=dt_range,
             query={"eo:cloud_cover":{"lt":30}},sortby=[{"field":"properties.eo:cloud_cover","direction":"asc"}],max_items=1).items())
    if not its: return None
    d=its[0].datetime.date()
    sib=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=rfc(d-dt.timedelta(days=2),d+dt.timedelta(days=2)),
             query={"eo:cloud_cover":{"lt":40}},max_items=10).items()) or its
    off=1000.0 if d>=dt.date(2022,1,25) else 0.0
    b={}
    for k,a in BANDS+[("SCL","scl")]:
        hrefs=[(x.assets[a].href if a in x.assets else x.assets.get(k).href) for x in sib if (a in x.assets or k in x.assets)]
        if not hrefs: return None
        gdal.Warp(f"/tmp/_kre_{k}.tif",[f"/vsicurl/{h}" for h in hrefs],dstSRS=f"EPSG:{UTM}",xRes=20,yRes=20,
                  outputBounds=ute,resampleAlg="average",format="GTiff",srcNodata=0)
        arr=gdal.Open(f"/tmp/_kre_{k}.tif").ReadAsArray().astype(np.float32)
        if k!="SCL": arr=np.clip(arr-off,0,None)
        b[k]=arr
    return b

def build_one(name, gu, fdate):
    e=gu.GetEnvelope(); cx=(e[0]+e[1])/2; cy=(e[2]+e[3])/2
    hx=max((e[1]-e[0])/2+500,CROP_M); hy=max((e[3]-e[2])/2+500,CROP_M)
    ute=[math.floor(cx-hx),math.floor(cy-hy),math.ceil(cx+hx),math.ceil(cy+hy)]
    dst=osr.SpatialReference(); dst.ImportFromEPSG(4326); dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src=osr.SpatialReference(); src.ImportFromEPSG(UTM); src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(src,dst)
    lo1,la1,_=ct.TransformPoint(e[0],e[2]); lo2,la2,_=ct.TransformPoint(e[1],e[3])
    bbox_ll=[min(lo1,lo2)-0.02,min(la1,la2)-0.02,max(lo1,lo2)+0.02,max(la1,la2)+0.02]
    pre_center=fdate.replace(year=fdate.year-1)
    pre=stac_refl(bbox_ll,rfc(pre_center-dt.timedelta(days=30),pre_center+dt.timedelta(days=45)),ute)
    post=stac_refl(bbox_ll,rfc(fdate+dt.timedelta(days=15),fdate+dt.timedelta(days=100)),ute)
    if pre is None or post is None: print(f"[{name}] 장면없음",flush=True); return 0
    H=min(pre['B02'].shape[0],post['B02'].shape[0]); W=min(pre['B02'].shape[1],post['B02'].shape[1])
    for d in (pre,post):
        for k in d: d[k]=d[k][:H,:W]
    srs=osr.SpatialReference(); srs.ImportFromEPSG(UTM)
    mem=ogr.GetDriverByName("Memory").CreateDataSource("m"); lyr=mem.CreateLayer("f",srs,ogr.wkbPolygon)
    ft=ogr.Feature(lyr.GetLayerDefn()); ft.SetGeometry(gu); lyr.CreateFeature(ft)
    rast=gdal.GetDriverByName("MEM").Create("",W,H,1,gdal.GDT_Byte); rast.SetGeoTransform((ute[0],20,0,ute[3],0,-20)); rast.SetProjection(srs.ExportToWkt())
    gdal.RasterizeLayer(rast,[1],lyr,burn_values=[1]); mask=rast.ReadAsArray()[:H,:W].astype(np.uint8)
    valid=(pre['B02']>0)&(pre['B8A']>0)&(pre['B12']>0)&(post['B02']>0)&(post['B8A']>0)&(post['B12']>0)
    valid&=~np.isin(pre['SCL'],[0,1,3,8,9,10,11])&~np.isin(post['SCL'],[0,1,3,8,9,10,11])
    x=np.stack([pre[k] for k in ORDER]+[post[k] for k in ORDER],0).astype(np.int16)  # 18ch
    y=(mask>0).astype(np.uint8)
    od=os.path.join(OUT,name); os.makedirs(od,exist_ok=True); T=256; n=0
    if H<T or W<T: print(f"[{name}] 크롭<{T}",flush=True); os.rmdir(od); return 0
    ys=sorted(set(min(yy,H-T) for yy in range(0,H-T+1,STRIDE))); xs=sorted(set(min(xx,W-T) for xx in range(0,W-T+1,STRIDE)))
    for yy2 in ys:
        for xx2 in xs:
            v=valid[yy2:yy2+T,xx2:xx2+T]
            if v.mean()<0.4: continue
            np.savez_compressed(os.path.join(od,f"{name}_{yy2}_{xx2}.npz"),
                x=x[:,yy2:yy2+T,xx2:xx2+T], y=(y[yy2:yy2+T,xx2:xx2+T]&v).astype(np.uint8),
                valid=v.astype(np.uint8)); n+=1
    if n==0: os.rmdir(od)
    print(f"[{name}] {fdate} | 타일 {n} | 불탄 {y[valid].mean()*100:.1f}%",flush=True); return n

def main():
    os.makedirs(OUT,exist_ok=True); ds=ogr.Open(SHP); ly=ds.GetLayer(); cand=[]
    for f in ly:
        if f.GetField("OCCU_YEAR") not in YEARS: continue
        g=f.GetGeometryRef()
        if not g: continue
        s=osr.SpatialReference(); s.ImportFromEPSG(5186); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        d=osr.SpatialReference(); d.ImportFromEPSG(UTM); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        gu=g.Clone(); gu.Transform(osr.CoordinateTransformation(s,d)); ha=gu.GetArea()/1e4
        if not(MIN_HA<=ha<=MAX_HA): continue
        od=f.GetField("OCCU_DATE")
        try: fdate=dt.date(int(od[:4]),int(od[4:6]),int(od[6:8]))
        except: continue
        cand.append((ha,fdate,gu.Clone()))
    cand.sort(key=lambda c:c[0],reverse=True)
    print(f"후보 {len(cand)}, 상위 {N_FIRES} 빌드(+RE 18ch)",flush=True); tot=built=0
    for i,(ha,fdate,gu) in enumerate(cand[:N_FIRES]):
        name=f"kf_{fdate.isoformat().replace('-','')}_{int(ha)}ha"
        if glob.glob(os.path.join(OUT,name,"*.npz")): print(f"[{name}] skip",flush=True); built+=1; continue
        try: n=build_one(name,gu,fdate); tot+=n; built+=(n>0)
        except Exception as ex: print(f"[{name}] 실패 {ex}",flush=True)
        if (i+1)%5==0: print(f"  진행 {i+1}/{min(N_FIRES,len(cand))} 누적타일 {tot}",flush=True)
    print(f"완성 화재 {built} | 총 타일 {len(glob.glob(os.path.join(OUT,'*','*.npz')))}",flush=True)

if __name__=="__main__": main()
