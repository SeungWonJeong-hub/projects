"""2025 대형산불 가상라벨(worldmodel) 빌드 — 실측GT 부재(사용자 전수확인) 대응.

한국 2022-2025 대형산불은 산림청 경계GT가 없다(MTfire.shp=2011-2021). 그래서
2025 역대최대 의성복합 등을 **가상라벨**로 합성:
  라벨 y = dNBR(불탄 물리신호 ≥THR) ∩ FIRMS 열점영역(실제 화재위치로 제약, dNBR 오탐 제거)
입력은 ko_build_refl와 동일한 원시반사율 6밴드×(pre,post)=12ch(×1e4, ≥2022 offset −1000).
→ 별도 폴더 ko_refl_2025/ 에 저장(train에만 추가, 고정 실측 홀드아웃 불변 = 테스트 정직).

사용: KO_OUT=../ko_refl_2025 venv/python ko_build_2025.py
"""
from __future__ import annotations
import os, math, json, datetime as dt
import numpy as np
from osgeo import gdal, ogr, osr
from pystac_client import Client
import socket
gdal.UseExceptions()
socket.setdefaulttimeout(90)
for _k,_v in {"GDAL_HTTP_TIMEOUT":"40","GDAL_HTTP_CONNECTTIMEOUT":"15","GDAL_HTTP_MAX_RETRY":"3",
              "GDAL_HTTP_RETRY_DELAY":"3","CPL_VSIL_CURL_USE_HEAD":"NO","GDAL_DISABLE_READDIR_ON_OPEN":"EMPTY_DIR"}.items():
    gdal.SetConfigOption(_k,_v)

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("KO_OUT",os.path.join(HERE,"..","ko_refl_2025"))
FIRMS=os.path.join(HERE,"..","..","산불데이터","firms_korea_bbox_2025.geojson")
CAT=Client.open("https://earth-search.aws.element84.com/v1")
UTM=32652
DNBR_THR=float(os.environ.get("DNBR_THR","0.20"))   # 심불 임계(USGS moderate)
FIRMS_BUF_M=float(os.environ.get("FIRMS_BUF_M","800"))  # 열점 버퍼(불탄영역 제약)
CROP_M=2650.0; STRIDE=256; T=256
BANDS=[("B02","blue"),("B03","green"),("B04","red"),("B8A","nir08"),("B11","swir16"),("B12","swir22")]

# 대형산불 (FIRMS 검증). bbox_ll=[W,S,E,N], fdate=화재 절정일
FIRES=[
    dict(name="kf2023_hongseong",       bbox=[126.42,36.48,126.60,36.63], fdate=dt.date(2023,4,2)),
    dict(name="kf2025_uiseong_48000ha", bbox=[128.46,36.23,128.95,36.55], fdate=dt.date(2025,3,25)),
    dict(name="kf2025_sancheong_hadong", bbox=[127.68,35.10,128.00,35.38], fdate=dt.date(2025,3,24)),  # 남쪽 수정
    dict(name="kf2025_ulju",            bbox=[129.14,35.34,129.32,35.50], fdate=dt.date(2025,3,24)),
]

def rfc(a,b): return f"{a}T00:00:00Z/{b}T23:59:59Z"

def otsu(v):
    """1D Otsu 임계 (반전-강건 char 분리용)."""
    v=v[np.isfinite(v)]
    if v.size<50: return None
    hist,edges=np.histogram(v,bins=128)
    p=hist/hist.sum(); w=np.cumsum(p); m=np.cumsum(p*((edges[:-1]+edges[1:])/2))
    mt=m[-1]; denom=w*(1-w)
    sb=np.where(denom>1e-9,(mt*w-m)**2/np.where(denom>1e-9,denom,1),0)
    return (edges[:-1]+edges[1:])[np.argmax(sb)]/2

def stac_refl(bbox_ll, dt_range, ute):
    its=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=dt_range,
             query={"eo:cloud_cover":{"lt":30}},sortby=[{"field":"properties.eo:cloud_cover","direction":"asc"}],max_items=1).items())
    if not its: return None
    d=its[0].datetime.date()
    sib=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=rfc(d-dt.timedelta(days=3),d+dt.timedelta(days=3)),
             query={"eo:cloud_cover":{"lt":45}},max_items=12).items()) or its
    off=1000.0 if d>=dt.date(2022,1,25) else 0.0
    b={}
    for k,a in BANDS+[("SCL","scl")]:
        hrefs=[(x.assets[a].href if a in x.assets else x.assets.get(k).href) for x in sib if (a in x.assets or k in x.assets)]
        if not hrefs: return None
        gdal.Warp(f"/tmp/_k25_{k}.tif",[f"/vsicurl/{h}" for h in hrefs],dstSRS=f"EPSG:{UTM}",xRes=20,yRes=20,
                  outputBounds=ute,resampleAlg="average",format="GTiff",srcNodata=0)
        arr=gdal.Open(f"/tmp/_k25_{k}.tif").ReadAsArray().astype(np.float32)
        if k!="SCL": arr=np.clip(arr-off,0,None)
        b[k]=arr
    b["_date"]=d
    return b

def firms_mask(bbox_ll, fdate, ute, H, W):
    """2025 FIRMS 열점(화재기간)을 UTM 래스터 버퍼로 → 불탄영역 제약 마스크."""
    ds=ogr.Open(FIRMS); ly=ds.GetLayer()
    s=osr.SpatialReference(); s.ImportFromEPSG(4326); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    d=osr.SpatialReference(); d.ImportFromEPSG(UTM); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(s,d)
    lo,la,hi,na=bbox_ll; d0=fdate-dt.timedelta(days=20); d1=fdate+dt.timedelta(days=20)
    mem=ogr.GetDriverByName("Memory").CreateDataSource("m"); srs=osr.SpatialReference(); srs.ImportFromEPSG(UTM)
    lyr=mem.CreateLayer("p",srs,ogr.wkbPolygon); n=0
    for f in ly:
        g=f.GetGeometryRef()
        if not g: continue
        x=g.GetX(); y=g.GetY()
        if not(lo<=x<=hi and la<=y<=na): continue
        dtc=str(f.GetField("acq_date") or "").replace("/","-")  # OGR 날짜필드는 YYYY/MM/DD 반환 → ISO로 정규화
        if not (d0.isoformat()<=dtc<=d1.isoformat()): continue
        px,py,_=ct.TransformPoint(x,y)
        ring=ogr.Geometry(ogr.wkbLinearRing)
        for dx,dy in [(-FIRMS_BUF_M,-FIRMS_BUF_M),(FIRMS_BUF_M,-FIRMS_BUF_M),(FIRMS_BUF_M,FIRMS_BUF_M),(-FIRMS_BUF_M,FIRMS_BUF_M),(-FIRMS_BUF_M,-FIRMS_BUF_M)]:
            ring.AddPoint(px+dx,py+dy)
        poly=ogr.Geometry(ogr.wkbPolygon); poly.AddGeometry(ring)
        ft=ogr.Feature(lyr.GetLayerDefn()); ft.SetGeometry(poly); lyr.CreateFeature(ft); n+=1
    rast=gdal.GetDriverByName("MEM").Create("",W,H,1,gdal.GDT_Byte)
    rast.SetGeoTransform((ute[0],20,0,ute[3],0,-20)); rast.SetProjection(srs.ExportToWkt())
    gdal.RasterizeLayer(rast,[1],lyr,burn_values=[1])
    return rast.ReadAsArray()[:H,:W].astype(np.uint8), n

def build_one(fire):
    name=fire["name"]; lo,la,hi,na=fire["bbox"]; fdate=fire["fdate"]
    # bbox → UTM outputBounds
    s=osr.SpatialReference(); s.ImportFromEPSG(4326); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    d=osr.SpatialReference(); d.ImportFromEPSG(UTM); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(s,d)
    xs=[];ys=[]
    for x,y in [(lo,la),(hi,la),(hi,na),(lo,na)]:
        px,py,_=ct.TransformPoint(x,y); xs.append(px); ys.append(py)
    ute=[math.floor(min(xs)),math.floor(min(ys)),math.ceil(max(xs)),math.ceil(max(ys))]
    bbox_ll=[lo,la,hi,na]
    pre_c=fdate.replace(year=fdate.year-1)
    pre=stac_refl(bbox_ll,rfc(pre_c-dt.timedelta(days=40),pre_c+dt.timedelta(days=30)),ute)
    post=stac_refl(bbox_ll,rfc(fdate+dt.timedelta(days=7),fdate+dt.timedelta(days=45)),ute)  # 화재직후로 당김
    if pre is None or post is None: print(f"[{name}] 장면없음",flush=True); return 0
    H=min(pre['B02'].shape[0],post['B02'].shape[0]); W=min(pre['B02'].shape[1],post['B02'].shape[1])
    for b in (pre,post):
        for k in list(b):
            if k!="_date": b[k]=b[k][:H,:W]
    # dNBR (반사율 비율이라 스케일 상쇄)
    nbr_pre=(pre['B8A']-pre['B12'])/(pre['B8A']+pre['B12']+1e-6)
    nbr_post=(post['B8A']-post['B12'])/(post['B8A']+post['B12']+1e-6)
    dnbr=nbr_pre-nbr_post
    valid=(pre['B02']>0)&(pre['B8A']>0)&(pre['B12']>0)&(post['B02']>0)&(post['B8A']>0)&(post['B12']>0)
    valid&=~np.isin(pre['SCL'],[0,1,3,8,9,10,11])&~np.isin(post['SCL'],[0,1,3,8,9,10,11])
    fmask,nfp=firms_mask(bbox_ll,fdate,ute,H,W)
    reg=(fmask>0)&valid
    # 반전-강건 char 신호: post SWIR2(B12) 상승 = 숯/재 (개엽 NIR오염과 무관). 반사율×1e4.
    dchar=post['B12']-pre['B12']
    thr=otsu(dchar[reg]) if reg.sum()>200 else None
    thr=max(thr if thr is not None else 200.0, 150.0)   # 최소 char 상승 하한(≈0.015 반사율)
    # 가상라벨 = char상승≥thr ∩ FIRMS영역 ∩ valid
    y=((dchar>=thr)&reg).astype(np.uint8)
    burn_ha=y.sum()*0.04
    corrob=y[reg].mean() if reg.sum()>0 else 0.0
    if reg.sum()>100:
        cv=dchar[reg]; dv=dnbr[reg]
        print(f"[{name}] FIRMS영역 {reg.sum()*0.04:.0f}ha | char(dB12) p50={np.median(cv):.0f} thr={thr:.0f} | "
              f"라벨 {burn_ha:.0f}ha ({corrob*100:.0f}% of FIRMS) | (참고 dNBR p50={np.median(dv):.3f})",flush=True)
    order=["B02","B03","B04","B8A","B11","B12"]
    x=np.stack([pre[k] for k in order]+[post[k] for k in order],0).astype(np.int16)
    od=os.path.join(OUT,name); os.makedirs(od,exist_ok=True); n=0
    if H<T or W<T: print(f"[{name}] 크롭<{T}",flush=True); os.rmdir(od); return 0
    yy_s=sorted(set(min(yy,H-T) for yy in range(0,H-T+1,STRIDE)))
    xx_s=sorted(set(min(xx,W-T) for xx in range(0,W-T+1,STRIDE)))
    for yy2 in yy_s:
        for xx2 in xx_s:
            v=valid[yy2:yy2+T,xx2:xx2+T]
            if v.mean()<0.4: continue
            yt=y[yy2:yy2+T,xx2:xx2+T]
            if yt.sum()<20: continue   # 불탄 거의없는 타일 skip(경계바깥)
            np.savez_compressed(os.path.join(od,f"{name}_{yy2}_{xx2}.npz"),
                x=x[:,yy2:yy2+T,xx2:xx2+T], y=(yt&v).astype(np.uint8), valid=v.astype(np.uint8)); n+=1
    if n==0:
        try: os.rmdir(od)
        except OSError: pass
    print(f"[{name}] pre {pre['_date']} post {post['_date']} | FIRMS {nfp}점 | 불탄 {burn_ha:.0f}ha | 타일 {n} | char라벨/FIRMS {corrob:.2f}",flush=True)
    return n

def main():
    os.makedirs(OUT,exist_ok=True); tot=0
    for fire in FIRES:
        try: tot+=build_one(fire)
        except Exception as ex: print(f"[{fire['name']}] 실패 {ex}",flush=True)
    print(f"완성 | 총 타일 {tot}",flush=True)

if __name__=="__main__": main()
