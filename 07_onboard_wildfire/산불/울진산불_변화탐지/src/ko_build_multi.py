"""한국 다(多)산불 native 학습셋 — 단일 산불 한계 해결.
산림청 MTfire.shp(5084 실측 경계) 중 S2시대(2016+) 대형산불 다수 → 13채널[pre6|post6|dNBR] 타일.
라벨=실측 경계(raw 픽셀, dNBR 독립). 화재별 서브디렉토리 → fire-disjoint 홀드아웃 가능.
전년동기 개엽기 pre(겨울 낙엽기 회피, uljin 검증). 사용: ko_build_multi.py [N_FIRES]
"""
from __future__ import annotations
import os, sys, math, datetime as dt
import numpy as np
from osgeo import gdal, ogr, osr
from pystac_client import Client
gdal.UseExceptions()
HERE=os.path.dirname(os.path.abspath(__file__))
SHP=os.path.join(HERE,"..","..","산불 데이터","korea","mtfire","MTfire.shp")
OUT=os.environ.get("KO_OUT",os.path.join(HERE,"..","ko_multi"))
CAT=Client.open("https://earth-search.aws.element84.com/v1")
UTM=32652
N_FIRES=int(sys.argv[1]) if len(sys.argv)>1 else 40
MIN_HA=float(os.environ.get("MIN_HA","600")); MAX_HA=float(os.environ.get("MAX_HA","8000"))
YEARS=set(os.environ.get("YEARS","2016,2017,2018,2019,2020,2021,2022,2023,2024").split(","))
CROP_M=float(os.environ.get("CROP_M","2650"))   # 화재당 크롭 최소 반폭(m) — 크게=타일 더 많이
STRIDE=int(os.environ.get("STRIDE","256"))      # 타일 stride(px) — <256=중첩 타일(데이터 증대)
JOINT_NORM=os.environ.get("JOINT_NORM","0")=="1"  # 전/후 공동 정규화(변화신호 보존)

def rfc(a,b):
    """date/date → RFC3339 구간 문자열 (Earth-Search가 date-only를 거부하므로 전체 타임스탬프)."""
    return f"{a}T00:00:00Z/{b}T23:59:59Z"

def stretch(ch,v,lo=2,hi=98):
    x=ch[v]; a,b=np.percentile(x,[lo,hi]) if x.size else (0,1); return np.clip((ch-a)/max(b-a,1e-6),0,1).astype(np.float32)
def ratio(x,y): return (x-y)/(x+y+1e-6)
def idx(x): return ((np.clip(x,-1,1)+1)/2).astype(np.float32)
def s6(B02,B03,B04,B08,B11,B12,v):
    return np.stack([stretch(B12,v),stretch(B08,v),stretch(B02,v),idx(ratio(B08,B12)),idx(ratio(B08,B04)),idx(ratio(B03,B08))],0)
def _sab(ch,a,b): return np.clip((ch-a)/max(b-a,1e-6),0,1).astype(np.float32)
def s6_joint(pre,post,v):
    """전/후 raw밴드(B12,B08,B02)를 '공동 백분위'로 정규화 → 절대 변화(연소 어두워짐) 보존.
    씬별 stretch가 변화신호를 지우던 결함(②) 수정. 비율채널은 원래 스케일불변."""
    ab={}
    for k in ("B12","B08","B02"):
        x=np.concatenate([pre[k][v],post[k][v]]) if v.any() else np.array([0.,1.])
        ab[k]=np.percentile(x,[2,98]) if x.size else (0,1)
    def b6(bd):
        return np.stack([_sab(bd['B12'],*ab['B12']),_sab(bd['B08'],*ab['B08']),_sab(bd['B02'],*ab['B02']),
                         idx(ratio(bd['B08'],bd['B12'])),idx(ratio(bd['B08'],bd['B04'])),idx(ratio(bd['B03'],bd['B08']))],0)
    return b6(pre),b6(post)

def stac_bands(bbox_ll, dt_range, ute):
    its=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=dt_range,
             query={"eo:cloud_cover":{"lt":30}},sortby=[{"field":"properties.eo:cloud_cover","direction":"asc"}],max_items=1).items())
    if not its: return None
    d=its[0].datetime.date()
    sib=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=rfc(d-dt.timedelta(days=2),d+dt.timedelta(days=2)),query={"eo:cloud_cover":{"lt":40}},max_items=10).items()) or its
    b={}
    for k,a in [("B02","blue"),("B03","green"),("B04","red"),("B08","nir"),("B11","swir16"),("B12","swir22"),("SCL","scl")]:
        hrefs=[(x.assets[a].href if a in x.assets else x.assets[k].href) for x in sib]
        gdal.Warp(f"/tmp/_km_{k}.tif",[f"/vsicurl/{h}" for h in hrefs],dstSRS=f"EPSG:{UTM}",xRes=20,yRes=20,outputBounds=ute,resampleAlg="average",format="GTiff",srcNodata=0)
        b[k]=gdal.Open(f"/tmp/_km_{k}.tif").ReadAsArray().astype(np.float32)
    return b

def build_one(name, gu, fdate):
    # 화재 중심으로 최소 5.3km(≥256px@20m) 크롭 강제 → 소형 화재도 1타일 확보(불탄 blob+주변 맥락).
    e=gu.GetEnvelope(); cx=(e[0]+e[1])/2; cy=(e[2]+e[3])/2
    hx=max((e[1]-e[0])/2+500,CROP_M); hy=max((e[3]-e[2])/2+500,CROP_M)
    ute=[math.floor(cx-hx),math.floor(cy-hy),math.ceil(cx+hx),math.ceil(cy+hy)]
    dst=osr.SpatialReference(); dst.ImportFromEPSG(4326); dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src=osr.SpatialReference(); src.ImportFromEPSG(UTM); src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(src,dst)
    lo1,la1,_=ct.TransformPoint(e[0],e[2]); lo2,la2,_=ct.TransformPoint(e[1],e[3])
    bbox_ll=[min(lo1,lo2)-0.02,min(la1,la2)-0.02,max(lo1,lo2)+0.02,max(la1,la2)+0.02]
    pre_center=fdate.replace(year=fdate.year-1)
    pre=stac_bands(bbox_ll,rfc(pre_center-dt.timedelta(days=30),pre_center+dt.timedelta(days=45)),ute)
    post=stac_bands(bbox_ll,rfc(fdate+dt.timedelta(days=15),fdate+dt.timedelta(days=100)),ute)
    if pre is None or post is None: print(f"[{name}] 장면없음 pre={bool(pre)} post={bool(post)}",flush=True); return 0
    H=min(pre['B02'].shape[0],post['B02'].shape[0]); W=min(pre['B02'].shape[1],post['B02'].shape[1])
    for d in (pre,post):
        for k in d: d[k]=d[k][:H,:W]
    mem=ogr.GetDriverByName("Memory").CreateDataSource("m"); srs=osr.SpatialReference(); srs.ImportFromEPSG(UTM)
    lyr=mem.CreateLayer("f",srs,ogr.wkbPolygon); ft=ogr.Feature(lyr.GetLayerDefn()); ft.SetGeometry(gu); lyr.CreateFeature(ft)
    rast=gdal.GetDriverByName("MEM").Create("",W,H,1,gdal.GDT_Byte); rast.SetGeoTransform((ute[0],20,0,ute[3],0,-20)); rast.SetProjection(srs.ExportToWkt())
    gdal.RasterizeLayer(rast,[1],lyr,burn_values=[1]); mask=rast.ReadAsArray()[:H,:W].astype(np.uint8)
    valid=(pre['B02']>0)&(pre['B08']>0)&(pre['B12']>0)&(post['B02']>0)&(post['B08']>0)&(post['B12']>0)
    valid&=~np.isin(pre['SCL'],[0,1,3,8,9,10,11])&~np.isin(post['SCL'],[0,1,3,8,9,10,11])
    if os.environ.get("MT_CLOUD","1")=="1":  # 다시점 구름·그림자 보강(번-안전) — Baetens+2019
        from mt_cloud import mt_cloud_shadow_bad; valid&=~mt_cloud_shadow_bad(pre,post)
    if JOINT_NORM:  # ② 공동 정규화
        pre6,post6=s6_joint(pre,post,valid)
    else:
        pre6=s6(pre['B02'],pre['B03'],pre['B04'],pre['B08'],pre['B11'],pre['B12'],valid)
        post6=s6(post['B02'],post['B03'],post['B04'],post['B08'],post['B11'],post['B12'],valid)
    dnbr=np.clip(ratio(pre['B08'],pre['B12'])-ratio(post['B08'],post['B12']),-1,1)
    x=np.concatenate([pre6,post6,idx(dnbr)[None]],0).astype(np.float32); y=(mask>0).astype(np.uint8)
    od=os.path.join(OUT,name); os.makedirs(od,exist_ok=True); T=256; n=0
    if H<T or W<T:
        print(f"[{name}] {fdate} | 타일 0 (크롭<{T}px)",flush=True); os.rmdir(od); return 0
    ys=sorted(set(min(yy,H-T) for yy in range(0,H-T+1,STRIDE)))  # ① 중첩 타일(stride<T)
    xs=sorted(set(min(xx,W-T) for xx in range(0,W-T+1,STRIDE)))
    for yy2 in ys:
        for xx2 in xs:
            v=valid[yy2:yy2+T,xx2:xx2+T]
            if v.mean()<0.4: continue
            np.savez_compressed(os.path.join(od,f"{name}_{yy2}_{xx2}.npz"),x=x[:,yy2:yy2+T,xx2:xx2+T],y=(y[yy2:yy2+T,xx2:xx2+T]&v).astype(np.uint8)); n+=1
    if n==0: os.rmdir(od)
    print(f"[{name}] {fdate} | 타일 {n} | 실측 불탄 {y[valid].mean()*100:.1f}%",flush=True); return n

def main():
    os.makedirs(OUT,exist_ok=True)
    ds=ogr.Open(SHP); ly=ds.GetLayer()
    cand=[]
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
    cand.sort(key=lambda c: c[0], reverse=True)
    print(f"후보 {len(cand)}건, 상위 {N_FIRES}건 빌드 (MIN_HA={MIN_HA})",flush=True)
    tot=0; built=0
    for i,(ha,fdate,gu) in enumerate(cand[:N_FIRES]):
        name=f"kf_{fdate.isoformat().replace('-','')}_{int(ha)}ha"
        import glob as _g
        if _g.glob(os.path.join(OUT,name,"*.npz")):  # 이미 빌드됨 → 건너뛰기(재다운로드 방지)
            print(f"[{name}] 이미 존재 → skip",flush=True); built+=1; continue
        try:
            n=build_one(name,gu,fdate); tot+=n; built+=(n>0)
        except Exception as ex: print(f"[{name}] 실패 {ex}",flush=True)
        if (i+1)%5==0: print(f"  진행 {i+1}/{min(N_FIRES,len(cand))} | 누적 타일 {tot} 화재 {built}",flush=True)
    import glob
    print(f"완성 화재 {built}개 | 총 타일 {len(glob.glob(os.path.join(OUT,'*','*.npz')))}",flush=True)

if __name__=="__main__": main()
