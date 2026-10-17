"""산불 재빌드 — 원시 반사율 6밴드(stretch/지수 없음).
[B02,B03,B04,<NIR>,B11,B12] 반사율(×10000)을 pre/post로 저장.
같은 MTfire 화재·같은 크롭 로직 → 고정홀드아웃 공정비교 유지. 라벨 y=실측 경계.

x = (12,T,T) int16 반사율 [pre 6밴드, post 6밴드],  y=(T,T) uint8

env (전부 기본값=기존 20m 베이스라인과 완전 동일):
  GSD=20|10        격자 해상도(m). 10이면 B02/B03/B04/B08이 네이티브, SWIR은 업샘플.
  NIR=B8A|B08      NIR 밴드. B8A=865nm 네이티브 20m(Prithvi 정합), B08=842nm 네이티브 10m.
  NO_OFFSET=0|1    1이면 ≥2022 씬의 -1000 오프셋을 빼지 않음(raw DN).
                   ※earth-search COG는 이미 harmonized라 raw DN이 옳음. 배포 ET-02(F1 0.828)가
                     raw DN 스케일로 학습됐는데 그 재빌드는 커밋되지 않았었다 → 여기서 플래그로 복원.
  T=256|512        타일 픽셀 수. **지상 시야 = T*GSD** — 10m에서 T=256이면 시야가 20m의
                   절반(2.56km vs 5.12km)이 되어 해상도 A/B에 문맥 변수가 섞인다.
                   해상도만 비교하려면 10m는 T=512로 시야를 맞춘다.
  KO_OUT=<dir>     출력 디렉터리(기본: 20m+B8A면 ../ko_refl, 그 외 ../ko_refl_<gsd>m_<nir>[_t<T>])

사용: 20m 베이스라인   python ko_build_refl.py [N_FIRES]
      10m ET-02용      GSD=10 NIR=B08 NO_OFFSET=1 python ko_build_refl.py [N_FIRES]
"""
from __future__ import annotations
import os, sys, math, datetime as dt, glob
import numpy as np
from osgeo import gdal, ogr, osr
from pystac_client import Client
import socket
gdal.UseExceptions()
# 네트워크 타임아웃 — /vsicurl COG 다운로드가 무한 poll로 hang나던 것 방지(→ 실패시 skip)
socket.setdefaulttimeout(90)   # pystac_client STAC 검색 backstop
for _k,_v in {"GDAL_HTTP_TIMEOUT":"40","GDAL_HTTP_CONNECTTIMEOUT":"15","GDAL_HTTP_MAX_RETRY":"3",
              "GDAL_HTTP_RETRY_DELAY":"3","CPL_VSIL_CURL_USE_HEAD":"NO",
              "GDAL_DISABLE_READDIR_ON_OPEN":"EMPTY_DIR",
              "AWS_NO_SIGN_REQUEST":"YES"}.items():      # 공개 S3 버킷 무서명 접근(/vsis3/)
    gdal.SetConfigOption(_k,_v)
HERE=os.path.dirname(os.path.abspath(__file__))
SHP=os.path.join(HERE,"..","..","산불데이터","korea","mtfire","MTfire.shp")  # 재정리로 공백없는 이름
GSD=float(os.environ.get("GSD","20"))
NIR=os.environ.get("NIR","B8A")                       # B8A=네이티브20m / B08=네이티브10m
NIR_ASSET={"B8A":"nir08","B08":"nir"}[NIR]            # earth-search asset key
NO_OFFSET=os.environ.get("NO_OFFSET","0")=="1"        # 1=raw DN(오프셋 미차감)
T=int(os.environ.get("T","256"))                      # 타일 픽셀. 지상 시야 = T*GSD (m)
_DEF_OUT="ko_refl" if (GSD==20 and NIR=="B8A") else f"ko_refl_{int(GSD)}m_{NIR.lower()}"
if T!=256: _DEF_OUT+=f"_t{T}"
OUT=os.environ.get("KO_OUT",os.path.join(HERE,"..",_DEF_OUT))
CAT=Client.open("https://earth-search.aws.element84.com/v1")
UTM=32652
N_FIRES=int(sys.argv[1]) if len(sys.argv)>1 else 50
MIN_HA=float(os.environ.get("MIN_HA","600")); MAX_HA=float(os.environ.get("MAX_HA","8000"))
YEARS=set(os.environ.get("YEARS","2016,2017,2018,2019,2020,2021,2022,2023,2024").split(","))
CROP_M=float(os.environ.get("CROP_M","2650"))
STRIDE=int(os.environ.get("STRIDE",str(T)))            # 기본 = 타일크기(비중첩)
SHARD_N=int(os.environ.get("SHARD_N","1")); SHARD_I=int(os.environ.get("SHARD_I","0"))  # 병렬 빌드 분할
# 6밴드 ← S2: blue,green,red,<NIR>,swir16(B11),swir22(B12)
BANDS=[("B02","blue"),("B03","green"),("B04","red"),(NIR,NIR_ASSET),("B11","swir16"),("B12","swir22")]
ORDER=["B02","B03","B04",NIR,"B11","B12"]

def rfc(a,b): return f"{a}T00:00:00Z/{b}T23:59:59Z"

def vsi(u):
    """STAC이 s3:// href를 주면 /vsicurl은 못 읽는다(Protocol "s3" not supported).
    https로 바꾸면 버킷 리전을 모를 때 301(PermanentRedirect)이 난다 → /vsis3/로 넘겨
    GDAL이 리전을 자동 해석하게 한다(공개 버킷이므로 무서명)."""
    if u.startswith("s3://"):
        return "/vsis3/"+u[5:]
    return "/vsicurl/"+u

def stac_refl(bbox_ll, dt_range, ute):
    """최저구름 장면 취득 → 반사율(×10000, ≥2022 offset 보정) 6밴드 + SCL + 날짜."""
    its=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=dt_range,
             query={"eo:cloud_cover":{"lt":30}},sortby=[{"field":"properties.eo:cloud_cover","direction":"asc"}],max_items=1).items())
    if not its: return None
    d=its[0].datetime.date()
    sib=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=rfc(d-dt.timedelta(days=2),d+dt.timedelta(days=2)),
             query={"eo:cloud_cover":{"lt":40}},max_items=10).items()) or its
    off=0.0 if NO_OFFSET else (1000.0 if d>=dt.date(2022,1,25) else 0.0)
    b={}
    for k,a in BANDS+[("SCL","scl")]:
        hrefs=[(x.assets[a].href if a in x.assets else x.assets.get(k).href) for x in sib if (a in x.assets or k in x.assets)]
        if not hrefs: return None
        tmp=f"/tmp/_kr_{os.getpid()}_{k}.tif"      # pid 포함: 병렬 빌드시 덮어쓰기 방지
        gdal.Warp(tmp,[vsi(h) for h in hrefs],dstSRS=f"EPSG:{UTM}",xRes=GSD,yRes=GSD,
                  outputBounds=ute,resampleAlg="average",format="GTiff",srcNodata=0)
        arr=gdal.Open(tmp).ReadAsArray().astype(np.float32)
        os.remove(tmp)
        if k!="SCL": arr=np.clip(arr-off,0,None)      # 반사율 offset 보정
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
    pre_src="prevyear"
    if pre is None:   # 전년도 pre 자료공백(2015-2016) → 같은 해 화재 직전으로 fallback (2016-2017 unlock)
        pre=stac_refl(bbox_ll,rfc(fdate-dt.timedelta(days=60),fdate-dt.timedelta(days=8)),ute); pre_src="sameyear"
    post=stac_refl(bbox_ll,rfc(fdate+dt.timedelta(days=15),fdate+dt.timedelta(days=100)),ute)
    if pre is None or post is None: print(f"[{name}] 장면없음(pre_src={pre_src})",flush=True); return 0
    H=min(pre['B02'].shape[0],post['B02'].shape[0]); W=min(pre['B02'].shape[1],post['B02'].shape[1])
    for d in (pre,post):
        for k in d: d[k]=d[k][:H,:W]
    srs=osr.SpatialReference(); srs.ImportFromEPSG(UTM)
    mem=ogr.GetDriverByName("Memory").CreateDataSource("m"); lyr=mem.CreateLayer("f",srs,ogr.wkbPolygon)
    ft=ogr.Feature(lyr.GetLayerDefn()); ft.SetGeometry(gu); lyr.CreateFeature(ft)
    rast=gdal.GetDriverByName("MEM").Create("",W,H,1,gdal.GDT_Byte); rast.SetGeoTransform((ute[0],GSD,0,ute[3],0,-GSD)); rast.SetProjection(srs.ExportToWkt())
    gdal.RasterizeLayer(rast,[1],lyr,burn_values=[1]); mask=rast.ReadAsArray()[:H,:W].astype(np.uint8)
    valid=(pre['B02']>0)&(pre[NIR]>0)&(pre['B12']>0)&(post['B02']>0)&(post[NIR]>0)&(post['B12']>0)
    valid&=~np.isin(pre['SCL'],[0,1,3,8,9,10,11])&~np.isin(post['SCL'],[0,1,3,8,9,10,11])
    x=np.stack([pre[k] for k in ORDER]+[post[k] for k in ORDER],0).astype(np.int16)  # 12ch 반사율
    y=(mask>0).astype(np.uint8)
    od=os.path.join(OUT,name); os.makedirs(od,exist_ok=True); n=0
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
    sel=cand[:N_FIRES][SHARD_I::SHARD_N]          # 샤딩: 여러 프로세스가 겹치지 않게 분할
    print(f"후보 {len(cand)}, 상위 {N_FIRES} 중 샤드 {SHARD_I}/{SHARD_N} = {len(sel)}건 빌드",flush=True); tot=built=0
    for i,(ha,fdate,gu) in enumerate(sel):
        name=f"kf_{fdate.isoformat().replace('-','')}_{int(ha)}ha"
        if glob.glob(os.path.join(OUT,name,"*.npz")): print(f"[{name}] skip",flush=True); built+=1; continue
        try: n=build_one(name,gu,fdate); tot+=n; built+=(n>0)
        except Exception as ex: print(f"[{name}] 실패 {ex}",flush=True)
        if (i+1)%5==0: print(f"  진행 {i+1}/{len(sel)} 누적타일 {tot}",flush=True)
    print(f"완성 화재 {built} | 총 타일 {len(glob.glob(os.path.join(OUT,'*','*.npz')))}",flush=True)

if __name__=="__main__": main()
