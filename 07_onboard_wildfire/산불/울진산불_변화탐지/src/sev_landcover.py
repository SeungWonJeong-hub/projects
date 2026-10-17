"""심각도 등급 × 토지피복 — 고성·양양의 등급-신호 역전이 '비산림'으로 설명되는가.

왜: 안동·강릉은 등급별 dNBR 이 심>중>경 순인데 고성(0.706<0.744)·양양(0.250<0.297)은 역전이다
(`sev_transfer.gate`). 가설 = "그 GT 구역엔 시가지·농경지가 섞여 등급 축이 산림 피해도라
분광과 어긋난다". 이걸 자료로 확인한다.

자 두 개를 같이 쓴다 — 한쪽만 보면 틀린다:
 ① ESA WorldCover 10m (공개 버킷, 인증 불요) — 시가지/농경지 같은 **이름 있는 분류**를 준다.
    ⚠ 2021년판이라 **화재 이후**다. 심하게 탄 산림이 관목·나대지로 재분류됐을 수 있어
      "심 등급에 비산림이 많다"가 결과인지 원인인지 이것만으론 못 가른다.
 ② 우리 캐시의 **화재 전(pre) 영상 NDVI** — 오염이 없다(화재 전에 찍힌 자료다).
    화재 전에 이미 식생이 없었다면 탈 것이 없었다는 뜻이고, 그게 낮은 dNBR 을 설명한다.

사용: conda run -n ml python sev_landcover.py [--sites andong gangneung goseong yangyang inje]
"""
import os, sys, argparse
import numpy as np
os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
from osgeo import gdal, osr; gdal.UseExceptions()
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN","EMPTY_DIR")

P=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GSD=20.0; PXA=GSD*GSD/1e4
WC="/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif"
NAME={10:"수목",20:"관목",30:"초지",40:"농경지",50:"시가지",60:"나대지",
      70:"설빙",80:"물",90:"습지",95:"맹그로브",100:"이끼"}

def wc_tile(lon,lat):
    """WorldCover 3°×3° 타일 이름 (남서 모서리, 3의 배수)."""
    return f"N{int(np.floor(lat/3)*3):02d}E{int(np.floor(lon/3)*3):03d}"

def load_site(site):
    C=os.path.join(P,"nifos_cache",site+"_doy")
    ute=np.load(os.path.join(C,"ute_20m.npy"))
    sev_p=os.path.join(C,"severity_20m.tif")
    if os.path.exists(sev_p):
        S=gdal.Open(sev_p).ReadAsArray()
    else:                                        # 안동은 기존 파일을 격자에 얹는다
        g=gdal.Open(f"{P}/../nifos_burn_gt/andong/andong_severity_20m.tif")
        gt_=g.GetGeoTransform(); A=g.ReadAsArray()
        gt20=np.load(os.path.join(C,"gt_20m.npy")); H,W=gt20.shape
        ox,oy=int(round((gt_[0]-ute[0])/GSD)),int(round((ute[3]-gt_[3])/GSD))
        S=np.zeros((H,W),A.dtype)
        ys=slice(max(oy,0),min(oy+A.shape[0],H)); xs=slice(max(ox,0),min(ox+A.shape[1],W))
        S[ys,xs]=A[max(-oy,0):max(-oy,0)+(ys.stop-ys.start),max(-ox,0):max(-ox,0)+(xs.stop-xs.start)]
    x=np.load(os.path.join(C,"scene_20m.npz"))["x"].astype("float32")/1e4
    H=min(S.shape[0],x.shape[1]); W=min(S.shape[1],x.shape[2])
    return S[:H,:W], x[:,:H,:W], ute

def landcover_on_grid(ute,shape):
    """WorldCover 를 평가격자(UTM52N 20m)로 최빈 리샘플."""
    utm=osr.SpatialReference(); utm.ImportFromEPSG(32652)
    wgs=osr.SpatialReference(); wgs.ImportFromEPSG(4326)
    for s in (utm,wgs): s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(utm,wgs)
    lon,lat,_=ct.TransformPoint(float(ute[0]),float(ute[1]))
    src=WC.format(wc_tile(lon,lat))
    ds=gdal.Warp("",src,format="MEM",dstSRS="EPSG:32652",xRes=GSD,yRes=GSD,
                 outputBounds=[float(v) for v in ute],resampleAlg="mode")
    a=ds.ReadAsArray()
    return a[:shape[0],:shape[1]]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sites",nargs="+",
                    default=["andong","gangneung","goseong","yangyang","inje"])
    a=ap.parse_args()
    for site in a.sites:
        try:
            S,x,ute=load_site(site)
        except Exception as e:
            print(f"[{site}] 건너뜀: {e}\n"); continue
        lc=landcover_on_grid(ute,S.shape)
        pre,post=x[:6],x[6:]
        f=lambda b,i,j:(b[i]-b[j])/(b[i]+b[j]+1e-6)
        dnbr=f(pre,3,5)-f(post,3,5)
        pre_ndvi=f(pre,3,2)
        print(f"■ {site}")
        print(f"  {'등급':4} {'면적ha':>8}  {'dNBR중앙':>9} {'화재전NDVI중앙':>14}  {'산림%':>6} {'농경%':>6} {'시가%':>6} {'초지·관목%':>10} {'나대지%':>7}")
        for k,lab in ((1,"심"),(2,"중"),(3,"경")):
            m=S==k
            if m.sum()==0: continue
            l=lc[m]
            pct=lambda ks: 100*np.isin(l,ks).mean()
            print(f"  {lab:4} {m.sum()*PXA:8.1f}  {np.median(dnbr[m]):9.3f} {np.median(pre_ndvi[m]):14.3f}"
                  f"  {pct([10]):6.1f} {pct([40]):6.1f} {pct([50]):6.1f} {pct([20,30]):10.1f} {pct([60]):7.1f}")
        m=np.isin(S,[1,2,3])
        l=lc[~m]
        pct=lambda ks: 100*np.isin(l,ks).mean()
        print(f"  {'(GT밖)':4} {(~m).sum()*PXA:8.1f}  {np.median(dnbr[~m]):9.3f} {np.median(pre_ndvi[~m]):14.3f}"
              f"  {pct([10]):6.1f} {pct([40]):6.1f} {pct([50]):6.1f} {pct([20,30]):10.1f} {pct([60]):7.1f}")
        print()

if __name__=="__main__": main()
