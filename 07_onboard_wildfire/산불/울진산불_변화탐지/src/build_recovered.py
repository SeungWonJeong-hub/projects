"""복구 화재의 학습타일(x 8ch + gt + valid) 빌드 → 각 화재 dir에 gt_stac_rb.npz.
재브래킷과 동일 브래킷(화재 직전 pre + 직후 post)으로 S2 8ch(RGB+NIR pre/post) + dNBR 소각.
학습에 넣을 수 있게 gt_stac 포맷과 동일 키(x,gt,valid,utm,ute).
"""
import os, glob, json, datetime as dt, numpy as np, rasterio
from rasterio.enums import Resampling
from pystac_client import Client
from pyproj import Transformer
from scipy import ndimage
PROJ=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
CIRD="/home/syntax_kmu/syntax_inference/Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires"
RB=os.path.join(PROJ,"runs","rebracket")
cat=Client.open("https://earth-search.aws.element84.com/v1"); THR=0.15
# ★10m 지원: REC_SCALE=2 → 20m 복구 burn을 2× 업샘플 + 밴드를 10m로 취득. 출력명 분리(20m 미덮어씀).
SCALE=int(os.environ.get("REC_SCALE","1"))
RB_NPZ=os.environ.get("RB_NPZ","gt_stac_rb.npz")
def pick(clon,clat,t0,t1):
    def cov(it): x0,y0,x1,y1=it.bbox; return x0<=clon<=x1 and y0<=clat<=y1
    its=[it for it in cat.search(collections=["sentinel-2-l2a"],bbox=[clon-.01,clat-.01,clon+.01,clat+.01],datetime=f"{t0}/{t1}",query={"eo:cloud_cover":{"lt":30}}).items() if cov(it)]
    return sorted(its,key=lambda x:x.properties["eo:cloud_cover"])[0] if its else None
def bands(it,b,shape):
    def rd(a):
        with rasterio.open(it.assets[a].href) as ds:
            w=rasterio.windows.from_bounds(*b,transform=ds.transform); return ds.read(1,window=w,out_shape=shape,resampling=Resampling.bilinear).astype(np.float32)/1e4
    return np.stack([rd("blue"),rd("green"),rd("red"),rd("nir")],0)  # B02,B03,B04,B08
built=0
for f in sorted(glob.glob(os.path.join(RB,"*.npz"))):
    n=os.path.splitext(os.path.basename(f))[0]
    z0=np.load(f); ute=[float(v) for v in z0["ute"]]; epsg=int(z0["utm"]); H0,W0=z0["burn"].shape
    H,W=H0*SCALE,W0*SCALE   # SCALE=2 → 10m
    fdate=dt.date(2000+int(n[:2]),int(n[2:4]),int(n[4:6]))
    tr=Transformer.from_crs(f"EPSG:{epsg}","EPSG:4326",always_xy=True); clon,clat=tr.transform((ute[0]+ute[2])/2,(ute[1]+ute[3])/2)
    b=(ute[0],ute[1],ute[2],ute[3])
    pre=pick(clon,clat,(fdate-dt.timedelta(days=35)).isoformat(),(fdate-dt.timedelta(days=1)).isoformat())
    post=pick(clon,clat,(fdate+dt.timedelta(days=7)).isoformat(),(fdate+dt.timedelta(days=40)).isoformat())
    if not pre or not post: print(f"{n}: skip"); continue
    xp=bands(pre,b,(H,W)); xq=bands(post,b,(H,W)); x=np.concatenate([xp,xq],0)  # 8ch @ SCALE res
    valid=np.all(np.isfinite(x),0)&(xp[3]>0.001)&(xq[3]>0.001)
    burn=z0["burn"].astype(bool)
    if SCALE>1: burn=np.repeat(np.repeat(burn,SCALE,0),SCALE,1)[:H,:W]  # 20m 라벨 nearest 업샘플
    gt=burn&valid
    out=os.path.join(CIRD,n,RB_NPZ)
    np.savez_compressed(out,x=x.astype(np.float32),gt=gt,valid=valid,utm=epsg,ute=z0["ute"])
    built+=1; print(f"{n}: x{x.shape} 소각 {int(gt.sum())}px → {os.path.basename(out)}")
print(f"\n빌드 {built}개 {RB_NPZ}")
