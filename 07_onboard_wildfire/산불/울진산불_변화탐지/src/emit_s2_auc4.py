"""단일시점 post-NBR 소각탐지 AUC: EMIT vs S2, 한국 봄화재 4개(n=4).
산청·의성(기존) + 안동길안·홍성서부(신규). 안동=의성과 동일 EMIT 씬.
심각도 의존적 하이퍼스펙트럴 우위 법칙을 n=4로 검증.
"""
import os, json, numpy as np, h5py, rasterio
from rasterio.enums import Resampling
from pystac_client import Client
from pyproj import Transformer
from sklearn.metrics import roc_auc_score
PROJ=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
CIRD="/home/syntax_kmu/syntax_inference/Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires"
cat=Client.open("https://earth-search.aws.element84.com/v1"); NIR_B,SWIR_B=64,244
E_UI=PROJ+"/data/emit_uiseong/EMIT_L2A_RFL_001_20250424T011718_2511401_011.nc"
FIRES={
 "산청": dict(cir="250321_산청시천면", emit=PROJ+"/data/emit_sancheong/EMIT_L2A_RFL_001_20250406T010521_2509601_003.nc", s2="2025-04-01/2025-04-12", ha=1127),
 "의성": dict(cir="250322_의성안평면", emit=E_UI, s2="2025-04-20/2025-04-30", ha=8981),
 "안동": dict(cir="250324_안동길안면", emit=E_UI, s2="2025-04-20/2025-04-30", ha=1714),
 "홍성": dict(cir="230402_홍성서부면", emit=PROJ+"/data/emit_hongseong/EMIT_L2A_RFL_001_20230428T011728_2311801_004.nc", s2="2023-04-22/2023-05-05", ha=638),
}
def auc(score,pos):
    m=np.isfinite(score); s=score[m]; p=pos[m]
    if p.sum()<20 or (~p).sum()<20: return float("nan"),int(p.sum())
    return roc_auc_score(p,s), int(p.sum())
def s2post(bbox,dr,b,s):
    its=list(cat.search(collections=["sentinel-2-l2a"],bbox=bbox,datetime=dr,query={"eo:cloud_cover":{"lt":25}}).items())
    it=sorted(its,key=lambda x:x.properties["eo:cloud_cover"])[0]
    def rd(h):
        with rasterio.open(h) as ds:
            w=rasterio.windows.from_bounds(*b,transform=ds.transform); return ds.read(1,window=w,out_shape=s,resampling=Resampling.bilinear).astype(np.float32)/1e4
    nir=rd(it.assets["nir"].href); sw=rd(it.assets["swir22"].href); v=(nir>.001)&(sw>.001); x=(nir-sw)/(nir+sw+1e-6); x[~v]=np.nan; return x,it.datetime.date()
def empost(path,b,s,tr):
    h=h5py.File(path,"r"); r=h["reflectance"]; nir=r[:,:,NIR_B].astype(np.float32); sw=r[:,:,SWIR_B].astype(np.float32)
    bad=(nir<=-.01)|(sw<=-.01); nb=(nir-sw)/(nir+sw+1e-6); nb[bad]=np.nan
    gx=h["location/glt_x"][:]; gy=h["location/glt_y"][:]; gt=h.attrs["geotransform"]; h.close()
    Ho,Wo=gx.shape; o=np.full((Ho,Wo),np.nan,np.float32); m=(gx>0)&(gy>0); o[m]=nb[gy[m]-1,gx[m]-1]
    lon0,rx,_,lat0,_,ry=[float(v) for v in gt[:6]]; xmin,ymin,xmax,ymax=b; H,W=s
    rr,cc=np.mgrid[0:H,0:W]; ux=xmin+(cc+.5)*(xmax-xmin)/W; uy=ymax-(rr+.5)*(ymax-ymin)/H
    lon,lat=tr.transform(ux.ravel(),uy.ravel()); ec=np.round((lon-lon0)/rx-.5).astype(int); er=np.round((lat-lat0)/ry-.5).astype(int)
    ok=(ec>=0)&(ec<Wo)&(er>=0)&(er<Ho); out=np.full(H*W,np.nan,np.float32); out[ok]=o[er[ok],ec[ok]]; return out.reshape(H,W)
res={}
print(f"{'화재':6s} {'ha':>6} {'S2 AUC':>8} {'EMIT AUC':>9} {'Δ':>7}  {'소각px(EMIT유효)'}")
for n,c in FIRES.items():
    d=np.load(os.path.join(CIRD,c["cir"],"gt_stac.npz")); gt=d["gt"].astype(bool); H,W=gt.shape
    xmin,ymin,xmax,ymax=[float(v) for v in d["ute"]]; b=(xmin,ymin,xmax,ymax)
    tr=Transformer.from_crs(f"EPSG:{int(d['utm'])}","EPSG:4326",always_xy=True)
    clon,clat=tr.transform((xmin+xmax)/2,(ymin+ymax)/2); bbox=[clon-.15,clat-.12,clon+.15,clat+.12]
    try:
        s2,_=s2post(bbox,c["s2"],b,(H,W)); em=empost(c["emit"],b,(H,W),tr)
    except Exception as e:
        print(f"{n}: 실패 {e}"); continue
    a_s2,_=auc(-s2,gt); a_em,ne=auc(-em,gt)
    res[n]=dict(ha=c["ha"],auc_s2=round(a_s2,3),auc_emit=round(a_em,3),emit_burn_px=ne)
    print(f"{n:6s} {c['ha']:6d} {a_s2:8.3f} {a_em:9.3f} {a_em-a_s2:+7.3f}  {ne}")
json.dump(res,open(os.path.join(PROJ,"runs","emit_s2_auc4.json"),"w"),ensure_ascii=False,indent=1)
print("\n저장: runs/emit_s2_auc4.json")
print("법칙: 저심각도(작은/미묘)=EMIT 우위 크게, 고심각도=둘다 높음(S2 해상도 우세)")
