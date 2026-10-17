"""단일시점 post-NBR 소각탐지 AUC: EMIT vs S2, 산청+의성 (cross-year 교란 없음, 지오 검증됨).
score = -post_NBR (char 낮음=소각). AUC = P(소각 score > 비소각 score) [Mann-Whitney].
EMIT AUC > S2 AUC 면 native SWIR가 더 잘 분리 = 하이퍼스펙트럴 우위 정량.
"""
import os, json, numpy as np, h5py, rasterio
from rasterio.enums import Resampling
from pystac_client import Client
from pyproj import Transformer
PROJ=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
CIRD="/home/syntax_kmu/syntax_inference/Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires"
cat=Client.open("https://earth-search.aws.element84.com/v1")
NIR_B,SWIR_B=64,244
FIRES={
 "산청": dict(cir="250321_산청시천면", emit=os.path.join(PROJ,"data/emit_sancheong/EMIT_L2A_RFL_001_20250406T010521_2509601_003.nc"), s2post="2025-04-01/2025-04-12"),
 "의성": dict(cir="250322_의성안평면", emit=os.path.join(PROJ,"data/emit_uiseong/EMIT_L2A_RFL_001_20250424T011718_2511401_011.nc"), s2post="2025-04-20/2025-04-30"),
}
def auc(score, pos):
    s=score; m=np.isfinite(s); s=s[m]; p=pos[m]
    if p.sum()<20 or (~p).sum()<20: return float("nan")
    r=np.argsort(np.argsort(s)); n1=p.sum(); n0=(~p).sum()
    return (r[p].sum()-n1*(n1-1)/2)/(n1*n0)
def s2_postnbr(bbox,dr,bounds,shape):
    its=list(cat.search(collections=["sentinel-2-l2a"],bbox=bbox,datetime=dr,query={"eo:cloud_cover":{"lt":20}}).items())
    it=sorted(its,key=lambda x:x.properties["eo:cloud_cover"])[0]
    def rd(h):
        with rasterio.open(h) as ds:
            w=rasterio.windows.from_bounds(*bounds,transform=ds.transform)
            return ds.read(1,window=w,out_shape=shape,resampling=Resampling.bilinear).astype(np.float32)/1e4
    nir=rd(it.assets["nir"].href); sw=rd(it.assets["swir22"].href)
    v=(nir>0.001)&(sw>0.001); x=(nir-sw)/(nir+sw+1e-6); x[~v]=np.nan; return x, it.datetime.date()
def emit_postnbr(path,bounds,shape,tr):
    h=h5py.File(path,"r"); refl=h["reflectance"]
    nir=refl[:,:,NIR_B].astype(np.float32); sw=refl[:,:,SWIR_B].astype(np.float32)
    bad=(nir<=-0.01)|(sw<=-0.01); nbr=(nir-sw)/(nir+sw+1e-6); nbr[bad]=np.nan
    gx=h["location/glt_x"][:]; gy=h["location/glt_y"][:]; gt=h.attrs["geotransform"]; h.close()
    Ho,Wo=gx.shape; o=np.full((Ho,Wo),np.nan,np.float32); m=(gx>0)&(gy>0); o[m]=nbr[gy[m]-1,gx[m]-1]
    lon0,rx,_,lat0,_,ry=[float(v) for v in gt[:6]]
    xmin,ymin,xmax,ymax=bounds; H,W=shape
    rows,cols=np.mgrid[0:H,0:W]
    ux=xmin+(cols+0.5)*(xmax-xmin)/W; uy=ymax-(rows+0.5)*(ymax-ymin)/H
    lon,lat=tr.transform(ux.ravel(),uy.ravel());
    ec=np.round((lon-lon0)/rx-0.5).astype(int); er=np.round((lat-lat0)/ry-0.5).astype(int)
    ok=(ec>=0)&(ec<Wo)&(er>=0)&(er<Ho); out=np.full(H*W,np.nan,np.float32); out[ok]=o[er[ok],ec[ok]]
    return out.reshape(H,W)
res={}
for n,c in FIRES.items():
    d=np.load(os.path.join(CIRD,c["cir"],"gt_stac.npz")); gt=d["gt"].astype(bool); H,W=gt.shape
    xmin,ymin,xmax,ymax=[float(v) for v in d["ute"]]; bounds=(xmin,ymin,xmax,ymax)
    tr=Transformer.from_crs(f"EPSG:{int(d['utm'])}","EPSG:4326",always_xy=True)
    clon,clat=tr.transform((xmin+xmax)/2,(ymin+ymax)/2); bbox=[clon-0.15,clat-0.12,clon+0.15,clat+0.12]
    s2,s2d=s2_postnbr(bbox,c["s2post"],bounds,(H,W))
    em=emit_postnbr(c["emit"],bounds,(H,W),tr)
    a_s2=auc(-s2,gt); a_em=auc(-em,gt)   # 낮은 NBR=소각 → score=-NBR
    print(f"{n} (소각 {int(gt.sum())}px): S2 post {s2d}")
    print(f"   소각탐지 AUC(단일시점 post-NBR):  S2 {a_s2:.3f}  |  EMIT {a_em:.3f}  → {'EMIT 우위 +'+format(a_em-a_s2,'.3f') if a_em>a_s2 else 'S2 우위'}")
    res[n]=dict(n_burn=int(gt.sum()),auc_s2=round(a_s2,3),auc_emit=round(a_em,3),s2_post=str(s2d))
print("\n=== 요약 ===")
for n,r in res.items(): print(f"  {n}: AUC S2 {r['auc_s2']} vs EMIT {r['auc_emit']}  (Δ{r['auc_emit']-r['auc_s2']:+.3f})")
json.dump(res,open(os.path.join(PROJ,"runs","emit_s2_auc.json"),"w"),ensure_ascii=False,indent=1)
print("저장: runs/emit_s2_auc.json")
