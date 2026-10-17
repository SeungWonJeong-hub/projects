"""시각 증거: 산청·의성 각각 S2 post-NBR | EMIT post-NBR | CIR-GT 소각 오버레이.
산청=S2 식생만/EMIT char 흉터(AUC 0.53 vs 0.86), 의성=둘다 흉터(0.93 vs 0.88).
"""
import os, numpy as np, h5py, rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from pystac_client import Client
from pyproj import Transformer
PROJ=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
CIRD="/home/syntax_kmu/syntax_inference/Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires"
cat=Client.open("https://earth-search.aws.element84.com/v1"); NIR_B,SWIR_B=64,244
F={"Sancheong (low-sev)":dict(cir="250321_산청시천면",emit=PROJ+"/data/emit_sancheong/EMIT_L2A_RFL_001_20250406T010521_2509601_003.nc",s2="2025-04-01/2025-04-12",auc="S2 0.53 / EMIT 0.86"),
   "Uiseong (high-sev)":dict(cir="250322_의성안평면",emit=PROJ+"/data/emit_uiseong/EMIT_L2A_RFL_001_20250424T011718_2511401_011.nc",s2="2025-04-20/2025-04-30",auc="S2 0.93 / EMIT 0.88")}
def s2nbr(bbox,dr,b,s):
    it=sorted(cat.search(collections=["sentinel-2-l2a"],bbox=bbox,datetime=dr,query={"eo:cloud_cover":{"lt":20}}).items(),key=lambda x:x.properties["eo:cloud_cover"])[0]
    def rd(h):
        with rasterio.open(h) as ds:
            w=rasterio.windows.from_bounds(*b,transform=ds.transform); return ds.read(1,window=w,out_shape=s,resampling=Resampling.bilinear).astype(np.float32)/1e4
    nir=rd(it.assets["nir"].href); sw=rd(it.assets["swir22"].href); v=(nir>.001)&(sw>.001); x=(nir-sw)/(nir+sw+1e-6); x[~v]=np.nan; return x
def emnbr(path,b,s,tr):
    h=h5py.File(path,"r"); r=h["reflectance"]; nir=r[:,:,NIR_B].astype(np.float32); sw=r[:,:,SWIR_B].astype(np.float32)
    bad=(nir<=-.01)|(sw<=-.01); nb=(nir-sw)/(nir+sw+1e-6); nb[bad]=np.nan
    gx=h["location/glt_x"][:]; gy=h["location/glt_y"][:]; gt=h.attrs["geotransform"]; h.close()
    Ho,Wo=gx.shape; o=np.full((Ho,Wo),np.nan,np.float32); m=(gx>0)&(gy>0); o[m]=nb[gy[m]-1,gx[m]-1]
    lon0,rx,_,lat0,_,ry=[float(v) for v in gt[:6]]; xmin,ymin,xmax,ymax=b; H,W=s
    rr,cc=np.mgrid[0:H,0:W]; ux=xmin+(cc+.5)*(xmax-xmin)/W; uy=ymax-(rr+.5)*(ymax-ymin)/H
    lon,lat=tr.transform(ux.ravel(),uy.ravel()); ec=np.round((lon-lon0)/rx-.5).astype(int); er=np.round((lat-lat0)/ry-.5).astype(int)
    ok=(ec>=0)&(ec<Wo)&(er>=0)&(er<Ho); out=np.full(H*W,np.nan,np.float32); out[ok]=o[er[ok],ec[ok]]; return out.reshape(H,W)
fig,axes=plt.subplots(2,3,figsize=(14,9))
for i,(name,c) in enumerate(F.items()):
    d=np.load(os.path.join(CIRD,c["cir"],"gt_stac.npz")); gt=d["gt"].astype(bool); H,W=gt.shape
    xmin,ymin,xmax,ymax=[float(v) for v in d["ute"]]; b=(xmin,ymin,xmax,ymax)
    tr=Transformer.from_crs(f"EPSG:{int(d['utm'])}","EPSG:4326",always_xy=True)
    clon,clat=tr.transform((xmin+xmax)/2,(ymin+ymax)/2); bbox=[clon-.15,clat-.12,clon+.15,clat+.12]
    s2=s2nbr(bbox,c["s2"],b,(H,W)); em=emnbr(c["emit"],b,(H,W),tr)
    a=axes[i]
    a[0].imshow(s2,cmap="RdYlGn",vmin=-.2,vmax=.6); a[0].set_title(f"{name}\nS2 post-NBR (20m)")
    a[1].imshow(em,cmap="RdYlGn",vmin=-.2,vmax=.6); a[1].set_title(f"EMIT post-NBR (60m native SWIR)\nAUC {c['auc']}")
    a[2].imshow(gt,cmap="hot"); a[2].set_title("CIR-GT burn polygon")
    for ax in a: ax.axis("off")
plt.tight_layout(); p=os.path.join(PROJ,"runs","emit_s2_compare.png"); plt.savefig(p,dpi=95); print("저장:",p)
