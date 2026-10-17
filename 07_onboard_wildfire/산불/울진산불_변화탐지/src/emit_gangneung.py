"""강릉난곡2023(CIR-GT 0ha=S2 dNBR 놓침)에 EMIT char가 뜨는지 = 신규 라벨 가능성.
GT 없음 → EMIT post-NBR(char) vs S2 dNBR(same-year 브래킷) 나란히.
EMIT은 흉터 보이고 S2는 안 보이면 = 하이퍼스펙트럴이 S2 놓친 화재 라벨 → 모델 개선 씨앗.
"""
import os, numpy as np, h5py, rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from pystac_client import Client
from pyproj import Transformer
PROJ=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
CIRD="/home/syntax_kmu/syntax_inference/Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires"
EMIT=PROJ+"/data/emit_gangneung/EMIT_L2A_RFL_001_20230421T034424_2311103_006.nc"
OUT=PROJ+"/runs/emit_gangneung"; os.makedirs(OUT,exist_ok=True)
NIR_B,SWIR_B=64,244
cat=Client.open("https://earth-search.aws.element84.com/v1")
d=np.load(os.path.join(CIRD,"230411_강릉난곡동","gt_stac.npz")); H,W=d["gt"].shape
xmin,ymin,xmax,ymax=[float(v) for v in d["ute"]]; b=(xmin,ymin,xmax,ymax)
tr=Transformer.from_crs(f"EPSG:{int(d['utm'])}","EPSG:4326",always_xy=True)
clon,clat=tr.transform((xmin+xmax)/2,(ymin+ymax)/2); bbox=[clon-.1,clat-.08,clon+.1,clat+.08]
def covers(it):
    x0,y0,x1,y1=it.bbox; return x0<=clon<=x1 and y0<=clat<=y1
def s2pick(dr):
    its=[it for it in cat.search(collections=["sentinel-2-l2a"],bbox=[clon-.01,clat-.01,clon+.01,clat+.01],datetime=dr,query={"eo:cloud_cover":{"lt":30}}).items() if covers(it)]
    return sorted(its,key=lambda x:x.properties["eo:cloud_cover"])[0]
def s2nbr(it):
    def rd(h):
        with rasterio.open(h) as ds:
            w=rasterio.windows.from_bounds(*b,transform=ds.transform); return ds.read(1,window=w,out_shape=(H,W),resampling=Resampling.bilinear).astype(np.float32)/1e4
    nir=rd(it.assets["nir"].href); sw=rd(it.assets["swir22"].href); v=(nir>.001)&(sw>.001); x=(nir-sw)/(nir+sw+1e-6); x[~v]=np.nan; return x
def empost():
    h=h5py.File(EMIT,"r"); r=h["reflectance"]; nir=r[:,:,NIR_B].astype(np.float32); sw=r[:,:,SWIR_B].astype(np.float32)
    bad=(nir<=-.01)|(sw<=-.01); nb=(nir-sw)/(nir+sw+1e-6); nb[bad]=np.nan
    gx=h["location/glt_x"][:]; gy=h["location/glt_y"][:]; gt=h.attrs["geotransform"]; h.close()
    Ho,Wo=gx.shape; o=np.full((Ho,Wo),np.nan,np.float32); m=(gx>0)&(gy>0); o[m]=nb[gy[m]-1,gx[m]-1]
    lon0,rx,_,lat0,_,ry=[float(v) for v in gt[:6]]
    rr,cc=np.mgrid[0:H,0:W]; ux=xmin+(cc+.5)*(xmax-xmin)/W; uy=ymax-(rr+.5)*(ymax-ymin)/H
    lon,lat=tr.transform(ux.ravel(),uy.ravel()); ec=np.round((lon-lon0)/rx-.5).astype(int); er=np.round((lat-lat0)/ry-.5).astype(int)
    ok=(ec>=0)&(ec<Wo)&(er>=0)&(er<Ho); out=np.full(H*W,np.nan,np.float32); out[ok]=o[er[ok],ec[ok]]; return out.reshape(H,W)
pre=s2pick("2023-03-20/2023-04-10"); post=s2pick("2023-04-15/2023-04-30")
s2d=s2nbr(pre)-s2nbr(post)   # S2 dNBR
em=empost()                  # EMIT post-NBR (char=low)
print(f"강릉난곡 (CIR-GT 0ha) | S2 pre {pre.datetime.date()} post {post.datetime.date()} | EMIT post 2023-04-21")
px_ha=(20*20)/1e4
s2burn=(s2d>0.15)&np.isfinite(s2d); emburn=(em<np.nanpercentile(em,20))&np.isfinite(em)
print(f"  S2 dNBR>0.15 (소각추정): {int(s2burn.sum())}px = {s2burn.sum()*px_ha:.0f}ha  (p90 {np.nanpercentile(s2d,90):+.3f})")
print(f"  EMIT post-NBR<p20 (char): {int(emburn.sum())}px = {emburn.sum()*px_ha:.0f}ha  (소각내 후보 p50 {np.nanmedian(em[emburn]):+.3f} vs 전체 {np.nanmedian(em):+.3f})")
# 공간 일치
both=np.isfinite(s2d)&np.isfinite(em)
if (emburn&both).sum()>10:
    print(f"  EMIT char 영역의 S2 dNBR p50 = {np.nanmedian(s2d[emburn&both]):+.3f} (양수 크면 S2도 봄 / 0근처면 S2 놓침)")
fig,ax=plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(s2d,cmap="inferno",vmin=0,vmax=0.4); ax[0].set_title(f"S2 dNBR ({pre.datetime.date()}→{post.datetime.date()})\nCIR-GT was 0 ha")
ax[1].imshow(em,cmap="RdYlGn",vmin=-.2,vmax=.5); ax[1].set_title("EMIT post-NBR (char=red, 2023-04-21)")
ax[2].imshow(emburn,cmap="hot"); ax[2].set_title("EMIT char mask (<p20)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(OUT+"/gangneung_emit_vs_s2.png",dpi=95); print("저장:",OUT+"/gangneung_emit_vs_s2.png")
