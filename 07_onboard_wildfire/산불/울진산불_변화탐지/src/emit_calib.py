"""캘리브: 의성2025 고심각도 화재(S2 dNBR 분리 +0.59, 확실)로 EMIT 지오로케이션 검증.
S2 소각 마스크(dNBR>0.2, S2가 확실히 봄) vs EMIT post-NBR char를 S2 20m 격자에 정합.
- EMIT post-NBR 소각내 vs 밖 (EMIT이 S2-확정 소각을 보나)
- 공간 IoU(EMIT low-NBR ∩ S2 소각) — 겹치면 EMIT 지오로케이션 OK → 산청 신호 신뢰가능.
"""
import os, json, numpy as np, h5py, rasterio
from rasterio.enums import Resampling
from pystac_client import Client
from pyproj import Transformer
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
OUT=os.path.join(PROJ,"runs","emit_uiseong"); os.makedirs(OUT,exist_ok=True)
CIRD="/home/syntax_kmu/syntax_inference/Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires"
EMIT=os.path.join(PROJ,"data","emit_uiseong","EMIT_L2A_RFL_001_20250424T011718_2511401_011.nc")
NIR_B,SWIR_B=64,244
cat=Client.open("https://earth-search.aws.element84.com/v1")

def s2_nbr(item,bounds,shape):
    def rd(h):
        with rasterio.open(h) as ds:
            w=rasterio.windows.from_bounds(*bounds,transform=ds.transform)
            return ds.read(1,window=w,out_shape=shape,resampling=Resampling.bilinear).astype(np.float32)/1e4
    nir=rd(item.assets["nir"].href); sw=rd(item.assets["swir22"].href)
    v=(nir>0.001)&(sw>0.001); x=(nir-sw)/(nir+sw+1e-6); x[~v]=np.nan; return x

def emit_postnbr_ortho():
    h=h5py.File(EMIT,"r"); refl=h["reflectance"]
    nir=refl[:,:,NIR_B].astype(np.float32); sw=refl[:,:,SWIR_B].astype(np.float32)
    bad=(nir<=-0.01)|(sw<=-0.01); nbr=(nir-sw)/(nir+sw+1e-6); nbr[bad]=np.nan
    gx=h["location/glt_x"][:]; gy=h["location/glt_y"][:]; gt=h.attrs["geotransform"]; h.close()
    Ho,Wo=gx.shape; o=np.full((Ho,Wo),np.nan,np.float32); m=(gx>0)&(gy>0); o[m]=nbr[gy[m]-1,gx[m]-1]
    lon0,rx,_,lat0,_,ry=[float(v) for v in gt[:6]]
    return o,(lon0,rx,lat0,ry)

def main():
    d=np.load(os.path.join(CIRD,"250322_의성안평면","gt_stac.npz")); gtp=d["gt"].astype(bool); H,W=gt_shape=d["gt"].shape
    xmin,ymin,xmax,ymax=[float(v) for v in d["ute"]]; bounds=(xmin,ymin,xmax,ymax); epsg=int(d["utm"])
    tr=Transformer.from_crs(f"EPSG:{epsg}","EPSG:4326",always_xy=True)
    clon,clat=tr.transform((xmin+xmax)/2,(ymin+ymax)/2); bbox=[clon-0.15,clat-0.12,clon+0.15,clat+0.12]
    def pick(dr):
        its=list(cat.search(collections=["sentinel-2-l2a"],bbox=bbox,datetime=dr,query={"eo:cloud_cover":{"lt":20}}).items())
        return sorted(its,key=lambda x:x.properties["eo:cloud_cover"])[0]
    pre=pick("2025-02-25/2025-03-19"); post=pick("2025-04-15/2025-04-30")
    s2dnbr=s2_nbr(pre,bounds,(H,W))-s2_nbr(post,bounds,(H,W))
    s2burn=(s2dnbr>0.2)&np.isfinite(s2dnbr)   # S2 확정 소각(고심각도)
    print(f"의성: S2 pre {pre.datetime.date()} post {post.datetime.date()} | S2소각(dNBR>0.2) {int(s2burn.sum())}px = {s2burn.sum()*400/1e4:.0f}ha")

    # EMIT post-NBR → S2 20m 격자로 샘플(각 S2 픽셀 lat/lon → EMIT ortho index)
    o,(lon0,rx,lat0,ry)=emit_postnbr_ortho(); Ho,Wo=o.shape
    rows,cols=np.mgrid[0:H,0:W]
    ux=xmin+(cols+0.5)*(xmax-xmin)/W; uy=ymax-(rows+0.5)*(ymax-ymin)/H
    lon,lat=tr.transform(ux.ravel(),uy.ravel()); lon=lon.reshape(H,W); lat=lat.reshape(H,W)
    ec=np.round((lon-lon0)/rx-0.5).astype(int); er=np.round((lat-lat0)/ry-0.5).astype(int)
    ok=(ec>=0)&(ec<Wo)&(er>=0)&(er<Ho)
    emit=np.full((H,W),np.nan,np.float32); emit[ok]=o[er[ok],ec[ok]]

    val=np.isfinite(emit)&np.isfinite(s2dnbr)
    inb=s2burn&val; outb=(~s2burn)&val
    ei=emit[inb]; eo=emit[outb]
    print(f"\n=== EMIT post-NBR: S2-확정소각 안 vs 밖 ===")
    print(f"  소각 안 p50 {np.median(ei):+.3f}  밖 p50 {np.median(eo):+.3f}  분리 {np.median(eo)-np.median(ei):+.3f} (밖이 높아야 정상)")
    # EMIT char 마스크 & 공간 IoU
    thr=np.nanpercentile(emit,30)  # 낮은 30% = char 후보
    emitburn=(emit<thr)&val
    inter=(emitburn&s2burn).sum(); uni=(emitburn|s2burn).sum()
    iou=inter/max(uni,1)
    rec=(emitburn&s2burn).sum()/max(s2burn.sum(),1)
    print(f"\n=== 공간 정합(지오로케이션 검증) ===")
    print(f"  EMIT char(<p30={thr:+.3f}) ∩ S2소각 IoU = {iou:.3f} | S2소각의 {rec*100:.0f}%를 EMIT char가 덮음")
    print(f"  판정: {'✅ EMIT 지오로케이션 신뢰(S2 소각과 공간겹침)' if rec>0.6 and np.median(eo)-np.median(ei)>0.1 else '❌ 정합 약함(EMIT 60m/지오 한계)'}")
    json.dump(dict(fire="uiseong",s2_sep=float(np.nanmedian(s2dnbr[s2burn])-np.nanmedian(s2dnbr[outb])) if outb.sum() else None,
                   emit_in_p50=float(np.median(ei)),emit_out_p50=float(np.median(eo)),
                   emit_sep=float(np.median(eo)-np.median(ei)),iou=float(iou),emit_covers_s2burn=float(rec)),
              open(os.path.join(OUT,"calib.json"),"w"),ensure_ascii=False,indent=1)
    print("저장: runs/emit_uiseong/calib.json")

if __name__=="__main__": main()
