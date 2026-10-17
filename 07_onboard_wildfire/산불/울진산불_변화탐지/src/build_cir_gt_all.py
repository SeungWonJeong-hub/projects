"""24개 화재 CIR-GT 일괄 생성. fires/<YYMMDD_지역>/ 각 폴더에서
Sentinel CIR(CompResult_SEN_<date>_CIR_s.tif) pre/post를 화재일 기준 자동선택 →
벡터 extent 지오레퍼런싱 → dNDVI 소각 GT 마스크 + QA.
pre/post 둘 다 있는 화재만 처리(없으면 skip, 1b서 S2로 보충).
사용: python build_cir_gt_all.py
"""
import os, re, glob, datetime as dt, numpy as np
from osgeo import gdal, ogr, osr
from scipy import ndimage as ndi
gdal.UseExceptions()
FIRES=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires")
DTHR=-0.12; POSTMAX=0.35; MINHA=5.0

def fire_date(name):
    m=re.match(r"(\d{6})_",name); s=m.group(1)
    return dt.date(2000+int(s[:2]),int(s[2:4]),int(s[4:6]))
def sen_date(fn):
    m=re.search(r"_SEN_(\d{8})_CIR",fn) or re.search(r"_SEN_(\d{6})_CIR",fn) or re.match(r"T\d{2}\w{3}_(\d{8})T",fn)
    if not m: return None
    d=m.group(1)
    return dt.date(int(d[:4]),int(d[4:6]),int(d[6:8])) if len(d)==8 else dt.date(2000+int(d[:2]),int(d[2:4]),int(d[4:6]))
def vector_extent(folder):
    shps=[s for s in glob.glob(os.path.join(folder,"*.shp")) if "burn" not in os.path.basename(s).lower()]
    exts=[]; srs_wkt=None
    for s in shps:
        ds=ogr.Open(s)
        if ds is None: continue
        ly=ds.GetLayer(); x0,x1,y0,y1=ly.GetExtent(); exts.append((x0,y0,x1,y1))
        if srs_wkt is None and ly.GetSpatialRef(): srs_wkt=ly.GetSpatialRef().ExportToWkt()
    if not exts: return None,None
    x0=min(e[0] for e in exts); y0=min(e[1] for e in exts); x1=max(e[2] for e in exts); y1=max(e[3] for e in exts)
    return (x0,y0,x1,y1),srs_wkt
def ndvi(a): return (a[0]-a[1])/(a[0]+a[1]+1e-6)

def process(folder,name):
    fd=fire_date(name)
    sens=[]
    for t in glob.glob(os.path.join(folder,"*CIR*.tif")):
        b=os.path.basename(t)
        if "_geo" in b or "mask" in b.lower(): continue
        d=sen_date(b)
        if d is not None:
            fam="comp" if "CompResult_SEN" in b else "ttile"
            sens.append((d,t,fam))
    # 같은 계열(동일 grid)서 pre/post 선택. CompResult 우선.
    chosen=None
    for fam in ("comp","ttile"):
        fs=[x for x in sens if x[2]==fam]
        pre=[x for x in fs if x[0]<fd]; post=[x for x in fs if x[0]>=fd]
        if pre and post: chosen=(max(pre,key=lambda x:x[0]),min(post,key=lambda x:x[0])); break
    if chosen is None: return f"skip (pre/post 동일계열 없음)"
    pre_t=chosen[0][1]; post_t=chosen[1][1]
    ext,srs_wkt=vector_extent(folder)
    if ext is None: return "skip (벡터없음)"
    X0,Y0,X1,Y1=ext
    src=gdal.Open(post_t); W,Ht=src.RasterXSize,src.RasterYSize
    px=(X1-X0)/W; py=(Y1-Y0)/Ht
    gt=(X0,px,0,Y1,0,-py)
    srs=osr.SpatialReference()
    if srs_wkt: srs.ImportFromWkt(srs_wkt)
    else: srs.ImportFromEPSG(5179)
    pre=gdal.Open(pre_t).ReadAsArray().astype(np.float32)
    post=gdal.Open(post_t).ReadAsArray().astype(np.float32)
    if pre.shape!=post.shape:
        h=min(pre.shape[1],post.shape[1]); w=min(pre.shape[2],post.shape[2])
        pre=pre[:,:h,:w]; post=post[:,:h,:w]; W,Ht=w,h; gt=(X0,(X1-X0)/w,0,Y1,0,-(Y1-Y0)/h); px=(X1-X0)/w; py=(Y1-Y0)/h
    dnd=ndvi(post)-ndvi(pre)
    burn=(dnd<DTHR)&(ndvi(post)<POSTMAX)
    burn=ndi.binary_opening(burn,iterations=2); burn=ndi.binary_closing(burn,iterations=3); burn=ndi.binary_fill_holes(burn)
    lab,n=ndi.label(burn)
    if n:
        sizes=ndi.sum(np.ones_like(lab),lab,range(1,n+1)); minpx=MINHA*10000/(px*py)
        burn=np.isin(lab,[i+1 for i,s in enumerate(sizes) if s>=minpx])
    area=float(burn.sum())*px*py/1e4
    drv=gdal.GetDriverByName("GTiff")
    mds=drv.Create(os.path.join(folder,f"burn_mask_{name}.tif"),W,Ht,1,gdal.GDT_Byte)
    mds.SetGeoTransform(gt); mds.SetProjection(srs.ExportToWkt()); mds.GetRasterBand(1).WriteArray(burn.astype(np.uint8)); mds=None
    # QA
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cir=np.dstack([post[0],post[1],post[2]]).astype(np.uint8); ext2=[X0,X1,Y0,Y1]
    fig,ax=plt.subplots(1,2,figsize=(16,5))
    ax[0].imshow(cir,extent=ext2); ax[0].set_title(f"{name} CIR post ({os.path.basename(post_t)})",fontsize=9)
    ax[1].imshow(cir,extent=ext2); ax[1].imshow(np.ma.masked_where(~burn,burn.astype(float)),extent=ext2,cmap="autumn",alpha=0.45)
    ax[1].set_title(f"burn ~{area:.0f}ha  pre={os.path.basename(pre_t)}",fontsize=9)
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(folder,f"qa_{name}.png"),dpi=80,bbox_inches="tight"); plt.close()
    return f"OK  {area:6.0f}ha  pre={sen_date(os.path.basename(pre_t)).strftime('%m%d')} post={sen_date(os.path.basename(post_t)).strftime('%m%d')} px={px:.1f}m"

def main():
    dirs=sorted(d for d in os.listdir(FIRES) if os.path.isdir(os.path.join(FIRES,d)))
    ok=0
    for d in dirs:
        try: r=process(os.path.join(FIRES,d),d)
        except Exception as e: r=f"ERR {str(e)[:50]}"
        if r.startswith("OK"): ok+=1
        print(f"{d:26s} {r}",flush=True)
    print(f"\n★ GT 생성 {ok}/{len(dirs)}",flush=True)

if __name__=="__main__": main()
