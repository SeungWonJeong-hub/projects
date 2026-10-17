"""24개 화재 GT를 STAC S2 실반사율(dNBR)로 생성 — 8bit CIR 프리뷰 대체.
번들에선 AOI(벡터 extent)+화재일만 사용. 화재별:
 AOI(5179 extent)→lonlat bbox·UTM존 → STAC서 pre/post 저구름 S2 6밴드+SCL 취득 →
 dNBR/dNDVI GT(라디오메트릭·SCL구름마스킹) → 학습npz(x 8ch rgbnir, gt, valid) + QA.
출력: fires/<name>/{gt_stac.npz, burn_mask_stac.tif, qa_stac.png}
env:
  GSD=20|10     격자 해상도. CIR 번들 원본이 10.16m라 10m가 네이티브에 가깝다.
  NIR=B8A|B08   B08(842nm)은 네이티브 10m, B8A(865nm)는 20m.
  NO_OFFSET=1   ≥2022 씬 -1000 오프셋 미차감(raw DN). 배포 ET-02가 raw DN 스케일이라
                CIR 파인튜닝도 같은 스케일이어야 한다. (step10_nonspring_cirgt.py가
                이 플래그를 세팅하지만 읽는 코드가 없었다 → 여기서 복원)
출력: GSD=20이면 gt_stac.npz(기존), 그 외 gt_stac_<gsd>m.npz — 20m 산출물 무손상.
사용: GSD=10 NIR=B08 NO_OFFSET=1 python build_cir_gt_stac.py   (오래걸림→백그라운드)
"""
import os, re, glob, datetime as dt, socket, sys, numpy as np
# conda activate 없이 절대경로 파이썬으로 부르면 activate.d 훅이 안 돌아 PROJ_DATA 가 비고
# osgeo 좌표변환이 통째로 죽는다(projinfo·pyproj 는 멀쩡해서 오해하기 쉽다). sys.prefix 로 채운다.
for _k,_d in (("PROJ_DATA","share/proj"),("GDAL_DATA","share/gdal")):
    if not os.environ.get(_k) and os.path.isdir(os.path.join(sys.prefix,_d)):
        os.environ[_k]=os.path.join(sys.prefix,_d)
from osgeo import gdal, ogr, osr
from scipy import ndimage as ndi
gdal.UseExceptions(); socket.setdefaulttimeout(120)
for k,v in {"GDAL_HTTP_TIMEOUT":"60","GDAL_HTTP_CONNECTTIMEOUT":"20","GDAL_HTTP_MAX_RETRY":"4",
            "GDAL_HTTP_RETRY_DELAY":"3","GDAL_DISABLE_READDIR_ON_OPEN":"EMPTY_DIR"}.items():
    gdal.SetConfigOption(k,v)
from pystac_client import Client
CAT=Client.open("https://earth-search.aws.element84.com/v1")
FIRES=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires")
GSD=float(os.environ.get("GSD","20"))
NIR=os.environ.get("NIR","B8A")
NIR_ASSET={"B8A":"nir08","B08":"nir"}[NIR]
NO_OFFSET=os.environ.get("NO_OFFSET","0")=="1"
PXA=GSD*GSD                                   # 픽셀 면적(m^2) — 최소면적·ha 환산에 사용
# 산출물 이름 = 설정별 분리(기존 20m+오프셋 GT를 덮어쓰지 않는다)
_SFX="" if (GSD==20 and not NO_OFFSET) else f"_{int(GSD)}m"+("_raw" if NO_OFFSET else "")
# BANDS12=1 → x 를 12ch(B02,B03,B04,NIR,B11,B12 ×pre/post)로 저장. SWIR 은 어차피 이미
# 내려받고 있었고(BANDS) GT 계산에도 쓰인다 — 저장만 안 했을 뿐이라 재취득 비용이 없다.
# ko_refl 의 12ch 순서(smp_cd.build_x 의 bitemporal)와 같아야 한다.
BANDS12=os.environ.get("BANDS12","0")=="1"
_SFX+="12" if BANDS12 else ""
NPZ=f"gt_stac{_SFX}.npz"; MSK=f"burn_mask_stac{_SFX}.tif"
BANDS=[("B02","blue"),("B03","green"),("B04","red"),(NIR,NIR_ASSET),("B11","swir16"),("B12","swir22")]
ORDER=["B02","B03","B04",NIR,"B11","B12"]
DNBR_THR=0.15; DNDVI_THR=0.08; POSTNDVI=0.45; MINHA=5.0
ADAPT=os.environ.get("ADAPT","0")=="1"   # 실험옵션(절대char OR). 겨울-pre 화재엔 무효로 판명→기본 off. 표준=고정 dNBR 0.15
ONLY=[x for x in os.environ.get("ONLY","").split(",") if x]  # 특정화재만 재생성
# ── 재브래킷 실험(2026-09-24): post 창을 tight·개엽전으로 좁히는 env 오버라이드 ──
# 기본 post 창(fd+6~+28)은 최저구름이 4월 중순 green-up 씬으로 드리프트→봄 흉터를 과소라벨.
# POST_LO/POST_HI 로 화재 직후 tight 창(예 +6~+13 = ~04-01)만 쓰게 해 라벨을 완전하게.
POST_LO=int(os.environ.get("POST_LO","6")); POST_HI=int(os.environ.get("POST_HI","28"))
NPZ=os.environ.get("NPZ_OUT",NPZ); MSK=os.environ.get("MSK_OUT",MSK)
RECIPE=os.environ.get("RECIPE","standard")   # standard(triple-AND) | rebracket(dNBR-only)
DNBR_THR=float(os.environ.get("DNBR_THR",DNBR_THR))
def otsu(v,lo=0.10,hi=0.25):
    v=v[np.isfinite(v)]
    if v.size<200: return DNBR_THR
    h,e=np.histogram(v,bins=128); c=(e[:-1]+e[1:])/2; w=h.astype(float)/h.sum()
    cw=np.cumsum(w); cm=np.cumsum(w*c); mg=cm[-1]
    with np.errstate(divide='ignore',invalid='ignore'):
        vb=(mg*cw-cm)**2/(cw*(1-cw)+1e-12)
    t=c[np.nanargmax(vb)]
    return float(min(max(t,lo),hi))
def rfc(a,b): return f"{a}T00:00:00Z/{b}T23:59:59Z"

def fire_date(name):
    s=re.match(r"(\d{6})_",name).group(1); return dt.date(2000+int(s[:2]),int(s[2:4]),int(s[4:6]))

def _sen_date(fn):
    m=re.search(r"_SEN_(\d{8})_CIR",fn) or re.search(r"_SEN_(\d{6})_CIR",fn) or re.match(r"T\d{2}\w{3}_(\d{8})T",fn)
    if not m: return None
    d=m.group(1)
    return dt.date(int(d[:4]),int(d[4:6]),int(d[6:8])) if len(d)==8 else dt.date(2000+int(d[:2]),int(d[2:4]),int(d[4:6]))
def platform_dates(folder,fd):
    """플랫폼 CIR의 타이트 pre/post 날짜(동일계열, 화재 최근접). 없으면 (None,None)."""
    sens=[]
    for t in glob.glob(os.path.join(folder,"*CIR*.tif")):
        b=os.path.basename(t)
        if "_geo" in b or "mask" in b.lower(): continue
        d=_sen_date(b)
        if d: sens.append((d,"comp" if "CompResult_SEN" in b else "ttile"))
    for fam in ("comp","ttile"):
        fs=[d for d,f in sens if f==fam]
        pre=[d for d in fs if d<fd]; post=[d for d in fs if d>=fd]
        if pre and post: return max(pre),min(post)
    return None,None

def aoi(folder):
    shps=[s for s in glob.glob(os.path.join(folder,"*.shp")) if "burn" not in os.path.basename(s).lower()]
    exts=[]
    for s in shps:
        ds=ogr.Open(s)
        if ds is None: continue
        ly=ds.GetLayer(); x0,x1,y0,y1=ly.GetExtent(); exts.append((x0,y0,x1,y1))
    if not exts: return None
    x0=min(e[0] for e in exts); y0=min(e[1] for e in exts); x1=max(e[2] for e in exts); y1=max(e[3] for e in exts)
    return (x0,y0,x1,y1)   # EPSG:5179

def to_ll_utm(ext5179):
    x0,y0,x1,y1=ext5179
    s=osr.SpatialReference(); s.ImportFromEPSG(5179); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ll=osr.SpatialReference(); ll.ImportFromEPSG(4326); ll.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    c2ll=osr.CoordinateTransformation(s,ll)
    pts=[c2ll.TransformPoint(x,y) for x in (x0,x1) for y in (y0,y1)]
    lons=[p[0] for p in pts]; lats=[p[1] for p in pts]
    lon0,lon1,lat0,lat1=min(lons),max(lons),min(lats),max(lats)
    utm=32652 if (lon0+lon1)/2>=126 else 32651
    u=osr.SpatialReference(); u.ImportFromEPSG(utm); u.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    c2u=osr.CoordinateTransformation(s,u)
    pu=[c2u.TransformPoint(x,y) for x in (x0,x1) for y in (y0,y1)]
    xs=[p[0] for p in pu]; ys=[p[1] for p in pu]
    ute=[int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))]
    g=int(GSD); ute=[u-u%g for u in ute]        # 격자 정렬(해상도 배수로 스냅)
    return (lon0,lat0,lon1,lat1),ute,utm

def stac_refl(bbox_ll,dtr,ute,utm):
    its=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=dtr,
        query={"eo:cloud_cover":{"lt":35}},sortby=[{"field":"properties.eo:cloud_cover","direction":"asc"}],max_items=1).items())
    if not its: return None
    d=its[0].datetime.date(); cc=its[0].properties.get("eo:cloud_cover",99)
    sib=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,
        datetime=rfc(d-dt.timedelta(days=2),d+dt.timedelta(days=2)),
        query={"eo:cloud_cover":{"lt":45}},max_items=12).items()) or its
    off=0.0 if NO_OFFSET else (1000.0 if d>=dt.date(2022,1,25) else 0.0); b={}; present=None
    for k,a in BANDS+[("SCL","scl")]:
        hrefs=[(x.assets[a].href if a in x.assets else x.assets.get(k).href) for x in sib if (a in x.assets or k in x.assets)]
        if not hrefs: return None
        _t=f"/tmp/_gs_{os.getpid()}_{k}.tif"     # pid 포함: 병렬 실행시 덮어쓰기 방지
        gdal.Warp(_t,[f"/vsicurl/{h}" for h in hrefs],dstSRS=f"EPSG:{utm}",xRes=GSD,yRes=GSD,
                  outputBounds=ute,resampleAlg="average",format="GTiff",srcNodata=0)
        arr=gdal.Open(_t).ReadAsArray().astype(np.float32); os.remove(_t)
        if k!="SCL":
            p=(arr>0)                       # 데이터 존재는 오프셋 차감 前 raw>0로 판정
            present=p if present is None else (present&p)
            arr=np.clip(arr-off,0,None)
        b[k]=arr
    b["_date"]=str(d); b["_cc"]=cc; b["_valid"]=present; return b

def process(folder,name):
    fd=fire_date(name); ext=aoi(folder)
    if ext is None: return "skip 벡터없음"
    bbox_ll,ute,utm=to_ll_utm(ext)
    # 타이트 브래킷: 플랫폼 CIR 날짜 있으면 그 날짜로, 없으면 화재 최근접 짧은 창
    D=dt.timedelta
    # post는 scar 발달 후·green-up 前(화재+6~28일) 저구름, pre는 화재 직전(-25~-2일) 저구름
    pre=stac_refl(bbox_ll,rfc(fd-D(25),fd-D(2)),ute,utm)
    post=stac_refl(bbox_ll,rfc(fd+D(POST_LO),fd+D(POST_HI)),ute,utm)
    src="auto"
    if not pre or not post: return f"skip S2취득실패(pre{bool(pre)}/post{bool(post)})"
    H=min(pre['B02'].shape[0],post['B02'].shape[0]); W=min(pre['B02'].shape[1],post['B02'].shape[1])
    for dd in (pre,post):
        for kk in list(dd):
            if kk not in("_date","_cc") and hasattr(dd[kk],"shape"): dd[kk]=dd[kk][:H,:W]
    def nbr(b): return (b[NIR]-b['B12'])/(b[NIR]+b['B12']+1e-6)
    def ndvi(b): return (b[NIR]-b['B04'])/(b[NIR]+b['B04']+1e-6)
    valid=pre['_valid']&post['_valid']
    valid&=~np.isin(pre['SCL'],[0,1,3,8,9,10,11])&~np.isin(post['SCL'],[0,1,3,8,9,10,11])
    dnbr=nbr(pre)-nbr(post); dndvi=ndvi(pre)-ndvi(post); nbrp=nbr(post)
    thr=0.14
    # 소각 = 식생소실(dndvi) AND (dNBR변화 OR 절대char[낮은 post-NBR]). 겨울-pre 저대비 화재는 절대char가 살림.
    if RECIPE=="rebracket":
        # 재브래킷 레시피(2026-09-24): 단일 dNBR 임계만(dNDVI·postNDVI 게이트 제거).
        # 봄 green-up이 ndvi 게이트로 진짜 소각을 잘라내는 벽을 우회 — Task1서 산청 101% 재현.
        burn=(dnbr>DNBR_THR)&valid
    elif ADAPT:
        burn=(dndvi>0.10)&(ndvi(post)<POSTNDVI)&valid&((dnbr>thr)|(nbrp<0.08))
    else:
        burn=(dnbr>DNBR_THR)&(dndvi>DNDVI_THR)&(ndvi(post)<POSTNDVI)&valid
    burn=ndi.binary_opening(burn,iterations=1); burn=ndi.binary_closing(burn,iterations=2); burn=ndi.binary_fill_holes(burn)
    lab,n=ndi.label(burn)
    if n:
        sizes=ndi.sum(np.ones_like(lab),lab,range(1,n+1)); minpx=MINHA*1e4/PXA
        burn=np.isin(lab,[i+1 for i,s in enumerate(sizes) if s>=minpx])
    area=float(burn.sum())*PXA/1e4
    # 학습 npz (rgbnir 8ch 0-1)
    four=lambda b:[b['B02'],b['B03'],b['B04'],b[NIR]]
    six =lambda b:[b[k] for k in ORDER]          # B02,B03,B04,NIR,B11,B12
    sel=six if BANDS12 else four
    x=np.stack([np.clip(c/1e4,0,1) for c in sel(pre)+sel(post)]).astype("float32")
    np.savez_compressed(os.path.join(folder,NPZ),x=x,gt=burn.astype(np.uint8),valid=valid.astype(np.uint8),
                        utm=utm,ute=np.array(ute),pre=pre['_date'],post=post['_date'])
    # mask tif
    drv=gdal.GetDriverByName("GTiff"); srs=osr.SpatialReference(); srs.ImportFromEPSG(utm)
    m=drv.Create(os.path.join(folder,MSK),W,H,1,gdal.GDT_Byte)
    m.SetGeoTransform((ute[0],GSD,0,ute[3],0,-GSD)); m.SetProjection(srs.ExportToWkt()); m.GetRasterBand(1).WriteArray(burn.astype(np.uint8)); m=None
    # QA
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def nz(a):
        v=a[a>0];
        if v.size==0: return a
        lo,hi=np.percentile(v,2),np.percentile(v,98); return np.clip((a-lo)/(hi-lo+1e-6),0,1)
    rgb=np.dstack([nz(post['B12']),nz(post[NIR]),nz(post['B04'])])
    fig,ax=plt.subplots(1,2,figsize=(14,5))
    ax[0].imshow(rgb); ax[0].set_title(f"post SWIR-NIR-R {post['_date']}(cc{post['_cc']:.0f}) pre{pre['_date']}",fontsize=8)
    ax[1].imshow(rgb); ax[1].imshow(np.ma.masked_where(~burn,burn.astype(float)),cmap="cool",alpha=0.5)
    ax[1].set_title(f"dNBR burn ~{area:.0f}ha valid{100*valid.mean():.0f}%",fontsize=8)
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(folder,"qa_stac.png"),dpi=80,bbox_inches="tight"); plt.close()
    return f"OK {area:7.0f}ha  thr={thr:.3f} pre={pre['_date']} post={post['_date']} ({src}) valid{100*valid.mean():.0f}%"

def main():
    dirs=sorted(d for d in os.listdir(FIRES) if os.path.isdir(os.path.join(FIRES,d)))
    if ONLY: dirs=[d for d in dirs if d in ONLY]
    ok=0
    for d in dirs:
        try: r=process(os.path.join(FIRES,d),d)
        except Exception as e: r=f"ERR {str(e)[:60]}"
        if r.startswith("OK"): ok+=1
        print(f"{d:26s} {r}",flush=True)
    print(f"\n★ STAC GT {ok}/{len(dirs)}",flush=True)

if __name__=="__main__": main()
