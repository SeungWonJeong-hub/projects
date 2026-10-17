"""트랙 B — 활성화재(active fire) 온보드 탐지. [[wildfire-cd-purpose]] 목적 B.
S2엔 열밴드 없지만 화염 고온이 SWIR(B11 1.6µm·B12 2.2µm)에 급등 → NHI(Normalized Hotspot
Indices, Marchese et al. 2019)로 '지금 타는 불'을 픽셀당 탐지. 학습불요·마이크로초 → 온보드 이상적.
  NHI_SWIR  = (B12-B11)/(B12+B11)     NHI_SWNIR = (B11-B8A)/(B11+B8A)
  활성화재 = NHI_SWNIR>0 (명확) 또는 NHI_SWIR>임계(후보) ; 물·구름·저휘도 오탐 제거.
진행중 overpass 다시점 → 확산 진행(최전선 이동) 정량화. FIRMS(VIIRS/MODIS 열탐지)로 독립검증.

사용:  python af_detect.py <YYYY-MM-DD> [YYYY-MM-DD ...]   # 없으면 울진 진행중창 기본
환경:  AOI="lo,la,LO,LA" (기본 울진) · FIRMS_KEY=<map_key>(검증, 선택)
"""
from __future__ import annotations
import os, sys, json, time, datetime as dt
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from stac import stac_search, AOI_BBOX, TARGET_GSD_M

def download_band(href, dest, resample="average"):
    """원격 COG를 내 AOI(env)로 클립·20m 리샘플. stac.download_band는 울진 bbox 하드코딩이라 자체 구현."""
    opts=gdal.TranslateOptions(xRes=TARGET_GSD_M,yRes=TARGET_GSD_M,resampleAlg=resample,
        projWin=[AOI[0],AOI[3],AOI[2],AOI[1]],projWinSRS="EPSG:4326",format="GTiff",
        creationOptions=["COMPRESS=DEFLATE","TILED=YES","BIGTIFF=IF_SAFER"])
    for att in range(4):
        try:
            d=gdal.Translate(dest,"/vsicurl/"+href,options=opts); d=None; return os.path.getsize(dest)
        except Exception as e:
            print(f"  band retry {att+1} {os.path.basename(dest)}: {e}"); time.sleep(5*(att+1))
    return 0

SITE=os.environ.get("SITE","uljin")   # 화재별 출력 분리
OUT=os.path.join(HERE,"..","active_fire",SITE); os.makedirs(OUT,exist_ok=True)
# NHI엔 B8A(nir08, 865nm 협대역 20m)가 표준. earth-search 자산키.
BANDS={"nir08":"B8A","swir16":"B11","swir22":"B12","red":"B04","green":"B03","blue":"B02","scl":"SCL"}
AOI=[float(v) for v in os.environ["AOI"].split(",")] if os.environ.get("AOI") else AOI_BBOX

def _read(p):
    ds=gdal.Open(p); a=ds.ReadAsArray().astype(np.float32); ds=None; return a

def fetch(date, sdir):
    """해당 날짜 장면 취득(AOI). 최저구름부터 시도하되, 클립 후 실제 유효픽셀이 있는
    장면으로 폴백(earth-search item bbox는 겹쳐 보고되나 실 데이터 풋프린트가 빈 이웃타일 배제)."""
    d=dt.date.fromisoformat(date)
    rng=f"{d}T00:00:00Z/{d+dt.timedelta(days=1)}T00:00:00Z"
    feats=[f for f in stac_search(AOI,rng,max_cloud=100,limit=30) if _covers(f)]
    feats.sort(key=lambda f:f["properties"].get("eo:cloud_cover",100))
    if not feats: print(f"[{date}] 장면 없음",flush=True); return None,None
    os.makedirs(sdir,exist_ok=True)
    off=1000.0 if d>=dt.date(2022,1,25) else 0.0   # baseline≥04.00 반사율 offset
    for feat in feats:
        sid=feat["id"]; b12a=feat["assets"].get("swir22")
        if not b12a: continue
        b12dest=os.path.join(sdir,"B12.tif")
        if not(os.path.exists(b12dest) and os.path.getsize(b12dest)>0):
            download_band(b12a["href"],b12dest,resample="average")
        b12=_read(b12dest)
        if (b12>0).mean() < 0.5:   # 유효<50% = 엉뚱한 타일 → 다음 후보
            os.remove(b12dest); continue
        bd={"B12":np.clip(b12-off,0,None)}
        for key,name in BANDS.items():
            if name=="B12": continue
            a=feat["assets"].get(key)
            if not a: print(f"  ! {sid} 자산없음 {key}"); continue
            dest=os.path.join(sdir,f"{name}.tif")
            if not(os.path.exists(dest) and os.path.getsize(dest)>0):
                download_band(a["href"],dest,resample="nearest" if name=="SCL" else "average")
            arr=_read(dest)
            bd[name]=arr if name=="SCL" else np.clip(arr-off,0,None)
        bd["_id"]=sid; bd["_cloud"]=feat["properties"].get("eo:cloud_cover")
        bd["_dt"]=feat["properties"].get("datetime"); bd["_geo"]=b12dest
        return bd,feat
    print(f"[{date}] AOI 덮는 유효 장면 없음",flush=True); return None,None

def _covers(f):
    """AOI 중심을 포함하는 장면만(모서리만 스치는 엉뚱한 타일 배제)."""
    b=f.get("bbox",[]); cx=(AOI[0]+AOI[2])/2; cy=(AOI[1]+AOI[3])/2
    return len(b)>=4 and b[0]<=cx<=b[2] and b[1]<=cy<=b[3]

def detect(bd):
    """활성화재 탐지(픽셀당, 온보드형). Murphy et al. 2016 S2 화재 판별.
    ★핵심 = 강한 SWIR2 열복사. 화염은 2.2µm(B12)가 비정상 치솟아 (a)절대 매우밝거나
    (b)SWIR1 대비 급증. 신선한 소각흔 char도 B12>B11이라(구식 NHI_SWIR>0은 char 오탐)
    → 절대 휘도 조건을 반드시 함께 걸어 char(중간 밝기)와 화염(고휘도)을 분리.
    (B12/B11은 offset보정 반사율×1e4. 0.28→2800, 0.15→1500)."""
    B11,B12,SCL=bd["B11"],bd["B12"],bd["SCL"]
    e=1e-6; t0=time.perf_counter()
    ratio=B12/(B11+e)                        # 열복사비(화염 >>1)
    bad=np.isin(SCL,[0,1,3,6,8,9,10,11])     # 0=nodata,3=shadow,6=water,8-10=cloud,11=snow
    fire = ((B12>=2800) | ((ratio>=1.4)&(B12>=1500))) & ~bad   # Murphy 활성화재
    mask=fire.astype(np.uint8)
    # 확신도: SWIR2 휘도 정규화(2800=포화급 1.0)
    conf=(np.clip(B12/2800,0,1).astype(np.float32))*mask
    core=int(((B12>=2800)&~bad).sum())       # 명확 화염 코어(포화급)
    ms=(time.perf_counter()-t0)*1000
    return mask,conf,ratio,B12,ms,core

def save_geo(ref_tif,arr,dest,dtype=gdal.GDT_Byte):
    r=gdal.Open(ref_tif); drv=gdal.GetDriverByName("GTiff")
    o=drv.Create(dest,r.RasterXSize,r.RasterYSize,1,dtype,["COMPRESS=DEFLATE"])
    o.SetGeoTransform(r.GetGeoTransform()); o.SetProjection(r.GetProjection())
    o.GetRasterBand(1).WriteArray(arr); o=None; r=None

def viz(bd,mask,conf,dest):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def s(x,p=98):
        hi=np.percentile(x,p) or 1; return np.clip(x/hi,0,1)
    # SWIR 합성(B12,B11,B8A)=화재가 붉게 두드러짐 + 탐지 오버레이
    swir=np.dstack([s(bd["B12"]),s(bd["B11"]),s(bd["B8A"])])
    rgb =np.dstack([s(bd["B04"]),s(bd["B03"]),s(bd["B02"])])
    ov=rgb.copy(); ov[mask>0]=[1,0,0]
    fig,ax=plt.subplots(1,3,figsize=(15,5))
    for a,im,t in zip(ax,[rgb,swir,ov],["RGB (true color)","SWIR (B12/B11/B8A)","Active fire (red)"]):
        a.imshow(im); a.set_title(t); a.axis("off")
    fig.suptitle(f"{bd['_id']}  cloud={bd['_cloud']:.0f}%  active-fire px={int(mask.sum())} ({mask.sum()*400/1e4:.0f} ha)")
    plt.tight_layout(); plt.savefig(dest,dpi=90); plt.close()

def main():
    dates=sys.argv[1:] or ["2022-03-05","2022-03-10","2022-03-15"]
    print(f"[활성화재 탐지] AOI={AOI} 날짜={dates}",flush=True)
    results=[]; masks={}
    for date in dates:
        sdir=os.path.join(OUT,date); bd,feat=fetch(date,sdir)
        if bd is None: continue
        mask,conf,nsw,nswn,ms,core=detect(bd)
        save_geo(bd["_geo"],mask,os.path.join(sdir,"active_fire_mask.tif"))
        save_geo(bd["_geo"],conf,os.path.join(sdir,"active_fire_conf.tif"),gdal.GDT_Float32)
        viz(bd,mask,conf,os.path.join(sdir,"detect.png"))
        px=int(mask.sum()); ha=px*(TARGET_GSD_M**2)/1e4; core_ha=core*(TARGET_GSD_M**2)/1e4
        r=dict(date=date,scene=bd["_id"],cloud=round(float(bd["_cloud"]),1),
               fire_px=px,fire_ha=round(ha,1),core_ha=round(core_ha,1),detect_ms=round(ms,2),
               px_per_ms=round(mask.size/ms,0))
        results.append(r); masks[date]=(mask,bd["_geo"])
        print(f"[{date}] {bd['_id']} cloud={bd['_cloud']:.0f}% → 활성화재 {px}px ({ha:.1f}ha, 코어 {core_ha:.1f}ha) | {ms:.1f}ms ({mask.size/ms:.0f}px/ms)",flush=True)
    # 확산 진행(다시점): 시점간 신규/소멸 최전선
    if len(masks)>=2:
        ds=sorted(masks); prog=[]
        for i in range(1,len(ds)):
            m0,_=masks[ds[i-1]]; m1,_=masks[ds[i]]
            H=min(m0.shape[0],m1.shape[0]);W=min(m0.shape[1],m1.shape[1])
            a=m0[:H,:W]>0; b=m1[:H,:W]>0
            new=int((b&~a).sum()); gone=int((a&~b).sum()); both=int((a&b).sum())
            prog.append(dict(frm=ds[i-1],to=ds[i],new_px=new,extinguished_px=gone,persist_px=both,
                             net_growth_ha=round((new-gone)*(TARGET_GSD_M**2)/1e4,1)))
            print(f"[확산 {ds[i-1]}→{ds[i]}] 신규 {new}px 소멸 {gone}px 지속 {both}px 순증 {(new-gone)*(TARGET_GSD_M**2)/1e4:+.1f}ha",flush=True)
        json.dump(prog,open(os.path.join(OUT,"spread_progression.json"),"w"),indent=2,ensure_ascii=False)
    json.dump(results,open(os.path.join(OUT,"detections.json"),"w"),indent=2,ensure_ascii=False)
    print("저장:",OUT,flush=True)

if __name__=="__main__": main()
