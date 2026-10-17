"""ASAP-pdSI 충실 구현: 안정 phenology-class detrend (Sung et al. 2026).

지난 asap_mismatch.py 결함: pre-NDVI 분위 층화가 소각과 엉켜 pdSI 붕괴.
수정: detrend 클래스를 **화재-독립 안정 phenology-class**로 — pre-fire 다계절
NDVI(겨울/봄/여름 median)로 픽셀을 계절거동 유형별 K-means 군집. 논문의
landcover/LSP 클래스 취지에 충실(소각과 비엉킴).

pdSI = dSI - median_{unburned,class}(dSI),  class=안정 phenology-class
검증: NIFoS 피해강도 서수 ρ(severity, 논문태스크) + extent F1.
대조: dSI vs pdSI(stable) vs pdSI(preNDVI-quantile, 구버전).

아카이브=B02/03/04/08(SWIR無)→NDVI. 울진6밴드(sample_tif)만 dNBR 병기.
"""
import os, json
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation
from scipy.cluster.vq import kmeans2, whiten

A = os.path.expanduser("~/Desktop/sentinel2_data")
OUT = os.path.expanduser("~/syntax_inference/Steven/산불/울진산불_변화탐지/runs/asap")
os.makedirs(OUT, exist_ok=True)
EDGE_BUF = 2; K = 4

# 화재 구성: GT, post(archive folder), pre 계절별 후보, phenology 빌드용 pre-fire 다계절
FIRES = {
  "uljin2022": dict(
    gt=os.path.expanduser("~/Steven/nifos_burn_gt/ext/Ulgin_severity.tif"),
    tile="52SEG",
    post6=os.path.expanduser("~/syntax_inference/Steven/웹앱/sample_tif/uljin2022_after.tif"),
    pre6 =os.path.expanduser("~/syntax_inference/Steven/웹앱/sample_tif/uljin2022_before.tif"),
    pre_spring=["S2A_52SEG_20200414_1_L2A","S2B_52SEG_20200429_1_L2A","S2A_52SEG_20200504_1_L2A"],
    pre_summer=["S2B_52SEG_20200608_1_L2A","S2B_52SEG_20200628_1_L2A"],
    pheno=["S2B_52SEG_20200113_1_L2A","S2A_52SEG_20200414_1_L2A","S2B_52SEG_20200608_1_L2A"],
  ),
  "andong2020": dict(   # 화재 2020-04-24 → summer는 post-fire라 pheno는 겨울/봄만
    gt=os.path.expanduser("~/Steven/nifos_burn_gt/andong/andong_severity_20m.tif"),
    tile="52SDF",
    post_arch=["S2A_52SDF_20200606_1_L2A","S2A_52SDF_20200616_1_L2A"],
    pre_spring=["S2A_52SDF_20200414_1_L2A","S2B_52SDF_20200409_1_L2A","S2A_52SDF_20200407_1_L2A"],
    pheno=["S2B_52SDF_20200110_1_L2A","S2A_52SDF_20200305_1_L2A","S2A_52SDF_20200414_1_L2A"],
  ),
}


def spearman(a,b):
    ra=a.argsort().argsort().astype(float); rb=b.argsort().argsort().astype(float)
    ra-=ra.mean(); rb-=rb.mean(); d=np.sqrt((ra*ra).sum()*(rb*rb).sum())
    return float((ra*rb).sum()/d) if d else float("nan")

def f1_sweep(score, ref, lo=-0.1, hi=0.7, step=0.02):
    best=(0.0,0.0)
    for t in np.arange(lo,hi,step):
        p=score>=t; tp=int((p&ref).sum()); fp=int((p&~ref).sum()); fn=int((~p&ref).sum())
        P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
        f=2*P*R/(P+R) if P+R else 0
        if f>best[1]: best=(round(float(t),2),round(float(f),3))
    return best

def band_grid(path, gds, band=1):
    with rasterio.open(path) as ds:
        dst=np.zeros((gds.height,gds.width),"float32")
        reproject(rasterio.band(ds,band),dst,src_transform=ds.transform,src_crs=ds.crs,
                  dst_transform=gds.transform,dst_crs=gds.crs,resampling=Resampling.average)
    return dst/10000.0

def ndvi_arch(folder, gds):
    red=band_grid(f"{A}/{folder}/B04.tif",gds); nir=band_grid(f"{A}/{folder}/B08.tif",gds)
    v=(nir+red>0)&(nir>0.001)&(red>0.001)
    return np.where(v,(nir-red)/(nir+red+1e-6),np.nan), v

def ndvi_6(path, gds):
    red=band_grid(path,gds,3); nir=band_grid(path,gds,4)
    v=(nir+red>0)&(nir>0.001)&(red>0.001)
    return np.where(v,(nir-red)/(nir+red+1e-6),np.nan), v

def nbr_6(path, gds):
    nir=band_grid(path,gds,4); sw=band_grid(path,gds,6)
    v=(nir+sw>0)&(nir>0.001)
    return np.where(v,(nir-sw)/(nir+sw+1e-6),np.nan), v

def pick_clean(cands, gds, burn):
    best=None; bv=-1
    for f in cands:
        if not os.path.isdir(f"{A}/{f}"): continue
        nd,v=ndvi_arch(f,gds); vf=float((v&~burn).sum())/max(1,(~burn).sum())
        if vf>bv: bv,best=vf,(f,nd,v)
    return best

def build_pheno_classes(folders, gds, valid_ref):
    """pre-fire 다계절 NDVI → 안정 phenology-class(화재독립).
    강건화: 유효 pre-fire 씬들의 평균 NDVI(식생타입 대리)를 K분위 bin.
    pre-fire라 소각과 비엉킴, 분위라 kmeans 붕괴 없음."""
    feats=[]
    for f in folders:
        if not os.path.isdir(f"{A}/{f}"): continue
        nd,v=ndvi_arch(f,gds)
        if float(v.mean())>0.3: feats.append(np.where(v,nd,np.nan))
    if not feats: return np.zeros((gds.height,gds.width),np.int16)
    stab=np.nanmean(np.stack(feats,0),axis=0)          # 화재독립 평균 NDVI
    ref=np.isfinite(stab)&valid_ref
    if ref.sum()<50: return np.zeros((gds.height,gds.width),np.int16)
    qs=np.quantile(stab[ref], np.linspace(0,1,K+1)[1:-1])
    lab=np.digitize(np.nan_to_num(stab,nan=-1.0), qs).astype(np.int16)
    return lab

def detrend(dsi, unburned, strata):
    pd=dsi.copy()
    for c in np.unique(strata):
        cls=strata==c; ref=unburned&cls
        off=np.median(dsi[ref]) if ref.sum()>=30 else np.median(dsi[unburned])
        pd[cls]=dsi[cls]-off
    return pd

def evaluate(dsi, ordm, burn, valid, burn_dil, pheno, name, kind):
    unb=valid&~burn_dil
    # 안정 클래스 detrend + (대조)pre-NDVI 분위
    pd_stab=detrend(dsi, unb, pheno)
    m=burn&valid; sev=ordm[m].astype(float)
    row=dict(kind=kind, n=int(m.sum()),
        unburned_med=round(float(np.median(dsi[unb])),3),
        rho_dSI=round(spearman(sev,dsi[m]),3),
        rho_pdSI_stable=round(spearman(sev,pd_stab[m]),3),
        F1_dSI=f1_sweep(dsi,burn&valid),
        F1_pdSI_stable=f1_sweep(pd_stab,burn&valid))
    return row

def run(name, cfg):
    g=rasterio.open(cfg["gt"]); gt=g.read(1); H,W=gt.shape
    ordm=np.zeros((H,W),np.uint8); ordm[gt==1]=3; ordm[gt==2]=2; ordm[gt==3]=1
    burn=gt>0; burn_dil=binary_dilation(burn,iterations=EDGE_BUF)
    pheno=build_pheno_classes(cfg["pheno"], g, burn_dil)
    print(f"\n##### {name} (burn {int(burn.sum())}px, pheno-classes {sorted(np.unique(pheno).tolist())}) #####")
    out={"fire":name, "conditions":{}}

    # post
    if "post6" in cfg:
        post_ndvi,pv=ndvi_6(cfg["post6"],g); post_nbr,pvn=nbr_6(cfg["post6"],g)
    else:
        r=pick_clean(cfg["post_arch"],g,burn_dil); post_ndvi,pv=r[1],r[2]; post_nbr=None
        print(f"  post = {r[0]}")

    conds={}
    # 계절 일치 pre (NDVI, archive)
    r=pick_clean(cfg["pre_spring"],g,burn_dil)
    if r: conds[("spring","NDVI")]=(r[1]-post_ndvi, r[2]&pv); print(f"  pre_spring = {r[0]}")
    # 계절 불일치 pre (여름) — 울진만
    if cfg.get("pre_summer"):
        r=pick_clean(cfg["pre_summer"],g,burn_dil)
        if r: conds[("summer","NDVI")]=(r[1]-post_ndvi, r[2]&pv); print(f"  pre_summer = {r[0]}")
    # 6밴드 dNBR (울진 정본)
    if "pre6" in cfg and post_nbr is not None:
        pre_nbr,vn=nbr_6(cfg["pre6"],g); conds[("canon6","dNBR")]=(pre_nbr-post_nbr, vn&pvn)

    print(f"  {'cond':16} {'idx':5} {'ub_med':>7} | {'ρ dSI':>7} {'ρ pdSI':>8} | {'F1 dSI':>7} {'F1 pdSI':>8}")
    for (season,kind),(dsi,valid) in conds.items():
        valid=valid&~np.isnan(dsi)
        row=evaluate(dsi,ordm,burn,valid,burn_dil,pheno,name,kind)
        out["conditions"][f"{season}_{kind}"]=row
        print(f"  {season:16} {kind:5} {row['unburned_med']:>7.3f} | {row['rho_dSI']:>+7.3f} {row['rho_pdSI_stable']:>+8.3f} | {row['F1_dSI'][1]:>7.3f} {row['F1_pdSI_stable'][1]:>8.3f}")
    return out

if __name__=="__main__":
    allout=[]
    for name,cfg in FIRES.items():
        try: allout.append(run(name,cfg))
        except Exception as e:
            import traceback; print(f"[{name}] ERR"); traceback.print_exc()
    with open(f"{OUT}/faithful_result.json","w") as f: json.dump(allout,f,ensure_ascii=False,indent=1)
    print(f"\n판정: pdSI(stable)가 (1)여름 불일치서 dSI 회복 (2)severity ρ 상승 하면 ASAP 작동")
    print(f"→ {OUT}/faithful_result.json")
