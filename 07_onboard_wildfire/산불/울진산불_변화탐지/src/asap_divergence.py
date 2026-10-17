"""severity↑ / extent↓ 발산 정식 실험 (ASAP-pdSI).

가설: phenology-detrend(pdSI)는
  H_sev : severity ρ 를 올린다 (Δρ = ρ_pdSI - ρ_dSI > 0)
  H_ext : extent F1 을 내린다 (ΔF1 = F1_pdSI - F1_dSI < 0)
같은 (화재×지수×계절) 셀에서 두 부호가 반대 = 발산.

설계: 4화재(울진·안동=severity+extent, 산청·영덕=extent) × 지수(NDVI, dNBR) ×
      계절조건(pre = 겨울/봄/여름 2020 아카이브 + 6밴드 정본). detrend=안정
      phenology-class(pre-fire 2020 다계절 NDVI 분위, 화재독립).
통계: 셀별 부트스트랩 95% CI(Δρ: 소각픽셀 재표집 B=200 / ΔF1: 전픽셀 B=100,
      각자 full-data 최적임계 고정) + 셀 전체 부호검정(sign test).
"""
import os, json, glob, math
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation

A = os.path.expanduser("~/Desktop/sentinel2_data")
S = os.path.expanduser("~/syntax_inference/Steven/웹앱/sample_tif")
NIF = os.path.expanduser("~/Steven/nifos_burn_gt")
OUT = os.path.expanduser("~/syntax_inference/Steven/산불/울진산불_변화탐지/runs/asap")
os.makedirs(OUT, exist_ok=True)
EDGE_BUF=2; K=4; rng=np.random.default_rng(0)

FIRES = {
 "uljin2022":  dict(tile="52SEG", fire="20220304", severity=f"{NIF}/ext/Ulgin_severity.tif",
                    pre6=f"{S}/uljin2022_before.tif", post6=f"{S}/uljin2022_after.tif"),
 "andong2020": dict(tile="52SDF", fire="20200424", severity=f"{NIF}/andong/andong_severity_20m.tif",
                    post_arch_season="summer"),   # post=화재직후 여름 아카이브
 "sancheong2025": dict(tile="52SCE", fire="20250321", extent=f"{S}/sancheong2025_GT.tif",
                    pre6=f"{S}/sancheong2025_before.tif", post6=f"{S}/sancheong2025_after.tif"),
}  # 영덕 제외(독립 extent GT 없음 → dNBR-GT는 순환)

def spearman(a,b):
    ra=a.argsort().argsort().astype(float); rb=b.argsort().argsort().astype(float)
    ra-=ra.mean(); rb-=rb.mean(); d=math.sqrt((ra*ra).sum()*(rb*rb).sum())
    return float((ra*rb).sum()/d) if d else float("nan")

def f1_at(score, ref, t):
    p=score>=t; tp=int((p&ref).sum()); fp=int((p&~ref).sum()); fn=int((~p&ref).sum())
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
    return 2*P*R/(P+R) if P+R else 0.0

def best_t(score, ref, lo=-0.1, hi=0.7, step=0.02):
    bt,bf=0.0,0.0
    for t in np.arange(lo,hi,step):
        f=f1_at(score,ref,t)
        if f>bf: bf,bt=f,float(t)
    return bt,bf

def band_grid(path, gds, band=1):
    with rasterio.open(path) as ds:
        dst=np.zeros((gds.height,gds.width),"float32")
        reproject(rasterio.band(ds,band),dst,src_transform=ds.transform,src_crs=ds.crs,
                  dst_transform=gds.transform,dst_crs=gds.crs,resampling=Resampling.average)
    return dst/10000.0

def ndvi_arch(folder,gds):
    r=band_grid(f"{A}/{folder}/B04.tif",gds); n=band_grid(f"{A}/{folder}/B08.tif",gds)
    v=(n+r>0)&(n>0.001)&(r>0.001); return np.where(v,(n-r)/(n+r+1e-6),np.nan),v
def ndvi6(path,gds):
    r=band_grid(path,gds,3); n=band_grid(path,gds,4)
    v=(n+r>0)&(n>0.001)&(r>0.001); return np.where(v,(n-r)/(n+r+1e-6),np.nan),v
def nbr6(path,gds):
    n=band_grid(path,gds,4); s=band_grid(path,gds,6)
    v=(n+s>0)&(n>0.001); return np.where(v,(n-s)/(n+s+1e-6),np.nan),v

def season_of(datestr): return {12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",
                                6:"summer",7:"summer",8:"summer",9:"fall",10:"fall",11:"fall"}[int(datestr[4:6])]

def scan_seasons(tile, gds, burn, before=None, after=None, year="2020"):
    """타일의 year 아카이브를 계절별 최청정 1씬씩."""
    best={}
    for d in sorted(glob.glob(f"{A}/S2*_{tile}_{year}*_L2A")):
        name=os.path.basename(d); date=name.split("_")[2]
        if before and date>=before: continue
        if after and date<after: continue
        if not os.path.exists(f"{d}/B04.tif"): continue
        se=season_of(date); nd,v=ndvi_arch(name,gds)
        vf=float((v&~burn).sum())/max(1,(~burn).sum())
        if vf>best.get(se,(None,-1))[1]: best[se]=(name,vf,nd,v)
    return best

def build_pheno(tile,gds,burn,before):
    ss=scan_seasons(tile,gds,burn,before=before)
    feats=[t[2] for t in ss.values() if t[1]>0.3]
    if not feats: return np.zeros((gds.height,gds.width),np.int16)
    stab=np.nanmean(np.stack([np.where(np.isfinite(x),x,np.nan) for x in feats],0),0)
    ref=np.isfinite(stab)&(~burn)
    if ref.sum()<50: return np.zeros((gds.height,gds.width),np.int16)
    qs=np.quantile(stab[ref],np.linspace(0,1,K+1)[1:-1])
    return np.digitize(np.nan_to_num(stab,nan=-1),qs).astype(np.int16)

def detrend(dsi,unb,strata):
    pd=dsi.copy()
    for c in np.unique(strata):
        cls=strata==c; ref=unb&cls
        off=np.median(dsi[ref]) if ref.sum()>=30 else np.median(dsi[unb])
        pd[cls]=dsi[cls]-off
    return pd

def cell(dsi, pheno, ordm, burn, valid, burn_dil, has_sev):
    unb=valid&~burn_dil
    pd=detrend(dsi,unb,pheno)
    out={}
    # extent (항상)
    ext=burn&valid
    td,fd=best_t(dsi,ext); tp,fp=best_t(pd,ext)
    out["F1_dSI"]=round(fd,3); out["F1_pdSI"]=round(fp,3); out["dF1"]=round(fp-fd,3)
    # bootstrap ΔF1 (전픽셀, 임계 고정)
    idx=np.flatnonzero(valid); dv=dsi[valid]; pv=pd[valid]; ev=ext[valid]; dF1s=[]
    for _ in range(100):
        s=rng.integers(0,idx.size,idx.size)
        dF1s.append(f1_at(pv[s],ev[s],tp)-f1_at(dv[s],ev[s],td))
    out["dF1_CI"]=[round(float(np.percentile(dF1s,2.5)),3),round(float(np.percentile(dF1s,97.5)),3)]
    # severity (GT 있을 때)
    if has_sev:
        m=burn&valid; sev=ordm[m].astype(float); a=dsi[m]; b=pd[m]
        rd=spearman(sev,a); rp=spearman(sev,b)
        out["rho_dSI"]=round(rd,3); out["rho_pdSI"]=round(rp,3); out["drho"]=round(rp-rd,3)
        drs=[]
        for _ in range(200):
            s=rng.integers(0,sev.size,sev.size)
            drs.append(spearman(sev[s],b[s])-spearman(sev[s],a[s]))
        out["drho_CI"]=[round(float(np.percentile(drs,2.5)),3),round(float(np.percentile(drs,97.5)),3)]
    return out

def load_gt(cfg,gds):
    """severity(서수 심3중2경1)+burn. 없으면 extent만."""
    if "severity" in cfg:
        g=rasterio.open(cfg["severity"]); gt=g.read(1)
        ordm=np.zeros_like(gt,np.uint8); ordm[gt==1]=3;ordm[gt==2]=2;ordm[gt==3]=1
        return ordm, gt>0, True
    # extent GT (0/1 or grid) — GT를 gds 격자로 재투영
    ext=np.zeros((gds.height,gds.width),"float32")
    with rasterio.open(cfg["extent"]) as ds:
        reproject(rasterio.band(ds,1),ext,src_transform=ds.transform,src_crs=ds.crs,
                  dst_transform=gds.transform,dst_crs=gds.crs,resampling=Resampling.nearest)
    return None, ext>0.5, False

def run_fire(name,cfg):
    # 격자 기준 = severity GT(있으면) / 아니면 post6
    gref = cfg["severity"] if "severity" in cfg else cfg["post6"]
    g=rasterio.open(gref)
    ordm,burn,has_sev=load_gt(cfg,g)
    burn_dil=binary_dilation(burn,iterations=EDGE_BUF)
    pheno=build_pheno(cfg["tile"],g,burn_dil,before=cfg["fire"])
    # post
    if "post6" in cfg: post,pv=ndvi6(cfg["post6"],g); post_nbr,pvn=nbr6(cfg["post6"],g)
    else:
        ss=scan_seasons(cfg["tile"],g,burn_dil,after=cfg["fire"])  # 화재 이후
        pf=ss.get("summer") or ss.get("fall") or ss.get("spring")
        post,pv=pf[2],pf[3]; post_nbr=None; post_folder=pf[0]
    conds={}
    # 아카이브 계절 pre (NDVI)
    ss=scan_seasons(cfg["tile"],g,burn_dil,before=cfg["fire"])
    for se,(fold,vf,nd,v) in ss.items():
        conds[(f"arch_{se}","NDVI")]=(nd-post, v&pv)
    # 6밴드 정본 (NDVI+dNBR)
    if "pre6" in cfg:
        pn,vn=ndvi6(cfg["pre6"],g); conds[("canon","NDVI")]=(pn-post, vn&pv)
        if post_nbr is not None:
            bn,vbn=nbr6(cfg["pre6"],g); conds[("canon","dNBR")]=(bn-post_nbr, vbn&pvn)
    res={"fire":name,"has_sev":has_sev,"n_burn":int(burn.sum()),
         "pheno_classes":sorted(np.unique(pheno).tolist()),"cells":{}}
    for (season,kind),(dsi,valid) in conds.items():
        valid=valid&np.isfinite(dsi)
        if (burn&valid).sum()<100: continue
        res["cells"][f"{season}_{kind}"]=cell(dsi,pheno,ordm,burn,valid,burn_dil,has_sev)
    return res

def signtest(vals, direction):
    """direction=+1 기대>0, -1 기대<0. 성공비율·이항검정 p."""
    vals=[v for v in vals if v is not None and not math.isnan(v)]
    n=len(vals); k=sum(1 for v in vals if (v>0)==(direction>0) and v!=0)
    # 양측 이항 p (귀무 0.5)
    from math import comb
    p=sum(comb(n,i) for i in range(0,n+1) if abs(i-n/2)>=abs(k-n/2))/2**n if n else 1.0
    return n,k,round(p,4)

if __name__=="__main__":
    allres=[]
    for name,cfg in FIRES.items():
        try: r=run_fire(name,cfg); allres.append(r)
        except Exception as e:
            import traceback; print(f"[{name}] ERR"); traceback.print_exc(); continue
        print(f"\n##### {name} (sev={r['has_sev']} burn={r['n_burn']} classes={r['pheno_classes']}) #####")
        for c,d in r["cells"].items():
            sev = f"Δρ={d['drho']:+.3f}{d['drho_CI']}" if r["has_sev"] else "Δρ=—"
            print(f"  {c:16} {sev:30}  ΔF1={d['dF1']:+.3f}{d['dF1_CI']}")
    # 집계 부호검정
    dr=[d["drho"] for r in allres if r["has_sev"] for d in r["cells"].values()]
    df=[d["dF1"] for r in allres for d in r["cells"].values()]
    div=[(d["drho"]>0 and d["dF1"]<0) for r in allres if r["has_sev"] for d in r["cells"].values()]
    print("\n"+"="*60+"\n집계 (셀 전체)")
    n,k,p=signtest(dr,+1); print(f"  severity Δρ>0 : {k}/{n} 셀  sign-test p={p}")
    n,k,p=signtest(df,-1); print(f"  extent  ΔF1<0 : {k}/{n} 셀  sign-test p={p}")
    print(f"  발산(Δρ>0 & ΔF1<0): {sum(div)}/{len(div)} severity셀")
    with open(f"{OUT}/divergence_result.json","w") as f: json.dump(allres,f,ensure_ascii=False,indent=1)
    print(f"→ {OUT}/divergence_result.json")
