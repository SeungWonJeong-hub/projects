"""불일치–이득 곡선: pdSI 이득이 계절 불일치 크기의 함수임을 정량화.

x = 계절 불일치 크기 = |median_{unburned}(dSI)|  (0=일치, 클수록 phenology 오염 큼)
y = pdSI 이득 = ΔF1(extent)  및  Δρ(severity, GT 있을 때)

점 생성: 화재별 pre-fire 아카이브 전 씬(date<fire, 청정만) 각각 = 1점.
   → 여러 계절·연도가 자연히 불일치 크기 범위를 채움. pool 후 선형회귀+영교차점.
가설: 이득은 불일치 크기와 함께 증가, 어떤 임계 이하선 detrend가 해롭고 이상서 이롭다.
"""
import os, glob, json, math
import numpy as np, rasterio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import sys; sys.path.insert(0, os.path.dirname(__file__))
from asap_divergence import (A, S, NIF, band_grid, ndvi_arch, ndvi6, nbr6, detrend,
                             best_t, f1_at, spearman, build_pheno, season_of, load_gt, K, EDGE_BUF)

OUT = os.path.expanduser("~/syntax_inference/Steven/산불/울진산불_변화탐지/runs/asap")
rng = np.random.default_rng(0)

FIRES = {
 "uljin2022":  dict(tile="52SEG", fire="20220304", severity=f"{NIF}/ext/Ulgin_severity.tif",
                    pre6=f"{S}/uljin2022_before.tif", post6=f"{S}/uljin2022_after.tif"),
 "andong2020": dict(tile="52SDF", fire="20200424", severity=f"{NIF}/andong/andong_severity_20m.tif",
                    post_after="20200424"),
 "sancheong2025": dict(tile="52SCE", fire="20250321", extent=f"{S}/sancheong2025_GT.tif",
                    pre6=f"{S}/sancheong2025_before.tif", post6=f"{S}/sancheong2025_after.tif"),
 # 2019-04 강원 대형산불(RapidEye NIFoS severity, 아카이브 2019 pre/post). post=화재직후 2019.
 "re2019_63": dict(tile="52SDG", fire="20190404", severity=f"{NIF}/ext/re2019_63_severity_20m.tif", post_after="20190404"),
 "re2019_71": dict(tile="52SDH", fire="20190404", severity=f"{NIF}/ext/re2019_71_severity_20m.tif", post_after="20190404"),
 "re2019_74": dict(tile="52SDH", fire="20190404", severity=f"{NIF}/ext/re2019_74_severity_20m.tif", post_after="20190404"),
}
MINVF = 0.6  # unburned 유효율 하한(구름 배제)
MIN_F1_BASE = 0.40  # 화재 가드: baseline median F1_dSI 이하면 post 불량으로 보고 제외

def all_pre_scenes(tile, before):
    out=[]
    for d in sorted(glob.glob(f"{A}/S2*_{tile}_*_L2A")):
        date=os.path.basename(d).split("_")[2]
        if date<before and os.path.exists(f"{d}/B04.tif"): out.append(os.path.basename(d))
    return out

def post_ndvi_for(cfg, g, burn_dil):
    if "post6" in cfg:
        nd,v=ndvi6(cfg["post6"],g); return nd,v
    # 안동: 화재 직후 최청정 아카이브
    best=None; bv=-1
    for d in sorted(glob.glob(f"{A}/S2*_{cfg['tile']}_*_L2A")):
        date=os.path.basename(d).split("_")[2]
        if date<=cfg["post_after"] or not os.path.exists(f"{d}/B04.tif"): continue
        if date[:4]!=cfg["post_after"][:4]: continue  # 같은해 직후
        nd,v=ndvi_arch(os.path.basename(d),g); vf=float((v&~burn_dil).sum())/max(1,(~burn_dil).sum())
        if vf>bv: bv,best=vf,(nd,v)
    return best

def run_fire(name,cfg,pts):
    gref=cfg["severity"] if "severity" in cfg else cfg["post6"]
    g=rasterio.open(gref); ordm,burn,has_sev=load_gt(cfg,g)
    bd=binary_dilation(burn,iterations=EDGE_BUF)
    pheno=build_pheno(cfg["tile"],g,bd,before=cfg["fire"])
    pr=post_ndvi_for(cfg,g,bd)
    if pr is None: print(f"  {name}: post 없음 → 제외"); return 0
    post,pv=pr
    local=[]
    for folder in all_pre_scenes(cfg["tile"], cfg["fire"]):
        nd,v=ndvi_arch(folder,g); valid=v&pv&np.isfinite(nd)&np.isfinite(post)
        unb=valid&~bd
        if float(unb.sum())/max(1,(~bd).sum())<MINVF: continue
        dsi=nd-post
        off=float(np.median(dsi[unb]))
        pd=detrend(dsi,unb,pheno)
        ext=burn&valid
        if ext.sum()<100: continue
        td,fd=best_t(dsi,ext); tp,fp=best_t(pd,ext)
        rec=dict(fire=name,folder=folder,season=season_of(folder.split("_")[2]),
                 mismatch=round(abs(off),4),offset=round(off,4),
                 dF1=round(fp-fd,4),F1_dSI=round(fd,3),F1_pdSI=round(fp,3),has_sev=has_sev)
        if has_sev:
            m=burn&valid; sev=ordm[m].astype(float)
            rec["drho"]=round(spearman(sev,pd[m])-spearman(sev,dsi[m]),4)
        local.append(rec)
    # 가드: baseline 신호 없으면(불량 post) 화재 통째 제외
    if not local:
        print(f"  {name}: 유효 pre-scene 0 → 제외"); return 0
    f1med=float(np.median([r["F1_dSI"] for r in local]))
    if f1med<MIN_F1_BASE:
        print(f"  {name}: baseline F1_dSI중앙={f1med:.2f}<{MIN_F1_BASE} → 제외(불량 post)"); return 0
    pts.extend(local)
    print(f"  {name}: {len(local)} pre-scenes (baseF1={f1med:.2f}, classes {sorted(np.unique(pheno).tolist())})")
    return len(local)

def fit(x,y,B=3000):
    x=np.array(x); y=np.array(y)
    if len(x)<3: return None
    m,b=np.polyfit(x,y,1); yh=m*x+b
    ss=1-((y-yh)**2).sum()/(((y-y.mean())**2).sum()+1e-9)
    zc=-b/m if abs(m)>1e-9 else float("nan")
    # 부트스트랩: 점 재표집 → slope·zero-cross 분포 → 95% CI
    sl=[]; zz=[]; n=len(x)
    for _ in range(B):
        s=rng.integers(0,n,n); mm,bb=np.polyfit(x[s],y[s],1)
        sl.append(mm)
        if abs(mm)>1e-9: zz.append(-bb/mm)
    ci=lambda a:[round(float(np.percentile(a,2.5)),4),round(float(np.percentile(a,97.5)),4)]
    return dict(slope=round(float(m),4),intercept=round(float(b),4),R2=round(float(ss),3),
                zero_cross=round(float(zc),4),n=n,
                slope_CI=ci(sl), zero_CI=ci([z for z in zz if -1<z<1]),
                slope_p=round(float(min(np.mean(np.array(sl)<=0),np.mean(np.array(sl)>=0))*2),4))

if __name__=="__main__":
    pts=[]
    print("pre-scene 열거·계산:")
    for name,cfg in FIRES.items():
        try: run_fire(name,cfg,pts)
        except Exception as e:
            import traceback; print(f"[{name}] ERR"); traceback.print_exc()
    json.dump(pts, open(f"{OUT}/curve_points.json","w"), ensure_ascii=False, indent=1)

    # 회귀
    ext_x=[p["mismatch"] for p in pts]; ext_y=[p["dF1"] for p in pts]
    sev_x=[p["mismatch"] for p in pts if p.get("has_sev")]; sev_y=[p["drho"] for p in pts if "drho" in p]
    fe=fit(ext_x,ext_y); fs=fit(sev_x,sev_y)
    print("\n=== 불일치–이득 회귀 (부트스트랩 95% CI) ===")
    for nm,ff in [("extent ΔF1",fe),("severity Δρ",fs)]:
        if ff: print(f"  {nm:12}: slope={ff['slope']} CI{ff['slope_CI']} (p={ff['slope_p']}) | "
                     f"임계 zero@|mm|={ff['zero_cross']} CI{ff['zero_CI']} | R²={ff['R2']} n={ff['n']}")
    json.dump({"extent":fe,"severity":fs}, open(f"{OUT}/curve_fit.json","w"), ensure_ascii=False, indent=1)

    # 그림
    colors={"uljin2022":"#1f77b4","andong2020":"#d62728","sancheong2025":"#2ca02c",
            "re2019_63":"#9467bd","re2019_71":"#ff7f0e","re2019_74":"#8c564b"}
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    for p in pts:
        ax[0].scatter(p["mismatch"],p["dF1"],c=colors.get(p["fire"],"gray"),s=32,alpha=0.8)
    if fe:
        xs=np.linspace(0,max(ext_x),50); ax[0].plot(xs,fe["slope"]*xs+fe["intercept"],"k--",lw=1.5)
        ax[0].axvspan(fe["zero_CI"][0],fe["zero_CI"][1],color="orange",alpha=0.15)
        ax[0].axvline(fe["zero_cross"],color="orange",lw=1,ls=":")
    ax[0].axhline(0,color="gray",lw=0.8); ax[0].set_xlabel("Seasonal mismatch  |median unburned dSI|")
    ax[0].set_ylabel("$\\Delta$F1  (pdSI $-$ dSI, extent)"); ax[0].set_title("Extent: mismatch vs gain"); ax[0].legend(fontsize=8)
    for p in pts:
        if "drho" in p: ax[1].scatter(p["mismatch"],p["drho"],c=colors.get(p["fire"]),s=32,alpha=0.8)
    if fs:
        xs=np.linspace(0,max(sev_x),50); ax[1].plot(xs,fs["slope"]*xs+fs["intercept"],"k--",lw=1.5)
        ax[1].axvspan(fs["zero_CI"][0],fs["zero_CI"][1],color="orange",alpha=0.15)
        ax[1].axvline(fs["zero_cross"],color="orange",lw=1,ls=":")
    ax[1].axhline(0,color="gray",lw=0.8); ax[1].set_xlabel("Seasonal mismatch  |median unburned dSI|")
    ax[1].set_ylabel("$\\Delta\\rho$  (pdSI $-$ dSI, severity)"); ax[1].set_title("Severity: mismatch vs gain")
    present={p["fire"] for p in pts}
    handles=[plt.Line2D([],[],marker="o",ls="",color=c,label=n) for n,c in colors.items() if n in present]
    fe_l=[plt.Line2D([],[],ls="--",color="k",label=f"fit R²={fe['R2']}  zero@{fe['zero_cross']} CI{fe['zero_CI']}")] if fe else []
    fs_l=[plt.Line2D([],[],ls="--",color="k",label=f"fit R²={fs['R2']}  zero@{fs['zero_cross']} CI{fs['zero_CI']}")] if fs else []
    ax[0].legend(handles=handles+fe_l,fontsize=7,loc="upper left")
    ax[1].legend(handles=fs_l,fontsize=8,loc="upper left")
    plt.tight_layout(); plt.savefig(f"{OUT}/mismatch_gain_curve.png",dpi=130)
    print(f"\n→ {OUT}/mismatch_gain_curve.png  ({len(pts)}점)")
    print(f"→ {OUT}/curve_points.json")
