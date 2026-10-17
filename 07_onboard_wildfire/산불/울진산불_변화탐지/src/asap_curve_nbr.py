"""dNBR severity 불일치-이득 곡선 (n 확대판).
severity GT 5화재(울진·안동·봉화·군위·밀양)의 다계절 SWIR pre/post(dl/<fire>_nbr/)로
dNBR·dNDVI 점 생성. post=burn 가장 선명(dNBR best-F1) 자동선택. pre별 점→
지수(dNBR/dNDVI)별 severity·extent 곡선 부트스트랩 적합. NDVI 대비 dNBR가
severity 구배 더 잘 살리는지 검증."""
import os, json
import numpy as np, rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import sys; sys.path.insert(0, os.path.dirname(__file__))
from asap_divergence import band_grid, detrend, best_t, spearman, load_gt, K, EDGE_BUF

NIF=os.path.expanduser("~/Steven/nifos_burn_gt")
DL =os.path.expanduser("~/syntax_inference/Steven/산불/울진산불_변화탐지/runs/asap/dl")
OUT=os.path.expanduser("~/syntax_inference/Steven/산불/울진산불_변화탐지/runs/asap")
rng=np.random.default_rng(0)
GT={"uljin":f"{NIF}/ext/Ulgin_severity.tif","andong":f"{NIF}/andong/andong_severity_20m.tif",
    "bonghwa":f"{NIF}/ext/봉화_severity_20m.tif","gunwi":f"{NIF}/ext/군위_severity_20m.tif",
    "miryang":f"{NIF}/ext/밀양_severity_20m.tif"}

def rd(sd,g,b): return band_grid(f"{sd}/{b}.tif",g)
def si(sd,g,kind):
    if kind=="NBR": n=rd(sd,g,"B08"); x=rd(sd,g,"B12"); v=(n+x>0)&(n>0.001)
    else: x=rd(sd,g,"B04"); n=rd(sd,g,"B08"); v=(n+x>0)&(n>0.001)&(x>0.001)
    return np.where(v,(n-x)/(n+x+1e-6),np.nan),v

def build_pheno(pren,g,burn):
    feats=[np.where(v,nd,np.nan) for nd,v in pren if float(v.mean())>0.3]
    if not feats: return np.zeros((g.height,g.width),np.int16)
    stab=np.nanmean(np.stack(feats,0),0); ref=np.isfinite(stab)&(~burn)
    if ref.sum()<50: return np.zeros((g.height,g.width),np.int16)
    qs=np.quantile(stab[ref],np.linspace(0,1,K+1)[1:-1])
    return np.digitize(np.nan_to_num(stab,nan=-1),qs).astype(np.int16)

def pt(dsi,pheno,ordm,burn,valid,bd):
    unb=valid&~bd; pd=detrend(dsi,unb,pheno); ext=burn&valid
    if ext.sum()<80: return None
    _,fd=best_t(dsi,ext); _,fp=best_t(pd,ext); m=burn&valid; sev=ordm[m].astype(float)
    return dict(mismatch=round(abs(float(np.median(dsi[unb]))),4),dF1=round(fp-fd,4),
                F1_dSI=round(fd,3),drho=round(spearman(sev,pd[m])-spearman(sev,dsi[m]),4),
                rho_dSI=round(spearman(sev,dsi[m]),3))

def run_fire(fire):
    g=rasterio.open(GT[fire]); ordm,burn,_=load_gt({"severity":GT[fire]},g)
    bd=binary_dilation(burn,iterations=EDGE_BUF)
    d=f"{DL}/{fire}_nbr";
    if not os.path.exists(f"{d}/picks.json"): return []
    picks=json.load(open(f"{d}/picks.json")); sdir=lambda l:f"{d}/{picks[l]['id']}"
    pre=[l for l,v in picks.items() if v["role"]=="pre" and os.path.exists(f"{sdir(l)}/B12.tif")]
    post=[l for l,v in picks.items() if v["role"]=="post" and os.path.exists(f"{sdir(l)}/B12.tif")]
    if not pre or not post: return []
    pren=[si(sdir(l),g,"NDVI") for l in pre]; pheno=build_pheno(pren,g,bd)
    refl=next((l for l in pre if "sp" in l),pre[0]); refn,_=si(sdir(refl),g,"NBR")
    bestp,bf=None,-1
    for pl in post:
        pn,pv=si(sdir(pl),g,"NBR"); dsi=refn-pn; _,f=best_t(dsi,burn&(pv&np.isfinite(dsi)))
        if f>bf: bf,bestp=f,pl
    if bf<0.3: print(f"  [{fire}] post 불명확 F1={bf:.2f} skip"); return []
    print(f"  [{fire}] post={picks[bestp]['dt']}(F1={bf:.2f}) pre={len(pre)}씬 classes={sorted(np.unique(pheno).tolist())}")
    out=[]
    for kind in ("NBR","NDVI"):
        postx,pvx=si(sdir(bestp),g,kind)
        for l in pre:
            prex,pv=si(sdir(l),g,kind); valid=pv&pvx&np.isfinite(prex)&np.isfinite(postx)
            r=pt(prex-postx,pheno,ordm,burn,valid,bd)
            if r: r.update(fire=fire,index=kind); out.append(r)
    return out

def fit(x,y,B=3000):
    x=np.array(x); y=np.array(y)
    if len(x)<3: return None
    m,b=np.polyfit(x,y,1); ss=1-(((y-(m*x+b))**2).sum())/(((y-y.mean())**2).sum()+1e-9)
    zc=-b/m if abs(m)>1e-9 else float("nan"); sl=[]; zz=[]; n=len(x)
    for _ in range(B):
        s=rng.integers(0,n,n); mm,bb=np.polyfit(x[s],y[s],1); sl.append(mm)
        if abs(mm)>1e-9: zz.append(-bb/mm)
    ci=lambda a:[round(float(np.percentile(a,2.5)),4),round(float(np.percentile(a,97.5)),4)]
    return dict(slope=round(float(m),4),R2=round(float(ss),3),zero_cross=round(float(zc),4),n=n,
                slope_CI=ci(sl),zero_CI=ci([z for z in zz if -1<z<1]),
                slope_p=round(float(min(np.mean(np.array(sl)<=0),np.mean(np.array(sl)>=0))*2),4))

if __name__=="__main__":
    print("dNBR severity 화재 점 생성:")
    pts=[]
    for f in GT: pts+=run_fire(f)
    for idx in ("NBR","NDVI"):
        n=sum(p["index"]==idx for p in pts); rbase=np.median([p["rho_dSI"] for p in pts if p["index"]==idx]) if n else 0
        print(f"  {idx}: {n}점  baseline ρ(dSI) 중앙={rbase:.3f}")
    json.dump(pts,open(f"{OUT}/curve_points_nbr.json","w"),ensure_ascii=False,indent=1)
    fits={}
    for idx in ("NBR","NDVI"):
        P=[p for p in pts if p["index"]==idx]
        fits[f"{idx}_severity"]=fit([p["mismatch"] for p in P],[p["drho"] for p in P])
        fits[f"{idx}_extent"]=fit([p["mismatch"] for p in P],[p["dF1"] for p in P])
    json.dump(fits,open(f"{OUT}/curve_fit_nbr.json","w"),ensure_ascii=False,indent=1)
    print("\n=== dNBR vs NDVI severity 곡선(부트스트랩 CI) ===")
    for k,f in fits.items():
        if f: print(f"  {k:14}: slope={f['slope']} CI{f['slope_CI']} p={f['slope_p']} | 임계 {f['zero_cross']} CI{f['zero_CI']} | R²={f['R2']} n={f['n']}")
    # 그림: dNBR severity(왼) vs NDVI severity(오)
    cols={"uljin":"#1f77b4","andong":"#d62728","bonghwa":"#9467bd","gunwi":"#ff7f0e","miryang":"#17becf"}
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    for i,idx in enumerate(("NBR","NDVI")):
        P=[p for p in pts if p["index"]==idx]; ff=fits[f"{idx}_severity"]
        for p in P: ax[i].scatter(p["mismatch"],p["drho"],c=cols.get(p["fire"],"gray"),s=28,alpha=0.8)
        if ff:
            xs=np.linspace(0,max(p["mismatch"] for p in P),50); ax[i].plot(xs,ff["slope"]*xs+(-ff["slope"]*ff["zero_cross"]),"k--",lw=1.5)
            ax[i].axvspan(*ff["zero_CI"],color="orange",alpha=0.15); ax[i].axvline(ff["zero_cross"],color="orange",ls=":",lw=1)
        ax[i].axhline(0,color="gray",lw=0.8); ax[i].set_title(f"{'dNBR' if idx=='NBR' else 'dNDVI'} severity (n={ff['n']}, R²={ff['R2']})")
        ax[i].set_xlabel("Seasonal mismatch |median unburned dSI|"); ax[i].set_ylabel("$\\Delta\\rho$ (pdSI-dSI)")
    present={p["fire"] for p in pts}
    ax[0].legend(handles=[plt.Line2D([],[],marker="o",ls="",color=c,label=n) for n,c in cols.items() if n in present],fontsize=8,loc="upper left")
    plt.tight_layout(); plt.savefig(f"{OUT}/dnbr_severity_curve.png",dpi=130)
    print(f"\n→ {OUT}/dnbr_severity_curve.png")
