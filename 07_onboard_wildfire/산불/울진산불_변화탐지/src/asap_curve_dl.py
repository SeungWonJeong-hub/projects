"""다운로드한 2022 NIFoS 4화재(봉화·군위·밀양·양구)를 불일치-이득 곡선에 합류.
각 화재: severity GT, pre 6씬(2019-21 계절)+post 후보 3씬(2022-23). post는 burn
가장 선명한 것 자동선택(dNBR best-F1). pre별 NDVI 점 생성→기존 curve_points와 병합→
재적합(부트스트랩 CI)+재플롯. SWIR 있으니 dNBR 점도 별도 기록."""
import os, json, glob
import numpy as np, rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import sys; sys.path.insert(0, os.path.dirname(__file__))
from asap_divergence import band_grid, detrend, best_t, f1_at, spearman, load_gt, K, EDGE_BUF

NIF=os.path.expanduser("~/Steven/nifos_burn_gt/ext")
DL =os.path.expanduser("~/syntax_inference/Steven/산불/울진산불_변화탐지/runs/asap/dl")
OUT=os.path.expanduser("~/syntax_inference/Steven/산불/울진산불_변화탐지/runs/asap")
rng=np.random.default_rng(0)
FIRE_GT={"bonghwa":"봉화_severity_20m.tif","gunwi":"군위_severity_20m.tif",
         "miryang":"밀양_severity_20m.tif","yanggu":"양구_severity_20m.tif"}

def rd(sdir,g,b): return band_grid(f"{sdir}/{b}.tif",g)
def ndvi(sdir,g):
    r=rd(sdir,g,"B04"); n=rd(sdir,g,"B08"); v=(n+r>0)&(n>0.001)&(r>0.001)
    return np.where(v,(n-r)/(n+r+1e-6),np.nan),v
def nbr(sdir,g):
    n=rd(sdir,g,"B08"); s=rd(sdir,g,"B12"); v=(n+s>0)&(n>0.001)
    return np.where(v,(n-s)/(n+s+1e-6),np.nan),v

def build_pheno(pre_ndvis,g,burn):
    feats=[np.where(v,nd,np.nan) for nd,v in pre_ndvis if float(v.mean())>0.3]
    if not feats: return np.zeros((g.height,g.width),np.int16)
    stab=np.nanmean(np.stack(feats,0),0); ref=np.isfinite(stab)&(~burn)
    if ref.sum()<50: return np.zeros((g.height,g.width),np.int16)
    qs=np.quantile(stab[ref],np.linspace(0,1,K+1)[1:-1])
    return np.digitize(np.nan_to_num(stab,nan=-1),qs).astype(np.int16)

def point(dsi,pheno,ordm,burn,valid,bd):
    unb=valid&~bd; pd=detrend(dsi,unb,pheno)
    ext=burn&valid
    if ext.sum()<80: return None
    _,fd=best_t(dsi,ext); _,fp=best_t(pd,ext)
    m=burn&valid; sev=ordm[m].astype(float)
    return dict(mismatch=round(abs(float(np.median(dsi[unb]))),4),
                dF1=round(fp-fd,4),F1_dSI=round(fd,3),
                drho=round(spearman(sev,pd[m])-spearman(sev,dsi[m]),4))

def run_fire(fire):
    gt=f"{NIF}/{FIRE_GT[fire]}"; g=rasterio.open(gt)
    ordm,burn,_=load_gt({"severity":gt},g); bd=binary_dilation(burn,iterations=EDGE_BUF)
    picks=json.load(open(f"{DL}/{fire}/picks.json"))
    sdir=lambda lab: f"{DL}/{fire}/{picks[lab]['id']}"
    pre=[l for l in picks if l.startswith("pre")]; post=[l for l in picks if l.startswith("post")]
    if not pre or not post: return []
    pre_ndvis=[ndvi(sdir(l),g) for l in pre]
    pheno=build_pheno(pre_ndvis,g,bd)
    # post 선택: burn 가장 선명(dNBR best-F1), 봄 pre 기준
    refl=next((l for l in pre if "sp" in l), pre[0]); ref_nbr,_=nbr(sdir(refl),g)
    bestp=None; bestf=-1
    for pl in post:
        pn,pv=nbr(sdir(pl),g); dsi=ref_nbr-pn; val=pv&np.isfinite(dsi)
        _,f=best_t(dsi,burn&val)
        if f>bestf: bestf,bestp=f,pl
    if bestf<0.3: print(f"  [{fire}] post burn 불명확(F1={bestf:.2f}) → skip"); return []
    post_nd,pvn=ndvi(sdir(bestp),g); post_nb,pvb=nbr(sdir(bestp),g)
    print(f"  [{fire}] post={picks[bestp]['dt']}(F1={bestf:.2f}) classes={sorted(np.unique(pheno).tolist())}")
    pts=[]
    for l,(nd,v) in zip(pre,pre_ndvis):
        for idx,(pre_x,pv,post_x) in [("NDVI",(nd,v,post_nd)),
                                       ("NBR",(nbr(sdir(l),g)[0],nbr(sdir(l),g)[1],post_nb))]:
            valid=pv&(pvn if idx=="NDVI" else pvb)&np.isfinite(pre_x)&np.isfinite(post_x)
            r=point(pre_x-post_x,pheno,ordm,burn,valid,bd)
            if r: r.update(fire=fire,season=l,index=idx,has_sev=True); pts.append(r)
    return pts

def fit(x,y,B=3000):
    x=np.array(x); y=np.array(y)
    if len(x)<3: return None
    m,b=np.polyfit(x,y,1); yh=m*x+b
    ss=1-((y-yh)**2).sum()/(((y-y.mean())**2).sum()+1e-9); zc=-b/m if abs(m)>1e-9 else float("nan")
    sl=[]; zz=[]; n=len(x)
    for _ in range(B):
        s=rng.integers(0,n,n); mm,bb=np.polyfit(x[s],y[s],1); sl.append(mm)
        if abs(mm)>1e-9: zz.append(-bb/mm)
    ci=lambda a:[round(float(np.percentile(a,2.5)),4),round(float(np.percentile(a,97.5)),4)]
    return dict(slope=round(float(m),4),intercept=round(float(b),4),R2=round(float(ss),3),
                zero_cross=round(float(zc),4),n=n,slope_CI=ci(sl),zero_CI=ci([z for z in zz if -1<z<1]),
                slope_p=round(float(min(np.mean(np.array(sl)<=0),np.mean(np.array(sl)>=0))*2),4))

if __name__=="__main__":
    print("다운로드 4화재 점 생성:")
    new=[]
    for fire in FIRE_GT: new+=run_fire(fire)
    print(f"  신규 점: {len(new)} (NDVI {sum(p['index']=='NDVI' for p in new)}, NBR {sum(p['index']=='NBR' for p in new)})")
    # 기존(아카이브 3화재, 전부 NDVI) 로드
    old=json.load(open(f"{OUT}/curve_points.json"))
    for p in old: p.setdefault("index","NDVI")
    allpts=old+new
    json.dump(allpts, open(f"{OUT}/curve_points_all.json","w"), ensure_ascii=False, indent=1)
    # NDVI-only 곡선(일관 x축)
    nd=[p for p in allpts if p["index"]=="NDVI"]
    ex=fit([p["mismatch"] for p in nd],[p["dF1"] for p in nd])
    sv=fit([p["mismatch"] for p in nd if p.get("has_sev")],[p["drho"] for p in nd if p.get("has_sev") and "drho" in p])
    # dNBR 곡선(다운로드 4화재만)
    nb=[p for p in new if p["index"]=="NBR"]
    exb=fit([p["mismatch"] for p in nb],[p["dF1"] for p in nb])
    svb=fit([p["mismatch"] for p in nb],[p["drho"] for p in nb])
    fits={"NDVI_extent":ex,"NDVI_severity":sv,"NBR_extent":exb,"NBR_severity":svb}
    json.dump(fits, open(f"{OUT}/curve_fit_all.json","w"), ensure_ascii=False, indent=1)
    print("\n=== 7화재 병합 회귀(부트스트랩 CI) ===")
    for k,f in fits.items():
        if f: print(f"  {k:14}: slope={f['slope']} CI{f['slope_CI']} p={f['slope_p']} | 임계 {f['zero_cross']} CI{f['zero_CI']} | R²={f['R2']} n={f['n']}")

    # 그림(NDVI, 7화재)
    cols={"uljin2022":"#1f77b4","andong2020":"#d62728","sancheong2025":"#2ca02c",
          "bonghwa":"#9467bd","gunwi":"#ff7f0e","miryang":"#17becf","yanggu":"#8c564b"}
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    for p in nd: ax[0].scatter(p["mismatch"],p["dF1"],c=cols.get(p["fire"],"gray"),s=26,alpha=0.75)
    if ex:
        xs=np.linspace(0,max(p["mismatch"] for p in nd),50); ax[0].plot(xs,ex["slope"]*xs+ex["intercept"],"k--",lw=1.5)
        ax[0].axvspan(*ex["zero_CI"],color="orange",alpha=0.15); ax[0].axvline(ex["zero_cross"],color="orange",ls=":",lw=1)
    ax[0].axhline(0,color="gray",lw=0.8); ax[0].set_title(f"Extent (NDVI, 7 fires n={ex['n']})")
    ax[0].set_xlabel("Seasonal mismatch |median unburned dSI|"); ax[0].set_ylabel("$\\Delta$F1 (pdSI-dSI)")
    for p in nd:
        if p.get("has_sev") and "drho" in p: ax[1].scatter(p["mismatch"],p["drho"],c=cols.get(p["fire"]),s=26,alpha=0.75)
    if sv:
        xs=np.linspace(0,max(p["mismatch"] for p in nd if p.get("has_sev")),50); ax[1].plot(xs,sv["slope"]*xs+sv["intercept"],"k--",lw=1.5)
        ax[1].axvspan(*sv["zero_CI"],color="orange",alpha=0.15); ax[1].axvline(sv["zero_cross"],color="orange",ls=":",lw=1)
    ax[1].axhline(0,color="gray",lw=0.8); ax[1].set_title(f"Severity (NDVI, 6 fires n={sv['n']})")
    ax[1].set_xlabel("Seasonal mismatch |median unburned dSI|"); ax[1].set_ylabel("$\\Delta\\rho$ (pdSI-dSI)")
    present={p["fire"] for p in nd}
    h=[plt.Line2D([],[],marker="o",ls="",color=c,label=n) for n,c in cols.items() if n in present]
    h+=[plt.Line2D([],[],ls="--",color="k",label=f"zero@{ex['zero_cross']} CI{ex['zero_CI']}")]
    ax[0].legend(handles=h,fontsize=7,loc="upper left")
    ax[1].legend(handles=[plt.Line2D([],[],ls="--",color="k",label=f"zero@{sv['zero_cross']} CI{sv['zero_CI']}")],fontsize=8,loc="upper left")
    plt.tight_layout(); plt.savefig(f"{OUT}/mismatch_gain_curve_7fires.png",dpi=130)
    print(f"\n→ {OUT}/mismatch_gain_curve_7fires.png ({len(nd)} NDVI점)")
