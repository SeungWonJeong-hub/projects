"""심각도 3등급을 SWIR 없이(RGB+NIR) 가를 수 있나 — 큐브위성엔 SWIR 이 없다.
현행은 dNBR(SWIR 필수) 0.27/0.44 절단. GT=NIFoS 안동 피해강도등급(1심/2중/3경)."""
import os, numpy as np
os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
from osgeo import gdal; gdal.UseExceptions()
P="/home/syntax-01/syntax_inference/Steven/산불/울진산불_변화탐지"
sev=gdal.Open(f"{P}/../nifos_burn_gt/andong/andong_severity_20m.tif")
g=sev.GetGeoTransform(); S=sev.ReadAsArray()
ute=np.load(f"{P}/nifos_cache/andong/ute_20m.npy")
x=np.load(f"{P}/nifos_cache/andong/scene_20m.npz")["x"].astype("float32")/10000.
# 등급 래스터를 씬 격자에 맞춰 오프셋 정렬
ox=int(round((g[0]-ute[0])/20)); oy=int(round((ute[3]-g[3])/20))
H,W=x.shape[1],x.shape[2]
S2=np.zeros((H,W),S.dtype)
ys,xs=slice(max(oy,0),min(oy+S.shape[0],H)), slice(max(ox,0),min(ox+S.shape[1],W))
S2[ys,xs]=S[max(-oy,0):max(-oy,0)+(ys.stop-ys.start), max(-ox,0):max(-ox,0)+(xs.stop-xs.start)]
pre,post=x[:6],x[6:]
idx=lambda a,b:(a-b)/(a+b+1e-6)
F={"dNBR (NIR,SWIR·현행)": idx(pre[3],pre[5])-idx(post[3],post[5]),
   "dNDVI (NIR,Red)":      idx(pre[3],pre[2])-idx(post[3],post[2]),
   "dGNDVI(NIR,Green)":    idx(pre[3],pre[1])-idx(post[3],post[1]),
   "dNIR  (NIR 차)":       pre[3]-post[3],
   "dBlue (NIR,Blue)":     idx(pre[3],pre[0])-idx(post[3],post[0])}
m=np.isin(S2,[1,2,3])
print(f"안동 20m 격자 · 등급화소 {m.sum():,} (심 {(S2==1).sum():,} 중 {(S2==2).sum():,} 경 {(S2==3).sum():,})\n")
from sklearn.metrics import roc_auc_score
print(f"{'지표':<22}{'심':>8}{'중':>8}{'경':>8}   {'단조':>5}{'AUC 심vs경':>11}")
for n,f in F.items():
    med=[float(np.median(f[S2==k])) for k in (1,2,3)]
    mono="예" if med[0]>med[1]>med[2] or med[0]<med[1]<med[2] else "아니오"
    sel=np.isin(S2,[1,3]); a=roc_auc_score((S2[sel]==1).astype(int), f[sel])
    print(f"{n:<22}{med[0]:>8.3f}{med[1]:>8.3f}{med[2]:>8.3f}   {mono:>5}{max(a,1-a):>11.3f}")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import cross_val_score
y=S2[m]
for name,keys in (("SWIR 포함 (dNBR+dNDVI)",["dNBR (NIR,SWIR·현행)","dNDVI (NIR,Red)"]),
                  ("RGB+NIR 만 (4지표)",["dNDVI (NIR,Red)","dGNDVI(NIR,Green)","dNIR  (NIR 차)","dBlue (NIR,Blue)"])):
    X=np.stack([F[k][m] for k in keys],1)
    s=cross_val_score(HGB(max_iter=120,random_state=0),X,y,cv=3)
    print(f"\n3등급 분류 정확도 · {name:<24} {s.mean():.3f} ± {s.std():.3f}")
print(f"(최빈등급만 찍기 = {np.bincount(y)[1:].max()/len(y):.3f})")
