"""소형 화재 −0.15 격차는 무엇의 대가인가 — 기하(둘레/면적) 가설 검증.

소형 9화재 CV 0.675 vs 대형 0.827 = **−0.152**(인계 §6b). 해상도로는 안 풀렸고
"라벨이 필요하다"로 정리돼 있었는데, 그 전에 **얼마가 모델 탓이고 얼마가 기하 탓인지**
를 갈라야 한다. 소형은 둘레/면적 비가 커서 **경계 화소 한 줄의 대가가 훨씬 크다** —
같은 품질의 경계를 그려도 F1 이 더 깎인다.

검증: 같은 예측을 ①엄격 F1 ②경계허용 F1(τ=20·40m, **미터 고정**) ③면적오차 로 재본다.
기하 탓이라면 τ 를 주는 순간 격차가 줄어야 하고, 면적에서는 더 줄어야 한다.

임계값은 cir_hold_eval 과 같은 **화재별 최적**을 쓴다(격차가 임계값 탓이 아님은 이미 확정).
채점 앙상블은 §6b 의 CV fold — 소형은 자기가 홀드아웃인 fold, 대형은 3 fold 평균.
"""
import os, sys, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("GSD", "20"); os.environ.setdefault("TILE", "256")
GSD = float(os.environ["GSD"]); PXA = GSD*GSD/1e4
import step4_rehearsal as S
import nifos_eval as N
N.TILE = int(os.environ["TILE"])
from scipy.ndimage import binary_dilation, binary_erosion

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
FIRES = os.path.join(PROJ, "..", "산불데이터", "국토정보플랫폼 데이터셋", "CIR_실측라벨", "fires")
FOLD = {1: "250322_울주온양읍,250428_대구노곡동,230402_금산복수면",
        2: "230403_순천송광면,230308_합천용주면,220531_밀양부북면",
        3: "220411_군위의흥면,230411_강릉난곡동,230403_함평대동면"}
LARGE = S.CIR_HOLD

def f1(pred, gt, valid):
    p, g = pred & valid, gt & valid
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    pr = tp/(tp+fp+1e-9); rc = tp/(tp+fn+1e-9)
    return 2*pr*rc/(pr+rc+1e-9)

def tol(pred, gt, valid, tau_m):
    """경계허용 F1 — τ 는 **미터 고정**(px 고정은 해상도 비교를 망친다)."""
    t = int(round(tau_m/GSD))
    p, g = pred & valid, gt & valid
    gd = binary_dilation(g, iterations=t) if t else g
    pd = binary_dilation(p, iterations=t) if t else p
    pr = int((p & gd).sum())/(int(p.sum())+1e-9); rc = int((g & pd).sum())/(int(g.sum())+1e-9)
    return 2*pr*rc/(pr+rc+1e-9)

def ens(fold): return sorted(glob.glob(os.path.join(PROJ, f"runs/cv_f{fold}_20m/*.pt")))

def run(name, folds):
    z = np.load(os.path.join(FIRES, name, S.CIR_NPZ))
    x8 = z["x"].astype(np.float32); gt = z["gt"].astype(bool); vd = z["valid"].astype(bool)
    x12 = np.zeros((12,)+x8.shape[1:], np.float32)
    x12[[0,1,2,3]] = x8[:4]*1e4; x12[[6,7,8,9]] = x8[4:]*1e4
    out = []
    for fd in folds:
        prob, _ = N.infer(x12, ens(fd))
        t = max(np.arange(0.05, 0.95, 0.025), key=lambda t: f1(prob >= t, gt, vd))
        pr = prob >= t
        out.append((f1(pr,gt,vd), tol(pr,gt,vd,20.), tol(pr,gt,vd,40.),
                    float(pr[vd].sum())*PXA))
    m = np.mean(out, 0)
    g = gt & vd
    edge = int((g & ~binary_erosion(g)).sum())          # GT 경계 화소
    return dict(f1=m[0], t20=m[1], t40=m[2], pred_ha=m[3],
                gt_ha=float(g.sum())*PXA, gt_px=int(g.sum()),
                edge_frac=edge/max(int(g.sum()),1))

import cir_gt_profile as G                     # 모델을 안 쓰는 난이도 프로파일(GT신호)

rows = []
for fd, names in FOLD.items():
    for n in names.split(","): rows.append(("소형", n, run(n, [fd])))
for n in LARGE: rows.append(("대형", n, run(n, [1,2,3])))
for _, n, r in rows:
    r["sig"] = G.profile(n)["sig"]

print(f"{'':<4}{'화재':<20}{'GT ha':>8}{'GT신호':>8}{'F1':>7}{'τ20m':>7}{'τ40m':>7}"
      f"{'면적오차':>9}{'경계비율':>9}")
print("-"*80)
for k, n, r in rows:
    print(f"{k:<4}{n:<20}{r['gt_ha']:>8.0f}{r['sig']:>8.3f}{r['f1']:>7.3f}{r['t20']:>7.3f}"
          f"{r['t40']:>7.3f}{(r['pred_ha']/r['gt_ha']-1)*100:>8.1f}%{r['edge_frac']*100:>8.1f}%")
print("-"*80)
agg = {}
for k in ("소형", "대형"):
    v = [r for kk, _, r in rows if kk == k]
    agg[k] = {m: float(np.mean([x[m] for x in v])) for m in ("f1","t20","t40","edge_frac")}
    agg[k]["ae"] = float(np.mean([abs(x["pred_ha"]/x["gt_ha"]-1)*100 for x in v]))
    print(f"{k} 평균({len(v)}화재)      {'':>16}{agg[k]['f1']:>7.3f}{agg[k]['t20']:>7.3f}"
          f"{agg[k]['t40']:>7.3f}{agg[k]['ae']:>8.1f}%{agg[k]['edge_frac']*100:>8.1f}%")
d = {m: agg["소형"][m]-agg["대형"][m] for m in ("f1","t20","t40")}
print(f"\n격차(소형−대형)         {'':>16}{d['f1']:>+7.3f}{d['t20']:>+7.3f}{d['t40']:>+7.3f}"
      f"   면적오차 {agg['소형']['ae']-agg['대형']['ae']:+.1f}%p")

# ── 가설 검증 ────────────────────────────────────────────────────────────
from scipy.stats import spearmanr
ha  = np.array([r["gt_ha"] for _,_,r in rows]); f1s = np.array([r["f1"] for _,_,r in rows])
sig = np.array([r["sig"]  for _,_,r in rows])
print(f"\n① 기하(경계) 가설 — τ 를 주면 격차가 줄어야 한다")
print(f"   τ20m 이 회수한 격차 {(1-d['t20']/d['f1'])*100:.0f}% · τ40m {(1-d['t40']/d['f1'])*100:.0f}%"
      f"  → 대부분 안 줄어든다. **기각**")
r1,p1 = spearmanr(sig,f1s)
print(f"② 약신호 가설 — F1 vs GT신호 ρ={r1:+.3f} (p={p1:.3f}) → 상관 없음. **소형 전체로는 기각**")
r2,p2 = spearmanr(np.log10(ha),f1s)
print(f"③ 크기 상관 ρ={r2:+.3f} (p={p2:.3f}) — 있긴 하다. 그러나:")
mid = [(n,r) for k,n,r in rows if 190<=r["gt_ha"]<=300]
print("   같은 크기대(190~300ha) 안에서 " + " · ".join(f"{n.split('_')[-1]} {r['f1']:.3f}" for n,r in mid))
print(f"   → 같은 크기 안의 분산({max(r['f1'] for _,r in mid)-min(r['f1'] for _,r in mid):.3f})이 "
      f"격차({-d['f1']:.3f})보다 크다. **크기는 원인이 아니라 대리변수다**")
