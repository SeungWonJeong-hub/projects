"""심각도 3등급을 RGB+NIR 로 **학습**한다 — 큐브위성엔 SWIR 이 없다 (§7 ①).

sev_nosw_probe.py 는 "가능한가"(화소단위 CV, 공간 자기상관으로 낙관적)를 봤다.
여기서는 **공간분리 CV** 로 정직한 숫자를 내고, 배포 규칙(dNBR 0.27/0.44 고정절단)과 겨룬다.
GT = NIFoS 안동 피해강도등급 1심/2중/3경 (KCBI 학습 자동분류 산출물, 원본 0.3m → 20m).

판정 지표는 정확도만이 아니다 — 이 제품의 산출물은 **등급별 면적(ha)** 이다.
"""
import os, sys, numpy as np
os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
os.environ.setdefault("OMP_NUM_THREADS", "4")   # 96코어에서 스레드 과다생성 → 무한정 느려진다
from osgeo import gdal; gdal.UseExceptions()

P   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GSD = 20.0
PXA = GSD * GSD / 1e4          # 화소면적 ha
SEED = 0

# ── 데이터 ────────────────────────────────────────────────────────────────
sev = gdal.Open(f"{P}/../nifos_burn_gt/andong/andong_severity_20m.tif")
g   = sev.GetGeoTransform(); S = sev.ReadAsArray()
ute = np.load(f"{P}/nifos_cache/andong/ute_20m.npy")
x   = np.load(f"{P}/nifos_cache/andong/scene_20m.npz")["x"].astype("float32") / 1e4

ox, oy = int(round((g[0]-ute[0])/GSD)), int(round((ute[3]-g[3])/GSD))
H, W = x.shape[1], x.shape[2]
S2 = np.zeros((H, W), S.dtype)
ys = slice(max(oy,0), min(oy+S.shape[0], H)); xs = slice(max(ox,0), min(ox+S.shape[1], W))
S2[ys, xs] = S[max(-oy,0):max(-oy,0)+(ys.stop-ys.start), max(-ox,0):max(-ox,0)+(xs.stop-xs.start)]

pre, post = x[:6], x[6:]                      # [B02,B03,B04,B8A,B11,B12]
idx = lambda a, b: (a-b)/(a+b+1e-6)
dNBR   = idx(pre[3], pre[5]) - idx(post[3], post[5])          # SWIR 필수
dNDVI  = idx(pre[3], pre[2]) - idx(post[3], post[2])
dGNDVI = idx(pre[3], pre[1]) - idx(post[3], post[1])
dNIR   = pre[3] - post[3]
dBlue  = idx(pre[3], pre[0]) - idx(post[3], post[0])

SETS = {
    "SWIR 포함 (dNBR+dNDVI)":  [dNBR, dNDVI],
    "RGB+NIR 4지표":           [dNDVI, dGNDVI, dNIR, dBlue],
    "RGB+NIR 4지표+원본반사율": [dNDVI, dGNDVI, dNIR, dBlue,
                                *pre[:4], *post[:4]],
}

m = np.isin(S2, [1, 2, 3])
y = S2[m].astype(int)
yy, xx = np.mgrid[0:H, 0:W]
print(f"안동 20m 격자 {H}×{W} · 등급화소 {m.sum():,} "
      f"(심 {(y==1).sum():,} 중 {(y==2).sum():,} 경 {(y==3).sum():,})")
print(f"GT 면적: 심 {(y==1).sum()*PXA:.1f} / 중 {(y==2).sum()*PXA:.1f} / 경 {(y==3).sum()*PXA:.1f} ha\n")

# ── 분할: 화소단위(낙관) vs 공간블록 vs 띠 ────────────────────────────────
from sklearn.model_selection import KFold, GroupKFold
BLK = 50                                       # 50px = 1km 블록
blk = ((yy//BLK)*(W//BLK + 1) + (xx//BLK))[m]
stripe = (xx[m] * 3 // W)                       # 서→동 3등분, 최대 분리

def splits(kind):
    if kind == "pixel":
        return list(KFold(3, shuffle=True, random_state=SEED).split(y))
    if kind == "block":
        return list(GroupKFold(3).split(y, y, blk))
    return [(np.where(stripe != k)[0], np.where(stripe == k)[0]) for k in range(3)]

# ── 배포 규칙(고정 절단) — 학습이 아니라 기준선 ──────────────────────────
def deployed(sel):
    d = dNBR[m][sel]
    return np.where(d >= 0.44, 1, np.where(d >= 0.27, 2, 3))

from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import f1_score

def area_err(pred, true):
    """등급별 면적오차(%)를 최대 절댓값으로 요약 — 제품 지표."""
    e = [(np.sum(pred == k) - np.sum(true == k)) / max(np.sum(true == k), 1) * 100 for k in (1, 2, 3)]
    return e, max(abs(v) for v in e)

def run(kind):
    sp = splits(kind)
    print(f"── {kind} 분할 ({'화소단위·낙관' if kind=='pixel' else '1km 블록' if kind=='block' else '서/중/동 3띠·최대분리'})")
    print(f"{'모델':<26}{'정확도':>9}{'macroF1':>9}   {'등급별 면적오차 %':>24}")
    rows = {}
    for name, feats in [("배포 규칙 dNBR 절단", None)] + list(SETS.items()):
        acc, mf1, ae = [], [], []
        for tr, te in sp:
            if feats is None:
                p = deployed(te)
            else:
                X = np.stack([f[m] for f in feats], 1)
                p = HGB(max_iter=120, random_state=SEED).fit(X[tr], y[tr]).predict(X[te])
            acc.append((p == y[te]).mean()); mf1.append(f1_score(y[te], p, average="macro"))
            ae.append(area_err(p, y[te])[0])
        a = np.mean(ae, 0)
        print(f"{name:<26}{np.mean(acc):>9.3f}{np.mean(mf1):>9.3f}   "
              f"심 {a[0]:>+6.1f} 중 {a[1]:>+6.1f} 경 {a[2]:>+6.1f}")
        rows[name] = (np.mean(acc), np.mean(mf1), a)
    print(f"{'(최빈등급만 찍기)':<26}{max(np.bincount(y)[1:])/len(y):>9.3f}\n")
    return rows

out = {k: run(k) for k in ("pixel", "block", "stripe")}
