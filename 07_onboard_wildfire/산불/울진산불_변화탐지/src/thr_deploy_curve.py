"""배포 평가 경로에서 thr 곡선을 낸다 — 면적오차 0 교차점을 배포 자로 잡는다 (§18 후속).

왜: §18 의 thr 곡선은 `nifos_cache/andong_doy`(DOY 정합 씬)에서 잰 값이라 thr 0.20 의
면적오차가 +8.6% 였는데, §1 에 적힌 배포 실측은 **+4.69%** 다. 절대 위치가 경로마다 다르므로
**배포가 실제로 쓰는 캐시**에서 다시 잡아야 반영할 수 있다(§18 말미).

지표는 qc_calibrate.score() 와 같은 정의를 쓴다(F1@thr · 면적오차 = (예측−GT)/GT).
면적은 전 영역 라벨(안동)에서만 의미가 있다 — 부분 라벨은 절대값으로 못 쓴다.

사용: conda run -n ml python src/thr_deploy_curve.py [캐시이름 ...]
"""
import os, sys, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
GSD = float(os.environ.get("GSD", "20")); PX = GSD * GSD / 1e4
RUNS = os.environ.get("RUNS", "runs/cir_reh_ens/*.pt")
CACHES = sys.argv[1:] or ["andong", "andong_doy"]
GRID = np.round(np.arange(0.05, 0.901, 0.0125), 4)


def curve(cache):
    os.environ.update(SITE="andong", GSD=str(int(GSD)), TILE="256", NIR="B8A")
    import importlib, nifos_eval as NE
    importlib.reload(NE)
    C = os.path.join(PROJ, "nifos_cache", cache)
    gt = np.load(os.path.join(C, f"gt_{int(GSD)}m.npy")).astype(bool)
    x = np.load(os.path.join(C, f"scene_{int(GSD)}m.npz"))["x"]
    H = min(gt.shape[0], x.shape[1]); W = min(gt.shape[1], x.shape[2])
    gt = gt[:H, :W]; x = x[:, :H, :W]
    prob, _ = NE.infer(x, sorted(glob.glob(os.path.join(PROJ, RUNS))))
    prob = prob[:H, :W]
    gt_ha = gt.sum() * PX
    f1, ae, rc = [], [], []
    for t in GRID:
        p = prob >= t; tp = (p & gt).sum()
        f1.append(2 * tp / max(p.sum() + gt.sum(), 1))
        ae.append((p.sum() * PX - gt_ha) / gt_ha * 100)
        rc.append(tp / max(gt.sum(), 1))
    return np.array(f1), np.array(ae), np.array(rc), gt_ha


def main():
    out = {}
    for c in CACHES:
        try:
            f1, ae, rc, gt_ha = curve(c)
        except FileNotFoundError as e:
            print(f"[{c}] 캐시 없음 — 건너뜀 ({e.filename})"); continue
        i0 = int(np.argmin(np.abs(GRID - 0.20)))
        iz = int(np.argmin(np.abs(ae)))                      # 면적오차 0 교차
        ib = int(np.argmax(f1))
        print(f"\n[{c}] GT {gt_ha:,.0f}ha · 가중치 {RUNS}")
        print(f"  {'thr':>6}{'F1':>8}{'재현율':>9}{'면적오차':>10}")
        for t in np.arange(0.10, 0.501, 0.025):
            k = int(np.argmin(np.abs(GRID - t)))
            mk = "  ← 현행" if k == i0 else ("  ← 면적 0" if k == iz else ("  ← F1 최대" if k == ib else ""))
            print(f"  {GRID[k]:>6.3f}{f1[k]:>8.4f}{rc[k]:>9.3f}{ae[k]:>+9.1f}%{mk}")
        print(f"  현행 0.200: F1 {f1[i0]:.4f} · 면적 {ae[i0]:+.1f}%"
              f"   |  면적 0 교차 thr {GRID[iz]:.3f}: F1 {f1[iz]:.4f} · 면적 {ae[iz]:+.1f}%"
              f"   |  F1 최대 thr {GRID[ib]:.3f}: F1 {f1[ib]:.4f} · 면적 {ae[ib]:+.1f}%")
        out[c] = dict(gt_ha=float(gt_ha), thr020=dict(f1=float(f1[i0]), area_err=float(ae[i0])),
                      zero=dict(thr=float(GRID[iz]), f1=float(f1[iz]), area_err=float(ae[iz])),
                      bestf1=dict(thr=float(GRID[ib]), f1=float(f1[ib]), area_err=float(ae[ib])))
    json.dump(out, open(os.path.join(PROJ, "data", "thr_deploy_curve.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n기록: data/thr_deploy_curve.json")


if __name__ == "__main__":
    main()
