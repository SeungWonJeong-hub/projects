"""오탐 덩어리의 **모양·크기·자리**를 본다 — 미소각 오탐 분해 방법 ② (§13.3).

세 가지를 묻는다:
  A. **크기 분포** — 국내 벌채 구획은 보통 수 ha 다. 오탐이 벌채라면 작은 덩어리에 몰려야 한다.
  B. **모양** — 벌채·수확지는 사람이 그은 경계라 직선·다각형, 소각지는 지형과 풍향을 따라 불규칙.
     기준선 없이는 숫자를 못 읽으므로 **실제 소각 덩어리와 나란히** 낸다(안동 GT + 산림청 폴리곤).
  C. ★**자리** — pre=post 인 여러 날짜에서 오탐 마스크가 **같은 자리**에 반복되면 원인은 날짜가
     아니라 **지표(지형·피복)** 다. 이게 원인 규명에 가장 직접적이다.

지표(0~1, 클수록 "인공적"): rect=면적/최소회전외접사각형, straight=경계 중 긴 직선 구간 비율.
compact=4πA/P² 는 반대로 낮을수록 불규칙하다.

사용: conda run -n ml python src/fp_shape_probe.py [--thrs 0.2 0.3 0.4] [--persist 6]
"""
import os, sys, json, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
GSD = 20; PXA = GSD * GSD / 1e4
MIN_HA = float(os.environ.get("MIN_HA", "0.5"))
RUN_L = int(os.environ.get("RUN_L", "5"))
BINS = [(0.5, 5), (5, 20), (20, 100), (100, 1e9)]      # ha — 벌채 규모는 앞의 두 칸이다


def shape_metrics(m):
    ys, xs = np.nonzero(m)
    A = len(ys) * PXA
    p = np.pad(m, 1)
    c = p[1:-1, 1:-1]
    per = int((c & ~p[:-2, 1:-1]).sum() + (c & ~p[2:, 1:-1]).sum() +
              (c & ~p[1:-1, :-2]).sum() + (c & ~p[1:-1, 2:]).sum())
    P = per * GSD
    compact = 4 * np.pi * (A * 1e4) / max(P * P, 1e-9)
    pts = np.stack([xs, ys], 1).astype(np.float64); pts -= pts.mean(0)
    best = np.inf
    for th in np.linspace(0, np.pi / 2, 46):
        cs, sn = np.cos(th), np.sin(th)
        r = pts @ np.array([[cs, -sn], [sn, cs]])
        best = min(best, (np.ptp(r[:, 0]) + 1) * (np.ptp(r[:, 1]) + 1))
    rect = len(ys) / max(best, 1e-9)
    st = 0
    for edge, ax in ((c & ~p[:-2, 1:-1], 1), (c & ~p[2:, 1:-1], 1),
                     (c & ~p[1:-1, :-2], 0), (c & ~p[1:-1, 2:], 0)):
        e = edge if ax == 1 else edge.T
        for row in e:
            if not row.any(): continue
            d = np.diff(np.concatenate([[0], row.view(np.int8), [0]]))
            for s0, e0 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                if e0 - s0 >= RUN_L: st += e0 - s0
    return A, float(rect), float(st / max(per, 1)), float(compact)


def blob_rows(mask):
    from scipy.ndimage import label
    lab, n = label(mask, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() * PXA < MIN_HA: continue
        out.append(shape_metrics(m))
    return out


def size_table(name, rows):
    tot = sum(r[0] for r in rows) or 1e-9
    print(f"\n── {name} (덩어리 {len(rows)} · 총 {tot:.0f}ha)")
    print(f"{'크기구간(ha)':<14}{'덩어리':>6}{'면적ha':>9}{'면적%':>8}{'rect':>8}{'straight':>10}{'compact':>9}")
    o = []
    for lo, hi in BINS:
        g = [r for r in rows if lo <= r[0] < hi]
        if not g:
            print(f"{f'{lo:g}~{hi:g}' if hi < 1e8 else f'{lo:g}+':<14}{0:>6}{0:>9.0f}{0:>7.1f}%")
            o.append(dict(lo=lo, hi=hi, n=0, ha=0.0)); continue
        A = np.array([r[0] for r in g]); w = A / A.sum()
        lab = f'{lo:g}~{hi:g}' if hi < 1e8 else f'{lo:g}+'
        print(f"{lab:<14}{len(g):>6}{A.sum():>9.0f}{A.sum()/tot*100:>7.1f}%"
              f"{np.average([r[1] for r in g], weights=w):>8.3f}"
              f"{np.average([r[2] for r in g], weights=w):>10.3f}"
              f"{np.average([r[3] for r in g], weights=w):>9.3f}")
        o.append(dict(lo=lo, hi=hi, n=len(g), ha=float(A.sum()), frac=float(A.sum() / tot),
                      rect=float(np.average([r[1] for r in g], weights=w)),
                      straight=float(np.average([r[2] for r in g], weights=w)),
                      compact=float(np.average([r[3] for r in g], weights=w))))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="andong")
    ap.add_argument("--pair", nargs=2, default=["2018-05-10", "2019-05-10"])
    ap.add_argument("--thrs", nargs="+", type=float, default=[0.20, 0.30, 0.40])
    ap.add_argument("--persist", type=int, default=6, help="자리 반복성 검사에 쓸 pre=post 날짜 수")
    a = ap.parse_args()

    os.environ.update(SITE=a.site, GSD=str(GSD), TILE="256", NIR="B8A")
    import nifos_eval as NE
    M = sorted(glob.glob(os.path.join(PROJ, "runs/cir_reh_ens/*.pt")))
    z = np.load(os.path.join(PROJ, "nifos_cache", f"{a.site}_ts", f"ts_{GSD}m.npz"), allow_pickle=True)
    ds = [str(s) for s in z["dates"]]; X = z["x"]; V = z["v"]
    out = {}

    # ── A·B. 문제의 쌍을 임계값별로 ────────────────────────────────────────
    i, j = ds.index(a.pair[0]), ds.index(a.pair[1])
    prob, _ = NE.infer(np.concatenate([X[i], X[j]], 0), M)
    gt = np.load(os.path.join(PROJ, "nifos_cache", a.site, f"gt_{GSD}m.npy")).astype(bool)
    H, W = min(prob.shape[0], gt.shape[0]), min(prob.shape[1], gt.shape[1])
    print(f"[{a.site}] 미소각 쌍 {a.pair[0]} → {a.pair[1]} · 덩어리 ≥{MIN_HA}ha")
    for t in a.thrs:
        out[f"fp@{t}"] = size_table(f"① 미소각 오탐 thr={t}", blob_rows(prob[:H, :W] >= t))
    out["gt_andong"] = size_table("② 안동 실제 소각 GT (정답)", blob_rows(gt[:H, :W]))

    # 실제 소각의 모양 기준선을 넓힌다 — 산림청 폴리곤 245화재
    rows = []
    for f in sorted(glob.glob(os.path.join(PROJ, "ko_refl_s50", "*", "*.npz")))[:120]:
        y = np.load(f)["y"].astype(bool)
        if y.sum() * PXA < MIN_HA: continue
        rows += blob_rows(y)
    out["gt_mtfire"] = size_table(f"③ 산림청 폴리곤 소각 덩어리 (학습풀, 참고)", rows)

    # ── C. ★자리 반복성 — pre=post 여러 날짜에서 같은 곳이 뜨는가 ──────────
    vf = V.reshape(len(ds), -1).mean(1)
    cand = [k for k in np.argsort(-vf)[:60]]
    sel = sorted(cand)[:: max(1, len(cand) // a.persist)][:a.persist]
    print(f"\n═══ C. 자리 반복성 — pre=post {len(sel)} 날짜 (각각 변화가 정의상 0) ═══")
    acc = np.zeros((H, W), np.int32); used = []
    for k in sel:
        p, _ = NE.infer(np.concatenate([X[k], X[k]], 0), M)
        m = p[:H, :W] >= 0.20
        acc += m; used.append(ds[k])
        print(f"  {ds[k]}  오탐 {m.sum()*PXA:>7.0f}ha", flush=True)
    n = len(sel)
    print(f"\n{'몇 개 날짜에서 오탐':<22}{'면적ha':>9}{'누적%':>8}   설명")
    tot = float((acc > 0).sum() * PXA)
    for k in range(n, 0, -1):
        ha = float((acc >= k).sum() * PXA)
        note = "★날짜와 무관 = 지표(지형·피복)가 원인" if k == n else ""
        print(f"{f'{k}/{n} 이상':<22}{ha:>9.0f}{ha/max(tot,1e-9)*100:>7.1f}%   {note}")
    # 늘 뜨는 자리의 성질 — 화재 전 NDVI 로 본다
    xf = X[sel[0]].astype(np.float32) / 1e4
    ndvi = (xf[3] - xf[2]) / (xf[3] + xf[2] + 1e-6)
    ndvi = ndvi[:H, :W]
    always, never = acc >= n, acc == 0
    print(f"\n늘 오탐인 곳 NDVI 중앙 {np.median(ndvi[always]) if always.any() else float('nan'):+.3f} "
          f"({always.sum()*PXA:.0f}ha) · 한 번도 아닌 곳 {np.median(ndvi[never]):+.3f} ({never.sum()*PXA:.0f}ha)")
    out["persistence"] = dict(dates=used, n=n, total_ha=tot,
                              by_count={int(k): float((acc >= k).sum() * PXA) for k in range(1, n + 1)},
                              ndvi_always=float(np.median(ndvi[always])) if always.any() else None,
                              ndvi_never=float(np.median(ndvi[never])))
    o = os.path.join(PROJ, "data", f"fp_shape_{a.site}.json")
    json.dump(out, open(o, "w"), ensure_ascii=False, indent=1)
    print(f"\n기록: data/fp_shape_{a.site}.json")


if __name__ == "__main__":
    main()
