"""후보 앙상블을 **재현율을 맞춘 뒤** 비교하고, thr 곡선에 **면적**을 얹는다 (§17.5~§18).

왜: §17.4 는 고정 thr 0.20 에서 점수를 비교했는데, 그러면 두 가지가 섞인다 —
  ① 모델이 실제로 나은가  ② 그 모델이 마침 그 thr 에서 P–R 곡선의 다른 자리에 앉는가.
확률 평균(앙상블)은 특히 곡선 위 위치가 달라서 **단일과 같은 thr 로 비교하면 안 된다**(함정 23).
그래서 각 후보의 thr 를 **목표 재현율(TARGET_R)에 맞춰 정렬**하고 그 지점의 오탐률만 읽는다.

지표 정의는 §16 과 같다(재현율=소각 화소, 오탐률=음성 화소, 모름은 제외). 자도 같다.
★면적은 **전 영역 라벨(안동)에서만** 낸다. RapidEye 3곳은 부분 라벨이라(GT_USAGE.md §8.4)
예측이 라벨보다 넓은 게 정상이고, 그걸 "과대"로 읽으면 자가 아니라 라벨의 성질을 재게 된다.
면적이 제품 지표이므로(함정 4) 운용점(thr) 결정은 **재현율·오탐률·면적을 한 표에서** 본다.

사용: conda run -n ml python src/thr_align.py --target 0.85 --unburned 4
      conda run -n ml python src/thr_align.py --curve "배포 현행 cir_reh_ens"
"""
import os, sys, json, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import unified_eval as UE                      # 3상태 라벨·화소 채점의 단일 소스

GSD, PXA = UE.GSD, UE.PXA
GRID = np.round(np.arange(0.05, 0.951, 0.0125), 4)


def collect(M, n_unburned):
    """앙상블 M 의 확률을 자별로 **한 번만** 계산해 (정답, 확률) 벡터로 모은다."""
    import importlib
    import nifos_eval as NE
    pairs = []
    for site, kind in UE.BURN_SITES:
        os.environ["SITE"] = site; importlib.reload(NE)
        C = os.path.join(PROJ, "nifos_cache", site + "_doy")
        try:
            x = np.load(os.path.join(C, f"scene_{GSD}m.npz"))["x"]
            gt = np.load(os.path.join(C, f"gt_{GSD}m.npy")).astype(bool)
        except FileNotFoundError:
            continue
        H, W = min(gt.shape[0], x.shape[1]), min(gt.shape[1], x.shape[2])
        gt = gt[:H, :W]; x = x[:, :H, :W]
        xf = x.astype(np.float32) / 1e4
        ix = lambda b, i, j: (b[i] - b[j]) / (b[i] + b[j] + 1e-6)
        dnbr = ix(xf[:6], 3, 5) - ix(xf[6:], 3, 5)
        lab, _ = UE.three_state(site, kind, gt, dnbr)
        prob, _ = NE.infer(x, M); prob = prob[:H, :W]
        v = lab != 255
        pairs.append((site, kind, (lab[v] == 2), prob[v]))
    ts = os.path.join(PROJ, "nifos_cache", "andong_ts", f"ts_{GSD}m.npz")
    if n_unburned > 0 and os.path.exists(ts):
        z = np.load(ts, allow_pickle=True)
        ds = [str(s) for s in z["dates"]]; X = z["x"]; V = z["v"]
        vf = V.reshape(len(ds), -1).mean(1)
        ok = [i for i in np.argsort(-vf)[:40]]
        sel = sorted(ok)[:: max(1, len(ok) // n_unburned)][:n_unburned]
        os.environ["SITE"] = "andong"; importlib.reload(NE)
        for i in sel:
            prob, _ = NE.infer(np.concatenate([X[i], X[i]], 0), M)   # pre=post → 변화 0
            pairs.append((f"미소각 {ds[i]}", "unburned", np.zeros(prob.size, bool), prob.ravel()))
    return pairs


def curve(pairs):
    """thr 격자에서 (재현율, 오탐률, 오탐ha) — 소각 화소와 음성 화소를 자 전체로 합산."""
    g = np.concatenate([p[2] for p in pairs]); pr = np.concatenate([p[3] for p in pairs])
    pos, neg = pr[g], pr[~g]
    npos, nneg = max(pos.size, 1), max(neg.size, 1)
    rc = np.array([(pos >= t).sum() / npos for t in GRID])
    fp = np.array([(neg >= t).sum() for t in GRID], dtype=float)
    return rc, fp / nneg, fp * PXA


def area_err(pairs):
    """전 영역 라벨 사이트만 — thr 격자에서 (예측 ha, GT ha, 오차 %). 부분 라벨은 뺀다."""
    full = [p for p in pairs if p[1] == "full"]
    if not full: return None, None, None
    g = np.concatenate([p[2] for p in full]); pr = np.concatenate([p[3] for p in full])
    gt_ha = g.sum() * PXA
    pred = np.array([(pr >= t).sum() for t in GRID], dtype=float) * PXA
    return pred, gt_ha, (pred - gt_ha) / max(gt_ha, 1e-9) * 100


def align(rc, target):
    """재현율 ≥ target 을 만족하는 **가장 높은** thr 의 격자 index. 없으면 None."""
    ok = np.nonzero(rc >= target)[0]
    return int(ok[-1]) if ok.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--unburned", type=int, default=4)
    ap.add_argument("--curve", default="", help="이 후보의 thr 곡선을 표로 낸다(면적 포함)")
    a = ap.parse_args()
    os.environ.update(SITE="andong", GSD=str(GSD), TILE="256", NIR="B8A", SCENE_PICK="doy")
    cands = json.load(open(os.path.join(PROJ, "data", "thr_align_cands.json"), encoding="utf-8"))
    print(f"재현율 {a.target:.2f} 에 thr 를 맞춘 뒤 오탐률을 비교한다 · 자 = §16 통합(미소각 {a.unburned}장)")
    print(f"{'후보':<24}{'n':>3}{'thr*':>7}{'재현율':>9}{'오탐률':>9}{'오탐ha':>10}{'안동면적':>10}   {'(thr 0.20 에서)':>22}")
    print("-" * 106)
    out = []
    for name, pat in cands.items():
        M = sorted(glob.glob(os.path.join(PROJ, pat)))
        if not M:
            print(f"{name:<24}(가중치 없음: {pat})"); continue
        pairs = collect(M, a.unburned)
        rc, fpr, fpha = curve(pairs)
        pred, gt_ha, aerr = area_err(pairs)
        if a.curve and name == a.curve:
            print(f"\n  [{name}] thr 곡선 — 면적은 안동(전 영역 라벨, GT {gt_ha:,.0f}ha)")
            print(f"  {'thr':>6}{'재현율':>9}{'오탐률':>9}{'오탐ha':>10}{'예측ha':>10}{'면적오차':>10}")
            for t in np.arange(0.05, 0.901, 0.05):
                k = int(np.argmin(np.abs(GRID - t)))
                print(f"  {GRID[k]:>6.3f}{rc[k]:>9.3f}{fpr[k]:>9.1%}{fpha[k]:>10,.0f}"
                      f"{pred[k]:>10,.0f}{aerr[k]:>9.1f}%")
            print()
        i = align(rc, a.target)
        j = int(np.argmin(np.abs(GRID - 0.20)))
        base = f"R={rc[j]:.3f} FPR={fpr[j]:.1%}"
        if i is None:
            print(f"{name:<24}{len(M):>3}{'—':>7}{rc[0]:>9.3f}{'':>9}{'':>10}{'':>10}   {base:>22}"
                  f"  ★최대 재현율({rc[0]:.3f})이 목표 미달")
            out.append(dict(name=name, n=len(M), thr=None, recall=float(rc[0]), fpr=None))
            continue
        ae = f"{aerr[i]:+.1f}%" if aerr is not None else "—"
        print(f"{name:<24}{len(M):>3}{GRID[i]:>7.3f}{rc[i]:>9.3f}{fpr[i]:>9.1%}{fpha[i]:>10,.0f}{ae:>10}   {base:>22}")
        out.append(dict(name=name, n=len(M), thr=float(GRID[i]), recall=float(rc[i]),
                        fpr=float(fpr[i]), fp_ha=float(fpha[i]),
                        andong_area_err=(float(aerr[i]) if aerr is not None else None),
                        at020=dict(recall=float(rc[j]), fpr=float(fpr[j]), fp_ha=float(fpha[j]))))
    print("\n  · thr* = 재현율 ≥ 목표를 만족하는 **가장 높은** thr(= 그 후보가 낼 수 있는 최저 오탐률 지점).")
    print("  · 안동면적 = 전 영역 라벨 사이트의 면적오차. 부분 라벨(RapidEye)은 면적에서 제외했다.")
    print("  · 이 표에서만 후보 간 오탐률을 비교할 것 — 고정 thr 비교는 P–R 곡선 위 위치가 섞인다(함정 23).")
    json.dump(dict(target=a.target, unburned=a.unburned, rows=out),
              open(os.path.join(PROJ, "data", "thr_align.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n기록: data/thr_align.json")


if __name__ == "__main__":
    main()
