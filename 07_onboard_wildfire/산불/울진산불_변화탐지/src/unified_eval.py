"""독립 자를 **하나로** 합친 평가 — 자별 숫자를 평균하지 않고 화소로 합친다 (§15.4).

왜: "미학습 6화재 평균 0.861" 은 독립 1건과 순환 5건을 섞은 값이었다(§15). 평균은 자를 감춘다.
자마다 **"라벨 밖"의 뜻이 다르기** 때문에 그냥 이어붙일 수도 없다 — 안동은 진짜 비소각이고
RapidEye 는 **모름**(부분 라벨, §8.4)이다. 그래서 화소를 **3상태**로 만들고 '모름'을 지표에서 뺀다.

  2=소각 · 0=비소각 · 255=모름(지표에서 제외)

사이트별 규칙
  안동      NIFoS 0.3m 전 영역 라벨 → GT=소각, 나머지=비소각. 전부 유효.
  RapidEye  폴리곤=소각. 라벨 밖은 두 규칙의 **합집합**을 모름으로 뺀다:
              ① GT 볼록껍질을 MARGIN_M 만큼 넓힌 안쪽 (라벨이 직선으로 잘린 구간을 덮는다)
              ② 라벨 안 됐지만 dNBR ≥ T0 (분광이 "탔다" 고 말하는 곳)
            그 밖은 비소각. ★②는 SWIR 을 쓰는데 모델은 RGB+NIR 이라 입력이 겹치지 않는다.
            그래도 **모델이 dNBR 과 일치하는 오답은 가려진다** — 낙관 쪽 편향이다(아래 경고 출력).
  미소각    화재 **이전** 날짜 쌍(정답 0ha) → 전부 비소각. §13 의 오탐 21~33% 가 지표로 들어온다.

인제는 전 학습셋 누출이라 제외한다(GT_USAGE.md).

사용: conda run -n ml python src/unified_eval.py [--unburned 4]
"""
import os, sys, json, glob, argparse, datetime as dt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
GSD = 20; PXA = GSD * GSD / 1e4
T0 = float(os.environ.get("T0", "0.18"))          # 분광상 "탔다" 하한 (§14 보정값)
MARGIN_M = float(os.environ.get("MARGIN_M", "600"))
USE_DNBR = os.environ.get("USE_DNBR_UNKNOWN", "1") == "1"   # 모름 규칙 ② 사용 여부(민감도 확인용)
THR = float(os.environ.get("THR", "0.20"))
RUNS = os.environ.get("RUNS", "runs/cir_reh_ens/*.pt")   # 채점할 가중치 글롭(단일 모델도 가능)
BURN_SITES = [("andong", "full"), ("gangneung", "partial"),
              ("goseong", "partial"), ("yangyang", "partial")]   # 인제 제외(누출)


def three_state(site, kind, gt, dnbr):
    """(라벨 2/0/255, 사유 문자열)."""
    lab = np.zeros(gt.shape, np.uint8)
    lab[gt] = 2
    if kind == "full":
        return lab, "전 영역 라벨 → 라벨 밖 = 비소각"
    import cv2
    ys, xs = np.nonzero(gt)
    hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
    m = np.zeros(gt.shape, np.uint8); cv2.fillConvexPoly(m, hull, 1)
    k = int(round(MARGIN_M / GSD))
    if k > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    unknown = (m.astype(bool) | ((dnbr >= T0) if USE_DNBR else np.zeros_like(gt))) & ~gt
    lab[unknown] = 255
    return lab, f"부분 라벨 → 껍질+{MARGIN_M:g}m" + (f" ∪ (dNBR≥{T0})" if USE_DNBR else "") + " = 모름"


def score(lab, prob):
    v = lab != 255
    p = (prob >= THR) & v
    g = (lab == 2) & v
    tp = int((p & g).sum()); fp = int((p & ~g & v).sum()); fn = int((~p & g).sum())
    return tp, fp, fn, int(v.sum()), int(g.sum())


def line(name, tp, fp, fn, nv, ng, extra=""):
    """★F1 을 대표값으로 쓰지 않는다.

    전부 음성인 장면에서 F1 은 TP=0 이라 **정의상 0** 이고, 음성 장면을 몇 장 넣느냐로
    합산 F1 이 얼마든지 움직인다 — 그건 모델의 성질이 아니라 **섞은 비율의 성질**이다
    (2026-09-10 사용자 지적). 그래서 비율에 안 흔들리는 두 값을 낸다:
      재현율 = TP/(TP+FN)         — 소각 화소에서만 계산, 음성 장면 수와 무관
      오탐률 = FP/(FP+TN)         — 음성 화소에서만 계산, 소각 사건 수와 무관
    F1 은 참고로만 적고, 그 값이 성립하는 **소각 비율**을 옆에 밝힌다.
    """
    neg = nv - ng
    rc = tp / max(tp + fn, 1)
    fpr = fp / max(neg, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    prev = ng / max(nv, 1)
    print(f"{name:<26}{rc:>8.3f}{fpr:>9.1%}{tp*PXA:>10,.0f}{fp*PXA:>10,.0f}{neg*PXA:>11,.0f}"
          f"{f1:>8.3f}{prev:>8.1%}  {extra}")
    return dict(recall=rc, fpr=fpr, f1=f1, prevalence=prev,
                tp_ha=tp * PXA, fp_ha=fp * PXA, fn_ha=fn * PXA, neg_ha=neg * PXA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unburned", type=int, default=4, help="미소각 장면 쌍 수(안동 시계열에서)")
    a = ap.parse_args()
    os.environ.update(SITE="andong", GSD=str(GSD), TILE="256", NIR="B8A", SCENE_PICK="doy")
    import nifos_eval as NE
    import importlib
    M = sorted(glob.glob(os.path.join(PROJ, RUNS)))
    assert M, f"가중치 없음: {RUNS}"
    print(f"가중치 {len(M)}개 — {RUNS}")
    ix = lambda b, i, j: (b[i] - b[j]) / (b[i] + b[j] + 1e-6)

    print(f"독립 자 통합 평가 · thr {THR} · 모름 규칙 = 껍질+{MARGIN_M:g}m ∪ dNBR≥{T0}")
    print(f"{'사이트':<26}{'재현율':>8}{'오탐률':>9}{'적중ha':>10}{'오탐ha':>10}{'음성ha':>11}"
          f"{'(F1)':>8}{'소각비':>8}  비고")
    tot = np.zeros(5, dtype=np.int64); rows = []
    for site, kind in BURN_SITES:
        os.environ["SITE"] = site; importlib.reload(NE)
        C = os.path.join(PROJ, "nifos_cache", site + "_doy")
        try:
            x = np.load(os.path.join(C, f"scene_{GSD}m.npz"))["x"]
            gt = np.load(os.path.join(C, f"gt_{GSD}m.npy")).astype(bool)
        except FileNotFoundError:
            print(f"{site:<26}(캐시 없음 — 건너뜀)"); continue
        H, W = min(gt.shape[0], x.shape[1]), min(gt.shape[1], x.shape[2])
        gt = gt[:H, :W]; x = x[:, :H, :W]
        xf = x.astype(np.float32) / 1e4
        dnbr = ix(xf[:6], 3, 5) - ix(xf[6:], 3, 5)
        lab, why = three_state(site, kind, gt, dnbr)
        prob, _ = NE.infer(x, M); prob = prob[:H, :W]
        r = score(lab, prob); tot += np.array(r + (0,))[:5] if False else np.array(r)
        m = line(f"{site} ({kind})", *r, extra=why)
        rows.append(dict(site=site, kind=kind, **m, tp=r[0], fp=r[1], fn=r[2],
                         valid_ha=r[3] * PXA, gt_ha=r[4] * PXA,
                         unknown_ha=float((lab == 255).sum() * PXA)))
    burn_tot = tot.copy()
    print("-" * 100)
    line("소계 (소각 사이트)", *burn_tot, extra="화소 합산 — 자별 평균이 아니다")

    # ── 미소각 장면: 정답 0ha. §13 의 오탐이 여기서 정밀도를 깎는다 ─────────
    ts = os.path.join(PROJ, "nifos_cache", "andong_ts", f"ts_{GSD}m.npz")
    if a.unburned > 0 and os.path.exists(ts):
        z = np.load(ts, allow_pickle=True)
        ds = [str(s) for s in z["dates"]]; X = z["x"]; V = z["v"]
        vf = V.reshape(len(ds), -1).mean(1)
        ok = [i for i in np.argsort(-vf)[:40]]
        sel = sorted(ok)[:: max(1, len(ok) // a.unburned)][:a.unburned]
        os.environ["SITE"] = "andong"; importlib.reload(NE)
        print()
        for i in sel:
            prob, _ = NE.infer(np.concatenate([X[i], X[i]], 0), M)   # pre=post → 변화 0
            lab = np.zeros(prob.shape, np.uint8)                      # 전부 비소각
            r = score(lab, prob); tot += np.array(r)
            line(f"미소각 {ds[i]}", *r, extra="pre=post, 정답 0ha")
    print("-" * 100)
    m_all = line("★통합 (독립 자 전부)", *tot, extra="자를 감추지 않는 단일 수치")

    print(f"\n읽는 법")
    print(f"  · ★**F1 을 대표값으로 읽지 말 것.** 음성 장면을 몇 장 넣느냐로 합산 F1 이 움직인다 —")
    print(f"    모델의 성질이 아니라 섞은 비율의 성질이다. 소각비(마지막 열)가 그 비율이다.")
    print(f"  · **재현율과 오탐률**이 비율에 안 흔들리는 두 값이다. 이 둘로 비교할 것.")
    print(f"  · 화소를 합산했다 — 자별 F1 의 평균이 아니다(평균은 자를 감춘다, §15).")
    print(f"  · '모름' 화소는 지표에서 뺐다(부분 라벨 바깥을 비소각으로 세면 실제 소각지가 오탐이 된다, §8.4).")
    print(f"  · CIR-GT·울진은 뺐다(순환). 산림청은 뺐다(과대 라벨).")
    json.dump(dict(thr=THR, t0=T0, margin_m=MARGIN_M, sites=rows,
                   burn_only=dict(zip(("tp","fp","fn","valid","gt"), map(int, burn_tot))),
                   total=dict(zip(("tp","fp","fn","valid","gt"), map(int, tot))),
                   total_metrics=m_all),
              open(os.path.join(PROJ, "data", "unified_eval.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n기록: data/unified_eval.json")


if __name__ == "__main__":
    main()
