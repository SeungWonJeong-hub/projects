"""자(尺)별 F1 을 **배포 임계값에서** 낸다 — 산림청 in-domain · 국토정보플랫폼 CIR-GT.

왜: 스펙에 적힌 산림청 0.682 / CIR 0.847 같은 값은 `step4_rehearsal.bestf1` 이 **화재별로
임계값을 훑어 최댓값**을 취한 것이라 배포에서 얻을 수 없는 수치다(스펙 f1_note 도 그렇게 적혀
있다). 배포 thr 이 20m 0.25 · 10m 0.20 으로 정해졌으니 **그 값에서** 다시 낸다.

두 값을 나란히 낸다 — 고정 thr(배포 실제) / 화재별 best-thr(기록 비교용).
★자의 성질은 GT_USAGE.md 를 볼 것: 산림청=과대 라벨(회귀 감지용) · CIR-GT=준순환.

사용: conda run -n ml python src/eval_rulers.py
"""
import os, sys, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
_ROOT = os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference"))
CIRD = os.path.join(_ROOT, "Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires")
CIR_LARGE = ["250324_안동길안면", "250321_산청시천면", "230402_홍성서부면", "220410_양구양구읍"]
CIR_SMALL = ["250322_울주온양읍", "250428_대구노곡동"]
_T20 = float(os.environ.get("THR20", "0.25"))   # 배포값. 교차검증용으로 덮어쓸 수 있다
ARMS = {"20m": dict(runs="runs/cir_reh_ens/*.pt", thr=_T20, gsd=20, tile=256, nir="B8A",
                    cir="gt_stac_20m_raw.npz", ko="ko_refl"),
        "10m": dict(runs="runs/cir_reh_ens_10m_s50_exp/*.pt", thr=0.20, gsd=10, tile=512, nir="B08",
                    cir="gt_stac_10m_raw.npz", ko="ko_refl_10m_b08_t512_s50c")}


def to12(x8):
    """CIR-GT 는 8ch(pre RGBN|post RGBN) float [0,1] 이다. nifos_eval.infer 는 12ch 정수
    ×1e4 를 받아 [0,1,2,3]+[6,7,8,9] 를 뽑으므로 그 자리에 맞춰 끼운다."""
    out = np.zeros((12,) + x8.shape[1:], np.float32)
    out[0:4] = x8[0:4]; out[6:10] = x8[4:8]
    return (out * 1e4).astype(np.int16)


def pr_at(P, Y, t):
    """(정밀도, 재현율, F1). ★F1 은 참고다 — 정밀도와 재현율이 다른 얘기를 하면 F1 이 그걸 감춘다."""
    p = P >= t; tp = int((p & (Y == 1)).sum())
    pr = tp / max(int(p.sum()), 1); rc = tp / max(int((Y == 1).sum()), 1)
    return pr, rc, 2 * pr * rc / max(pr + rc, 1e-9)


def f1_at(P, Y, t):
    return pr_at(P, Y, t)[2]


def best_f1(P, Y):
    return max(f1_at(P, Y, t) for t in np.arange(0.05, 0.9, 0.025))


def run(arm):
    c = ARMS[arm]
    os.environ.update(SITE="andong", GSD=str(c["gsd"]), TILE=str(c["tile"]), NIR=c["nir"])
    import importlib, nifos_eval as NE; importlib.reload(NE)
    M = sorted(glob.glob(os.path.join(PROJ, c["runs"])))
    assert M, c["runs"]
    out = {}
    # ── 국토정보플랫폼 CIR-GT 홀드아웃 ─────────────────────────────
    #    ★한 숫자로 낼 땐 **화소 합산**이 기본이다 — 6화재를 한 장의 지도로 채점한 값이고
    #    제품 지표(면적)와 가중이 같다. 화재평균은 작은 화재에 같은 무게를 줘 다른 질문에 답한다.
    PA, YA = [], []
    for grp, names in (("CIR 대형4", CIR_LARGE), ("CIR 소형2", CIR_SMALL)):
        fx, bs = [], []
        for n in names:
            p = os.path.join(CIRD, n, c["cir"])
            if not os.path.exists(p): continue
            z = np.load(p); x = z["x"].astype("float32"); gt = z["gt"].astype(bool); vd = z["valid"].astype(bool)
            prob, _ = NE.infer(to12(x), M)
            H, W = gt.shape; prob = prob[:H, :W]
            fx.append(f1_at(prob[vd], gt[vd].astype(int), c["thr"])); bs.append(best_f1(prob[vd], gt[vd].astype(int)))
            PA.append(prob[vd]); YA.append(gt[vd].astype(int))
        out[grp] = (float(np.mean(fx)), float(np.mean(bs)), len(fx))
    P, Y = np.concatenate(PA), np.concatenate(YA)
    out["CIR 전체(화소합산)"] = (float(f1_at(P, Y, c["thr"])), float(best_f1(P, Y)), len(PA))
    out["_PR_CIR"] = pr_at(P, Y, c["thr"])
    out["CIR 전체(화재평균)"] = (float(np.mean([out["CIR 대형4"][0]]*4 + [out["CIR 소형2"][0]]*2)),
                             float(np.mean([out["CIR 대형4"][1]]*4 + [out["CIR 소형2"][1]]*2)), 6)
    out["_소각화소비"] = (float(sum(y.sum() for y in YA[:4]) / max(sum(y.sum() for y in YA), 1)), 0.0, 0)
    # ── 산림청 in-domain 홀드아웃 ─────────────────────────────────
    hold = [l.strip() for l in open(os.path.join(PROJ, "data", "holdout_fixed.txt")) if l.strip()]
    fx, bs, PF, YF = [], [], [], []
    for h in hold:
        ds = [d for d in glob.glob(os.path.join(PROJ, c["ko"], "*"))
              if h.replace("kf_", "") in os.path.basename(d) and os.path.isdir(d)]
        if not ds: continue
        P, Y = [], []
        for f in glob.glob(os.path.join(ds[0], "*.npz")):
            z = np.load(f)
            pr, _ = NE.infer(z["x"], M)     # ko_refl 은 12ch int16 ×1e4 — 그대로 넣는다
            v = z["valid"].astype(bool) if "valid" in z else np.ones(z["y"].shape, bool)
            P.append(pr[:v.shape[0], :v.shape[1]][v]); Y.append(z["y"][v].astype(int))
        if not P: continue
        P = np.concatenate(P); Y = np.concatenate(Y)
        fx.append(f1_at(P, Y, c["thr"])); bs.append(best_f1(P, Y)); PF.append(P); YF.append(Y)
    out["산림청(화재평균)"] = (float(np.mean(fx)), float(np.mean(bs)), len(fx))
    P, Y = np.concatenate(PF), np.concatenate(YF)
    out["산림청(화소합산)"] = (float(f1_at(P, Y, c["thr"])), float(best_f1(P, Y)), len(PF))
    out["_PR_산림청"] = pr_at(P, Y, c["thr"])
    return out


def main():
    res = {}
    print(f"{'자':<20}{'해상도':>6}{'배포 thr':>9}{'F1@배포thr':>12}{'F1@best-thr':>13}{'n':>4}")
    print("-" * 64)
    for arm in ("20m", "10m"):
        r = run(arm); res[arm] = r
        for k, (fx, bs, n) in r.items():
            if k.startswith("_"): continue
            print(f"{k:<20}{arm:>6}{ARMS[arm]['thr']:>9.2f}{fx:>12.3f}{bs:>13.3f}{n:>4}")
        print(f"  (CIR 전체 소각화소 중 대형4 의 몫 {r['_소각화소비'][0]:.1%})")
        for nm, key in (("CIR 전체", "_PR_CIR"), ("산림청", "_PR_산림청")):
            pr, rc, f1 = r[key]
            print(f"  ★{nm:<8} 화소합산 정밀도 {pr:.3f} · 재현율 {rc:.3f}  (F1 {f1:.3f})")
    json.dump(res, open(os.path.join(PROJ, "data", "eval_rulers.json"), "w"), ensure_ascii=False, indent=1)
    print("\n※best-thr 는 화재별로 임계값을 훑어 최댓값을 취한 것 — 배포에서 얻을 수 없다(비교용).")
    print("※산림청은 과대 라벨이라 회귀 감지용이고, CIR-GT 는 S2 지수 절단이라 준순환이다(GT_USAGE.md).")


if __name__ == "__main__":
    main()
