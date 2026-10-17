"""확률표본 판독 — 표본 설계·추출 (GT_USAGE.md 말미가 예고한 것, 미결 최상위).

왜: 독립 자가 안동 1건뿐이고(§15), 학습 라벨은 화소 단위로 못 쓴다(§21). 전수 경계를 새로
그리는 대신 **확률표본**을 뽑아 독립 영상에서 판독하면, 추출확률을 반영해 면적·오차행렬을
**편향 없이** 추정할 수 있다(Olofsson et al. 2014 good practice).

★이 설계의 핵심 이점: 판독 대상이 **라벨이 아니라 영상**이라, RapidEye 부분 라벨의 바깥
(씬의 92~99%)도 판정이 된다 — "라벨 밖을 모름으로 빼는" 문제가 구조적으로 사라진다(§16).

모집단
  강릉 2019_63 · 고성 2019_74 · 양양 2018_33 의 RapidEye 5m 씬 footprint 안 **20m 화소**.
  ※인제 2019_71_1 은 전 학습셋 누출이라 제외(채점 금지). 2016년 3건은 S2 씬이 없어 제외.

층(strata) — 배포 20m 앙상블 @thr 0.25 의 예측으로 나눈다
  1 소각      p ≥ thr
  2 경계      예측 경계에서 ±BUF_M 이내(양쪽) — 오차가 몰리는 곳이라 따로 뽑는다
  3 비소각    나머지
층은 서로 겹치지 않고 씬을 덮는다. 각 층에서 **단순무작위 비복원** 추출한다.

판독 항목: 비소각 / 경 / 중 / 심 / 판독불가  (기존 RapidEye 라벨의 degree 와 같은 눈금)

사용: conda run -n ml python src/sample_design.py --n 1000 --target 0.85
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
GSD = 20.0; PXA = GSD * GSD / 1e4
THR = float(os.environ.get("THR", "0.25"))          # 배포 20m 절단(2026-09-10 반영)
RUNS = os.environ.get("RUNS", "runs/cir_reh_ens/*.pt")
BUF_M = float(os.environ.get("BUF_M", "100"))       # 경계층 폭(양쪽)
# ★사이트 × 층 **교차** 층화. 사이트마다 n 을 따로 준다 — 통합값이 아니라 **사이트별**이
#   보고 단위이기 때문이다(강릉만 누출 0 이라 헤드라인이고, 소각 비중이 0.2~7.4% 로 제각각이라
#   통합 평균은 해석이 어렵다). 사이트별 추정이 각자 성립해야 한다.
SITES = [("gangneung", "2019_63", 700),   # 누출 0 — **주 결과**
         ("goseong",   "2019_74", 300),   # s50 누출 — 보조
         ("yangyang",  "2018_33", 200)]   # s50 누출 — 보조
SEED = int(os.environ.get("SEED", "0"))


def strata(site):
    """(층 라벨 1/2/3, 확률맵, 지오변환) — 배포 예측으로 층을 만든다."""
    import glob, importlib
    os.environ.update(SITE=site, GSD="20", TILE="256", NIR="B8A", SCENE_PICK="doy")
    import nifos_eval as NE; importlib.reload(NE)
    C = os.path.join(PROJ, "nifos_cache", site + "_refoot")   # ★RapidEye 씬 전체(§표본 모집단)
    z = np.load(os.path.join(C, "scene_20m.npz"))
    x = z["x"]
    prob, _ = NE.infer(x, sorted(glob.glob(os.path.join(PROJ, RUNS))))
    burn = prob >= THR
    # S2 가 구름인 화소는 예측이 쓰레기라 **모집단에서 뺀다**(판독은 RapidEye 로 하지만 층이 무의미).
    bad = [0, 1, 3, 8, 9, 10, 11]
    ok = ~np.isin(z["scl"][0], bad) & ~np.isin(z["scl"][1], bad)
    ok &= (x[0] > 0) & (x[6] > 0)
    from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
    k = int(round(BUF_M / GSD))
    st = generate_binary_structure(2, 2)
    grow = binary_dilation(burn, st, iterations=k)
    shrink = binary_erosion(burn, st, iterations=k)
    lab = np.where(burn, 1, 3).astype(np.uint8)
    lab[grow & ~shrink] = 2                      # 경계 = 팽창 ∩ 침식여집합(소각·비소각 양쪽)
    lab[~ok] = 0                                 # 0 = 모집단 제외(구름·무효)
    ute = np.load(os.path.join(C, "ute_20m.npy"))
    return lab, prob, ute


def n_olofsson(W, U, se_target):
    """Olofsson 2014 (13): n = (Σ W_i S_i)² / S(Ô)² · S_i = √(U_i(1−U_i))."""
    S = np.sqrt(np.array(U) * (1 - np.array(U)))
    return float((np.dot(W, S) / se_target) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", default="0.30,0.35,0.35", help="사이트 안 층 배분(소각,경계,비소각)")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    labs, meta, rows = {}, {}, []
    tot = np.zeros(4, np.int64)
    print(f"층화 — 배포 20m @thr {THR} · 경계 ±{BUF_M:g}m")
    print(f"{'사이트':<12}{'RE 씬':<10}{'씬 화소':>10}{'소각':>10}{'경계':>10}{'비소각':>10}")
    for site, re_id, _n in SITES:
        lab, prob, ute = strata(site)
        labs[site] = lab
        os.makedirs(os.path.join(PROJ, "sample"), exist_ok=True)
        np.savez_compressed(os.path.join(PROJ, "sample", f"strata_{site}.npz"),
                            lab=lab, ute=ute)      # 음성 QC 가 "예측 소각에서 얼마나 먼가" 를 쓴다
        c = np.array([0] + [int((lab == i).sum()) for i in (1, 2, 3)])
        tot += c
        meta[site] = dict(re_scene=re_id, shape=list(lab.shape), counts=c[1:].tolist(),
                          ute=[float(v) for v in ute], excluded_px=int((lab == 0).sum()))
        print(f"{site:<12}{re_id:<10}{lab.size:>10,}{c[1]:>10,}{c[2]:>10,}{c[3]:>10,}"
              f"{'  (구름제외 '+format(int((lab==0).sum()),',')+'px)' if (lab==0).any() else ''}")
    N = tot[1:].sum(); W = tot[1:] / N
    print(f"{'합계':<12}{'':<10}{N:>10,}{tot[1]:>10,}{tot[2]:>10,}{tot[3]:>10,}")
    print(f"{'층 비중 W':<22}{'':>10}{W[0]:>10.3f}{W[1]:>10.3f}{W[2]:>10.3f}   (총 {N*PXA:,.0f}ha)")

    # ── 규모: **사이트 단독** 추정에 필요한 n ────────────────────────
    U = [0.85, 0.50, 0.99]      # 층별 예상 사용자정확도(소각·경계·비소각)
    print(f"\n규모(Olofsson 2014) · 가정 사용자정확도 {U} · **사이트 단독** 기준")
    print(f"  {'사이트':<12}{'±20%':>8}{'±10%':>8}{'±5%':>8}")
    for site, _r, _n in SITES:
        cnt = np.array(meta[site]["counts"], float); Ws = cnt / cnt.sum()
        p_s = float(Ws[0] + 0.5 * Ws[1])
        need = [n_olofsson(Ws, U, p_s * rel / 1.96) for rel in (0.20, 0.10, 0.05)]
        print(f"  {site:<12}{need[0]:>8,.0f}{need[1]:>8,.0f}{need[2]:>8,.0f}")

    # ── 추출: 사이트 × 층 교차 ───────────────────────────────────
    al = np.array([float(x) for x in a.alloc.split(",")]); al /= al.sum()
    print(f"\n추출 — 사이트 × 층 교차 층화 · 층 배분 {a.alloc}")
    print(f"{'사이트':<12}{'n':>5}{'소각':>7}{'경계':>7}{'비소각':>8}{'소각면적 95%CI':>16}")
    site_se = {}
    for site, re_id, n_site in SITES:
        lab = labs[site]; cnt = np.array(meta[site]["counts"], float)
        Wsite = cnt / cnt.sum()
        n_c = np.maximum(np.round(al * n_site).astype(int), 8)      # 층당 최소 8점
        for s_ in (1, 2, 3):
            idx = np.flatnonzero(lab.ravel() == s_)
            k = min(int(n_c[s_-1]), idx.size)
            if k == 0: continue
            pick = rng.choice(idx, size=k, replace=False)
            ys, xs = np.unravel_index(pick, lab.shape)
            for y, x in zip(ys, xs):
                ux = meta[site]["ute"][0] + (x + 0.5) * GSD
                uy = meta[site]["ute"][3] - (y + 0.5) * GSD
                rows.append(dict(site=site, re_scene=re_id, stratum=int(s_),
                                 row=int(y), col=int(x), utm_x=round(ux, 1), utm_y=round(uy, 1),
                                 pi=float(k / cnt[s_-1]), read="", grade="", note=""))
        se_s = np.sqrt(sum(Wsite[i]**2 * U[i]*(1-U[i]) / max(n_c[i], 1) for i in range(3)))
        p_s = float(Wsite[0] + 0.5 * Wsite[1])
        site_se[site] = (float(se_s), p_s)
        print(f"{site:<12}{int(n_c.sum()):>5}{n_c[0]:>7}{n_c[1]:>7}{n_c[2]:>8}"
              f"{'±'+format(1.96*se_s/max(p_s,1e-9)*100, '.1f')+'%':>16}")
    print(f"  ※CI 는 그 사이트 **단독** 추정의 것이다. 통합값은 내지 않는다 — 사이트별로 보고한다.")
    rng.shuffle(rows)                    # 판독 순서에서 층을 숨긴다(편향 방지)
    # ※판독자 검증(QC) 점은 gdal 이 필요해 별도다 — src/sample_qc.py (geo) 가 섞어 넣는다.
    for i, r in enumerate(rows, 1): r["id"] = f"P{i:04d}"
    out = os.path.join(PROJ, "sample", "sample_points.json")
    json.dump(dict(thr=THR, buf_m=BUF_M, seed=SEED, gsd=GSD, sites=meta,
                   strata_counts=tot[1:].tolist(), W=W.tolist(),
                   site_n={s_: n_ for s_, _r, n_ in SITES},
                   headline="gangneung",           # 누출 0 — 주 결과
                   n=len(rows), points=rows),
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n표본 {len(rows)}점 → sample/sample_points.json (판독 순서는 섞었다 — 층을 모르게)")


if __name__ == "__main__":
    main()
