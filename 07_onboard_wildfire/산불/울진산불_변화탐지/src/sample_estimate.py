"""확률표본 판독 — 층화 추정(사이트별 면적·오차행렬·CI)과 판독자 게이트.

Olofsson et al. 2014 good practice. 층(모델 예측)과 판독(참값)의 교차표를 **층 비중 W_h 로
가중**한다. 층별로 다르게 뽑았어도(비비례 배분) 추정은 편향되지 않는다.

  p̂_hq = W_h · n_hq / n_h · 면적 = Σ_h p̂_hq · A
  SE(p̂_q) = √( Σ_h W_h²·(p̂_hq/W_h)(1 − p̂_hq/W_h)/(n_h − 1) )

★**사이트별로 보고한다.** 강릉만 누출 0 이라 주 결과이고, 소각 비중이 0.2~7.4% 로 제각각이라
통합 평균은 해석이 어렵다. 통합값은 내지 않는다.

판독자 게이트(양측):  균형정확도 ≥ 0.85  ·  등급 선형가중 kappa ≥ 0.60
  균형정확도 = (양성 QC 민감도 + 음성 QC 특이도)/2 — 양성만 보면 "전부 소각" 판독자가 만점이다.

사용: conda run -n ml python src/sample_estimate.py [--form sample/sample_form.csv]
"""
import os, sys, csv, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
PXA = 20.0 * 20.0 / 1e4
READS = ["소각", "벌채·개간", "비변화", "판정불가"]
GRADES = ["경", "중", "심"]


def wkappa(a, b, cats):
    """선형 가중 kappa — 3급 불균형에서 원시 일치율은 우연만으로 0.6~0.7 이 나온다."""
    k = len(cats); i = {c: n for n, c in enumerate(cats)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        if x in i and y in i: O[i[x], i[y]] += 1
    if O.sum() == 0: return float("nan")
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    W = 1 - np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    po = (W * O).sum(); pe = (W * E).sum()
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default=os.path.join(PROJ, "sample", "sample_form.csv"))
    ap.add_argument("--reads", nargs="*", default=None,
                    help="판독 기록 jsonl (기본: sample/claude_read_main.jsonl + claude_read*.jsonl). "
                         "★기록의 단일 소스는 jsonl 이다 — csv 기입표는 사람이 손으로 채울 때만 쓴다.")
    a = ap.parse_args()
    S = json.load(open(os.path.join(PROJ, "sample", "sample_points.json"), encoding="utf-8"))
    by = {p["id"]: p for p in S["points"]}

    # ── 판독 기록 읽기 ─────────────────────────────────────────────
    # jsonl 한 줄 = {"id","read","grade","basis","conf","note"} — **append-only**. 같은 id 가
    # 여러 번 나오면 **마지막 줄이 이긴다**(재확인이 원판을 덮어쓴다: recheck·dNBR 재검).
    src = a.reads
    if src is None:
        src = [f for f in (os.path.join(PROJ, "sample", "claude_read_main.jsonl"),)
               if os.path.exists(f)]
    read, conf = {}, {}
    for f in src:
        with open(f, encoding="utf-8") as fh:
            for l in fh:
                l = l.strip()
                if not l: continue
                r = json.loads(l)
                if r.get("id") in by and (r.get("read") or "").strip():
                    read[r["id"]] = (r["read"].strip(), (r.get("grade") or "").strip(),
                                     (r.get("basis") or "").strip())
                    conf[r["id"]] = (r.get("conf") or "").strip()
    if not read and os.path.exists(a.form):
        with open(a.form, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                i = (r.get("id") or "").strip()
                if i.startswith("P") and (r.get("판정") or "").strip():
                    read[i] = ((r["판정"]).strip(), (r.get("등급") or "").strip(),
                               (r.get("근거") or "").strip())
                    conf[i] = (r.get("확신") or "").strip()
    if not read:
        print(f"판독 결과가 비어 있다 — {src or a.form} 를 채운 뒤 다시 실행할 것."); return
    n_main = sum(1 for i in read if not by[i].get("qc"))
    n_tot = sum(1 for p in S["points"] if not p.get("qc"))
    print(f"판독 기록 {len(read)}점 (비-QC {n_main}/{n_tot}) ← {', '.join(os.path.basename(f) for f in src)}")
    if n_main < n_tot:
        print(f"  ⚠아직 {n_tot - n_main}점이 안 읽혔다 — 아래 추정은 **중간값**이다.\n")
    else:
        print()

    # ── 판독자 게이트(양측) ────────────────────────────────────────
    pos = [(by[i], v) for i, v in read.items() if by[i].get("qc") and by[i].get("truth") == "소각"]
    neg = [(by[i], v) for i, v in read.items() if by[i].get("qc") and by[i].get("truth") == "비변화"]
    if pos or neg:
        sens = sum(1 for p, v in pos if v[0] == "소각") / max(len(pos), 1)
        spec = sum(1 for p, v in neg if v[0] != "소각") / max(len(neg), 1)
        bal = (sens + spec) / 2
        kg = wkappa([p.get("truth_grade") for p, v in pos if v[0] == "소각"],
                    [v[1] for p, v in pos if v[0] == "소각"], GRADES)
        print(f"판독자 게이트")
        print(f"  민감도(양성 QC n={len(pos)})  {sens:.3f}   — 탄 곳을 탔다고 하는가")
        print(f"  특이도(음성 QC n={len(neg)})  {spec:.3f}   — 안 탄 곳을 안 탔다고 하는가")
        print(f"  ★균형정확도 {bal:.3f}  (기준 ≥0.85)  {'통과' if bal >= 0.85 else '미달 — 이 판독은 자로 쓰지 않는다'}")
        print(f"  등급 선형가중 kappa {kg:.3f}  (기준 ≥0.60)  {'통과' if kg >= 0.60 else '미달 — 등급 추정은 보류'}")
        if len(neg) < 20:
            print(f"  ※음성 QC 가 {len(neg)}점뿐이라 특이도 신뢰구간이 넓다.")
        print()

    # ── 사이트별 층화 추정 ─────────────────────────────────────────
    ri = {c: n for n, c in enumerate(READS)}
    out = {}            # ★표에 찍은 값을 그대로 파일로도 남긴다(문서가 손으로 옮긴 숫자가 되지 않게)
    for site, m in S["sites"].items():
        cnt = np.array(m["counts"], float); W = cnt / cnt.sum(); A = cnt.sum() * PXA
        n_h = np.zeros(3); tab = np.zeros((3, len(READS)))
        for i, (v, g, b) in read.items():
            p = by[i]
            if p.get("qc") or p["site"] != site or v not in ri: continue
            h = p["stratum"] - 1; n_h[h] += 1; tab[h, ri[v]] += 1
        if n_h.sum() == 0: continue
        ph = np.where(n_h[:, None] > 0, tab / np.maximum(n_h[:, None], 1), 0) * W[:, None]
        pq = ph.sum(0)
        se = np.array([np.sqrt(sum(W[h]**2 * (ph[h, q]/W[h]) * (1 - ph[h, q]/W[h]) / (n_h[h]-1)
                                   for h in range(3) if n_h[h] > 1)) for q in range(len(READS))])
        tag = " ★주 결과(누출 0)" if site == S.get("headline") else " (s50 누출 — 보조)"
        print(f"[{site}]{tag}  전체 {A:,.0f}ha · 모델 예측 소각 {W[0]*A:,.0f}ha")
        print(f"  {'층':<8}{'n':>5}" + "".join(f"{c:>10}" for c in READS))
        for h, nm in enumerate(["1 소각", "2 경계", "3 비소각"]):
            print(f"  {nm:<8}{int(n_h[h]):>5}" + "".join(f"{ph[h,q]:>10.4f}" for q in range(len(READS))))
        print(f"  {'추정 면적':<13}" + "".join(f"{pq[q]*A:>10,.0f}" for q in range(len(READS))) + "  ha")
        print(f"  {'95% CI':<13}" + "".join(f"{'±'+format(1.96*se[q]*A, ',.0f'):>10}" for q in range(len(READS))))
        b_ha = pq[ri["소각"]] * A
        out[site] = dict(
            total_ha=round(A, 1), pred_burn_ha=round(W[0] * A, 1),
            n=dict(zip(("s1", "s2", "s3"), n_h.astype(int).tolist())),
            est_ha={c: round(pq[q] * A, 1) for c, q in ri.items()},
            ci95_ha={c: round(1.96 * se[q] * A, 1) for c, q in ri.items()},
            area_bias_pct=round((W[0] * A - b_ha) / max(b_ha, 1e-9) * 100, 1),
            headline=(site == S.get("headline")))
        print(f"  → 소각 면적오차 {(W[0]*A - b_ha)/max(b_ha, 1e-9)*100:+.1f}%"
              f"  ·  벌채로 판독된 면적 {pq[ri['벌채·개간']]*A:,.0f}ha"
              f"  ·  판정불가 {pq[ri['판정불가']]*A:,.0f}ha\n")

    # ── 근거 민감도: RapidEye 만으로 판정한 점만 다시 집계 ─────────────
    only_re = {i: v for i, v in read.items() if v[2].upper() == "RE"}
    print(f"근거 분포 — RE 단독 {len(only_re)} / 전체 판독 {len(read)}"
          f" ({len(only_re)/max(len(read),1):.0%})")
    print("  ※RE 단독만으로 다시 집계했을 때 결론이 바뀌면, 그 결론은 S2(모델 입력)에 기댄 것이다.")
    json.dump(dict(n_read=len(read), n_main=n_main, n_main_total=n_tot,
                   complete=(n_main >= n_tot), sites=out,
                   only_re=len(only_re), note="등급은 추정에서 제외(§23.2). 판정불가는 소각에 넣지 않은 하한이다."),
              open(os.path.join(PROJ, "sample", "sample_estimate.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
