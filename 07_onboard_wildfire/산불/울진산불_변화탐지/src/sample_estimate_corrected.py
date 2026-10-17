"""확률표본 판독 — 판독자 오분류(민감도·특이도) 보정 면적추정.

`sample_estimate.py` 는 판독자를 참값으로 놓는다(Olofsson 2014 층화추정). 판독자 민감도가
1 이 아니면(탄 곳을 놓치면) 그 면적추정은 참 소각면적을 **아래로** 편향한다. 이 스크립트는
QC 로 잰 민감도·특이도로 그 편향을 되돌린다(Rogan–Gladen).

  p_reader = sens·p_true + (1−spec)·(1−p_true)
  →  p_true = (p_reader − (1−spec)) / (sens + spec − 1)      # spec=1 이면 p_true = p_reader/sens

불확실성은 몬테카를로로 전파한다: 층화추정 SE, sens~Beta, spec~Beta 를 함께 뽑아
보정 소각면적의 분포를 만든다. **판정불가**는 별도 카테고리라, "전부 비소각" 하한과
"전부 소각" 상한을 같이 낸다 — 이게 보통 민감도보다 결론을 더 크게 흔든다.

원본 층화추정 코어는 sample_estimate.py 와 동일하게 유지하고 보정층만 얹었다.

사용:
  python src/sample_estimate_corrected.py --form sample/sample_form_filled.csv
  python src/sample_estimate_corrected.py --jsonl sample/claude_read_main.jsonl   # 같은 id 마지막 줄 승
"""
import os, sys, csv, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
PXA = 20.0 * 20.0 / 1e4
READS = ["소각", "벌채·개간", "비변화", "판정불가"]
N_MC = 200_000


def load_reads(a):
    """판독을 {id: (판정, 등급, 근거)} 로. jsonl 우선(같은 id 마지막 줄 승), 없으면 csv."""
    read = {}
    if a.jsonl and os.path.exists(a.jsonl):
        for line in open(a.jsonl, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            i = (r.get("id") or "").strip()
            v = (r.get("판정") or r.get("verdict") or "").strip()
            if i.startswith("P") and v:
                read[i] = (v, (r.get("등급") or r.get("grade") or "").strip(),
                           (r.get("근거") or r.get("basis") or "").strip())
        return read, f"jsonl:{os.path.basename(a.jsonl)}"
    with open(a.form, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            i = (r.get("id") or "").strip()
            if i.startswith("P") and (r.get("판정") or "").strip():
                read[i] = ((r["판정"]).strip(), (r.get("등급") or "").strip(),
                           (r.get("근거") or "").strip())
    return read, f"csv:{os.path.basename(a.form)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default=os.path.join(PROJ, "sample", "sample_form.csv"))
    ap.add_argument("--jsonl", default=os.path.join(PROJ, "sample", "claude_read_main.jsonl"))
    ap.add_argument("--seed", type=int, default=20260910)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    S = json.load(open(os.path.join(PROJ, "sample", "sample_points.json"), encoding="utf-8"))
    by = {p["id"]: p for p in S["points"]}
    read, src = load_reads(a)
    if not read:
        print("판독이 비어 있다."); return
    print(f"판독 원천: {src} · 판독점 {len(read)}\n")

    # ── QC 로 민감도·특이도 (Jeffreys Beta 사후) ────────────────────
    pos = [(by[i], v) for i, v in read.items() if by[i].get("qc") and by[i].get("truth") == "소각"]
    neg = [(by[i], v) for i, v in read.items() if by[i].get("qc") and by[i].get("truth") == "비변화"]
    n_pos, n_neg = len(pos), len(neg)
    k_sens = sum(1 for p, v in pos if v[0] == "소각")
    k_spec = sum(1 for p, v in neg if v[0] != "소각")
    if n_pos == 0 or n_neg == 0:
        print("QC 양성/음성 중 하나가 0 — 보정 불가."); return
    sens_hat, spec_hat = k_sens / n_pos, k_spec / n_neg
    print(f"판독자 게이트 (QC 양성 {n_pos} / 음성 {n_neg})")
    print(f"  민감도 {sens_hat:.3f} ({k_sens}/{n_pos})   특이도 {spec_hat:.3f} ({k_spec}/{n_neg})")
    print(f"  ★spec=1 이면 보정계수 ≈ 1/민감도 = {1/max(sens_hat,1e-9):.3f}\n")
    sens_mc = rng.beta(k_sens + 0.5, n_pos - k_sens + 0.5, N_MC)
    spec_mc = rng.beta(k_spec + 0.5, n_neg - k_spec + 0.5, N_MC)

    # ── 사이트별 층화추정 + 보정 ────────────────────────────────────
    ri = {c: n for n, c in enumerate(READS)}
    for site, m in S["sites"].items():
        cnt = np.array(m["counts"], float); W = cnt / cnt.sum(); A = cnt.sum() * PXA
        n_h = np.zeros(3); tab = np.zeros((3, len(READS)))
        for i, (v, g, b) in read.items():
            p = by[i]
            if p.get("qc") or p["site"] != site or v not in ri:
                continue
            h = p["stratum"] - 1; n_h[h] += 1; tab[h, ri[v]] += 1
        if n_h.sum() == 0:
            continue
        ph = np.where(n_h[:, None] > 0, tab / np.maximum(n_h[:, None], 1), 0) * W[:, None]
        pq = ph.sum(0)
        se = np.array([np.sqrt(sum(W[h]**2 * (ph[h, q]/W[h]) * (1 - ph[h, q]/W[h]) / (n_h[h]-1)
                                   for h in range(3) if n_h[h] > 1)) for q in range(len(READS))])
        model_ha = W[0] * A
        p_burn, se_burn = pq[ri["소각"]], se[ri["소각"]]
        p_indet = pq[ri["판정불가"]]
        tag = " ★주 결과(누출 0)" if site == S.get("headline") else " (s50 누출 — 보조)"
        print(f"[{site}]{tag}  전체 {A:,.0f}ha · 모델 예측 소각 {model_ha:,.0f}ha")
        print(f"  판독 소각 {p_burn*A:,.0f} ±{1.96*se_burn*A:,.0f}ha · 판정불가 {p_indet*A:,.0f}ha")

        # 몬테카를로: 층화추정 표본오차 + sens/spec 오차 전파
        pr = rng.normal(p_burn, se_burn, N_MC).clip(0, 1)               # 판독 소각 비율
        denom = (sens_mc + spec_mc - 1).clip(1e-6, None)
        p_true = ((pr - (1 - spec_mc)) / denom).clip(0, 1)             # 하한: 판정불가=비소각
        p_true_hi = ((pr + p_indet - (1 - spec_mc)) / denom).clip(0, 1)  # 상한: 판정불가=소각
        for name, pt in [("민감도보정(판정불가=비소각)", p_true),
                         ("민감도보정(판정불가=소각, 상한)", p_true_hi)]:
            b = pt * A
            lo, med, hi = np.percentile(b, [2.5, 50, 97.5])
            op = (model_ha - b) / np.maximum(b, 1e-9) * 100
            olo, omed, ohi = np.percentile(op, [2.5, 50, 97.5])
            print(f"  {name}")
            print(f"    참 소각 {med:,.0f}ha (95% {lo:,.0f}–{hi:,.0f}) · "
                  f"모델 오차 {omed:+.1f}% (95% {olo:+.1f}~{ohi:+.1f}%)")
        # 판정불가를 안 건드린 원시(보정 전) 오차도 같이
        raw_op = (model_ha - p_burn * A) / max(p_burn * A, 1e-9) * 100
        print(f"  (보정 전 원시 오차 {raw_op:+.1f}%)\n")

    json.dump(dict(n_read=len(read), sens=sens_hat, spec=spec_hat,
                   n_qc_pos=n_pos, n_qc_neg=n_neg),
              open(os.path.join(PROJ, "sample", "sample_estimate_corrected.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
