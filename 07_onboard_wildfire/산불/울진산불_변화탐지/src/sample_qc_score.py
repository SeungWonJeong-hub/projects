"""QC 게이트 채점 — 판독자가 자로 쓸 만한가.

★게이트는 **양측**이다(§SAMPLING_PROTOCOL). 양성 QC(폴리곤 안)만 보면 "전부 소각" 판독자가
만점을 받는다. 그래서 원시 일치율이 아니라 **균형정확도**와 **등급 선형가중 kappa** 로 잰다.
통과선: 균형정확도 ≥0.85 · 등급 kappa ≥0.60.

사용: conda run -n ml python src/sample_qc_score.py [--read sample/claude_read.jsonl]
"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
GRADES = ["경", "중", "심"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", default="sample/claude_read.jsonl")
    ap.add_argument("--points", default="sample/sample_points.json",
                    help="표본 정의(1차: sample_points.json · 재시험: sample_qc2.json)")
    ap.add_argument("--who", default="claude")
    a = ap.parse_args()
    pts = {p["id"]: p for p in json.load(open(os.path.join(PROJ, a.points)))["points"]}
    rd = {}
    for ln in open(os.path.join(PROJ, a.read), encoding="utf-8"):
        ln = ln.strip()
        if ln: r = json.loads(ln); rd[r["id"]] = r
    qc = [i for i, p in pts.items() if p.get("qc")]
    done = [i for i in qc if i in rd]
    print(f"QC {len(qc)}점 중 판독 {len(done)}점 · 판독자 {a.who}")

    # ── 이진(소각/비소각) 혼동행렬 ─────────────────────────────────────
    #    판정불가는 **분모에서 뺀다**(판독 거부는 오답이 아니다) — 몇 점인지는 따로 보고한다.
    tp = fp = tn = fn = 0; nd = 0; rows = []
    for i in done:
        t = pts[i]["truth"]; r = rd[i]["read"]
        t_burn = (t == "소각")
        if r == "판정불가":
            nd += 1; rows.append((i, t, r, "—")); continue
        r_burn = (r == "소각")
        if r_burn and t_burn: tp += 1; m = "○"
        elif r_burn and not t_burn: fp += 1; m = "✗과탐"
        elif not r_burn and t_burn: fn += 1; m = "✗누락"
        else: tn += 1; m = "○"
        rows.append((i, t, r, m))
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    bal = (sens + spec) / 2
    raw = (tp + tn) / max(tp + tn + fp + fn, 1)
    print(f"\n{'':<6}{'정답':<10}{'판독':<10}")
    for i, t, r, m in rows:
        if m != "○": print(f"{i:<6}{t:<10}{r:<10}{m}")
    print(f"\n양성 QC {sum(1 for i in done if pts[i]['truth']=='소각')} · "
          f"음성 QC {sum(1 for i in done if pts[i]['truth']!='소각')} · 판정불가 {nd}")
    print(f"  민감도(소각을 소각으로) {sens:.3f}   [{tp}/{tp+fn}]")
    print(f"  특이도(비소각을 비소각으로) {spec:.3f}   [{tn}/{tn+fp}]")
    print(f"  ★균형정확도 {bal:.3f}  (통과선 0.85)   원시일치율 {raw:.3f}")

    # ── 등급 선형가중 kappa (양성 QC 중 양쪽이 소각이고 등급이 있는 것) ────
    pair = [(pts[i].get("truth_grade"), rd[i].get("grade")) for i in done
            if pts[i]["truth"] == "소각" and rd[i]["read"] == "소각"
            and pts[i].get("truth_grade") in GRADES and rd[i].get("grade") in GRADES]
    if len(pair) >= 5:
        y = np.array([GRADES.index(t) for t, _ in pair]); x = np.array([GRADES.index(r) for _, r in pair])
        n = len(pair); k = 3
        O = np.zeros((k, k)); 
        for i2 in range(n): O[y[i2], x[i2]] += 1
        O /= n
        E = np.outer(O.sum(1), O.sum(0))
        W = np.array([[abs(i2 - j2) / (k - 1) for j2 in range(k)] for i2 in range(k)])
        kap = 1 - (W * O).sum() / max((W * E).sum(), 1e-9)
        print(f"\n  등급 선형가중 kappa {kap:.3f}  (n={n}, 통과선 0.60)")
        print("   정답\\판독 " + "".join(f"{g:>6}" for g in GRADES))
        for i2, g in enumerate(GRADES):
            print(f"   {g:<8}" + "".join(f"{int(O[i2,j2]*n):>6}" for j2 in range(k)))
    else:
        kap = None; print(f"\n  등급 kappa: 짝 {len(pair)}개 — 산출 불가")
    print(f"\n★게이트: 균형정확도 {'통과' if bal>=0.85 else '실패'}"
          + (f" · 등급 kappa {'통과' if (kap or 0)>=0.60 else '실패'}" if kap is not None else ""))
    json.dump(dict(who=a.who, n=len(done), sens=sens, spec=spec, balanced=bal, raw=raw,
                   kappa=kap, undecided=nd, tp=tp, fp=fp, tn=tn, fn=fn),
              open(os.path.join(PROJ, "data", f"qc_score_{a.who}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
