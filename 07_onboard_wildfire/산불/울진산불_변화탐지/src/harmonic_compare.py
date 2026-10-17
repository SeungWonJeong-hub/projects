"""조화모델 기저선 vs 직전 pre 영상 — 착수 순서 ①. 이득이 있는지 여기서 판명된다.

두 가지를 잰다(둘 다 §11 의 미결이다):
  A. **pre 민감도** — 배포 앙상블은 pre 를 뭘 고르냐에 따라 F1 0.855~0.912 · 면적 +2.8~+15.2%p
     로 흔들린다(data/qc_calibrate_andong.json). 조화 기저선은 pre 선택 변수를 **정의상 없앤다**.
     남는 질문은 "어느 실제 pre 만큼 좋은가" — **최선의 pre 와 겨뤄야** 의미가 있다.
  B. **post 민감도** — §11.3 의 진짜 미결. 같은 pre 로 post 만 바꿔도 면적이 크게 흔들렸다.
     조화 기저선은 post 날짜에 맞춰 pre 를 다시 예측하므로 이 흔들림도 줄어야 한다.

★한계(해석 시 반드시 같이 읽을 것): 배포 가중치는 **실제 pre 관측**으로 학습됐다. 매끄럽고
  구름 없는 합성 pre 는 학습분포 밖이다 — 조화 쪽이 지면 "생각이 틀렸다" 가 아니라 "재학습 없이는
  안 된다" 일 수 있다. 이 실험은 **재학습 없는 즉시 이득**만 판정한다.

사용: conda run -n ml python src/harmonic_compare.py --posts 2020-06-08 2020-05-21 --ref-pre 2019-05-10
"""
import os, sys, json, glob, argparse, subprocess, datetime as dt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
GSD = 20


def harm_cache(site, post):
    """조화 기저선 캐시(없으면 geo 환경에서 생성)."""
    dst = os.path.join(PROJ, "nifos_cache", f"{site}_harm_q{post.replace('-','')}")
    if os.path.exists(os.path.join(dst, f"scene_{GSD}m.npz")):
        return dst
    env = dict(os.environ, SITE=site); env.pop("PROJ_DATA", None)
    r = subprocess.run(["conda", "run", "-n", "geo", "python",
                        os.path.join(HERE, "harmonic_baseline.py"), "--site", site, "--post", post],
                       env=env, capture_output=True, text=True, cwd=PROJ)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(f"조화 기저선 생성 실패 {post}")
    print(r.stdout.strip().splitlines()[-2] if r.stdout.strip() else "", flush=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="andong")
    ap.add_argument("--posts", nargs="+", default=["2020-06-08"])
    ap.add_argument("--ref-pre", default="2019-05-10", help="post 민감도 비교용 고정 실제 pre")
    a = ap.parse_args()

    import qc_calibrate as QC
    QC.SITE = a.site

    px = GSD * GSD / 1e4
    rows = []

    # ── A. pre 민감도 ─────────────────────────────────────────────────────
    j = os.path.join(PROJ, "data", f"qc_calibrate_{a.site}.json")
    base = json.load(open(j)) if os.path.exists(j) else []
    p0 = a.posts[0]
    base = [b for b in base]                       # post 는 qc_calibrate 가 고정해 둔 값
    hc = harm_cache(a.site, p0)
    h = QC.score(hc); h["pre"] = f"조화@{p0}"; h["ddoy"] = 0

    print(f"\n═══ A. pre 민감도 — {a.site} · post 고정 {p0} · 배포 20m 앙상블 ═══")
    print(f"{'pre':<14}{'ΔDOY':>5}{'배경dNDVI':>11}{'F1@0.2':>9}{'예측ha':>9}"
          f"{'면적오차':>9}{'최적thr':>8}{'최적F1':>8}")
    for r in sorted(base, key=lambda r: r["pre"]) + [h]:
        print(f"{r['pre']:<14}{r['ddoy']:>5}{r['bg']:>+11.3f}{r['f1']:>9.4f}{r['pred_ha']:>9.0f}"
              f"{r['area_err']:>+8.1f}%{r['best_thr']:>8.3f}{r['best_f1']:>8.4f}")
    if base:
        f1s = [r["f1"] for r in base]; aes = [r["area_err"] for r in base]
        print(f"\n  실제 pre {len(base)}종: F1 {min(f1s):.4f}~{max(f1s):.4f} (폭 {max(f1s)-min(f1s):.4f}) · "
              f"면적오차 {min(aes):+.1f}~{max(aes):+.1f}%p (폭 {max(aes)-min(aes):.1f}%p)")
        bi = max(range(len(base)), key=lambda i: f1s[i])
        print(f"  최선의 실제 pre  {base[bi]['pre']}: F1 {f1s[bi]:.4f} · 면적 {aes[bi]:+.1f}%")
        print(f"  조화 기저선      : F1 {h['f1']:.4f} ({h['f1']-f1s[bi]:+.4f}) · "
              f"면적 {h['area_err']:+.1f}% ({abs(h['area_err'])-abs(aes[bi]):+.1f}%p 절댓값)")
        print(f"  평균 실제 pre 대비: F1 {h['f1']-np.mean(f1s):+.4f} · "
              f"면적 절댓값 {abs(h['area_err'])-np.mean(np.abs(aes)):+.1f}%p")
    rows.append({"exp": "pre_sensitivity", "post": p0, "real": base, "harmonic": h})

    # ── B. post 민감도 ────────────────────────────────────────────────────
    if len(a.posts) > 1:
        print(f"\n═══ B. post 민감도 — pre 조건 2종 × post {len(a.posts)}종 ═══")
        print(f"{'post':<14}{'조건':<16}{'배경dNDVI':>11}{'F1@0.2':>9}{'예측ha':>9}{'면적오차':>9}")
        out = {"real": [], "harm": []}
        for q in a.posts:
            c = QC.prep(a.ref_pre, q); r = QC.score(c); r["post"] = q
            print(f"{q:<14}{'실제 pre '+a.ref_pre[-5:]:<16}{r['bg']:>+11.3f}{r['f1']:>9.4f}"
                  f"{r['pred_ha']:>9.0f}{r['area_err']:>+8.1f}%")
            out["real"].append(r)
            hq = QC.score(harm_cache(a.site, q)); hq["post"] = q
            print(f"{'':<14}{'조화@post':<16}{hq['bg']:>+11.3f}{hq['f1']:>9.4f}"
                  f"{hq['pred_ha']:>9.0f}{hq['area_err']:>+8.1f}%")
            out["harm"].append(hq)
        for k, lab in (("real", f"실제 pre {a.ref_pre}"), ("harm", "조화 기저선")):
            ha = [r["pred_ha"] for r in out[k]]; f1s = [r["f1"] for r in out[k]]
            print(f"\n  {lab:<22} 면적 {min(ha):.0f}~{max(ha):.0f}ha "
                  f"(폭 {(max(ha)-min(ha))/min(ha)*100:.1f}%) · F1 폭 {max(f1s)-min(f1s):.4f}")
        rows.append({"exp": "post_sensitivity", "ref_pre": a.ref_pre, **out})

    o = os.path.join(PROJ, "data", f"harmonic_compare_{a.site}.json")
    json.dump(rows, open(o, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n기록: data/harmonic_compare_{a.site}.json")


if __name__ == "__main__":
    main()


def fp_score(cache, runs=None):
    """대조군 채점 — GT 가 0ha 이므로 F1·면적오차가 정의되지 않는다. 예측 면적(=전부 오탐)만 낸다."""
    import glob as _g, importlib
    os.environ.update(SITE=os.environ.get("SITE", "andong"), GSD=str(GSD), TILE="256", NIR="B8A")
    import nifos_eval as NE; importlib.reload(NE)
    x = np.load(os.path.join(cache, f"scene_{GSD}m.npz"))["x"]
    prob, _ = NE.infer(x, sorted(_g.glob(os.path.join(PROJ, "runs/cir_reh_ens/*.pt"))))
    px = GSD * GSD / 1e4
    return {"fp_ha@0.20": float((prob >= 0.20).sum() * px),
            "fp_ha@0.30": float((prob >= 0.30).sum() * px),
            "prob_median": float(np.median(prob)), "prob_p99": float(np.percentile(prob, 99))}
