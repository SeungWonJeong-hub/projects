"""미소각 오탐 분해 — 1,292ha 중 실제 토지변화는 얼마인가 (§12.5).

왜: 화재 없는 쌍(pre 2018-05-10 · post 2019-05-10)에서 배포 앙상블이 장면의 24% 를 소각으로
찍었다. 그런데 **1년 간격**이라 그 사이 실제 벌채·수확이 섞여 있다 — 전부가 오탐은 아니다.
분해하지 않으면 "모델을 고쳐야 하나 자료를 고쳐야 하나" 가 안 갈린다.

방법: **간격을 줄인다.** 같은 계절 2~4주 간격이면 벌채가 끼어들 여지가 거의 없다. 거기 남는 게
**순수 모델 오탐**이고, 1년 간격과의 차이가 **실제 변화분**이다.

자료는 재취득하지 않는다 — `nifos_cache/{SITE}_ts` 에 조화모델용으로 모아둔 243 날짜가
같은 격자·같은 밴드 순서라 임의의 쌍을 즉석에서 만들 수 있다. **전부 화재 이전 날짜**이므로
정답은 언제나 0ha 다.

★교란 주의: 봄은 2~4주만에도 녹화가 크게 진행된다. 그래서 계절을 **여름(DOY 170~260)** 과
  **봄(DOY 90~160)** 으로 갈라서 따로 낸다 — 짧은 간격의 잔여 신호가 녹화 탓인지 모델 탓인지
  구분하기 위해서다.

사용: conda run -n ml python src/nonfire_fp_probe.py [--site andong]
"""
import os, sys, json, glob, argparse, datetime as dt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
GSD = 20; PXA = GSD * GSD / 1e4
RUNS = "runs/cir_reh_ens/*.pt"
# 목표 간격(일)과 허용 오차 — 짧은 쪽은 엄격히, 1·2년은 DOY 정합을 위해 넉넉히
TARGETS = [(0, 0), (5, 2), (10, 3), (20, 4), (30, 5), (60, 7), (120, 10), (365, 12), (730, 12)]
SEASONS = {"봄 (DOY 90~160)": (90, 160), "여름 (DOY 170~260)": (170, 260)}


def load_ts(site):
    z = np.load(os.path.join(PROJ, "nifos_cache", f"{site}_ts", f"ts_{GSD}m.npz"), allow_pickle=True)
    ds = [dt.date.fromisoformat(s) for s in z["dates"]]
    return ds, z["x"], z["v"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="andong")
    ap.add_argument("--per-bin", type=int, default=3, help="간격·계절 조합당 최대 쌍 수")
    ap.add_argument("--min-valid", type=float, default=0.97, help="두 씬 모두 요구하는 유효화소 비율")
    a = ap.parse_args()

    os.environ.update(SITE=a.site, GSD=str(GSD), TILE="256", NIR="B8A")
    import nifos_eval as NE
    models = sorted(glob.glob(os.path.join(PROJ, RUNS)))
    assert models, "배포 앙상블 가중치 없음"

    ds, X, V = load_ts(a.site)
    vf = V.reshape(len(ds), -1).mean(1)
    ok = np.where(vf >= a.min_valid)[0]
    print(f"[{a.site}] 시계열 {len(ds)} 날짜 중 유효≥{a.min_valid:.0%} 인 {len(ok)} 날짜 사용 "
          f"({ds[0]} ~ {ds[-1]}, 전부 화재 이전)", flush=True)
    doy = np.array([d.timetuple().tm_yday for d in ds])

    idx = lambda b, i, j: (b[i] - b[j]) / (b[i] + b[j] + 1e-6)
    rows = []
    for sname, (d0, d1) in SEASONS.items():
        pool = [i for i in ok if d0 <= doy[i] <= d1]
        print(f"\n═══ {sname} — 후보 {len(pool)} 날짜 ═══")
        print(f"{'목표Δ':>6}{'실제Δ':>7}  {'pre':<12}{'post':<12}"
              f"{'배경dNDVI':>10}{'예측ha@0.20':>12}{'@0.30':>9}{'장면%':>7}")
        for tgt, tol in TARGETS:
            pairs = []
            for i in pool:
                for j in pool:
                    if j <= i and tgt > 0: continue
                    if tgt == 0 and j != i: continue
                    if abs((ds[j] - ds[i]).days - tgt) <= tol:
                        pairs.append((i, j))
            # 간격 오차가 작은 순, 그리고 서로 다른 시기가 골고루 섞이도록 앞에서 자른다
            pairs.sort(key=lambda p: (abs((ds[p[1]] - ds[p[0]]).days - tgt), ds[p[0]]))
            seen, sel = set(), []
            for i, j in pairs:
                if len(sel) >= a.per_bin: break
                if i in seen or j in seen: continue
                sel.append((i, j)); seen.update((i, j))
            if not sel:
                print(f"{tgt:>6}{'—':>7}  (해당 간격 쌍 없음)"); continue
            for i, j in sel:
                x = np.concatenate([X[i], X[j]], 0)
                prob, _ = NE.infer(x, models)
                xf = x.astype(np.float32) / 1e4
                m = V[i] & V[j]
                bg = float(np.median((idx(xf[:6], 3, 2) - idx(xf[6:], 3, 2))[m]))
                h20 = float((prob >= 0.20).sum() * PXA); h30 = float((prob >= 0.30).sum() * PXA)
                tot = prob.size * PXA
                print(f"{tgt:>6}{(ds[j]-ds[i]).days:>7}  {ds[i].isoformat():<12}{ds[j].isoformat():<12}"
                      f"{bg:>+10.3f}{h20:>12.0f}{h30:>9.0f}{h20/tot*100:>6.1f}%", flush=True)
                rows.append(dict(season=sname, target=tgt, delta=(ds[j]-ds[i]).days,
                                 pre=ds[i].isoformat(), post=ds[j].isoformat(),
                                 bg=bg, ha20=h20, ha30=h30, scene_ha=tot))

    # ── 요약: 간격이 늘면 면적이 얼마나 늘어나는가 ─────────────────────────
    print("\n═══ 요약 — 간격별 예측면적 중앙값(정답은 언제나 0ha) ═══")
    print(f"{'계절':<20}{'Δ(일)':>7}{'n':>3}{'예측ha@0.20':>12}{'@0.30':>9}")
    for sname in SEASONS:
        for tgt, _ in TARGETS:
            g = [r for r in rows if r["season"] == sname and r["target"] == tgt]
            if not g: continue
            print(f"{sname:<20}{tgt:>7}{len(g):>3}{np.median([r['ha20'] for r in g]):>12.0f}"
                  f"{np.median([r['ha30'] for r in g]):>9.0f}")
    json.dump(rows, open(os.path.join(PROJ, "data", f"nonfire_fp_{a.site}.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n기록: data/nonfire_fp_{a.site}.json")


if __name__ == "__main__":
    main()
