"""dNBR 심각도 라벨 대량 생산 — 라벨 없는 화재까지 학습에 넣는다 (착수 순서 ②-2).

구조(사용자 제안): **학습 라벨만 SWIR(dNBR)로 만들고, 모델 입력은 RGB+NIR 로 둔다.**
가중치에 SWIR 경로가 생기지 않으므로 온보드 탑재에 영향이 없다. 상한은 dNBR 이다.

★절단선은 **재보정값**을 쓴다. 배포 고정절단 0.27/0.44 로 라벨을 만들면 중등급 면적이
−64% 틀린 라벨을 245화재분 대량 생산하게 된다(`src/dnbr_threshold_calib.py` §2).
기본값 0.19/0.60 = 안동 NIFoS 등급에 **면적**을 맞춘 값(강릉 전이 정확도 0.738).

산림청 폴리곤은 **과대(경계) 라벨**이라 그대로 소각으로 쓰면 안 된다([[wildfire-gt-usage-scope]]).
폴리곤 **안이면서** dNBR ≥ t_lo 인 화소만 등급을 준다 — 나머지는 0(비소각)이다.

원본을 건드리지 않는다 — `ko_refl_s50/` 는 그대로 두고 `ko_sev_s50/` 에 사이드카로 쓴다.

QC 는 **차단하지 않고 기록만** 한다(§11: 어느 선에서 잘라도 정상 봄 화재를 대량으로 막는다).
학습 쪽에서 매니페스트를 보고 걸러 쓰라는 뜻이다.

★★2026-09-09 실행 결과: **이 풀에서는 만들 수 없다.** ko_refl 계열의 폴리곤 안 dNBR 중앙값이
   **음수**다(ko_refl −0.235 / ko_refl_s50 −0.371). dNDVI 도 같이 음수(−0.223/−0.374)이므로
   부호 버그가 아니라 **계절 어긋난 쌍**이다 — pre 가 잎 전, post 가 만엽이라 녹화 신호가 소각
   신호를 덮는다(학습셋 245화재 중 54%가 그 구간, §11.1). 안동 참조값은 +0.545 다.
   → 절단선을 어떻게 잡아도 폴리곤 통과율 중앙 0.001 이 나온다. **학습셋을 다시 짝지어야
   쓸 수 있다**(§12 조화 기저선을 pre 로 쓰면 가능해진다). 스크립트는 그때를 위해 남긴다.

사용: conda run -n ml python src/dnbr_labelset.py [--pool ko_refl_s50] [--t0 0.18 --lo 0.26 --hi 0.61]
"""
import os, sys, csv, json, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
GSD = 20.0; PXA = GSD * GSD / 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="ko_refl_s50")
    ap.add_argument("--out", default=None)
    ap.add_argument("--t0", type=float, default=0.18, help="소각/무시 절단 (안동 NIFoS 소각마스크 F1 최적)")
    ap.add_argument("--lo", type=float, default=0.26, help="경|중 절단 (t0 위에서 면적 최적)")
    ap.add_argument("--hi", type=float, default=0.61, help="중|심 절단")
    ap.add_argument("--bg-warn", type=float, default=0.10, help="배경 dNDVI 절댓값 경고선(§11)")
    a = ap.parse_args()
    out = os.path.join(PROJ, a.out or a.pool.replace("ko_refl", "ko_sev"))
    tiles = sorted(glob.glob(os.path.join(PROJ, a.pool, "*", "*.npz")))
    assert tiles, f"타일 없음: {a.pool}"
    print(f"[{a.pool}] 타일 {len(tiles)} · 절단 t0={a.t0} 등급 {a.lo}/{a.hi} → {os.path.basename(out)}")
    print(f"  ⚠ 이 절단으로 안동 NIFoS 경등급의 **71%가 무시 구간으로 빠진다** — 경등급 라벨은 "
          f"희소하고 밝은 쪽으로 편향된다. 심·중만 신뢰할 것.")

    idx = lambda b, i, j: (b[i] - b[j]) / (b[i] + b[j] + 1e-6)
    rows, tot = [], np.zeros(4)
    for t in tiles:
        z = np.load(t)
        x = z["x"].astype(np.float32) / 1e4
        y = z["y"].astype(bool); v = z["valid"].astype(bool)
        pre, post = x[:6], x[6:]
        dnbr = idx(pre, 3, 5) - idx(post, 3, 5)          # NBR = (NIR-B12)/(NIR+B12)
        dndvi = idx(pre, 3, 2) - idx(post, 3, 2)
        # ★4단 분류. t0 는 등급 절단 lo 와 **다른 값**이다 —
        #   lo 를 소각 절단으로 쓰면 경등급이 통째로 비소각이 된다(보정기 §4).
        #   0=비소각 · 1=심 · 2=중 · 3=경 · 255=무시(폴리곤 안인데 dNBR 이 낮은 구간)
        sev = np.zeros(y.shape, np.uint8)
        inp = y & v                                       # 산림청 폴리곤 안(과대 라벨)
        burn = inp & (dnbr >= a.t0)
        sev[inp & ~burn] = 255                            # 경등급과 폴리곤 과대분이 섞이는 구간
        sev[burn] = 3
        sev[burn & (dnbr >= a.lo)] = 2
        sev[burn & (dnbr >= a.hi)] = 1
        sev[~v] = 255                                     # 구름·무효 화소
        bg = float(np.median(dndvi[v & ~y])) if (v & ~y).sum() else float("nan")
        keep = float(burn.sum() / max(inp.sum(), 1))      # 폴리곤 중 dNBR 소각 절단을 통과한 비율
        cnt = np.array([(sev == k).sum() for k in (0, 1, 2, 3)], float)
        n_ign = int((sev == 255).sum())
        tot += cnt
        fire = os.path.basename(os.path.dirname(t))
        d = os.path.join(out, fire); os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, os.path.basename(t)),
                            y_sev=sev, valid=v.astype(np.uint8), dnbr=dnbr.astype(np.float32))
        rows.append(dict(fire=fire, tile=os.path.basename(t), bg_dndvi=round(bg, 4),
                         poly_keep=round(keep, 4), ha_severe=round(cnt[1] * PXA, 1),
                         ha_mod=round(cnt[2] * PXA, 1), ha_low=round(cnt[3] * PXA, 1),
                         ha_ignore=round(n_ign * PXA, 1),
                         seasonal_warn=int(abs(bg) >= a.bg_warn)))

    with open(os.path.join(out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    warn = sum(r["seasonal_warn"] for r in rows)
    keeps = np.array([r["poly_keep"] for r in rows])
    print(f"\n등급 면적 합계: 심 {tot[1]*PXA:,.0f} / 중 {tot[2]*PXA:,.0f} / 경 {tot[3]*PXA:,.0f} ha "
          f"(비소각 {tot[0]*PXA:,.0f})")
    print(f"산림청 폴리곤 중 dNBR≥{a.lo} 통과 비율: 중앙 {np.median(keeps):.3f} · "
          f"10% 분위 {np.percentile(keeps,10):.3f} · 90% 분위 {np.percentile(keeps,90):.3f}")
    print(f"  → 폴리곤이 실제 소각보다 넓다는 뜻이다. 중앙값 기준 {(1-np.median(keeps))*100:.0f}% 가 걸러졌다.")
    print(f"계절 경고(|배경 dNDVI|≥{a.bg_warn}) 타일: {warn}/{len(rows)} ({warn/len(rows)*100:.0f}%) "
          f"— ★차단하지 않는다(§11), 학습 쪽에서 매니페스트로 걸러 쓸 것")
    json.dump(dict(pool=a.pool, t0=a.t0, lo=a.lo, hi=a.hi, n_tiles=len(rows),
                   ha=dict(severe=tot[1] * PXA, mod=tot[2] * PXA, low=tot[3] * PXA),
                   poly_keep_median=float(np.median(keeps)), seasonal_warn=warn),
              open(os.path.join(PROJ, "data", "dnbr_labelset.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n기록: {os.path.basename(out)}/manifest.csv · data/dnbr_labelset.json")


if __name__ == "__main__":
    main()
