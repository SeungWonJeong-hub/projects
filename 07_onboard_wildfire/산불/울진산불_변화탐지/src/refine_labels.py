"""라벨 품질 정제 — 경계 폴리곤 안의 '안 탄 섬' 제거 (화재별 Otsu, 봄산불 대응).

문제: ko_multi 라벨=산림청 경계 폴리곤 래스터화 → 경계 안 미연소 패치도 '탔다'로 포함(노이즈).
해법: 경계 내부 픽셀의 dNBR 분포에 **화재별 Otsu** → 상위 클러스터(=실제 연소)만 유지.
      한국 봄산불은 dNBR 반전(불탄곳도 음수)이나, 배경이 개엽 초록화되면 불탄 픽셀이
      경계 내에서 *상대적으로* 높은 dNBR → 부호 무관, 화재별 상대 기준으로 분리됨.

안전장치: Otsu가 경계의 30~90%를 남길 때만 적용(명확한 이봉 분리). 그 밖(균일 연소·불신뢰)엔
          경계 원본 유지 → 실제 연소를 과잉 삭제하지 않음.

입력: ko_multi/*/*.npz (x[13,H,W] ch12=idx(dnbr), y[H,W]=경계라벨).  재다운로드 불필요.
출력: ko_multi_refined/<fire>/*.npz (동일 x, 정제된 y).
"""
from __future__ import annotations
import os, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "ko_multi")
OUT = os.path.join(HERE, "..", "ko_multi_refined")
KEEP_LO, KEEP_HI = 0.30, 0.90   # Otsu 유지비율이 이 범위일 때만 정제
MIN_BURN_PX = 200               # 경계 연소픽셀 이보다 적으면 정제 안 함


def otsu(vals, bins=256):
    """연속값 Otsu 임계 — between-class variance 최대화."""
    h, edges = np.histogram(vals, bins=bins)
    h = h.astype(np.float64); p = h / max(h.sum(), 1)
    cx = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); w1 = 1 - w0
    mu = np.cumsum(p * cx); mut = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mut * w0 - mu) ** 2 / (w0 * w1 + 1e-12)
    i = int(np.nanargmax(sb))
    return float(cx[i])


def dnbr_of(x):
    return 2.0 * x[12].astype(np.float32) - 1.0   # idx(dnbr)=(clip(dnbr,-1,1)+1)/2 역변환


def main():
    os.makedirs(OUT, exist_ok=True)
    fires = sorted(d for d in glob.glob(os.path.join(SRC, "*")) if glob.glob(os.path.join(d, "*.npz")))
    print(f"화재 {len(fires)}개 정제 시작", flush=True)
    stats = []
    tot_before = tot_after = 0
    for d in fires:
        name = os.path.basename(d)
        tiles = sorted(glob.glob(os.path.join(d, "*.npz")))
        data = [(f, np.load(f)) for f in tiles]
        pool = np.concatenate([dnbr_of(z["x"])[z["y"] > 0] for _, z in data]) if data else np.array([])
        od = os.path.join(OUT, name); os.makedirs(od, exist_ok=True)
        thr = None; action = "keep-full"
        if pool.size >= MIN_BURN_PX:
            t = otsu(pool)
            keep = float((pool >= t).mean())
            if KEEP_LO <= keep <= KEEP_HI:
                thr = t; action = f"refine(thr={t:.3f} keep={keep:.2f})"
        # 적용
        for f, z in data:
            x = z["x"]; y = z["y"].astype(np.uint8)
            b0 = int((y > 0).sum())
            if thr is not None:
                y = (y & (dnbr_of(x) >= thr)).astype(np.uint8)
            b1 = int((y > 0).sum())
            tot_before += b0; tot_after += b1
            np.savez_compressed(os.path.join(od, os.path.basename(f)), x=x, y=y)
        rm = 100 * (1 - (sum(int((np.load(os.path.join(od, os.path.basename(f)))["y"] > 0).sum()) for f, _ in data)
                         / max(sum(int((z["y"] > 0).sum()) for _, z in data), 1)))
        stats.append(dict(fire=name, tiles=len(tiles), action=action, removed_pct=round(rm, 1)))
        print(f"  [{name}] {action} 제거 {rm:.1f}%", flush=True)
    json.dump(stats, open(os.path.join(OUT, "refine_stats.json"), "w"), indent=2, ensure_ascii=False)
    refined = sum(1 for s in stats if s["action"].startswith("refine"))
    print(f"\n완료: {len(fires)}화재 중 {refined}개 정제, 경계 연소픽셀 {tot_before}→{tot_after} "
          f"({100*(1-tot_after/max(tot_before,1)):.1f}% 미연소섬 제거)", flush=True)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
