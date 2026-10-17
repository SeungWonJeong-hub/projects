"""CIR-GT 화재별 난이도 프로파일 — **모델 출력을 쓰지 않고** 라벨·영상만으로 채점 난이도를 잰다.

점수가 나쁜 화재를 사후에 골라내면 체리피킹이다. 이 스크립트는 모델을 전혀 참조하지 않으므로
채점 전에 "이 화재는 장면 제약(scene-limited)"을 선언할 수 있다.

지표 (gt_stac[_<gsd>m_raw].npz 의 x=8ch rgbnir, gt, valid 만 사용):
  GT신호   GT 내부 dNDVI 중앙값. 낮을수록 소각이 흐려 **미탐(recall)** 이 는다.
  오염배수 GT 밖에서 GT 하위25%와 같은 세기의 소각유사 신호가 GT 면적의 몇 배인가.
           GT 생성식과 맞추려 dNDVI 뿐 아니라 post-NDVI<0.45 도 요구하고,
           3x3 열림으로 산발 픽셀을 턴다(이 두 보정 없이는 값이 몇 배로 부풀려진다).
  최대덩어리 오염 덩어리 중 최대 면적. 크면 "라벨 안 된 이웃 화재" 를 의심해야 한다.

판정 규칙(2026-09-08 확정): **오염배수 > 1.0x 또는 GT신호 < 0.20 이면 scene-limited 로 표시**하고
화재평균에서 따로 보고한다. 24화재 중 250322_울주온양읍 만 걸린다(GT신호 0.180).

사용: python cir_gt_profile.py [화재명...]     (인자 없으면 CIR_HOLD + SMALL_HOLD 전체)
env: GSD=20|10  NPZ=<파일명>
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
FIRES = os.path.join(PROJ, "..", "산불데이터", "국토정보플랫폼 데이터셋", "CIR_실측라벨", "fires")
GSD = float(os.environ.get("GSD", "20"))
NPZ = os.environ.get("NPZ", "gt_stac_20m_raw.npz" if GSD == 20 else f"gt_stac_{int(GSD)}m_raw.npz")
PXHA = GSD * GSD / 1e4
POSTNDVI = 0.45                      # build_cir_gt_stac.py 와 같은 값
FLAG_CONTAM, FLAG_SIGNAL = 1.0, 0.20


def ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-6)


def profile(name):
    p = os.path.join(FIRES, name, NPZ)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    x = z["x"].astype(np.float32); gt = z["gt"].astype(bool); vd = z["valid"].astype(bool)
    if gt.sum() == 0 or vd.mean() < 0.2:
        return {"name": name, "skip": f"무효(valid {vd.mean()*100:.0f}%, GT {int(gt.sum())}px)"}
    pre, post = x[:4], x[4:]
    dnd = ndvi(pre[3], pre[2]) - ndvi(post[3], post[2])
    npost = ndvi(post[3], post[2])
    sig = float(np.median(dnd[gt]))
    contam = ndimage.binary_opening(
        (dnd > np.percentile(dnd[gt], 25)) & vd & (~gt) & (npost < POSTNDVI), np.ones((3, 3)))
    lab, k = ndimage.label(contam)
    sz = ndimage.sum(contam, lab, range(1, k + 1)) if k else [0]
    ratio = float(contam.sum() / gt.sum())
    flags = []
    if ratio > FLAG_CONTAM:
        flags.append("오염")
    if sig < FLAG_SIGNAL:
        flags.append("약신호")
    return {"name": name, "ha": gt.sum() * PXHA, "sig": sig, "contam_ha": contam.sum() * PXHA,
            "ratio": ratio, "biggest": max(sz) * PXHA, "n_blob": k,
            "flag": "scene-limited(" + "/".join(flags) + ")" if flags else ""}


def main():
    names = sys.argv[1:]
    if not names:
        sys.path.insert(0, HERE)
        import step4_rehearsal as S
        names = list(S.CIR_HOLD) + list(S.SMALL_HOLD)
    print(f"GT={NPZ} GSD={GSD:.0f}m  (모델 미사용)\n")
    print(f"{'화재':<20}{'GT ha':>7}{'GT신호':>8}{'오염ha':>8}{'오염배수':>9}{'최대덩어리':>10}  표시")
    print("-" * 78)
    rows = []
    for n in names:
        r = profile(n)
        if r is None:
            print(f"{n:<20}   GT 없음"); continue
        if "skip" in r:
            print(f"{n:<20}   {r['skip']}"); continue
        rows.append(r)
        print(f"{r['name']:<20}{r['ha']:>7.0f}{r['sig']:>8.3f}{r['contam_ha']:>8.0f}"
              f"{r['ratio']:>8.2f}x{r['biggest']:>9.1f}ha  {r['flag']}")
    if rows:
        import statistics as st
        print(f"\n중앙값 GT신호 {st.median([r['sig'] for r in rows]):.3f} | "
              f"오염배수 {st.median([r['ratio'] for r in rows]):.2f}x")
        f = [r["name"] for r in rows if r["flag"]]
        print(f"scene-limited {len(f)}건: {', '.join(f) if f else '없음'}")


if __name__ == "__main__":
    main()
