"""다시점(multi-temporal) 구름·그림자 마스크 — SCL(Sen2Cor) 보강.

배경: Baetens+2019 (Remote Sensing 11:433, `Steven/논문/구름마스킹_논문`).
 - 단일시점 SCL(Sen2Cor)은 얇은 구름·구름그림자를 놓치고 밝은 지표를 과탐.
   (우리 실측: Railroad Valley 밝은모래 씬 SCL IoU 0.048 = 논문 지적 재현)
 - MAJA식 '다시점'이 더 정확(ALCD 벤치 90.8% > Sen2Cor 84%).
 - 우리는 전/후(bi-temporal) 쌍을 이미 보유 → 시점 간 변화로 SCL 누락분을 보강.

번(burn)-안전 설계 (산불 변화탐지이므로 실제 불탄 픽셀을 지우면 안 됨):
 - 구름: 전대역(특히 blue)을 '밝힌다'.
 - 그림자: 전대역(SWIR 포함)을 '어둡힌다'.
 - 번:   blue를 밝히지 않고(오히려 유지·감소) + SWIR(B12)는 char/ash로 '유지·상승'.
 → blue가 급격히 '밝아짐' = 구름,  SWIR까지 함께 '어두워짐' = 그림자 로 번과 분리.

입력 밴드는 원시 DN(0~10000). 반사율 r=DN/1e4 로 판정.
build_dataset.py(울진), ko_build_multi.py(한국 다지역) 공통 사용.
"""
from __future__ import annotations
import numpy as np

# nodata, saturated, cloud-shadow, cloud-medium, cloud-high, thin-cirrus, snow
SCL_DROP = (0, 1, 3, 8, 9, 10, 11)


def _refl(b):
    return np.asarray(b, np.float32) / 1e4


def mt_cloud_shadow_bad(pre, post,
                        blue_abs=0.20, blue_delta=0.06,
                        dark_abs=0.06, dark_blue=0.10, dark_delta=0.05):
    """전/후 씬 dict(B02,B08,B12 필요) → SCL이 놓친 구름·그림자 bad 마스크(bool 배열).

    반환 True = SCL 외에 추가로 구름/그림자로 판정되어 제외할 픽셀.
    임계값은 반사율 기준(경험적, 한국 봄 씬 튜닝 여지 있음).
    """
    pb, pg, p12 = _refl(pre['B02']), _refl(pre['B08']), _refl(pre['B12'])
    ob, og, o12 = _refl(post['B02']), _refl(post['B08']), _refl(post['B12'])
    dblue = ob - pb       # +면 post가 밝아짐
    dnir = og - pg

    # 다시점 구름: 한 시점에서 blue가 절대적으로 밝고 + 상대적으로 급격히 밝아짐
    cloud_post = (ob > blue_abs) & (dblue > blue_delta)
    cloud_pre = (pb > blue_abs) & (dblue < -blue_delta)

    # 다시점 그림자: 한 시점에서 NIR·SWIR·blue 모두 어둡고 + 그 시점으로 급격히 어두워짐.
    #   SWIR(B12)까지 어두움을 요구 → char/ash로 SWIR 유지되는 '번'은 제외(번-안전).
    shadow_post = (og < dark_abs) & (o12 < dark_abs) & (ob < dark_blue) & (dnir < -dark_delta)
    shadow_pre = (pg < dark_abs) & (p12 < dark_abs) & (pb < dark_blue) & (dnir > dark_delta)

    return cloud_post | cloud_pre | shadow_post | shadow_pre


def scl_bad(pre, post):
    """단일시점 SCL 기반 bad(bool) — 기존 파이프라인과 동일 기준."""
    return (np.isin(np.asarray(pre['SCL']).astype(np.int16), SCL_DROP)
            | np.isin(np.asarray(post['SCL']).astype(np.int16), SCL_DROP))


def data_valid(pre, post):
    """밴드 값이 유효(>0)한 픽셀."""
    return ((np.asarray(pre['B02']) > 0) & (np.asarray(pre['B08']) > 0) & (np.asarray(pre['B12']) > 0)
            & (np.asarray(post['B02']) > 0) & (np.asarray(post['B08']) > 0) & (np.asarray(post['B12']) > 0))


def valid_multitemporal(pre, post, use_mt=True):
    """최종 valid(bool) = 데이터유효 ∧ ¬SCL_bad ∧ ¬다시점_bad.

    use_mt=False 면 기존 단일시점 SCL 동작으로 폴백(대조 실험용).
    """
    v = data_valid(pre, post) & ~scl_bad(pre, post)
    if use_mt:
        v &= ~mt_cloud_shadow_bad(pre, post)
    return v


def report(pre, post):
    """SCL 대비 다시점이 추가로 잡는 구름/그림자 비율을 리턴(딕셔너리)."""
    data = data_valid(pre, post)
    sb = scl_bad(pre, post) & data
    mb = mt_cloud_shadow_bad(pre, post) & data
    extra = mb & ~sb
    n = data.sum() + 1e-9
    return dict(
        scl_bad_pct=round(float(sb.sum() / n * 100), 3),
        mt_bad_pct=round(float(mb.sum() / n * 100), 3),
        extra_over_scl_pct=round(float(extra.sum() / n * 100), 3),
    )


if __name__ == "__main__":
    # 울진 raw 씬으로 셀프테스트: SCL이 놓친 구름/그림자를 얼마나 추가로 잡는지.
    import os
    from osgeo import gdal

    HERE = os.path.dirname(os.path.abspath(__file__))
    RAW = os.path.join(HERE, "..", "data", "raw")
    PRE = os.path.join(RAW, "S2A_52SEG_20210419_2_L2A")
    POST = os.path.join(RAW, "S2B_52SEG_20220409_0_L2A")

    def load(d):
        return {b: gdal.Open(os.path.join(d, f"{b}.tif")).ReadAsArray().astype(np.float32)
                for b in ("B02", "B03", "B04", "B08", "B11", "B12", "SCL")}

    pre, post = load(PRE), load(POST)
    print("[울진 pre=2021-04-19 post=2022-04-09]")
    print(" ", report(pre, post))
    v_old = valid_multitemporal(pre, post, use_mt=False)
    v_new = valid_multitemporal(pre, post, use_mt=True)
    print(f"  valid  SCL만={v_old.mean()*100:.2f}%  다시점={v_new.mean()*100:.2f}%  "
          f"(추가 제외 {((v_old & ~v_new).sum())} px)")
