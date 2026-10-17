"""ASAP-pdSI 재현 실험 (Sung et al. 2026, RSE 343:115488).

phenology-detrended differenced Spectral Index:
  dSI   = SI_pre - SI_post                (표준, burn>0)
  pdSI  = dSI - median_{unburned,class}(dSI)
          = 주변 unburned의 계절 dSI(=phenology 성분)를 클래스별로 제거
소각 픽셀엔 진짜 소각 신호만 남음 → 계절 불일치·개엽 반전에 강건(하다는 주장).

검증: NIFoS 피해강도(심/중/경) 서수 vs (dSI, pdSI) Spearman ρ.
가설 H1: ρ(pdSI) >= ρ(dSI)  / H2: 봄 반전 억제 / H3: extent F1 개선.
baseline 기지: 울진 20m ρ(dNBR)=0.093 (xval_uljin.py).

밴드순 B02,B03,B04,B08(NIR),B11,B12(SWIR2), /10000.
"""
import os, json
import numpy as np, rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import binary_dilation

FIRES = {
    "uljin2022": dict(
        bef=os.path.expanduser("~/syntax_inference/Steven/웹앱/sample_tif/uljin2022_before.tif"),
        aft=os.path.expanduser("~/syntax_inference/Steven/웹앱/sample_tif/uljin2022_after.tif"),
        gt =os.path.expanduser("~/Steven/nifos_burn_gt/ext/Ulgin_severity.tif"),  # 1심 2중 3경
    ),
}
OUT = os.path.expanduser("~/syntax_inference/Steven/산불/울진산불_변화탐지/runs/asap")
os.makedirs(OUT, exist_ok=True)
N_STRATA = 5          # pre-NDVI 분위 pseudo-class 개수
EDGE_BUF = 2          # 소각 경계 buffer(px) — unburned 참조서 혼합 경계 제외


def spearman(a, b):
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra*ra).sum()*(rb*rb).sum())
    return float((ra*rb).sum()/d) if d else float("nan")


def f1_sweep(score, ref_burn, lo=0.02, hi=0.6, step=0.02):
    best = (0.0, 0.0)
    for t in np.arange(lo, hi, step):
        pred = score >= t
        tp = int((pred & ref_burn).sum()); fp = int((pred & ~ref_burn).sum()); fn = int((~pred & ref_burn).sum())
        P = tp/(tp+fp) if tp+fp else 0.0; R = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*P*R/(P+R) if P+R else 0.0
        if f1 > best[1]: best = (float(t), float(f1))
    return best  # (t*, F1*)


def read_index(path, bounds, H, W, kind):
    ds = rasterio.open(path)
    win = from_bounds(*bounds, transform=ds.transform)
    a = ds.read(window=win, boundless=True, fill_value=0).astype("float32")/10000.0
    a = a[:, :H, :W]
    blue, green, red, nir, swir1, swir2 = a[0], a[1], a[2], a[3], a[4], a[5]
    valid = (nir + swir2 > 0) & (nir + red > 0)
    if kind == "NBR":
        si = (nir - swir2)/(nir + swir2 + 1e-6)
    elif kind == "NDVI":
        si = (nir - red)/(nir + red + 1e-6)
    else:
        raise ValueError(kind)
    return si, valid, nir, red


def detrend(dsi, unburned, strata):
    """pdSI = dsi - class-median(dsi over unburned). strata: int label map."""
    pd = dsi.copy()
    for c in np.unique(strata):
        cls = strata == c
        ref = unburned & cls
        off = np.median(dsi[ref]) if ref.sum() >= 30 else np.median(dsi[unburned])
        pd[cls] = dsi[cls] - off
    return pd


def run_fire(name, cfg):
    g = rasterio.open(cfg["gt"]); gt = g.read(1); bounds = g.bounds; H, W = gt.shape
    # NIFoS: 1심 2중 3경 → 서수 심3 중2 경1
    ordm = np.zeros((H, W), np.uint8); ordm[gt==1]=3; ordm[gt==2]=2; ordm[gt==3]=1
    burn = gt > 0
    # unburned 참조: 소각 아님 & 소각경계 buffer 밖
    burn_dil = binary_dilation(burn, iterations=EDGE_BUF)

    res = {"fire": name, "grid": [int(W), int(H)], "n_burn": int(burn.sum())}
    for kind in ("NBR", "NDVI"):
        si_b, vb, nir_b, red_b = read_index(cfg["bef"], bounds, H, W, kind)
        si_a, va, _, _ = read_index(cfg["aft"], bounds, H, W, kind)
        valid = vb & va
        dsi = si_b - si_a                       # burn>0
        # pre-NDVI 분위 pseudo-class (phenology/식생타입 층화)
        ndvi_pre = (nir_b - red_b)/(nir_b + red_b + 1e-6)
        unb = valid & ~burn_dil
        qs = np.quantile(ndvi_pre[unb], np.linspace(0, 1, N_STRATA+1)[1:-1]) if unb.sum() else []
        strata = np.digitize(ndvi_pre, qs).astype(np.int16)
        glob1 = np.zeros((H, W), np.int16)      # 단일 클래스

        pd_glob = detrend(dsi, valid & ~burn_dil, glob1)
        pd_strat = detrend(dsi, valid & ~burn_dil, strata)

        m = burn & valid                        # 소각 픽셀서 severity 상관
        sev = ordm[m].astype(float)
        row = {
            "rho_dSI":     spearman(sev, dsi[m]),
            "rho_pdSI_glob":  spearman(sev, pd_glob[m]),
            "rho_pdSI_strat": spearman(sev, pd_strat[m]),
        }
        # unburned 계절 잔차: pd가 0에 붙어야(=phenology 제거 확인)
        ub = valid & ~burn_dil
        row["unburned_median_dSI"]  = float(np.median(dsi[ub]))
        row["unburned_median_pdSI_strat"] = float(np.median(pd_strat[ub]))
        # extent F1 (Phase D): score 임계 sweep vs NIFoS burn
        row["F1_dSI"]  = f1_sweep(dsi, burn & valid)
        row["F1_pdSI_strat"] = f1_sweep(pd_strat, burn & valid)
        # 등급별 신호 중앙값(물리 일관성: 심>중>경)
        row["by_grade_dSI"]  = {n: float(np.median(dsi[(gt==c)&valid])) for c,n in [(1,"심"),(2,"중"),(3,"경")]}
        row["by_grade_pdSI"] = {n: float(np.median(pd_strat[(gt==c)&valid])) for c,n in [(1,"심"),(2,"중"),(3,"경")]}
        res[kind] = row
    return res


if __name__ == "__main__":
    out = []
    for name, cfg in FIRES.items():
        if not all(os.path.exists(cfg[k]) for k in ("bef","aft","gt")):
            print(f"[skip] {name}: 파일 없음"); continue
        r = run_fire(name, cfg); out.append(r)
        print(f"\n===== {name}  (burn {r['n_burn']}px, grid {r['grid']}) =====")
        for kind in ("NBR","NDVI"):
            d = r[kind]
            print(f"  [{kind}]  ρ:  dSI={d['rho_dSI']:+.3f}   pdSI(glob)={d['rho_pdSI_glob']:+.3f}   pdSI(strat)={d['rho_pdSI_strat']:+.3f}")
            print(f"         unburned median:  dSI={d['unburned_median_dSI']:+.3f} → pdSI={d['unburned_median_pdSI_strat']:+.3f}  (0에 붙어야 계절제거)")
            print(f"         등급별 dSI 심/중/경:  {d['by_grade_dSI']['심']:+.3f}/{d['by_grade_dSI']['중']:+.3f}/{d['by_grade_dSI']['경']:+.3f}")
            print(f"         extent F1:  dSI={d['F1_dSI'][1]:.3f}(t={d['F1_dSI'][0]:.2f})   pdSI={d['F1_pdSI_strat'][1]:.3f}(t={d['F1_pdSI_strat'][0]:.2f})")
    with open(os.path.join(OUT, "pdsi_result.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n→ {OUT}/pdsi_result.json")
