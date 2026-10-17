"""ASAP-pdSI 계절 불일치 테스트 (공정한 판정).

논문 핵심 주장: pdSI 이득은 pre/post가 계절 불일치일 때, dNDVI 등 phenology-민감
지수서 최대. 앞선 asap_pdsi.py의 울진 페어는 봄-봄(계절 일치, unburned median≈0)
이라 pdSI가 고칠 게 없었음 → 불일치 페어로 재검.

설계: post=실제 post-fire(uljin2022_after, 봄2022) 고정.
      pre를 계절별로 교체 (모두 pre-fire, GT격자로 재투영):
        - canon_spring : sample_tif/uljin2022_before (봄, 정본)
        - arch_spring  : 광역아카이브 52SCH 2020 봄
        - arch_summer  : 광역아카이브 52SCH 2020 여름  ← 불일치(여름 leaf-on vs 봄 post)
아카이브는 B02/03/04/08만(SWIR無) → NDVI만. NDVI=(B08-B04)/(B08+B04).

가설: dNDVI(summer) << dNDVI(spring)  [계절오염]
      pdNDVI(summer) ≈ dNDVI(spring)  [pdSI 회복]  ← 회복하면 ASAP 작동 입증
검증: NIFoS 피해강도(심3중2경1) Spearman ρ + extent F1 + unburned median(계절 크기).
"""
import os, json, glob
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation

A = os.path.expanduser("~/Desktop/sentinel2_data")
GTP = os.path.expanduser("~/Steven/nifos_burn_gt/ext/Ulgin_severity.tif")
POST = os.path.expanduser("~/syntax_inference/Steven/웹앱/sample_tif/uljin2022_after.tif")
CANON_PRE = os.path.expanduser("~/syntax_inference/Steven/웹앱/sample_tif/uljin2022_before.tif")
OUT = os.path.expanduser("~/syntax_inference/Steven/산불/울진산불_변화탐지/runs/asap")
os.makedirs(OUT, exist_ok=True)

# pre 후보(계절). 여러 개 두고 유효율 보고 골라씀.
ARCH_SPRING = ["S2A_52SEG_20200407_1_L2A", "S2A_52SEG_20200414_1_L2A", "S2B_52SEG_20200429_1_L2A", "S2A_52SEG_20200504_1_L2A"]
ARCH_SUMMER = ["S2B_52SEG_20200608_1_L2A", "S2B_52SEG_20200628_1_L2A", "S2A_52SEG_20200716_1_L2A", "S2A_52SEG_20200726_1_L2A"]
N_STRATA = 5
EDGE_BUF = 2


def spearman(a, b):
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra*ra).sum()*(rb*rb).sum())
    return float((ra*rb).sum()/d) if d else float("nan")


def f1_sweep(score, ref, lo=0.0, hi=0.6, step=0.02):
    best = (0.0, 0.0)
    for t in np.arange(lo, hi, step):
        pred = score >= t
        tp = int((pred & ref).sum()); fp = int((pred & ~ref).sum()); fn = int((~pred & ref).sum())
        P = tp/(tp+fp) if tp+fp else 0.0; R = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*P*R/(P+R) if P+R else 0.0
        if f1 > best[1]: best = (round(float(t),2), round(float(f1),3))
    return best


def band_on_grid(path, gds, resamp=Resampling.average):
    with rasterio.open(path) as ds:
        dst = np.zeros((gds.height, gds.width), "float32")
        reproject(rasterio.band(ds, 1), dst, src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=gds.transform, dst_crs=gds.crs, resampling=resamp)
    return dst / 10000.0


def ndvi_from_archive(folder, gds):
    red = band_on_grid(f"{A}/{folder}/B04.tif", gds)
    nir = band_on_grid(f"{A}/{folder}/B08.tif", gds)
    valid = (nir + red > 0) & (nir > 0.001) & (red > 0.001)
    ndvi = np.where(valid, (nir - red)/(nir + red + 1e-6), np.nan)
    return ndvi, valid


def ndvi_from_6band(path, gds):
    # B02,B03,B04,B08,B11,B12 → red idx2, nir idx3
    with rasterio.open(path) as ds:
        red = np.zeros((gds.height, gds.width), "float32"); nir = red.copy()
        reproject(rasterio.band(ds,3), red, src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=gds.transform, dst_crs=gds.crs, resampling=Resampling.average)
        reproject(rasterio.band(ds,4), nir, src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=gds.transform, dst_crs=gds.crs, resampling=Resampling.average)
    red/=10000.0; nir/=10000.0
    valid = (nir + red > 0) & (nir > 0.001) & (red > 0.001)
    ndvi = np.where(valid, (nir - red)/(nir + red + 1e-6), np.nan)
    return ndvi, valid


def detrend(dsi, unburned, strata):
    pd = dsi.copy()
    for c in np.unique(strata):
        cls = strata == c; ref = unburned & cls
        off = np.median(dsi[ref]) if ref.sum() >= 30 else np.median(dsi[unburned])
        pd[cls] = dsi[cls] - off
    return pd


def pick_clean(cands, gds, burn):
    """AOI(비소각 영역)서 유효율 가장 높은 씬 선택."""
    best = None; bestv = -1
    for f in cands:
        if not os.path.isdir(f"{A}/{f}"): continue
        ndvi, valid = ndvi_from_archive(f, gds)
        vf = float((valid & ~burn).sum()) / max(1, (~burn).sum())
        mn = float(np.nanmedian(ndvi[valid & ~burn])) if (valid & ~burn).sum() else float("nan")
        print(f"    cand {f}: 유효율={vf:.2f} unburned NDVI중앙={mn:+.3f}")
        if vf > bestv: bestv, best = vf, (f, ndvi, valid)
    return best


def main():
    g = rasterio.open(GTP); gt = g.read(1); H, W = gt.shape
    ordm = np.zeros((H,W), np.uint8); ordm[gt==1]=3; ordm[gt==2]=2; ordm[gt==3]=1
    burn = gt > 0
    burn_dil = binary_dilation(burn, iterations=EDGE_BUF)

    post_ndvi, post_valid = ndvi_from_6band(POST, g)

    conds = {}
    print("[canon_spring] pre = uljin2022_before")
    ns, vs = ndvi_from_6band(CANON_PRE, g); conds["canon_spring"] = (ns, vs)
    print("[arch_spring] 후보 스캔"); r = pick_clean(ARCH_SPRING, g, burn_dil)
    if r: print(f"  → 선택 {r[0]}"); conds["arch_spring"] = (r[1], r[2])
    print("[arch_summer] 후보 스캔"); r = pick_clean(ARCH_SUMMER, g, burn_dil)
    if r: print(f"  → 선택 {r[0]}"); conds["arch_summer"] = (r[1], r[2])

    results = {}
    print("\n" + "="*78)
    print(f"{'condition':14} {'unburned dNDVI':>14} | {'ρ dSI':>8} {'ρ pdSI':>8} | {'F1 dSI':>8} {'F1 pdSI':>8}")
    print(f"{'(pre season)':14} {'(계절크기)':>14} | {'(NIFoS)':>8} {'(strat)':>8} | {'sweep':>8} {'(strat)':>8}")
    print("-"*78)
    for name, (pre_ndvi, pre_valid) in conds.items():
        valid = pre_valid & post_valid & ~np.isnan(pre_ndvi) & ~np.isnan(post_ndvi)
        dsi = pre_ndvi - post_ndvi          # burn>0 (식생손실)
        unb = valid & ~burn_dil
        qs = np.quantile(pre_ndvi[unb], np.linspace(0,1,N_STRATA+1)[1:-1]) if unb.sum() else []
        strata = np.digitize(np.nan_to_num(pre_ndvi), qs).astype(np.int16)
        pd_strat = detrend(dsi, unb, strata)

        m = burn & valid
        sev = ordm[m].astype(float)
        r_dsi = spearman(sev, dsi[m]); r_pd = spearman(sev, pd_strat[m])
        f1_dsi = f1_sweep(dsi, burn & valid); f1_pd = f1_sweep(pd_strat, burn & valid)
        umed = float(np.median(dsi[unb]))
        results[name] = dict(unburned_median=umed, rho_dSI=r_dsi, rho_pdSI=r_pd,
                             F1_dSI=f1_dsi, F1_pdSI=f1_pd, n_burn=int(m.sum()))
        print(f"{name:14} {umed:>+14.3f} | {r_dsi:>+8.3f} {r_pd:>+8.3f} | {f1_dsi[1]:>8.3f} {f1_pd[1]:>8.3f}")

    with open(f"{OUT}/mismatch_result.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("-"*78)
    print("판정: arch_summer(불일치)서 dSI 붕괴 & pdSI가 arch_spring 수준 회복하면 → ASAP 작동")
    print(f"→ {OUT}/mismatch_result.json")


if __name__ == "__main__":
    main()
