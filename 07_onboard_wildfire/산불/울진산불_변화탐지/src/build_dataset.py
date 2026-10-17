"""Step 4 — refine the pseudo-label and build the 13-channel CD training set.

Improvements over the fixed dNBR>0.27 label (for accuracy):
  - RBR (Relativized Burn Ratio, Parks 2014) computed for reference.
  - Data-driven Otsu threshold on dNBR (restricted to pre-fire vegetated valid pixels,
    NDVI_pre>0.2), clamped to [0.15, 0.35] for stability.
  - Morphological open+close + min-area filter (drop components < 0.5 ha) -> coherent scar.

Input tensor per tile: 13 channels = pre[6] | post[6] | dNBR[1]  (all in [0,1]).
Spatial block split by column to reduce train/test leakage within the single scene:
  left 60% -> train, next 20% -> val, right 20% -> test  (overlapping tiles in train only).
"""
import os, sys, json
import numpy as np
import cv2
from osgeo import gdal
gdal.UseExceptions()

HERE = os.path.dirname(__file__)
# GSD 파라미터화: 기본 20 → 기존 20m 산출물과 완전 동일(베이스라인 무손상).
# GSD=10 → data/raw_10m 읽고 data/cd_10m·outputs_10m 로 저장.
GSD = float(os.environ.get("GSD", "20"))
_SUF = "" if GSD == 20 else f"_{int(GSD)}m"
RAW_BASE = os.path.join(HERE, "..", "data", "raw" + _SUF)
PAIR = os.path.join(HERE, "..", "data", "scene_pair.json")
CD = os.path.join(HERE, "..", "data", "cd" + _SUF)
OUTTIF = os.path.join(HERE, "..", "data", "outputs" + _SUF)
for split in ("train", "val", "test"):
    os.makedirs(os.path.join(CD, split), exist_ok=True)
os.makedirs(OUTTIF, exist_ok=True)

SCL_DROP = {0, 1, 3, 8, 9, 10, 11}
SEV_BREAKS = [0.10, 0.27, 0.44, 0.66]
TILE = 256
PX_HA = (GSD * GSD) / 1e4
MIN_AREA_HA = 0.5


def load_scene(d):
    bands = {}
    gt = proj = None
    for b in ("B02", "B03", "B04", "B08", "B11", "B12", "SCL"):
        ds = gdal.Open(os.path.join(d, f"{b}.tif"))
        bands[b] = ds.ReadAsArray().astype(np.float32)
        if b == "B02":
            gt, proj = ds.GetGeoTransform(), ds.GetProjection()
    return bands, gt, proj


def ratio(a, b, eps=1e-6):
    return (a - b) / (a + b + eps)


def stretch(chan, valid, lo=2, hi=98):
    v = chan[valid]
    a, b = np.percentile(v, lo), np.percentile(v, hi)
    if b <= a: b = a + 1
    return np.clip((chan - a) / (b - a), 0, 1).astype(np.float32)


def scene_valid(bd):
    data = (bd["B02"] > 0) & (bd["B08"] > 0) & (bd["B12"] > 0)
    return data & ~np.isin(bd["SCL"].astype(np.int16), list(SCL_DROP))


def stack6(bd, valid):
    idx = lambda x: np.clip((x + 1) / 2, 0, 1).astype(np.float32)
    return np.stack([stretch(bd["B12"], valid), stretch(bd["B08"], valid), stretch(bd["B02"], valid),
                     idx(ratio(bd["B08"], bd["B12"])), idx(ratio(bd["B08"], bd["B04"])),
                     idx(ratio(bd["B03"], bd["B08"]))], 0)


def write_gtiff(path, arr, gt, proj, nodata=None):
    drv = gdal.GetDriverByName("GTiff")
    h, w = arr.shape
    ds = drv.Create(path, w, h, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(gt); ds.SetProjection(proj)
    ds.GetRasterBand(1).WriteArray(arr.astype(np.float32))
    if nodata is not None: ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds = None


def refine_label(dnbr, valid, ndvi_pre):
    veg = valid & (ndvi_pre > 0.2)
    vals = np.clip(dnbr[veg], 0, 1)
    u8 = (vals * 255).astype(np.uint8)
    otsu, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(otsu / 255.0, 0.15, 0.35))
    raw = (valid & (dnbr >= thr)).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
    # min-area filter
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_px = int(MIN_AREA_HA / PX_HA)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            keep[lab == i] = 1
    return keep.astype(np.uint8), thr


def main():
    pair = json.load(open(PAIR))
    # scene id 재사용, RAW_BASE(20m=data/raw, 10m=data/raw_10m)에서 로드 → GSD A/B 공정.
    pre, gt, proj = load_scene(os.path.join(RAW_BASE, pair["pre"]["id"]))
    post, _, _ = load_scene(os.path.join(RAW_BASE, pair["post"]["id"]))
    valid = scene_valid(pre) & scene_valid(post)
    # 다시점 구름·그림자 보강(SCL 누락분 제거, 번-안전) — Baetens+2019 시사점 적용.
    if os.environ.get("MT_CLOUD", "1") == "1":
        from mt_cloud import mt_cloud_shadow_bad
        mtb = mt_cloud_shadow_bad(pre, post)
        print(f"[다시점 구름/그림자] SCL 외 추가 제외 {int((mtb & valid).sum())} px")
        valid &= ~mtb

    nbr_pre = ratio(pre["B08"], pre["B12"]); nbr_post = ratio(post["B08"], post["B12"])
    ndvi_pre = ratio(pre["B08"], pre["B04"])
    dnbr = (nbr_pre - nbr_post).astype(np.float32); dnbr[~valid] = 0
    rbr = (dnbr / (nbr_pre + 1.001)).astype(np.float32); rbr[~valid] = 0

    label, thr = refine_label(dnbr, valid, ndvi_pre)
    label[~valid] = 0
    sev = np.zeros_like(dnbr, np.uint8)
    for i, t in enumerate(SEV_BREAKS, 1): sev[dnbr >= t] = i
    sev[~valid] = 255

    pre6, post6 = stack6(pre, valid), stack6(post, valid)
    dnbr_n = ((np.clip(dnbr, -1, 1) + 1) / 2).astype(np.float32)
    image = np.concatenate([pre6, post6, dnbr_n[None]], 0)  # (13,H,W)

    # crop to valid bbox
    ys, xs = np.where(valid)
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = lambda a: a[..., r0:r1, c0:c1]
    image, label, valid, dnbr, rbr, sev = map(crop, (image, label, valid, dnbr, rbr, sev))
    gt_c = (gt[0] + c0 * gt[1], gt[1], 0, gt[3] + r0 * gt[5], 0, gt[5])
    H, W = label.shape

    # georeferenced refined outputs
    write_gtiff(os.path.join(OUTTIF, "dnbr.tif"), dnbr, gt_c, proj, 0)
    write_gtiff(os.path.join(OUTTIF, "rbr.tif"), rbr, gt_c, proj, 0)
    write_gtiff(os.path.join(OUTTIF, "pseudolabel_refined.tif"), label.astype(np.float32), gt_c, proj, 0)
    write_gtiff(os.path.join(OUTTIF, "severity.tif"), sev.astype(np.float32), gt_c, proj, 255)
    np.savez_compressed(os.path.join(CD, "uljin_full.npz"),
                        image=image, mask=label, valid=valid.astype(np.uint8),
                        dnbr=dnbr, geotransform=np.array(gt_c), projection=proj,
                        crop=np.array([r0, r1, c0, c1]))

    # spatial ROW-block split (burn is evenly spread by row; columns leave test empty).
    # Bands are >= TILE tall and tiles are ANCHORED within each band (no global-grid
    # filtering, which starved the narrow bands).  train 0-56% / val 56-74% / test 74-100%.
    h_train, h_val = int(H * 0.56), int(H * 0.74)
    row_bands = {"train": (0, h_train), "val": (h_train, h_val), "test": (h_val, H)}
    counts = {"train": 0, "val": 0, "test": 0}
    for split, (r_lo, r_hi) in row_bands.items():
        stride = 128 if split == "train" else 256
        rs = list(range(r_lo, r_hi - TILE + 1, stride))
        if (r_hi - TILE) >= r_lo and (not rs or rs[-1] != r_hi - TILE):
            rs.append(r_hi - TILE)                      # cover band bottom edge
        cs = list(range(0, W - TILE + 1, 128))
        if cs and cs[-1] != W - TILE:
            cs.append(W - TILE)
        for r in rs:
            for c in cs:
                v = valid[r:r+TILE, c:c+TILE]
                if v.mean() < 0.6:
                    continue
                np.savez_compressed(os.path.join(CD, split, f"t_{r}_{c}.npz"),
                                    image=image[:, r:r+TILE, c:c+TILE].astype(np.float32),
                                    mask=label[r:r+TILE, c:c+TILE].astype(np.uint8),
                                    valid=v.astype(np.uint8))
                counts[split] += 1

    stats = {
        "otsu_threshold": thr, "burn_px": int(label.sum()),
        "burn_area_ha": float(label.sum() * PX_HA),
        "burn_pct_valid": float(100 * label.sum() / valid.sum()),
        "crop_shape": [int(H), int(W)], "in_channels": 13, "tiles": counts,
        "dnbr_mean": float(dnbr[valid].mean()), "rbr_mean": float(rbr[valid].mean()),
    }
    json.dump(stats, open(os.path.join(CD, "dataset_stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
