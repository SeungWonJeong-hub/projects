"""Step 3 — build the change-detection tensors for the Uljin fire.

For pre and post scenes we build a 6-channel stack
    [fR=B12, fG=B08, fB=B02, NBR, NDVI, NDWI]
then dNBR = NBR_pre - NBR_post (standard burn-severity differencing; positive = burned).

Label = dNBR pseudo-label (USGS thresholds).  Valid mask excludes nodata + cloud +
cloud-shadow + snow in EITHER date (SCL).  Everything is cropped to the common valid
bbox, saved as (a) georeferenced GeoTIFFs for QGIS, (b) a full-scene npz, (c) 256x256
tiles for training, and (d) a preview PNG for eyeballing.

Sentinel-2 20 m bands loaded: B02,B03,B04,B08,B11,B12 (uint16 DN) + SCL.
Indices use raw-DN ratios; a constant reflectance offset largely cancels in dNBR.
"""
import os
import sys
import json
import numpy as np
import cv2
from osgeo import gdal
gdal.UseExceptions()

HERE = os.path.dirname(__file__)
PAIR = os.path.join(HERE, "..", "data", "scene_pair.json")
CD = os.path.join(HERE, "..", "data", "cd")
TILES = os.path.join(CD, "tiles")
OUTTIF = os.path.join(HERE, "..", "data", "outputs")
FIGS = os.path.join(HERE, "..", "figs")
for d in (CD, TILES, OUTTIF, FIGS):
    os.makedirs(d, exist_ok=True)

# SCL classes to DROP (invalid): 0 nodata,1 saturated,3 cloud-shadow,8/9 cloud,10 cirrus,11 snow
SCL_DROP = {0, 1, 3, 8, 9, 10, 11}
# USGS dNBR burn-severity breakpoints
SEV_BREAKS = [0.10, 0.27, 0.44, 0.66]   # unburned/low/mod-low/mod-high/high
BURN_THR = 0.27                          # binary pseudo-label threshold (>= moderate-low)
TILE = 256
STRIDE = 256


def load_scene(scene_dir):
    bands = {}
    for b in ("B02", "B03", "B04", "B08", "B11", "B12", "SCL"):
        ds = gdal.Open(os.path.join(scene_dir, f"{b}.tif"))
        bands[b] = ds.ReadAsArray().astype(np.float32)
        if b == "B02":
            gt, proj = ds.GetGeoTransform(), ds.GetProjection()
    return bands, gt, proj


def ratio(a, b, eps=1e-6):
    return (a - b) / (a + b + eps)


def stretch(chan, valid, lo=2, hi=98):
    v = chan[valid]
    if v.size == 0:
        return np.zeros_like(chan, np.uint8)
    a, b = np.percentile(v, lo), np.percentile(v, hi)
    if b <= a:
        b = a + 1
    return (np.clip((chan - a) / (b - a), 0, 1) * 255).astype(np.uint8)


def build_stack(bd, valid):
    nbr = ratio(bd["B08"], bd["B12"])
    ndvi = ratio(bd["B08"], bd["B04"])
    ndwi = ratio(bd["B03"], bd["B08"])
    fr = stretch(bd["B12"], valid); fg = stretch(bd["B08"], valid); fb = stretch(bd["B02"], valid)
    idx8 = lambda x: (np.clip((x + 1) / 2, 0, 1) * 255).astype(np.uint8)
    stack = np.stack([fr, fg, fb, idx8(nbr), idx8(ndvi), idx8(ndwi)], 0)  # (6,H,W) uint8
    return stack, nbr


def write_gtiff(path, arr, gt, proj, dtype=gdal.GDT_Float32, nodata=None):
    drv = gdal.GetDriverByName("GTiff")
    h, w = arr.shape
    ds = drv.Create(path, w, h, 1, dtype, options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(gt); ds.SetProjection(proj)
    ds.GetRasterBand(1).WriteArray(arr)
    if nodata is not None:
        ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds = None


def main():
    pair = json.load(open(PAIR))
    pre_bd, gt, proj = load_scene(os.path.expanduser(pair["pre"]["dir"]))
    post_bd, _, _ = load_scene(os.path.expanduser(pair["post"]["dir"]))

    # valid = data present + not cloud/shadow/snow in BOTH dates
    def scene_valid(bd):
        data = (bd["B02"] > 0) & (bd["B08"] > 0) & (bd["B12"] > 0)
        sclok = ~np.isin(bd["SCL"].astype(np.int16), list(SCL_DROP))
        return data & sclok
    valid = scene_valid(pre_bd) & scene_valid(post_bd)

    pre6, nbr_pre = build_stack(pre_bd, valid)
    post6, nbr_post = build_stack(post_bd, valid)
    dnbr = (nbr_pre - nbr_post).astype(np.float32)
    dnbr[~valid] = 0

    # pseudo-label + severity
    label = ((dnbr >= BURN_THR) & valid).astype(np.uint8)
    sev = np.zeros_like(dnbr, np.uint8)
    for i, t in enumerate(SEV_BREAKS, start=1):
        sev[dnbr >= t] = i
    sev[~valid] = 255

    # crop to common valid bbox
    ys, xs = np.where(valid)
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = lambda a: a[..., r0:r1, c0:c1]
    pre6, post6, dnbr, label, sev, valid = map(crop, (pre6, post6, dnbr, label, sev, valid))
    gt_c = (gt[0] + c0 * gt[1], gt[1], 0, gt[3] + r0 * gt[5], 0, gt[5])
    H, W = dnbr.shape

    # (a) georeferenced GeoTIFFs for QGIS
    write_gtiff(os.path.join(OUTTIF, "dnbr.tif"), dnbr, gt_c, proj, nodata=0)
    write_gtiff(os.path.join(OUTTIF, "pseudolabel.tif"), label.astype(np.float32), gt_c, proj, nodata=0)
    write_gtiff(os.path.join(OUTTIF, "severity.tif"), sev.astype(np.float32), gt_c, proj, nodata=255)

    # (b) full-scene npz
    np.savez_compressed(os.path.join(CD, "uljin_prepost.npz"),
                        pre=pre6, post=post6, dnbr=dnbr, mask=label, valid=valid.astype(np.uint8),
                        geotransform=np.array(gt_c), projection=proj)

    # (c) 256x256 tiles (keep tiles with >=60% valid)
    ntile = kept = 0
    for r in range(0, H - TILE + 1, STRIDE):
        for c in range(0, W - TILE + 1, STRIDE):
            ntile += 1
            v = valid[r:r+TILE, c:c+TILE]
            if v.mean() < 0.60:
                continue
            np.savez_compressed(os.path.join(TILES, f"tile_{r}_{c}.npz"),
                                pre=pre6[:, r:r+TILE, c:c+TILE],
                                post=post6[:, r:r+TILE, c:c+TILE],
                                dnbr=dnbr[r:r+TILE, c:c+TILE],
                                mask=label[r:r+TILE, c:c+TILE],
                                valid=v.astype(np.uint8))
            kept += 1

    # (d) preview PNG: pre-fc | post-fc | dNBR | pseudolabel-overlay
    def fc(stack):
        return cv2.merge([stack[2], stack[1], stack[0]])  # BGR for cv2 (fB,fG,fR)
    dn_vis = np.clip((dnbr + 0.5), 0, 1)
    dn_col = cv2.applyColorMap((dn_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    dn_col[~valid] = 0
    overlay = fc(post6).copy()
    overlay[label == 1] = (0, 0, 255)  # red burn on post
    strip = np.hstack([fc(pre6), fc(post6), dn_col, overlay])
    cv2.imwrite(os.path.join(FIGS, "preview_preprocess.png"), strip)

    # stats
    burn_pct = 100 * label.sum() / max(valid.sum(), 1)
    sev_counts = {int(k): int((sev == k).sum()) for k in range(5)}
    px_ha = (20 * 20) / 1e4
    stats = {
        "pre": pair["pre"]["id"], "post": pair["post"]["id"],
        "crop_shape": [int(H), int(W)], "valid_pct": float(100 * valid.mean()),
        "burn_pixels": int(label.sum()), "burn_pct_of_valid": float(burn_pct),
        "burn_area_ha": float(label.sum() * px_ha),
        "dnbr_min": float(dnbr[valid].min()), "dnbr_max": float(dnbr[valid].max()),
        "dnbr_mean": float(dnbr[valid].mean()),
        "severity_pixel_counts_0to4": sev_counts,
        "tiles_total": ntile, "tiles_kept": kept,
    }
    json.dump(stats, open(os.path.join(CD, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nGeoTIFFs -> {os.path.abspath(OUTTIF)}")
    print(f"npz+tiles -> {os.path.abspath(CD)}  (kept {kept}/{ntile} tiles)")
    print(f"preview  -> {os.path.abspath(os.path.join(FIGS, 'preview_preprocess.png'))}")


if __name__ == "__main__":
    main()
