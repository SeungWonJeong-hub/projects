"""Step 6 — full-scene inference + post-processing -> final Uljin burn products.

  1. Sliding-window (256, stride 128, cosine-blended) inference over the whole scene.
  2. Threshold (val-optimal from training) -> raw prob mask.
  3. Physical dNBR fusion: keep predictions only where dNBR >= 0.05 (removes
     non-physical false positives; burn must show a real NBR drop).
  4. Morphological open+close + min-area filter (drop < 0.5 ha components).
  5. Severity map (USGS dNBR classes) inside the burn.
  6. Vectorize to GeoJSON polygons (GDAL Polygonize).
  7. Metrics vs the dNBR-refined pseudo-label + final 4-panel figure.
"""
import os, sys, json
import numpy as np
import cv2
import torch
from osgeo import gdal, ogr, osr
gdal.UseExceptions()

sys.path.insert(0, os.environ.get("WILDFIRE_SEG_ROOT", os.path.expanduser("~/wildfire-seg")))
from src.model import UNetResNet50

HERE = os.path.dirname(__file__)
CD = os.path.join(HERE, "..", "data", "cd")
RUN = os.path.join(HERE, "..", "runs", "cd_unet")
OUTTIF = os.path.join(HERE, "..", "data", "outputs")
FIGS = os.path.join(HERE, "..", "figs")
TILE, STRIDE = 256, 128
PX_HA = (20 * 20) / 1e4
MIN_AREA_HA = 0.5
SEV_BREAKS = [0.10, 0.27, 0.44, 0.66]


def cosine_window(n=TILE):
    w = np.hanning(n)
    return np.clip(np.outer(w, w), 1e-3, 1).astype(np.float32)


def write_gtiff(path, arr, gt, proj, dt=gdal.GDT_Float32, nodata=None):
    drv = gdal.GetDriverByName("GTiff"); h, w = arr.shape
    ds = drv.Create(path, w, h, 1, dt, options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(gt); ds.SetProjection(proj); ds.GetRasterBand(1).WriteArray(arr)
    if nodata is not None: ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds = None


def polygonize(mask, gt, proj, out_geojson):
    drv = gdal.GetDriverByName("GTiff")
    src = drv.Create("/vsimem/m.tif", mask.shape[1], mask.shape[0], 1, gdal.GDT_Byte)
    src.SetGeoTransform(gt); src.SetProjection(proj); src.GetRasterBand(1).WriteArray(mask.astype(np.uint8))
    band = src.GetRasterBand(1)
    if os.path.exists(out_geojson): os.remove(out_geojson)
    ogr_ds = ogr.GetDriverByName("GeoJSON").CreateDataSource(out_geojson)
    srs = osr.SpatialReference(); srs.ImportFromWkt(proj)
    lyr = ogr_ds.CreateLayer("burn", srs=srs, geom_type=ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("val", ogr.OFTInteger))
    gdal.Polygonize(band, band, lyr, 0, [], callback=None)  # use mask as its own validity
    n = lyr.GetFeatureCount(); ogr_ds = None; src = None
    gdal.Unlink("/vsimem/m.tif")
    return n


@torch.no_grad()
def infer_full(model, image, valid, dev):
    C, H, W = image.shape
    prob = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    win = cosine_window()
    xs = list(range(0, max(W - TILE, 0) + 1, STRIDE)) + ([W - TILE] if (W - TILE) % STRIDE else [])
    ys = list(range(0, max(H - TILE, 0) + 1, STRIDE)) + ([H - TILE] if (H - TILE) % STRIDE else [])
    for r in sorted(set(ys)):
        for c in sorted(set(xs)):
            tile = image[:, r:r+TILE, c:c+TILE]
            t = torch.from_numpy(np.ascontiguousarray(tile)[None]).to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=(dev.type == "cuda")):
                p = torch.sigmoid(model(t).float())[0, 0].cpu().numpy()
            prob[r:r+TILE, c:c+TILE] += p * win; wsum[r:r+TILE, c:c+TILE] += win
    prob /= np.maximum(wsum, 1e-6); prob[~valid] = 0
    return prob


def clean(mask):
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    min_px = int(MIN_AREA_HA / PX_HA); keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_px: keep[lab == i] = 1
    return keep


def stats(pred, gt, valid):
    p = (pred > 0) & valid; g = (gt > 0) & valid
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    eps = 1e-7; prec = tp/(tp+fp+eps); rec = tp/(tp+fn+eps)
    return {"precision": prec, "recall": rec, "f1": 2*prec*rec/(prec+rec+eps),
            "iou": tp/(tp+fp+fn+eps)}


def main():
    d = np.load(os.path.join(CD, "uljin_full.npz"), allow_pickle=True)
    image = d["image"].astype(np.float32); label = d["mask"]; valid = d["valid"].astype(bool)
    dnbr = d["dnbr"]; gt = tuple(d["geotransform"].tolist()); proj = str(d["projection"])
    res = json.load(open(os.path.join(RUN, "results.json")))
    thr = res.get("threshold", 0.5)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet50(classes=1, pretrained=False, in_channels=13, attention=True).to(dev)
    model.load_state_dict(torch.load(os.path.join(RUN, "best.pt"), map_location=dev)); model.eval()

    prob = infer_full(model, image, valid, dev)
    raw = (prob >= thr) & valid
    fused = raw & (dnbr >= 0.05)          # physical dNBR gate
    final = clean(fused)

    sev = np.zeros_like(dnbr, np.uint8)
    for i, t in enumerate(SEV_BREAKS, 1): sev[(dnbr >= t) & (final > 0)] = i
    sev[~valid] = 255

    write_gtiff(os.path.join(OUTTIF, "pred_prob.tif"), prob.astype(np.float32), gt, proj, nodata=0)
    write_gtiff(os.path.join(OUTTIF, "burn_final.tif"), final.astype(np.float32), gt, proj, nodata=0)
    write_gtiff(os.path.join(OUTTIF, "severity_pred.tif"), sev.astype(np.float32), gt, proj, nodata=255)
    npoly = polygonize(final, gt, proj, os.path.join(OUTTIF, "burn_final.geojson"))

    m_pred = stats(final, label, valid)
    m_raw = stats(raw, label, valid)
    out = {
        "threshold": thr, "polygons": npoly,
        "pred_area_ha": float(final.sum() * PX_HA),
        "label_area_ha": float(label.sum() * PX_HA),
        "metrics_final_vs_label": m_pred,
        "metrics_raw_vs_label": m_raw,
        "train_test_f1": res.get("test", {}).get("f1"),
        "severity_ha": {int(k): float((sev == k).sum() * PX_HA) for k in range(5)},
    }
    json.dump(out, open(os.path.join(RUN, "postprocess.json"), "w"), indent=2)

    # final 4-panel figure with headers + legends
    def fc():
        pr = image[6:9]  # post fR,fG,fB (post6[0:3] = channels 6,7,8)
        return cv2.merge([(pr[2]*255).astype(np.uint8), (pr[1]*255).astype(np.uint8), (pr[0]*255).astype(np.uint8)])
    base = fc()
    prob_c = cv2.applyColorMap((np.clip(prob, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO); prob_c[~valid] = 0
    ov = base.copy(); ov[final > 0] = (0, 0, 255)
    palette = {1: (0, 255, 255), 2: (0, 165, 255), 3: (0, 80, 255), 4: (0, 0, 200)}  # BGR: low->high
    sev_col = np.zeros((*sev.shape, 3), np.uint8)
    for k, col in palette.items(): sev_col[sev == k] = col

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    def header(img, title):
        h, w = img.shape[:2]
        bar = np.full((44, w, 3), 30, np.uint8)
        cv2.putText(bar, title, (12, 30), FONT, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([bar, img])

    def swatch(img, items, x=12, y0=None):
        """draw legend swatches (label, BGR) at bottom-left."""
        h, w = img.shape[:2]; y = (h - 20 - 30*len(items)) if y0 is None else y0
        for label, col in items:
            cv2.rectangle(img, (x, y), (x+26, y+22), col, -1)
            cv2.rectangle(img, (x, y), (x+26, y+22), (255, 255, 255), 1)
            cv2.putText(img, label, (x+34, y+19), FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            y += 30
        return img

    ov = swatch(ov, [("Burn (predicted)", (0, 0, 255))])
    sev_col = swatch(sev_col, [("Low (dNBR>0.10)", palette[1]), ("Mod-low (>0.27)", palette[2]),
                               ("Mod-high (>0.44)", palette[3]), ("High (>0.66)", palette[4])])
    # probability colorbar legend (dark=0 -> bright=1)
    cbar_lbl = np.zeros((prob_c.shape[0], 1, 3), np.uint8)
    prob_c = swatch(prob_c, [("Low P", (30, 30, 30)), ("High P", (60, 240, 255))])

    strip = np.hstack([header(base, "AFTER 2022-04-09 (post-fire)"),
                       header(prob_c, "U-Net burn probability"),
                       header(ov, "Predicted burn overlay"),
                       header(sev_col, "Burn severity (dNBR)")])
    cv2.imwrite(os.path.join(FIGS, "final_result.png"), strip)

    print(json.dumps(out, indent=2))
    print(f"\nGeoJSON polygons: {npoly} -> data/outputs/burn_final.geojson")
    print(f"figure -> figs/final_result.png")


if __name__ == "__main__":
    main()
