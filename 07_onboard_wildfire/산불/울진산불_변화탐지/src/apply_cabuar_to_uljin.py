"""Apply the CaBuAr REAL-label-trained model to Uljin (transfer) and validate vs FIRMS.
Non-circular Uljin burn map (model learned from real California burn masks).

Usage: python apply_cabuar_to_uljin.py [full|phys] [FIRMS_MAP_KEY]
  phys = scale-invariant physical-index model (domain-consistent 7ch)
"""
import os, sys, json
import numpy as np, torch, cv2
from osgeo import gdal, osr
gdal.UseExceptions()
sys.path.insert(0, os.environ.get("WILDFIRE_SEG_ROOT", os.path.expanduser("~/wildfire-seg")))
from src.model import UNetResNet50

_args = sys.argv[1:]
MODE = _args.pop(0) if (_args and _args[0] in ("full", "phys")) else "full"
KEY = _args[0].strip() if _args else None
CH = [3, 4, 5, 9, 10, 11, 12] if MODE == "phys" else list(range(13))
TAG = "cabuar_phys" if MODE == "phys" else "cabuar_cd"

HERE = os.path.dirname(__file__)
CD = os.path.join(HERE, "..", "data", "cd")
OUT = os.path.join(HERE, "..", "data", "outputs")
FIGS = os.path.join(HERE, "..", "figs")
CKPT = os.path.join(HERE, "..", "runs", TAG, "best.pt")
RES = os.path.join(HERE, "..", "runs", TAG, "results.json")
BURN_TIF = os.path.join(OUT, f"burn_{TAG}.tif")
TILE, STRIDE = 256, 128


def cos_win(n=TILE):
    w = np.hanning(n); return np.clip(np.outer(w, w), 1e-3, 1).astype(np.float32)


@torch.no_grad()
def infer(model, image, valid, dev):
    C, H, W = image.shape; prob = np.zeros((H, W), np.float32); ws = np.zeros((H, W), np.float32); win = cos_win()
    xs = sorted(set(list(range(0, W-TILE+1, STRIDE)) + [W-TILE]))
    ys = sorted(set(list(range(0, H-TILE+1, STRIDE)) + [H-TILE]))
    for r in ys:
        for c in xs:
            t = torch.from_numpy(np.ascontiguousarray(image[:, r:r+TILE, c:c+TILE])[None]).to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=(dev.type == "cuda")):
                p = torch.sigmoid(model(t).float())[0, 0].cpu().numpy()
            prob[r:r+TILE, c:c+TILE] += p*win; ws[r:r+TILE, c:c+TILE] += win
    prob /= np.maximum(ws, 1e-6); prob[~valid] = 0; return prob


def clean(m):
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)); k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k3); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8); keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 13: keep[lab == i] = 1
    return keep


def main():
    d = np.load(os.path.join(CD, "uljin_full.npz"), allow_pickle=True)
    image = d["image"][CH].astype(np.float32); valid = d["valid"].astype(bool); dnbr = d["dnbr"]
    gt = tuple(d["geotransform"].tolist()); proj = str(d["projection"])
    thr = json.load(open(RES)).get("threshold", 0.5)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet50(classes=1, pretrained=False, in_channels=len(CH), attention=True).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev)); model.eval()

    prob = infer(model, image, valid, dev)
    burn = clean((prob >= thr) & valid & (dnbr >= 0.05))
    px_ha = 0.04
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(BURN_TIF, burn.shape[1], burn.shape[0], 1, gdal.GDT_Byte,
                    ["COMPRESS=DEFLATE"]); ds.SetGeoTransform(gt); ds.SetProjection(proj)
    ds.GetRasterBand(1).WriteArray(burn); ds = None

    # compare vs pseudo-label-trained final
    pl = gdal.Open(os.path.join(OUT, "burn_final.tif")).ReadAsArray() > 0
    inter = int((burn.astype(bool) & pl).sum()); uni = int((burn.astype(bool) | pl).sum())
    out = {"mode": MODE, "threshold": thr, "cabuar_burn_ha": round(int(burn.sum())*px_ha, 0),
           "pseudolabel_burn_ha": round(int(pl.sum())*px_ha, 0),
           "iou_cabuar_vs_pseudolabel": round(inter/max(uni, 1), 3)}

    # optional FIRMS containment
    if KEY:
        import io, csv, requests
        key = KEY; pts = []
        for src in ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP"]:
            for st in ["2022-03-04", "2022-03-09"]:
                r = requests.get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/129.13,36.85,129.57,37.22/5/{st}", timeout=120)
                for row in csv.DictReader(io.StringIO(r.text)):
                    try: pts.append((float(row["latitude"]), float(row["longitude"])))
                    except: pass
        dst = osr.SpatialReference(); dst.ImportFromWkt(proj)
        s = osr.SpatialReference(); s.ImportFromEPSG(4326); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(s, dst); H, W = burn.shape; ins = tot = 0; b = 9
        for lat, lon in pts:
            ux, uy, _ = ct.TransformPoint(lon, lat); px = int((ux-gt[0])/gt[1]); py = int((uy-gt[3])/gt[5])
            if not (0 <= px < W and 0 <= py < H): continue
            tot += 1
            if burn[max(0, py-b):py+b+1, max(0, px-b):px+b+1].any(): ins += 1
        out["firms_containment_pct"] = round(100*ins/max(tot, 1), 1); out["firms_points"] = tot

    json.dump(out, open(os.path.join(HERE, "..", "runs", TAG, "uljin_apply.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
