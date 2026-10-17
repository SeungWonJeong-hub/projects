"""Independent validation against NASA FIRMS VIIRS active-fire detections.

FIRMS thermal active-fire points (375 m, independent satellite + algorithm) for the
Uljin fire window are pulled from the FIRMS Area API and checked against our U-Net
burn mask: how many independent fire detections fall inside our detected burn.

Usage:  python validate_firms.py <MAP_KEY>
Get a free MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/map_key
"""
import os, sys, io, csv, json
import numpy as np
import cv2
import requests
from osgeo import gdal, osr
gdal.UseExceptions()

HERE = os.path.dirname(__file__)
OUTTIF = os.path.join(HERE, "..", "data", "outputs")
CD = os.path.join(HERE, "..", "data", "cd")
FIGS = os.path.join(HERE, "..", "figs")
RUN = os.path.join(HERE, "..", "runs", "cd_unet")

AOI = "129.13,36.85,129.57,37.22"           # west,south,east,north
# FIRMS Area API archive allows <=5 days/request -> two windows cover 03-04..03-13
WINDOWS = [("2022-03-04", 5), ("2022-03-09", 5)]
SOURCES = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP"]   # standard-processing archive


def fetch(map_key):
    pts = []
    for src in SOURCES:
        n = 0
        for start, days in WINDOWS:
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{src}/{AOI}/{days}/{start}"
            try:
                r = requests.get(url, timeout=120)
            except Exception as e:
                print(f"  {src} {start}: request error {e}"); continue
            if r.status_code != 200 or r.text.lstrip().lower().startswith("invalid"):
                print(f"  {src} {start}: HTTP {r.status_code} / {r.text[:80]!r}"); continue
            for row in csv.DictReader(io.StringIO(r.text)):
                try:
                    pts.append((float(row["latitude"]), float(row["longitude"]),
                                src, row.get("acq_date", ""), row.get("frp", "")))
                    n += 1
                except (KeyError, ValueError):
                    pass
        print(f"  {src}: {n} detections")
    return pts


def main():
    if len(sys.argv) < 2:
        print("usage: python validate_firms.py <MAP_KEY>"); return
    key = sys.argv[1].strip()
    pts = fetch(key)
    print(f"total FIRMS active-fire points: {len(pts)}")
    if not pts:
        print("no points (bad key / no data) — aborting."); return

    burn_ds = gdal.Open(os.path.join(OUTTIF, "burn_final.tif"))
    burn = burn_ds.ReadAsArray() > 0
    gt = burn_ds.GetGeoTransform(); proj = burn_ds.GetProjection()
    H, W = burn.shape

    # lon/lat (EPSG:4326) -> raster UTM pixel
    dst = osr.SpatialReference(); dst.ImportFromWkt(proj)
    src = osr.SpatialReference(); src.ImportFromEPSG(4326)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)

    # reference masks: our U-Net burn, and the dNBR fire-affected areas (context)
    dd = np.load(os.path.join(CD, "uljin_full.npz"), allow_pickle=True)
    dnbr = dd["dnbr"]; valid = dd["valid"].astype(bool)
    masks = {"unet_burn": burn, "dnbr_0.27": (dnbr >= 0.27) & valid,
             "dnbr_0.10": (dnbr >= 0.10) & valid}
    counts = {k: 0 for k in masks}
    total = 0; px_list = []
    buf = 9  # ~180 m tolerance (VIIRS pixel is 375 m)
    for lat, lon, srcname, date, frp in pts:
        ux, uy, _ = ct.TransformPoint(lon, lat)
        px = int((ux - gt[0]) / gt[1]); py = int((uy - gt[3]) / gt[5])
        if not (0 <= px < W and 0 <= py < H):
            continue
        total += 1; px_list.append((px, py))
        y0, y1 = max(0, py-buf), min(H, py+buf+1)
        x0, x1 = max(0, px-buf), min(W, px+buf+1)
        for k, m in masks.items():
            if m[y0:y1, x0:x1].any():
                counts[k] += 1
    inside = counts["unet_burn"]
    pct = 100 * inside / max(total, 1)
    containment = {k: round(100*counts[k]/max(total, 1), 1) for k in masks}

    # figure: burn overlay (post) + FIRMS points
    d = np.load(os.path.join(CD, "uljin_full.npz"), allow_pickle=True); img = d["image"]
    base = cv2.merge([(img[8]*255).astype(np.uint8), (img[7]*255).astype(np.uint8), (img[6]*255).astype(np.uint8)])
    base = (base*0.6).astype(np.uint8)
    base[burn] = (0, 90, 255)                     # our burn = orange
    for px, py in px_list:
        cv2.circle(base, (px, py), 5, (255, 255, 255), -1)
        cv2.circle(base, (px, py), 5, (0, 0, 0), 1)   # FIRMS point = white dot
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    bar = np.full((46, base.shape[1], 3), 25, np.uint8)
    cv2.putText(bar, f"FIRMS validation: {inside}/{total} active-fire pts inside our burn ({pct:.0f}%)",
                (12, 32), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    out = np.vstack([bar, base])
    for i, (lab, col) in enumerate([("Our detected burn", (0, 90, 255)), ("FIRMS active fire", (255, 255, 255))]):
        y = out.shape[0]-60+i*28
        cv2.rectangle(out, (16, y-16), (44, y+6), col, -1)
        cv2.putText(out, lab, (52, y+2), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(FIGS, "firms_validation.png"), out)

    res = {"firms_points_total": total, "inside_unet_burn": inside, "containment_pct": pct,
           "containment_by_ref_pct": containment, "tolerance_m": buf*20,
           "windows": WINDOWS, "sources": SOURCES}
    json.dump(res, open(os.path.join(RUN, "firms_validation.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("figure -> figs/firms_validation.png")


if __name__ == "__main__":
    main()
