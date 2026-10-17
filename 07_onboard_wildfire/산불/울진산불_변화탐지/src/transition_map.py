"""Semantic 'from-to' transition map (SCD style): what changed INTO what.

Same per-date semantic classes as before_after_seg.py, then per pixel we label the
land-cover TRANSITION (pre-class -> post-class) into fire-relevant categories:

    Veg -> Burned scar      (severe: forest charred)            red
    Healthy -> Sparse veg   (canopy scorch / partial damage)    orange
    Veg -> Bare soil        (vegetation lost to soil)           yellow
    No significant change   (dimmed post terrain shown)         --
    Cloud / invalid                                             gray

Outputs a colour map (over dimmed post image) + a georeferenced code raster.
"""
import os
import numpy as np
import cv2
from osgeo import gdal
gdal.UseExceptions()

HERE = os.path.dirname(__file__)
CD = os.path.join(HERE, "..", "data", "cd")
OUTTIF = os.path.join(HERE, "..", "data", "outputs")
FIGS = os.path.join(HERE, "..", "figs")

d = np.load(os.path.join(CD, "uljin_full.npz"), allow_pickle=True)
img = d["image"]; valid = d["valid"].astype(bool)
gt = tuple(d["geotransform"].tolist()); proj = str(d["projection"])


def idx(ch): return ch * 2 - 1

def classify(base):
    ndvi, nbr, nir = idx(base[4]), idx(base[3]), base[1]
    cls = np.zeros(ndvi.shape, np.uint8)
    cls[ndvi >= 0.55] = 5      # healthy veg
    cls[(ndvi >= 0.25) & (ndvi < 0.55)] = 4   # sparse veg
    cls[ndvi < 0.25] = 3       # bare/soil
    cls[nbr < -0.10] = 2       # burned scar
    cls[nir < 0.12] = 1        # water
    cls[~valid] = 0
    return cls

pre, post = classify(img[0:6]), classify(img[6:12])
veg_pre = np.isin(pre, [4, 5])
# Gate transitions by the U-Net DETECTED burn mask so a 10-day phenology mismatch
# (pre later in spring) is not mislabelled as fire damage: transitions are only
# classified INSIDE the semantically-segmented burn area. This ties the from-to map
# to our segmentation output and removes the seasonal artefact.
fire = gdal.Open(os.path.join(OUTTIF, "burn_final.tif")).ReadAsArray() > 0

# transition codes (only inside the detected burn)
trans = np.zeros(pre.shape, np.uint8)                    # 0 = no significant change
trans[(pre == 5) & (post == 4) & fire] = 2              # healthy -> sparse (scorch)
trans[veg_pre & (post == 3) & fire] = 3                 # veg -> bare
trans[veg_pre & (post == 2) & fire] = 1                 # veg -> burned scar (override)
trans[~valid] = 4                                        # cloud/invalid

# georeferenced code raster
drv = gdal.GetDriverByName("GTiff")
ds = drv.Create(os.path.join(OUTTIF, "transition.tif"), trans.shape[1], trans.shape[0], 1,
                gdal.GDT_Byte, options=["COMPRESS=DEFLATE", "TILED=YES"])
ds.SetGeoTransform(gt); ds.SetProjection(proj); ds.GetRasterBand(1).WriteArray(trans); ds = None

# colour over dimmed post false-colour
post_fc = cv2.merge([(img[8]*255).astype(np.uint8), (img[7]*255).astype(np.uint8), (img[6]*255).astype(np.uint8)])
canvas = (post_fc * 0.45).astype(np.uint8)
COL = {1: (0, 0, 220), 2: (0, 140, 255), 3: (0, 220, 220), 4: (150, 150, 150)}
for k, c in COL.items():
    canvas[trans == k] = c

FONT = cv2.FONT_HERSHEY_SIMPLEX
bar = np.full((48, canvas.shape[1], 3), 25, np.uint8)
cv2.putText(bar, "URUJIN 2021-04-19 -> 2022-04-09  land-cover transition (from -> to)",
            (14, 33), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
canvas = np.vstack([bar, canvas])
LAB = [("Veg -> Burned scar", COL[1]), ("Healthy -> Sparse veg (scorch)", COL[2]),
       ("Veg -> Bare soil", COL[3]), ("No significant change", (90, 90, 90)),
       ("Cloud / invalid", COL[4])]
x, y = 18, canvas.shape[0] - 20 - 36*len(LAB)
cv2.rectangle(canvas, (x-10, y-26), (x-10+430, y-26+36*len(LAB)+14), (20, 20, 20), -1)
cv2.rectangle(canvas, (x-10, y-26), (x-10+430, y-26+36*len(LAB)+14), (255, 255, 255), 1)
for lab, c in LAB:
    cv2.rectangle(canvas, (x, y-20), (x+32, y+6), c, -1)
    cv2.rectangle(canvas, (x, y-20), (x+32, y+6), (255, 255, 255), 1)
    cv2.putText(canvas, lab, (x+44, y+2), FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    y += 36
cv2.imwrite(os.path.join(FIGS, "transition_map.png"), canvas)

px_ha = 0.04
areas = {1: "Veg->Burned", 2: "Healthy->Sparse", 3: "Veg->Bare"}
print("transition_map -> figs/transition_map.png  + data/outputs/transition.tif")
for k, nm in areas.items():
    print(f"  {nm:16}: {int((trans==k).sum())*px_ha:8.0f} ha")
print(f"  total changed   : {int(np.isin(trans,[1,2,3]).sum())*px_ha:8.0f} ha")
