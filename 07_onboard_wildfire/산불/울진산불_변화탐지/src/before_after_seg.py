"""Same semantic-segmentation scheme applied to EACH date, compared side by side.

Only the acquisition time differs (2021-04-19 pre vs 2022-04-09 post); the per-pixel
land-cover classification is identical, so the wildfire shows up as a class change
(healthy vegetation -> bare/burned) when you compare the two panels.

Per-pixel classes (from spectral indices, applied identically to both dates):
    Burned scar      NBR < -0.10           (charred surface: low NIR, high SWIR)
    Water            NDWI > 0.10 & low SWIR (SWIR test excludes burned scars)
    Bare / soil      NDVI < 0.25 (not burned)
    Sparse veg       0.25 <= NDVI < 0.55
    Healthy veg      NDVI >= 0.55
The fire appears as green (before) -> red burned scar (after).
"""
import os
import numpy as np
import cv2

HERE = os.path.dirname(__file__)
CD = os.path.join(HERE, "..", "data", "cd")
FIGS = os.path.join(HERE, "..", "figs")

d = np.load(os.path.join(CD, "uljin_full.npz"), allow_pickle=True)
img = d["image"]; valid = d["valid"].astype(bool)
# channels per date: [B12,B08,B02, NBR, NDVI, NDWI] normalised to [0,1] (0.5 == index 0)
def idx(ch):  # back to [-1,1]
    return ch * 2 - 1

def classify(base):     # base = 6-channel stack for one date
    ndvi, nbr = idx(base[4]), idx(base[3])
    nir = base[1]                               # stretched B08 [0,1]: water darkest, veg brightest
    cls = np.zeros(ndvi.shape, np.uint8)        # 0 = unclassified/invalid
    cls[ndvi >= 0.55] = 5                        # healthy veg
    cls[(ndvi >= 0.25) & (ndvi < 0.55)] = 4      # sparse veg
    cls[ndvi < 0.25] = 3                         # bare / soil
    cls[nbr < -0.10] = 2                         # burned scar (charred: overrides veg/bare)
    cls[nir < 0.12] = 1                          # water (sea is darkest in NIR; overrides all)
    cls[~valid] = 0
    return cls

PAL = {1: (150, 90, 30), 2: (0, 0, 210), 3: (120, 190, 210), 4: (120, 210, 120), 5: (40, 120, 40)}  # BGR
GRAY = (150, 150, 150)   # cls 0 = invalid (cloud / cloud-shadow / cirrus / nodata, via SCL)
LABELS = [("Healthy vegetation", PAL[5]), ("Sparse vegetation", PAL[4]),
          ("Bare / soil", PAL[3]), ("Burned scar", PAL[2]), ("Water", PAL[1]),
          ("Cloud / invalid", GRAY)]

def colorize(cls):
    out = np.full((*cls.shape, 3), GRAY, np.uint8)   # invalid(0) -> gray by default
    for k, c in PAL.items():
        out[cls == k] = c
    return out

pre_cls = classify(img[0:6])
post_cls = classify(img[6:12])
pre_col, post_col = colorize(pre_cls), colorize(post_cls)

FONT = cv2.FONT_HERSHEY_SIMPLEX
def panel(im, title):
    bar = np.full((46, im.shape[1], 3), 25, np.uint8)
    cv2.putText(bar, title, (14, 32), FONT, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    im = np.vstack([bar, im])
    # legend bottom-left
    x, y = 16, im.shape[0] - 20 - 34*len(LABELS)
    cv2.rectangle(im, (x-10, y-24), (x-10+330, y-24+34*len(LABELS)+14), (20, 20, 20), -1)
    cv2.rectangle(im, (x-10, y-24), (x-10+330, y-24+34*len(LABELS)+14), (255, 255, 255), 1)
    for lab, col in LABELS:
        cv2.rectangle(im, (x, y-18), (x+30, y+6), col, -1)
        cv2.rectangle(im, (x, y-18), (x+30, y+6), (255, 255, 255), 1)
        cv2.putText(im, lab, (x+40, y+2), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 34
    return im

sbs = np.hstack([panel(pre_col, "BEFORE 2021-04-19  (semantic segmentation)"),
                 panel(post_col, "AFTER 2022-04-09  (semantic segmentation)")])
out = os.path.join(FIGS, "before_after_segmentation.png")
cv2.imwrite(out, sbs)

# how much vegetation turned into burned scar
veg_pre = (pre_cls == 5) | (pre_cls == 4)
burned_post = (post_cls == 2)
changed = veg_pre & burned_post & valid
names = {1: "water", 2: "burned", 3: "bare", 4: "sparse", 5: "healthy"}
print("before_after_segmentation ->", out)
print("vegetation -> burned scar:", round(int(changed.sum())*0.04, 0), "ha")
for nm, cls in (("BEFORE", pre_cls), ("AFTER", post_cls)):
    print(nm, {names[k]: round(int((cls == k).sum())*0.04, 0) for k in (1, 2, 3, 4, 5)}, "ha")
