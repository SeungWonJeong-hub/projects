"""Single color-coded CHANGE MAP: where + how-severely each area changed.

U-Net segmentation says WHERE it burned; dNBR severity says HOW severely.  We paint
the burned area by severity class (green->yellow->orange->red->darkred) over the
post-fire image, with a large legend.  This is the 'how did it change, in colour' view.
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
img = d["image"]
post = cv2.merge([(img[8]*255).astype(np.uint8), (img[7]*255).astype(np.uint8), (img[6]*255).astype(np.uint8)])  # BGR
sev = gdal.Open(os.path.join(OUTTIF, "severity_pred.tif")).ReadAsArray().astype(np.int16)

# severity palette (BGR): 1 low, 2 mod-low, 3 mod-high, 4 high
PAL = {1: (80, 240, 255), 2: (0, 165, 255), 3: (0, 70, 255), 4: (0, 0, 150)}
LABELS = [("Unburned / no change", (60, 90, 60)),
          ("Low severity  (dNBR 0.10-0.27)", PAL[1]),
          ("Moderate-low  (0.27-0.44)", PAL[2]),
          ("Moderate-high (0.44-0.66)", PAL[3]),
          ("High severity (> 0.66)", PAL[4])]

# dim the base so coloured burn pops
base = (post * 0.55).astype(np.uint8)
overlay = base.copy()
for k, col in PAL.items():
    overlay[sev == k] = col
blend = cv2.addWeighted(base, 0.35, overlay, 0.65, 0)
blend[sev == 0] = base[sev == 0]   # keep unburned as dimmed terrain

# legend box (top-left)
FONT = cv2.FONT_HERSHEY_SIMPLEX
x0, y0, dy = 20, 30, 46
w_box = 560
cv2.rectangle(blend, (x0-12, y0-24), (x0-12+w_box, y0-24+dy*len(LABELS)+16), (20, 20, 20), -1)
cv2.rectangle(blend, (x0-12, y0-24), (x0-12+w_box, y0-24+dy*len(LABELS)+16), (255, 255, 255), 1)
cv2.putText(blend, "URUJIN FIRE  change (how it changed)", (x0, y0+4), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
y = y0 + 44
for label, col in LABELS:
    cv2.rectangle(blend, (x0, y-20), (x0+34, y+6), col, -1)
    cv2.rectangle(blend, (x0, y-20), (x0+34, y+6), (255, 255, 255), 1)
    cv2.putText(blend, label, (x0+46, y+2), FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    y += dy

out = os.path.join(FIGS, "change_map.png")
cv2.imwrite(out, blend)
# stats
px_ha = 0.04
tot = int((sev >= 1).sum()) * px_ha
print("change map ->", out)
print("burned total ha:", round(tot, 0),
      "| by class ha:", {k: round(int((sev == k).sum())*px_ha, 0) for k in (1, 2, 3, 4)})
