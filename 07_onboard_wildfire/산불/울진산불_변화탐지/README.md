# Uljin Wildfire Change Detection (20 m GSD, Sentinel-2)

Bi-temporal semantic-segmentation change detection of the **2022 Uljin–Samcheok
wildfire** (ignition 2022-03-04, contained ~03-13, ~20,900 ha), Korea's largest.

## Approach
- **Sensor / GSD:** Sentinel-2 L2A, all bands resampled to **20 m** (SWIR B11/B12 are
  native 20 m — the burn-scar–sensitive bands drive the choice).
- **Input channels (per date):** 6-band stack `[B12, B08, B02, NBR, NDVI, NDWI]`,
  early-fusion `pre|post` + `dNBR`.
- **Label:** dNBR pseudo-label (unsupervised), USGS severity breaks; binary burn at
  `dNBR ≥ 0.27` (moderate-low+).
- **AOI:** Uljin burn-scar–centred bbox `[129.13, 36.85, 129.57, 37.22]` (~40×40 km).

## Scene pair (auto-selected, lowest cloud, tile 52SEG)
| role | scene | date | cloud |
|---|---|---|---|
| **pre** (leaf-on, phenology-matched) | S2A_52SEG_20210419_2_L2A | 2021-04-19 | 0.0% |
| **post** | S2B_52SEG_20220409_0_L2A | 2022-04-09 | 0.2% |
| pre_winter (kept, unused) | S2A_52SEG_20220203_0_L2A | 2022-02-03 | 12% |

> A previous-year **leaf-on** pre is used instead of the immediate winter (2022-02) pre:
> the Feb scene was leaf-off + partly snow-covered, which depressed dNBR and undercut
> the burn area (3,278 ha). The phenology-matched 2021-04 pre recovers it to **12,712 ha**
> at dNBR≥0.27 (valid area 79.7% → 97.9%).

## Pipeline (`src/`)
1. `search_scenes.py` — STAC search pre/post windows, pick lowest-cloud pair.
2. `add_pre2021.py` — swap in the phenology-matched 2021 leaf-on pre.
3. `download_scenes.py` — pull B02/B03/B04/B08/B11/B12/SCL, clip to AOI, 20 m (GDAL /vsicurl).
4. `preprocess_cd.py` — build 6-band stacks + dNBR, dNBR pseudo-label + severity,
   valid mask (nodata/cloud/shadow/snow via SCL), crop to valid bbox, tile 256², preview.
- `stac.py` — shared STAC + COG helpers.

## Outputs
- `data/outputs/{dnbr,pseudolabel,severity}.tif` — georeferenced, open in QGIS.
- `data/cd/uljin_prepost.npz` — full-scene arrays (pre, post, dnbr, mask, valid + geo).
- `data/cd/tiles/*.npz` — 35× 256×256 training tiles (pre, post, dnbr, mask, valid).
- `data/cd/stats.json` — burn area, severity histogram, dNBR stats.
- `figs/preview_preprocess.png` — pre | post | dNBR | label-overlay.

## Result (final)
**Semantic-segmentation U-Net** (13ch pre|post|dNBR → burn), trained on the dNBR-refined
pseudo-label (Otsu 0.306 + morphology + min-area, 9,532 ha).

- **Held-out metrics** (row-block split, best-val checkpoint ep34):
  VAL **F1 0.872 / IoU 0.774 / OA 0.963 / Kappa 0.851**,
  TEST **F1 0.772 / IoU 0.629 / OA 0.988 / Kappa 0.766**.
- **Full-scene** (pred vs refined label): **F1 0.879 / IoU 0.784 / OA 0.979 / Kappa 0.868**.
  (Kappa/OA per burned-area paper convention; `src/metrics_extra.py`, `runs/cd_unet/metrics_extra.json`)
- **Detected burn: 9,949 ha, 307 polygons** (largest 3,665 ha). Severity (ha):
  low 1,360 / mod-low 4,113 / mod-high 2,572 / high 1,884.

### Independent validation — NASA FIRMS active fire (`src/validate_firms.py`)
2,342 independent VIIRS/MODIS thermal active-fire detections (2022-03-04..13, Uljin AOI)
checked against our masks (180 m tolerance):
- **99.2%** fall within dNBR≥0.10 (any fire effect), **97.6%** within dNBR≥0.27
  → the change-detection **physical basis is validated**: fire occurred where we detect spectral change.
- **66%** fall within the U-Net final mask → our cleaned map is a **conservative severe-core**
  (min-area filter drops the fragmented western low-severity spread; matches the 9,949 vs 20,923 ha gap).
Figure: `figs/firms_validation.png` · run needs a free FIRMS MAP_KEY (not committed).

### Real-label experiment — CaBuAr (California) → Uljin transfer (honest negative result)
To escape the circular dNBR pseudo-label, we trained the SAME 13-ch CD U-Net on **CaBuAr
real burn masks** (bi-temporal subset, 780/356/280 tiles; `prepare_cabuar_cd.py`,
`train_cabuar.py`) and applied it to Uljin (`apply_cabuar_to_uljin.py`).
- **Genuine accuracy** on CaBuAr real test: **F1 0.648 / IoU 0.479 / OA 0.943 / Kappa 0.62**
  (precision 0.50, recall 0.92 — over-predicts). This is the *honest* number; the 0.88 vs the
  Uljin pseudo-label was circular/inflated.
- **Transfer to Uljin FAILED**: FIRMS containment **9.3%** (vs 66% for the Uljin-tuned model),
  IoU-vs-pseudolabel 0.32, 3,196 ha mislocated (`figs/model_compare.png`). A California-trained
  model does **not generalize to the Korean fire** — the key blocker for onboard deployment.
- Likely cause: domain gap + normalization mismatch (CaBuAr uint8 per-patch vs Uljin uint16
  per-scene percentile stretch). Fix = physically-consistent reflectance normalization +
  multi-region training / domain adaptation, or fine-tune on real Uljin ground truth.

### Notes / limitations (honest)
- Label = dNBR pseudo-label (no independent ground truth) → F1 is vs that reference.
- train/val/test are **row blocks of ONE scene** → optimistic; earlier column split left the
  test block over sea (F1 0). Row split (train 0-56% / val 56-74% / test 74-100%) fixed it.
- **60-epoch training repeatedly OOMs on the 8 GB Jetson** (memory grows per epoch even at
  bs4/workers0); the **best-val checkpoint (ep34)** is used — equivalent to early stopping.

## QGIS output (final goal)
- `src/make_qgis.py` (run with **system python3** + PyQGIS, headless) builds **`uljin.qgz`** —
  a styled project that opens with: post false-colour basemap + Burn perimeter (GeoJSON) +
  Land-cover transition + Burn severity + **Cloud/invalid mask** layers, CRS EPSG:32652.
- Open: `./open_qgis.sh`  (or `qgis --project uljin.qgz`).

**Cloud masking** is applied automatically in preprocessing: Sentinel-2 **SCL** classes
{0 nodata,1 saturated,3 cloud-shadow,8/9 cloud,10 cirrus,11 snow} in *either* date are
dropped from the valid mask, so clouds are never mistaken for burn. Exposed explicitly as the
`data/outputs/cloudmask.tif` layer.

## Change visualization
- `figs/before_after_segmentation.png` — the SAME per-pixel semantic-segmentation scheme
  (healthy veg / sparse veg / bare-soil / burned-scar / water) applied to each date, side by
  side; only the acquisition time differs, so the fire reads as green (before) -> red burned
  scar (after). Legend embedded in both panels. (`src/before_after_seg.py`)
- `figs/change_map.png` — single map, burn coloured by dNBR severity over the post image
  (`src/change_map.py`).
- `figs/transition_map.png` + `data/outputs/transition.tif` — semantic **from-to** transition
  (SCD style) INSIDE the detected burn: Veg→Burned scar (2,599 ha) / Healthy→Sparse veg-scorch
  (5,112 ha) / Veg→Bare (1,701 ha). Gated by the U-Net burn mask so the 10-day phenology
  mismatch is not mislabelled as change (`src/transition_map.py`).

## Next (optional)
- Independent-fire validation or official burn-perimeter ground truth (biggest accuracy lever).
- Train to full 60 epochs on a larger-memory GPU.

Env: reuse `~/wildfire-seg/.venv` (torch 2.5, GDAL 3.4). Run scripts with absolute paths.
