"""Shared STAC + COG helpers for the Uljin wildfire change-detection pipeline.

Reuses the same approach as ~/wildfire-seg/sentinel_downloader.py:
  - AWS Earth Search STAC (public, no credentials), collection sentinel-2-l2a
  - remote COG read via GDAL /vsicurl, area-average resampled to 20 m GSD

Bands we pull (Earth-Search asset key -> Sentinel-2 band), enough to build the
6-channel burn stack [B12,B08,B02,NBR,NDVI,NDWI] + dNBR and an SCL cloud mask:

    blue=B02  green=B03  red=B04  nir=B08  swir16=B11  swir22=B12  scl=SCL
"""
import os
import time
import json
import requests
from osgeo import gdal

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "4")
gdal.SetConfigOption("GDAL_HTTP_RETRY_DELAY", "5")
gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", "180")
gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
gdal.SetConfigOption("VSI_CACHE", "TRUE")

STAC_URL   = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
REQ_TIMEOUT = 180
MAX_RETRIES = 4
TARGET_GSD_M = 20

# Uljin (2022-03-04..03-13) burn-scar AOI, ~40x40 km centred on the scar.
AOI_BBOX = [129.13, 36.85, 129.57, 37.22]

# Asset keys -> output band filename.  SCL is categorical (nearest, no resample avg).
WANT_BANDS = {"blue": "B02", "green": "B03", "red": "B04",
              "nir": "B08", "swir16": "B11", "swir22": "B12"}
WANT_MASK = {"scl": "SCL"}


def stac_search(bbox, datetime_range, max_cloud=100, limit=100):
    """POST search; datetime_range is an RFC3339 interval 'start/end'."""
    body = {
        "collections": [COLLECTION],
        "bbox": bbox,
        "datetime": datetime_range,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": limit,
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(STAC_URL, json=body, timeout=REQ_TIMEOUT)
            if r.status_code == 200:
                return r.json().get("features", [])
            print(f"  STAC HTTP {r.status_code} (attempt {attempt+1})")
        except Exception as e:
            print(f"  STAC error {e} (attempt {attempt+1})")
        time.sleep(5 * (attempt + 1))
    return []


def scene_summary(feat):
    p = feat["properties"]
    return {
        "id": feat["id"],
        "datetime": p.get("datetime"),
        "cloud": p.get("eo:cloud_cover"),
        "tile": p.get("grid:code") or p.get("s2:mgrs_tile"),
        "bbox": feat.get("bbox"),
        "assets": sorted(feat.get("assets", {}).keys()),
    }


def download_band(href, dest, resample="average"):
    """Read remote COG, clip to AOI_BBOX (EPSG:4326), resample to 20 m, write GTiff."""
    src = "/vsicurl/" + href
    opts = gdal.TranslateOptions(
        xRes=TARGET_GSD_M, yRes=TARGET_GSD_M, resampleAlg=resample,
        projWin=[AOI_BBOX[0], AOI_BBOX[3], AOI_BBOX[2], AOI_BBOX[1]],
        projWinSRS="EPSG:4326", format="GTiff",
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"])
    for attempt in range(MAX_RETRIES):
        tmp = dest + ".part"
        try:
            ds = gdal.Translate(tmp, src, options=opts)
            ds = None
            n = os.path.getsize(tmp)
            os.replace(tmp, dest)
            return n
        except Exception as e:
            print(f"  band retry {attempt+1} ({os.path.basename(dest)}): {e}")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            time.sleep(5 * (attempt + 1))
    return 0


def fetch_scene(feat, out_dir):
    """Download all WANT_BANDS + SCL for one scene into out_dir/<id>/."""
    sid = feat["id"]
    assets = feat.get("assets", {})
    scene_dir = os.path.join(out_dir, sid)
    os.makedirs(scene_dir, exist_ok=True)
    total = 0
    for key, name in {**WANT_BANDS, **WANT_MASK}.items():
        asset = assets.get(key)
        if not asset or not asset.get("href"):
            print(f"  ! missing asset {key} for {sid}")
            continue
        dest = os.path.join(scene_dir, f"{name}.tif")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            total += os.path.getsize(dest)
            continue
        alg = "nearest" if name == "SCL" else "average"
        total += download_band(asset["href"], dest, resample=alg)
    json.dump(scene_summary(feat), open(os.path.join(scene_dir, "metadata.json"), "w"), indent=2)
    return scene_dir, total
