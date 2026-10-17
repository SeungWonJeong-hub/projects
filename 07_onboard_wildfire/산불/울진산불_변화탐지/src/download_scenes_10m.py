"""Step 2 (10 m variant) — re-download the SAME Uljin pre/post scene pair on a
10 m grid instead of 20 m, into data/raw_10m/ (20 m baseline in data/raw/ stays intact).

물리 현실: S2에서 blue/green/red/nir(B02,B03,B04,B08)만 네이티브 10 m.
swir16/swir22(B11,B12)는 네이티브 20 m → gdal이 10 m 격자로 업샘플(area-average).
따라서 RGB+NIR은 진짜 10 m 디테일, SWIR는 20 m가 촘촘한 격자에 올라갈 뿐.
같은 scene_pair.json(pre 2021-04-19 / post 2022-04-09)을 재사용 → 공정 A/B.

사용: /usr/bin/python3 download_scenes_10m.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import stac
from stac import stac_search, AOI_BBOX, fetch_scene

# --- 10 m 격자로 재타깃 (stac.download_band이 읽는 모듈 전역을 덮어씀) ---
stac.TARGET_GSD_M = 10

HERE = os.path.dirname(__file__)
PAIR = os.path.join(HERE, "..", "data", "scene_pair.json")
RAW10 = os.path.join(HERE, "..", "data", "raw_10m")


def refetch_feature(sid, dt):
    """정확한 scene id로 STAC 재질의 → asset href 확보."""
    day = dt[:10]
    win = f"{day}T00:00:00Z/{day}T23:59:59Z"
    for f in stac_search(AOI_BBOX, win, max_cloud=100, limit=50):
        if f["id"] == sid:
            return f
    return None


def main():
    pair = json.load(open(PAIR))
    os.makedirs(RAW10, exist_ok=True)
    out = {}
    for role in ("pre", "post"):
        sel = pair[role]
        print(f"\n--- {role.upper()} @10m: {sel['id']} ({sel['datetime'][:10]}, cloud {sel['cloud']}%) ---")
        feat = refetch_feature(sel["id"], sel["datetime"])
        if feat is None:
            print(f"  !! could not re-query {sel['id']}")
            continue
        scene_dir, total = fetch_scene(feat, RAW10)
        bands = sorted(f for f in os.listdir(scene_dir) if f.endswith(".tif"))
        print(f"  saved {len(bands)} bands ({total/1e6:.1f} MB) -> {scene_dir}")
        # 밴드별 실제 픽셀 크기 확인(10 m면 20 m 대비 폭·높이 2배)
        from osgeo import gdal
        for b in ("B02", "B08", "B12"):
            p = os.path.join(scene_dir, f"{b}.tif")
            if os.path.exists(p):
                ds = gdal.Open(p); gt = ds.GetGeoTransform()
                print(f"    {b}: {ds.RasterXSize}x{ds.RasterYSize} px, pixel={gt[1]:.1f}m")
        out[role] = {**sel, "dir_10m": scene_dir}
    json.dump({**pair, **{f"{k}_10m_dir": v["dir_10m"] for k, v in out.items()}},
              open(os.path.join(HERE, "..", "data", "scene_pair_10m.json"), "w"), indent=2)
    print("\ndone (10 m).")


if __name__ == "__main__":
    main()
