"""Step 2 — download the chosen pre/post scenes (B02,B03,B04,B08,B11,B12,SCL),
clipped to the Uljin AOI and resampled to 20 m GSD."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from stac import fetch_scene, stac_search, AOI_BBOX

HERE = os.path.dirname(__file__)
PAIR = os.path.join(HERE, "..", "data", "scene_pair.json")
RAW = os.path.join(HERE, "..", "data", "raw")


def refetch_feature(sid, dt):
    """Re-query STAC for the exact scene id to get its asset hrefs."""
    day = dt[:10]
    win = f"{day}T00:00:00Z/{day}T23:59:59Z"
    for f in stac_search(AOI_BBOX, win, max_cloud=100, limit=50):
        if f["id"] == sid:
            return f
    return None


def main():
    pair = json.load(open(PAIR))
    os.makedirs(RAW, exist_ok=True)
    for role in ("pre", "post"):
        sel = pair[role]
        print(f"\n--- {role.upper()}: {sel['id']} ({sel['datetime'][:10]}, cloud {sel['cloud']}%) ---")
        feat = refetch_feature(sel["id"], sel["datetime"])
        if feat is None:
            print(f"  !! could not re-query {sel['id']}")
            continue
        scene_dir, total = fetch_scene(feat, RAW)
        bands = sorted(f for f in os.listdir(scene_dir) if f.endswith(".tif"))
        print(f"  saved {len(bands)} bands ({total/1e6:.1f} MB) -> {scene_dir}")
        print(f"  bands: {bands}")
        # tag which scene dir is pre/post for the next step
        pair[role]["dir"] = scene_dir
    json.dump(pair, open(PAIR, "w"), indent=2)
    print("\ndone.")


if __name__ == "__main__":
    main()
