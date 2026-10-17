"""Step 1b — swap in a previous-year leaf-on PRE to match the post phenology.

The 2022-02 winter pre depressed dNBR (leaf-off + snow).  Post is 2022-04-09, so we
pick a low-cloud 2021 scene near the same day-of-year (2021-03-20..2021-04-30) over the
SAME tile (52SEG) for pixel alignment, download it, and rewrite scene_pair.json so
'pre' points to it.  The old winter pre is kept as 'pre_winter' for reference.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from stac import stac_search, scene_summary, fetch_scene, AOI_BBOX

HERE = os.path.dirname(__file__)
PAIR = os.path.join(HERE, "..", "data", "scene_pair.json")
RAW = os.path.join(HERE, "..", "data", "raw")
WINDOW = "2021-03-20T00:00:00Z/2021-04-30T23:59:59Z"
TILE = "MGRS-52SEG"


def main():
    pair = json.load(open(PAIR))
    feats = stac_search(AOI_BBOX, WINDOW, max_cloud=100, limit=50)
    rows = [(f, scene_summary(f)) for f in feats]
    same_tile = [(f, s) for f, s in rows if s["tile"] == TILE]
    print("=== 2021 leaf-on PRE candidates on tile 52SEG (cloud-sorted, top 8) ===")
    for _, s in same_tile[:8]:
        print(f"  {s['datetime'][:10]}  cloud={s['cloud']:>5.1f}%  {s['id']}")
    if not same_tile:
        print("!! no same-tile candidate; widen window"); return
    feat, sel = same_tile[0]
    print(f"\n>>> SELECTED 2021 PRE: {sel['datetime'][:10]} cloud={sel['cloud']:.1f}% {sel['id']}")
    scene_dir, total = fetch_scene(feat, RAW)
    print(f"  downloaded {total/1e6:.1f} MB -> {scene_dir}")

    pair["pre_winter"] = {k: pair["pre"][k] for k in pair["pre"]}  # keep old
    pair["pre"] = {**sel, "dir": scene_dir}
    json.dump(pair, open(PAIR, "w"), indent=2)
    print(f"\nscene_pair.json updated: pre = {sel['id']} (winter pre kept as pre_winter)")


if __name__ == "__main__":
    main()
