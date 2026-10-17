"""Step 1 — auto-select the pre/post Sentinel-2 scene pair for the Uljin fire.

Fire: 2022-03-04 ignition, ~03-13 contained. We search
  PRE  window: 2022-01-15 .. 2022-03-03  (before ignition)
  POST window: 2022-03-14 .. 2022-05-10  (after containment, snow gone)
over the Uljin AOI and pick the lowest-cloud scene in each window.
Candidates are printed so the choice is auditable.
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from stac import stac_search, scene_summary, AOI_BBOX

PRE_WINDOW = "2022-01-15T00:00:00Z/2022-03-03T23:59:59Z"
POST_WINDOW = "2022-03-14T00:00:00Z/2022-05-10T23:59:59Z"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "scene_pair.json")


def show(feats, tag):
    print(f"\n=== {tag} candidates (cloud-sorted, top 8) ===")
    rows = [scene_summary(f) for f in feats]
    for r in rows[:8]:
        print(f"  {r['datetime'][:10]}  cloud={r['cloud']:>5.1f}%  tile={r['tile']}  {r['id']}")
    return rows


def main():
    pre = stac_search(AOI_BBOX, PRE_WINDOW, max_cloud=100, limit=50)
    post = stac_search(AOI_BBOX, POST_WINDOW, max_cloud=100, limit=50)
    pre_rows = show(pre, "PRE (before 2022-03-04)")
    post_rows = show(post, "POST (after 2022-03-13)")
    if not pre_rows or not post_rows:
        print("\n!! empty result in one window — widen the search window.")
        return
    chosen = {"pre": pre_rows[0], "post": post_rows[0]}
    print("\n>>> AUTO-SELECTED PAIR (lowest cloud in each window):")
    print(f"    PRE : {chosen['pre']['datetime'][:10]}  cloud={chosen['pre']['cloud']:.1f}%  {chosen['pre']['id']}")
    print(f"    POST: {chosen['post']['datetime'][:10]}  cloud={chosen['post']['cloud']:.1f}%  {chosen['post']['id']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(chosen, open(OUT, "w"), indent=2)
    print(f"\nsaved -> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
