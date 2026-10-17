"""판독 정합성 — 표본점의 **RapidEye 화소가 실제로 있는가**를 전수로 검사한다.

판정면은 RE CIR 이다. 상자가 RE 씬 밖이거나 채움값(65535)·0 이면 **판정불가**여야 하는데,
칩 그림만 보면 400m 창 가장자리에서 눈으로 놓치기 쉽다(P0064). 사람 눈 대신 원본 래스터로
전수 검사해 **같은 규칙을 모든 점에 똑같이** 적용한다.

사용: ~/miniforge3/envs/geo/bin/python src/sample_re_valid.py
"""
import os, json
os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
RE_DIR = os.path.join(PROJ, "..", "nifos_burn_gt", "ext", "re_img")

S = json.load(open(f"{PROJ}/sample/sample_points.json", encoding="utf-8"))
R = {}
for l in open(f"{PROJ}/sample/claude_read_main.jsonl", encoding="utf-8"):
    l = l.strip()
    if l:
        r = json.loads(l); R[r["id"]] = r
cache = {}
bad = []
for p in S["points"]:
    if p["id"] not in R: continue
    re_id = p["re_scene"]
    if re_id not in cache:
        d = gdal.Open(os.path.join(RE_DIR, f"{re_id}.tif"))
        cache[re_id] = (d, d.GetGeoTransform())
    d, g = cache[re_id]
    cx = int((p["utm_x"] - g[0]) / g[1]); cy = int((p["utm_y"] - g[3]) / g[5])
    k = 2                                    # 20m 화소 = RE 5m 4x4, 가장자리 포함해 조금 넓게
    x0, y0 = cx - k, cy - k
    if x0 < 0 or y0 < 0 or x0 + 2*k > d.RasterXSize or y0 + 2*k > d.RasterYSize:
        bad.append((p["id"], p["site"], R[p["id"]]["read"], "씬 밖")); continue
    a = d.ReadAsArray(x0, y0, 2*k, 2*k).astype(np.float32)
    v = a[4]                                  # NIR
    if not np.isfinite(v).any() or (v <= 0).all() or (v >= 60000).all():
        bad.append((p["id"], p["site"], R[p["id"]]["read"], "채움값")); continue
    frac = float(((v > 0) & (v < 60000)).mean())
    if frac < 0.5:
        bad.append((p["id"], p["site"], R[p["id"]]["read"], f"유효 {frac:.0%}"))
print(f"판독된 {len(R)}점 중 RE 화소가 없는 점 {len(bad)}")
for i, s, rd, why in bad:
    print(f"  {i} {s:<10} 판독 {rd:<6} — {why}" + ("  ←판정불가로 고칠 것" if rd != "판정불가" else "  (이미 판정불가)"))
json.dump([dict(id=i, site=s, read=rd, why=w) for i, s, rd, w in bad],
          open(f"{PROJ}/sample/re_invalid.json", "w"), ensure_ascii=False, indent=1)
