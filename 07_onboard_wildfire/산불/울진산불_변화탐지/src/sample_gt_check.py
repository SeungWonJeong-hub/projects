"""판독 정합성 점검 — RapidEye GT 폴리곤 안의 표본점이 어떻게 읽혔나.

GT 는 **부분 라벨**이라 면적의 하한만 준다(폴리곤 밖도 탔을 수 있다). 그러나 **폴리곤 안**은
탄 곳이므로, 그 안에 떨어진 본표본 점이 대부분 '비변화'로 읽혔다면 그 사이트의 판독이 틀린 것이다.
양양 층1(예측 소각) 46점이 0/46 소각으로 읽힌 것이 진짜인지 여기서 가른다.
"""
import os, json
os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
import numpy as np
from osgeo import ogr, gdal
gdal.UseExceptions()
PROJ = "/home/syntax-01/syntax_inference/Steven/산불/울진산불_변화탐지"
GSD = 20.0
SITES = [("gangneung", "2019_63"), ("goseong", "2019_74"), ("yangyang", "2018_33")]

S = json.load(open(f"{PROJ}/sample/sample_points.json", encoding="utf-8"))
by = {p["id"]: p for p in S["points"]}
R = {}
for l in open(f"{PROJ}/sample/claude_read_main.jsonl", encoding="utf-8"):
    l = l.strip()
    if l:
        r = json.loads(l); R[r["id"]] = r

for site, re_id in SITES:
    shp = f"{PROJ}/../nifos_burn_gt/ext/re_labels/{re_id}.shp"
    ute = np.load(f"{PROJ}/nifos_cache/{site}_refoot/ute_20m.npy")
    lab = np.load(f"{PROJ}/sample/strata_{site}.npz")["lab"]
    H, W = lab.shape
    # 폴리곤 → 20m 래스터(footprint 격자에 맞춰)
    drv = gdal.GetDriverByName("MEM")
    ds = drv.Create("", W, H, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((float(ute[0]), GSD, 0, float(ute[3]), 0, -GSD))
    v = ogr.Open(shp); ly = v.GetLayer()
    gdal.RasterizeLayer(ds, [1], ly, burn_values=[1])
    g = ds.ReadAsArray().astype(bool)
    gt_ha = g.sum() * GSD * GSD / 1e4
    pred_ha = (lab == 1).sum() * GSD * GSD / 1e4
    inter = (g & (lab == 1)).sum() * GSD * GSD / 1e4
    print(f"[{site}] GT 폴리곤 {gt_ha:,.0f}ha · 모델 예측소각 {pred_ha:,.0f}ha · 겹침 {inter:,.0f}ha"
          f"  (GT 재현 {inter/max(gt_ha,1e-9):.2f})")
    # GT 안에 떨어진 본표본 점의 판독
    ids = [i for i, p in by.items() if p["site"] == site and not p.get("qc") and i in R
           and 0 <= p["row"] < H and 0 <= p["col"] < W and g[p["row"], p["col"]]]
    from collections import Counter
    c = Counter(R[i]["read"] for i in ids)
    print(f"   GT 폴리곤 안 본표본 {len(ids)}점 판독 → {dict(c)}")
    if ids and c.get("소각", 0) / len(ids) < 0.5:
        print("   ⚠GT 안인데 절반 넘게 소각이 아니다 — 이 사이트의 판독을 의심할 것")
        for i in sorted(ids)[:8]:
            print(f"      {i} {R[i]['read']}  conf={R[i].get('conf')}  {R[i].get('note','')}")
    # 층1 점 중 GT 안 비율
    s1 = [i for i, p in by.items() if p["site"] == site and not p.get("qc") and i in R
          and p["stratum"] == 1]
    s1g = [i for i in s1 if g[by[i]["row"], by[i]["col"]]]
    print(f"   층1 표본 {len(s1)}점 중 GT 안 {len(s1g)}점 · "
          f"층1 소각판독 {sum(1 for i in s1 if R[i]['read']=='소각')}점\n")
