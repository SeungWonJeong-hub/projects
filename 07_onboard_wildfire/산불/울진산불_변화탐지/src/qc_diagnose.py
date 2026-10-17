"""QC 게이트 실패의 원인을 가른다 — 판독자가 놓친 건가, 폴리곤이 과대포함인가 (§23).

Claude 단독 판독이 QC 게이트에서 **누락 23/60**(과탐은 0)을 냈다. 두 해석이 가능한데
판독 결과만으로는 못 가른다. 그래서 **독립 분광 증거**로 본다 — dNBR 은 SWIR(B12)을 쓰므로
모델 입력(RGB+NIR)과 겹치지 않고, §21 에서 안동 실측 GT 를 **AUC 0.916** 으로 갈라낸 자다.

  ⓐ 폴리곤 안 · 판독 소각   (37점)  ← 둘 다 "탔다"
  ⓑ 폴리곤 안 · 판독 비소각 (23점)  ← 불일치. **이게 어디에 앉는가**
  ⓒ 음성 QC                 (20점)  ← 둘 다 "안 탔다"

ⓑ 가 ⓒ 쪽이면 **폴리곤 과대포함**(게이트가 순환), ⓐ 쪽이면 **판독자가 놓친 것**이다.

②전수 검사도 같이 낸다: RapidEye 폴리곤 **안−밖** dNBR AUC(§21 과 같은 지표).
  안동 실측 0.916 · 산림청 0.480 이 비교 기준선이다.

사용: conda run -n geo python src/qc_diagnose.py
"""
import os, sys, json
os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
import numpy as np
from osgeo import ogr, gdal
gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
GSD = 20.0
SITES = [("gangneung", "2019_63"), ("goseong", "2019_74"), ("yangyang", "2018_33")]


def idx(b, i, j):
    return (b[i] - b[j]) / (b[i] + b[j] + 1e-6)


def load(site):
    C = os.path.join(PROJ, "nifos_cache", f"{site}_refoot")
    z = np.load(os.path.join(C, "scene_20m.npz")); x = z["x"].astype(np.float32) / 1e4
    ute = np.load(os.path.join(C, "ute_20m.npy"))
    pre, post = x[:6], x[6:]
    dnbr = idx(pre, 3, 5) - idx(post, 3, 5)      # NIR vs B12 → SWIR 사용(모델 입력과 독립)
    dndvi = idx(pre, 3, 2) - idx(post, 3, 2)
    scl = z["scl"]; bad = [0, 1, 3, 8, 9, 10, 11]
    ok = ~np.isin(scl[0], bad) & ~np.isin(scl[1], bad)
    return dnbr, dndvi, ok, ute


def rasterize(site, re_id, shape, ute):
    shp = os.path.join(PROJ, "..", "nifos_burn_gt", "ext", "re_labels", f"{re_id}.shp")
    if not os.path.exists(shp): return None
    ds = ogr.Open(shp); ly = ds.GetLayer()
    H, W = shape
    drv = gdal.GetDriverByName("MEM")
    r = drv.Create("", W, H, 1, gdal.GDT_Byte)
    r.SetGeoTransform((float(ute[0]), GSD, 0.0, float(ute[3]), 0.0, -GSD))
    gdal.RasterizeLayer(r, [1], ly, burn_values=[1])
    return r.ReadAsArray().astype(bool)


def auc(pos, neg):
    """안−밖 분리도. ★순위는 **비교 집합 안에서만** 매긴다(함정 26)."""
    if len(pos) < 20 or len(neg) < 20: return float("nan")
    v = np.concatenate([pos, neg]); r = v.argsort().argsort().astype(np.float64) + 1
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    pts = {p["id"]: p for p in json.load(open(os.path.join(PROJ, "sample/sample_points.json")))["points"]}
    rd = {}
    for ln in open(os.path.join(PROJ, "sample/claude_read.jsonl"), encoding="utf-8"):
        if ln.strip(): r = json.loads(ln); rd[r["id"]] = r

    global cache_g
    cache = cache_g = {s: load(s) for s, _ in SITES}
    G = {"ⓐ 폴리곤안·판독 소각": [], "ⓑ 폴리곤안·판독 비소각": [], "ⓒ 음성 QC": [], "판정불가": []}
    for i, p in pts.items():
        if not p.get("qc") or i not in rd: continue
        dnbr, dndvi, ok, _ = cache[p["site"]]
        r, c = p["row"], p["col"]
        if not (0 <= r < dnbr.shape[0] and 0 <= c < dnbr.shape[1]): continue
        rec = (float(dnbr[r, c]), float(dndvi[r, c]), bool(ok[r, c]), i)
        read = rd[i]["read"]
        if read == "판정불가": G["판정불가"].append(rec)
        elif p["truth"] != "소각": G["ⓒ 음성 QC"].append(rec)
        elif read == "소각": G["ⓐ 폴리곤안·판독 소각"].append(rec)
        else: G["ⓑ 폴리곤안·판독 비소각"].append(rec)

    print("═══ ① QC 점의 독립 분광 증거 (dNBR = NIR vs SWIR, 모델 입력과 무관) ═══")
    print(f"{'무리':<22}{'n':>4}{'dNBR 중앙':>11}{'[25%':>8}{'75%]':>8}{'dNDVI 중앙':>12}{'dNBR≥0.18':>11}")
    stat = {}
    for k, v in G.items():
        if not v: continue
        a = np.array([x[0] for x in v]); b = np.array([x[1] for x in v])
        q = np.percentile(a, [25, 75])
        print(f"{k:<22}{len(v):>4}{np.median(a):>11.3f}{q[0]:>8.3f}{q[1]:>8.3f}"
              f"{np.median(b):>12.3f}{(a >= 0.18).mean():>10.0%}")
        stat[k] = dict(n=len(v), dnbr_med=float(np.median(a)), dndvi_med=float(np.median(b)),
                       pass18=float((a >= 0.18).mean()))
    print("\n  ★기준: 0.18 = 배포 extent 절단(§14.5). 안동 실측 폴리곤 안 dNBR 중앙 +0.545(§21).")

    print("\n═══ ② RapidEye 폴리곤 자체의 화소 정확도 — 안−밖 dNBR AUC ═══")
    print(f"  비교 기준선: 안동 NIFoS 0.3m 실측 **0.916** · 산림청 폴리곤 **0.480** (§21)")
    print(f"{'사이트':<12}{'폴리곤ha':>10}{'AUC':>8}{'안 dNBR':>10}{'밖 dNBR':>10}{'차':>8}")
    aucs = {}
    for site, re_id in SITES:
        dnbr, dndvi, ok, ute = cache[site]
        m = rasterize(site, re_id, dnbr.shape, ute)
        if m is None: print(f"{site:<12}(shp 없음)"); continue
        inn = dnbr[m & ok]; out = dnbr[(~m) & ok]
        a = auc(inn, out)
        print(f"{site:<12}{m.sum()*GSD*GSD/1e4:>10,.0f}{a:>8.3f}"
              f"{np.median(inn):>10.3f}{np.median(out):>10.3f}{np.median(inn)-np.median(out):>+8.3f}")
        aucs[site] = dict(auc=float(a), ha=float(m.sum() * GSD * GSD / 1e4),
                          in_med=float(np.median(inn)), out_med=float(np.median(out)))

    print("\n═══ ③ 놓친 23건 개별 ═══")
    print(f"{'id':<7}{'사이트':<11}{'dNBR':>8}{'dNDVI':>8}  판독 사유")
    for d, dv, okk, i in sorted(G["ⓑ 폴리곤안·판독 비소각"], key=lambda x: x[0]):
        print(f"{i:<7}{pts[i]['site']:<11}{d:>8.3f}{dv:>8.3f}  {rd[i]['note'][:44]}")
    json.dump(dict(groups=stat, poly_auc=aucs),
              open(os.path.join(PROJ, "data", "qc_diagnose.json"), "w"), ensure_ascii=False, indent=1)
    grade_check()
    print("\n기록: data/qc_diagnose.json")




def grade_check():
    """등급 kappa 0.171 이 판독자 탓인지 — 폴리곤 `degree` 자체가 신호 순서를 지키는지 본다.

    ★새 GT 는 등급-신호 ①순서 ②눈금을 먼저 검사한다(함정 5, `src/sev_transfer.py:gate`).
    degree 가 dNBR 순서를 안 지키면 등급 정답이 먼저 의심 대상이다.
    """
    import collections
    print("\n═══ ④ 폴리곤 degree 의 등급-신호 순서 검사 ═══")
    print(f"{'사이트':<12}{'degree':>8}{'ha':>9}{'dNBR 중앙':>11}{'dNDVI 중앙':>12}")
    for site, re_id in SITES:
        dnbr, dndvi, ok, ute = cache_g[site]
        shp = os.path.join(PROJ, "..", "nifos_burn_gt", "ext", "re_labels", f"{re_id}.shp")
        ds = ogr.Open(shp); ly = ds.GetLayer()
        by = collections.defaultdict(list)
        for f in ly:
            g = f.GetGeometryRef()
            if g is None: continue
            by[str(f.GetField("degree"))].append(g.Clone())
        for deg in sorted(by):
            H, W = dnbr.shape
            r = gdal.GetDriverByName("MEM").Create("", W, H, 1, gdal.GDT_Byte)
            r.SetGeoTransform((float(ute[0]), GSD, 0.0, float(ute[3]), 0.0, -GSD))
            mem = ogr.GetDriverByName("Memory").CreateDataSource("")
            lay = mem.CreateLayer("l", ly.GetSpatialRef(), ogr.wkbPolygon)
            for g in by[deg]:
                ft = ogr.Feature(lay.GetLayerDefn()); ft.SetGeometry(g); lay.CreateFeature(ft)
            gdal.RasterizeLayer(r, [1], lay, burn_values=[1])
            m = r.ReadAsArray().astype(bool) & ok
            if m.sum() < 25: continue
            print(f"{site:<12}{deg:>8}{m.sum()*GSD*GSD/1e4:>9,.0f}"
                  f"{np.median(dnbr[m]):>11.3f}{np.median(dndvi[m]):>12.3f}")


if __name__ == "__main__":
    main()
