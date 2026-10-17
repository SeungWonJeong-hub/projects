"""확률표본 판독 — **판독자 검증(QC) 점**을 섞어 넣는다.

왜: 판독 결과를 자로 쓰려면 **판독자가 얼마나 맞히는지**를 먼저 알아야 한다. 정답을 아는 점을
본 표본에 섞고(판독자는 어느 것이 QC 인지 모른다) 일치율을 잰다.

★**양성만으로는 게이트가 작동하지 않는다**(2026-09-10 사용자 지적, 맞다). 정답이 전부
폴리곤 안에 있으면 **"전부 소각"이라고 찍는 판독자가 일치율 1.00** 을 받는다 — 이 표본이
재려는 것은 **과탐**인데 게이트는 반대 방향만 본다. 그래서 **음성 QC 를 같이** 넣는다.

  양성 QC — 기존 RapidEye 폴리곤 **내부**. 정답 = 소각 + `degree` 등급(경/중/심).
  음성 QC — S2 에서 **규칙이 극단적으로 명확한 곳**만. 정답 = 비변화.
            ①수체: NDWI > 0.10 (pre·post 둘 다)  ②지속 나지: NDVI < 0.20 (둘 다)
            ③무변화 산림: NDVI > 0.50 (둘 다) & |dNBR| < 0.08
            셋 다 **예측 소각 층에서 500m 이상** 떨어진 곳에서만 뽑는다.

음성 QC 의 정답은 S2 지수로 정의했지만 판독은 RapidEye 로 한다 — 검증 대상이 **모델이 아니라
판독자**이므로 순환이 아니다. 다만 규칙이 극단적이어야 "정답"이라 부를 수 있다.
벌채 QC 는 자동으로 못 고른다(그래서 없다) — 기입표의 `벌채·개간` 항목으로 받는다.

사용: conda run -n geo python src/sample_qc.py --qc 60
"""
import os, sys, json, argparse
os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
import numpy as np
from osgeo import ogr, gdal
gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
GSD = 20.0
SITES = [("gangneung", "2019_63"), ("goseong", "2019_74"), ("yangyang", "2018_33")]


def neg_points(k, rng, S):
    """음성 QC — 규칙이 극단적으로 명확하고 화재에서 먼 곳(정답 = 비변화)."""
    if k <= 0: return []
    from scipy.ndimage import distance_transform_edt
    per = max(1, k // len(SITES)); out = []
    for site, re_id in SITES:
        C = os.path.join(PROJ, "nifos_cache", f"{site}_refoot")
        z = np.load(os.path.join(C, "scene_20m.npz")); x = z["x"].astype(np.float32) / 1e4
        ute = np.load(os.path.join(C, "ute_20m.npy"))
        ndvi = lambda b: (b[3] - b[2]) / (b[3] + b[2] + 1e-6)
        ndwi = lambda b: (b[1] - b[3]) / (b[1] + b[3] + 1e-6)
        nbr = lambda b: (b[3] - b[5]) / (b[3] + b[5] + 1e-6)
        pre, post = x[:6], x[6:]
        d = nbr(pre) - nbr(post)
        water = (ndwi(pre) > 0.10) & (ndwi(post) > 0.10)
        bare = (ndvi(pre) < 0.20) & (ndvi(post) < 0.20) & ~water
        forest = (ndvi(pre) > 0.50) & (ndvi(post) > 0.50) & (np.abs(d) < 0.08)
        # ★거리는 **모델이 소각이라 한 층**에서 잰다(dNBR 대용이 아니라 실제 층 지도).
        L = np.load(os.path.join(PROJ, "sample", f"strata_{site}.npz"))["lab"]
        far = distance_transform_edt(L != 1) * 20.0 >= 500.0
        far = far[:d.shape[0], :d.shape[1]]
        scl = z["scl"]; bad = [0, 1, 3, 8, 9, 10, 11]
        ok = ~np.isin(scl[0], bad) & ~np.isin(scl[1], bad)
        for kind, m in (("수체", water), ("지속나지", bare), ("무변화산림", forest)):
            idx = np.flatnonzero((m & far & ok).ravel())
            take = max(1, per // 3)
            if idx.size == 0:
                print(f"[{site}] 음성QC {kind}: 후보 0 — 건너뜀"); continue
            pick = rng.choice(idx, size=min(take, idx.size), replace=False)
            ys, xs = np.unravel_index(pick, m.shape)
            for y, xx in zip(ys, xs):
                out.append(dict(site=site, re_scene=re_id, stratum=0, qc=True,
                                row=int(y), col=int(xx),
                                utm_x=round(float(ute[0] + (xx + 0.5) * GSD), 1),
                                utm_y=round(float(ute[3] - (y + 0.5) * GSD), 1), pi=0.0,
                                truth="비변화", truth_kind=kind, read="", grade="", note=""))
            print(f"[{site}] 음성QC {kind} {min(take, idx.size)}점 (후보 {idx.size:,})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", type=int, default=60, help="양성 QC(폴리곤 안)")
    ap.add_argument("--neg", type=int, default=30, help="음성 QC(정답=비변화)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed + 777)
    P = os.path.join(PROJ, "sample", "sample_points.json")
    S = json.load(open(P, encoding="utf-8"))
    if any(p.get("qc") for p in S["points"]):
        print("이미 QC 가 섞여 있다 — sample_design 을 다시 돌린 뒤 실행할 것"); return
    per = max(1, a.qc // len(SITES)); out = []
    for site, re_id in SITES:
        shp = os.path.join(PROJ, "..", "nifos_burn_gt", "ext", "re_labels", f"{re_id}.shp")
        if not os.path.exists(shp): continue
        ds = ogr.Open(shp); ly = ds.GetLayer()
        polys = [(f.GetGeometryRef().Clone(), f.GetField("degree")) for f in ly if f.GetGeometryRef()]
        if not polys: continue
        ute = np.load(os.path.join(PROJ, "nifos_cache", f"{site}_refoot", "ute_20m.npy"))
        got = 0
        while got < per:
            g, deg = polys[int(rng.integers(len(polys)))]
            e = g.GetEnvelope(); hit = None
            for _ in range(300):
                ux = rng.uniform(e[0], e[1]); uy = rng.uniform(e[2], e[3])
                pt = ogr.Geometry(ogr.wkbPoint); pt.AddPoint(float(ux), float(uy))
                if g.Contains(pt): hit = (ux, uy); break
            if hit is None: continue
            ux, uy = hit
            out.append(dict(site=site, re_scene=re_id, stratum=0, qc=True,
                            row=int((ute[3] - uy) / GSD), col=int((ux - ute[0]) / GSD),
                            utm_x=round(float(ux), 1), utm_y=round(float(uy), 1), pi=0.0,
                            truth="소각", truth_grade=str(deg), read="", grade="", note=""))
            got += 1
        print(f"[{site}] QC {got}점 · 등급값 {sorted({p[1] for p in polys})}")
    out += neg_points(a.neg, rng, S)
    pts = S["points"] + out
    rng.shuffle(pts)
    for i, p in enumerate(pts, 1): p["id"] = f"P{i:04d}"
    S["points"] = pts; S["n"] = len(pts); S["n_qc"] = len(out)
    json.dump(S, open(P, "w"), ensure_ascii=False, indent=1)
    npos = sum(1 for p in out if p.get("truth") == "소각")
    print(f"본표본 {len(pts)-len(out)} + 양성QC {npos} + 음성QC {len(out)-npos} = {len(pts)}점"
          f" → sample/sample_points.json")


if __name__ == "__main__":
    main()
