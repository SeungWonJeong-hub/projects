"""재보정용 **눈금 예시** + **새 QC 40점** — 게이트 재시험 (§23.1 의 다음 수).

왜: 1차 QC 84 에서 균형정확도 0.808 · 등급 kappa 0.171 로 떨어졌고, 원인이 갈렸다(§23.1) —
폴리곤이 아니라 **판독자**다. 실패의 정체는 **"탄화 = 전 밴드 어두움"** 기준이었다. 초지·경작지·
촌락이 타면 재와 노출 토양이 남아 **밝아진다**(놓친 16건이 전부 그것, dNBR 0.42~0.86).

그래서 두 가지를 만든다.
  ① **눈금 예시**(`data/calib_chips/`) — 판독 **전에** 본다. 등급 기준선이 없어서 전부 `중` 으로
     쏠렸다(강릉 kappa 0.242). 예시는 **강릉에서만** 뽑는다 — degree 의 등급-신호 순서가
     유효한 유일한 사이트다(고성 중>심, 양양 중>심 역전, 함정 5·§23.1).
     각 예시의 dNBR 을 같이 적는다. 이건 **학습셋**이지 시험이 아니다.
  ② **새 QC 40점**(`sample/sample_qc2.json`) — 1차 84점·눈금 예시와 **화소가 겹치지 않게** 뽑는다.
     같은 84점 재채점은 정답을 본 뒤라 독립성이 없다.

사용: conda run -n geo python src/sample_qc2.py [--pos 24 --neg 16 --calib 5]
"""
import os, sys, json, csv, argparse
os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
import numpy as np
from osgeo import ogr, gdal
gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
GSD = 20.0
SITES = [("gangneung", "2019_63"), ("goseong", "2019_74"), ("yangyang", "2018_33")]
CALIB_SITE = "gangneung"          # degree 순서가 dNBR 과 일치하는 유일한 사이트(§23.1 ④)


def dnbr_of(site, cache={}):
    if site not in cache:
        C = os.path.join(PROJ, "nifos_cache", f"{site}_refoot")
        x = np.load(os.path.join(C, "scene_20m.npz"))["x"].astype(np.float32) / 1e4
        ix = lambda b, i, j: (b[i] - b[j]) / (b[i] + b[j] + 1e-6)
        cache[site] = ix(x[:6], 3, 5) - ix(x[6:], 3, 5)
    return cache[site]


def poly_points(site, re_id, want, rng, taken, by_degree=None):
    """폴리곤 안 무작위 점. by_degree 를 주면 그 등급 폴리곤에서만 뽑는다."""
    shp = os.path.join(PROJ, "..", "nifos_burn_gt", "ext", "re_labels", f"{re_id}.shp")
    ds = ogr.Open(shp); ly = ds.GetLayer()
    polys = [(f.GetGeometryRef().Clone(), str(f.GetField("degree"))) for f in ly if f.GetGeometryRef()]
    if by_degree: polys = [p for p in polys if p[1] == by_degree]
    if not polys: return []
    ute = np.load(os.path.join(PROJ, "nifos_cache", f"{site}_refoot", "ute_20m.npy"))
    D = dnbr_of(site); out = []
    for _ in range(20000):
        if len(out) >= want: break
        g, deg = polys[int(rng.integers(len(polys)))]
        e = g.GetEnvelope(); hit = None
        for _ in range(300):
            ux = rng.uniform(e[0], e[1]); uy = rng.uniform(e[2], e[3])
            pt = ogr.Geometry(ogr.wkbPoint); pt.AddPoint(float(ux), float(uy))
            if g.Contains(pt): hit = (ux, uy); break
        if hit is None: continue
        ux, uy = hit
        r = int((ute[3] - uy) / GSD); c = int((ux - ute[0]) / GSD)
        if (site, r, c) in taken: continue
        if not (0 <= r < D.shape[0] and 0 <= c < D.shape[1]): continue
        taken.add((site, r, c))
        out.append(dict(site=site, re_scene=re_id, stratum=0, qc=True, row=r, col=c,
                        utm_x=round(float(ux), 1), utm_y=round(float(uy), 1), pi=0.0,
                        truth="소각", truth_grade=deg, dnbr=round(float(D[r, c]), 3),
                        read="", grade="", note=""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=24); ap.add_argument("--neg", type=int, default=16)
    ap.add_argument("--calib", type=int, default=5); ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    S = json.load(open(os.path.join(PROJ, "sample/sample_points.json"), encoding="utf-8"))
    taken = {(p["site"], p["row"], p["col"]) for p in S["points"]}   # 1차 표본 전부 회피
    import sample_chips as SC

    # ── ① 눈금 예시 (강릉, 등급별) ─────────────────────────────────────
    calib = []
    for deg in ("경", "중", "심"):
        got = poly_points(CALIB_SITE, dict(SITES)[CALIB_SITE], a.calib, rng, taken, by_degree=deg)
        for i, p in enumerate(got, 1): p["id"] = f"C{deg}{i}"
        calib += got
        print(f"[눈금] {deg} {len(got)}점 · dNBR {[p['dnbr'] for p in got]}")
    json.dump(calib, open(os.path.join(PROJ, "sample/sample_calib.json"), "w"),
              ensure_ascii=False, indent=1)

    # ── ② 새 QC 40점 ───────────────────────────────────────────────
    per = a.pos // len(SITES); pos = []
    for site, re_id in SITES:
        pos += poly_points(site, re_id, per, rng, taken)
    from sample_qc import neg_points
    neg = neg_points(a.neg, rng, S)
    neg = [p for p in neg if (p["site"], p["row"], p["col"]) not in taken]
    for p in neg:
        taken.add((p["site"], p["row"], p["col"]))
        p["dnbr"] = round(float(dnbr_of(p["site"])[p["row"], p["col"]]), 3)
    qc2 = pos + neg
    rng.shuffle(qc2)
    for i, p in enumerate(qc2, 1): p["id"] = f"Q{i:03d}"
    json.dump(dict(n=len(qc2), n_pos=len(pos), n_neg=len(neg), seed=a.seed, points=qc2),
              open(os.path.join(PROJ, "sample/sample_qc2.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[QC2] 양성 {len(pos)} + 음성 {len(neg)} = {len(qc2)}점")

    # ── chip 생성 (★토지피복을 제목에 얹는다 — §23.1 실패의 절반이 피복 오판이었다) ────
    # ★칩 제목은 matplotlib DejaVu Sans 로 그려져 한글이 □ 로 깨진다 — 영문으로 적는다.
    WC = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up",
          60: "bare/sparse", 70: "snow/ice", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
    wcache = {}
    def lcname(site, r, c):
        if site not in wcache:
            f = os.path.join(PROJ, "nifos_cache", f"{site}_refoot", "wc_20m.npy")
            wcache[site] = np.load(f) if os.path.exists(f) else None
        a = wcache[site]
        if a is None or not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]): return "n/a"
        return WC.get(int(a[r, c]), f"code{int(a[r,c])}")

    for name, pts, od in (("calib", calib, "data/calib_chips"), ("qc2", qc2, "data/qc2_chips")):
        o = os.path.join(PROJ, od); os.makedirs(o, exist_ok=True)
        for p in pts:
            cir, nat, b1 = SC.re_chip(p["re_scene"], p["utm_x"], p["utm_y"])
            if cir is None: print(f"  {p['id']}: RE 범위 밖 — 건너뜀"); continue
            pre, post, b2 = SC.s2_chip(p["site"], p["row"], p["col"])
            extra = (f"   [calib · degree={p['truth_grade']} · dNBR={p['dnbr']}]"
                     if name == "calib" else "")
            extra += f"   landcover(WorldCover 2021): {lcname(p['site'], p['row'], p['col'])}"
            SC.draw([cir, nat, pre, post], [b1, b1, b2, b2], os.path.join(o, f"{p['id']}.png"),
                    f"{p['id']}   {p['site']} / {p['re_scene']}   "
                    f"yellow box = the 20 m pixel to judge  (400 m window){extra}")
        print(f"chip {len(pts)}장 → {od}/")


if __name__ == "__main__":
    main()
