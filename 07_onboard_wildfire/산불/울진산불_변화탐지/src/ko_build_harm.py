"""학습셋 재빌드 — pre 를 **조화 계절 기저선**으로 바꾼다 (§14.2 가 예고한 것, 미결 #9).

왜: 현 학습풀 `ko_refl` 은 pre 를 "화재 1년 전 ±한두 달의 최저구름 씬" 으로 고르고 post 는
"화재 +15~100일의 최저구름 씬" 으로 고른다. 두 창이 독립이라 **DOY 가 최대 100일 어긋난다**
(§11.1: 245화재 중 54%). 그 대가가 실측됐다 — **폴리곤 안 dNBR 중앙값이 음수**(−0.235)라
녹화가 소각을 덮고, 이 풀로는 dNBR 라벨을 만들 수 없다(§14.2).

조화 기저선을 pre 로 쓰면 **계절이 정의상 맞는다** — pre 를 post 와 **같은 날짜**에 대해
예측하기 때문이다(§12). 고를 씬이 없으니 선택 변수도 사라진다.

★한계: 피팅에 화재 이전 다년 관측이 필요하다. S2 국내 안정 커버리지가 2017 부터라
**2017~2018 초 화재는 이력이 모자라 만들 수 없다**(MIN_OK 관문에서 걸러진다).
그래서 이 풀은 원본보다 **작다** — 무엇이 빠졌는지 manifest 에 남긴다.

산출: ko_refl_harm/{화재}/{화재}_{y}_{x}.npz  — `ko_refl` 과 **같은 규격**(x 12ch int16
      [pre6|post6] · y · valid)이라 step4_rehearsal 의 KO_DIR 로 바로 바꿔 낄 수 있다.
      data/ko_harm_manifest.json 에 화재별 성공/실패 사유와 **폴리곤 안 dNBR 중앙값**을 남긴다.

사용: conda run -n geo python src/ko_build_harm.py [N_FIRES]
      FIRES=kf_20200424_147ha,...  로 특정 화재만(파일럿)
"""
from __future__ import annotations
import os, sys, json, math, time, datetime as dt, glob

os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
import numpy as np
from osgeo import gdal, ogr, osr

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import ko_build_refl as KB          # 후보 선택·씬 취득·타일링을 그대로 쓴다(같은 화재 = 공정 A/B)
import harmonic_baseline as HB      # build_ts / fit / predict

OUT = os.environ.get("KO_OUT", os.path.join(PROJ, "ko_refl_harm"))
TS_DIR = os.environ.get("TS_DIR", os.path.join(PROJ, "nifos_cache"))
GAP = int(os.environ.get("GAP", "14"))        # 화재 며칠 전까지만 피팅에 쓸지(피해 학습 방지)
DROP_TS = os.environ.get("DROP_TS", "0") == "1"   # 피팅 후 시계열 캐시 삭제(디스크 절약)
N_FIRES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
ONLY = [x for x in os.environ.get("FIRES", "").split(",") if x]
T, STRIDE, GSD, NIR = KB.T, KB.STRIDE, KB.GSD, KB.NIR
ORDER = KB.ORDER


def post_pick(bbox_ll, fdate):
    """post 씬의 **날짜**를 먼저 확정한다 — 조화 pre 를 그 날짜에 예측해야 하기 때문."""
    its = list(KB.CAT.search(collections=["sentinel-2-l2a"], bbox=bbox_ll,
                             datetime=KB.rfc(fdate + dt.timedelta(days=15), fdate + dt.timedelta(days=100)),
                             query={"eo:cloud_cover": {"lt": 30}},
                             sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
                             max_items=1).items())
    return its[0].datetime.date() if its else None


def dnbr_in_polygon(x, y, valid):
    """폴리곤 안 dNBR 중앙값 — 이 풀로 dNBR 라벨을 만들 수 있는지의 단일 판정값(§14.2)."""
    f = x.astype(np.float32) / 1e4
    nbr = lambda b: (b[3] - b[5]) / (b[3] + b[5] + 1e-6)      # (NIR − SWIR22)
    d = nbr(f[:6]) - nbr(f[6:])
    m = (y > 0) & valid
    return float(np.median(d[m])) if m.any() else float("nan")


def build_one(name, gu, fdate):
    e = gu.GetEnvelope(); cx = (e[0] + e[1]) / 2; cy = (e[2] + e[3]) / 2
    hx = max((e[1] - e[0]) / 2 + 500, KB.CROP_M); hy = max((e[3] - e[2]) / 2 + 500, KB.CROP_M)
    ute = [math.floor(cx - hx), math.floor(cy - hy), math.ceil(cx + hx), math.ceil(cy + hy)]
    dst = osr.SpatialReference(); dst.ImportFromEPSG(4326); dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src = osr.SpatialReference(); src.ImportFromEPSG(KB.UTM); src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    lo1, la1, _ = ct.TransformPoint(e[0], e[2]); lo2, la2, _ = ct.TransformPoint(e[1], e[3])
    bb = [min(lo1, lo2) - 0.02, min(la1, la2) - 0.02, max(lo1, lo2) + 0.02, max(la1, la2) + 0.02]

    d_post = post_pick(bb, fdate)
    if d_post is None: return dict(ok=False, why="post 씬 없음")
    post = KB.stac_refl(bb, KB.rfc(d_post, d_post), ute)
    if post is None: return dict(ok=False, why="post 취득 실패")

    t_end = fdate - dt.timedelta(days=GAP)
    cache = os.path.join(TS_DIR, f"{name}_ts")
    t0 = time.time()
    try:
        dates, xs, vs = HB.build_ts(name, ute, bb, t_end, cache)
    except AssertionError as ex:
        return dict(ok=False, why=f"시계열 부족: {str(ex)[:80]}")
    if len(dates) < int(os.environ.get("MIN_DATES", "60")):
        return dict(ok=False, why=f"피팅 날짜 {len(dates)} < MIN_DATES")
    try:
        C, ok_m, nobs = HB.fit(dates, xs, vs)
    except AssertionError as ex:
        return dict(ok=False, why=f"MIN_OK 미달: {str(ex)[:90]}", n_dates=len(dates))
    pre_hw = HB.predict(C, ok_m, d_post)                    # (H,W,6) 반사율×1e4
    fit_s = time.time() - t0
    if DROP_TS:
        for f in glob.glob(os.path.join(cache, "*.npz")): os.remove(f)

    pre = {k: pre_hw[:, :, i].astype(np.float32) for i, k in enumerate(ORDER)}
    H = min(pre[ORDER[0]].shape[0], post['B02'].shape[0]); W = min(pre[ORDER[0]].shape[1], post['B02'].shape[1])
    for d in (pre, post):
        for k in d: d[k] = d[k][:H, :W]
    srs = osr.SpatialReference(); srs.ImportFromEPSG(KB.UTM)
    mem = ogr.GetDriverByName("Memory").CreateDataSource("m"); lyr = mem.CreateLayer("f", srs, ogr.wkbPolygon)
    ft = ogr.Feature(lyr.GetLayerDefn()); ft.SetGeometry(gu); lyr.CreateFeature(ft)
    rast = gdal.GetDriverByName("MEM").Create("", W, H, 1, gdal.GDT_Byte)
    rast.SetGeoTransform((ute[0], GSD, 0, ute[3], 0, -GSD)); rast.SetProjection(srs.ExportToWkt())
    gdal.RasterizeLayer(rast, [1], lyr, burn_values=[1]); mask = rast.ReadAsArray()[:H, :W].astype(np.uint8)

    # ★pre 는 합성이라 SCL 이 없다 — 유효성은 post 의 SCL 과 **계수 유효 화소**로 정의한다.
    valid = (post['B02'] > 0) & (post[NIR] > 0) & (post['B12'] > 0)
    valid &= ~np.isin(post['SCL'], [0, 1, 3, 8, 9, 10, 11])
    valid &= ok_m[:H, :W]
    x = np.stack([pre[k] for k in ORDER] + [post[k] for k in ORDER], 0).astype(np.int16)
    y = (mask > 0).astype(np.uint8)
    dn = dnbr_in_polygon(x, y, valid)

    od = os.path.join(OUT, name); os.makedirs(od, exist_ok=True); n = 0
    if H < T or W < T:
        os.rmdir(od); return dict(ok=False, why=f"크롭<{T}")
    ys = sorted(set(min(yy, H - T) for yy in range(0, H - T + 1, STRIDE)))
    xs2 = sorted(set(min(xx, W - T) for xx in range(0, W - T + 1, STRIDE)))
    for yy2 in ys:
        for xx2 in xs2:
            v = valid[yy2:yy2 + T, xx2:xx2 + T]
            if v.mean() < 0.4: continue
            np.savez_compressed(os.path.join(od, f"{name}_{yy2}_{xx2}.npz"),
                                x=x[:, yy2:yy2 + T, xx2:xx2 + T],
                                y=(y[yy2:yy2 + T, xx2:xx2 + T] & v).astype(np.uint8),
                                valid=v.astype(np.uint8)); n += 1
    if n == 0:
        os.rmdir(od); return dict(ok=False, why="유효 타일 0")
    print(f"[{name}] {fdate} post={d_post} | 타일 {n} | 불탄 {y[valid].mean()*100:.1f}%"
          f" | 폴리곤 dNBR 중앙 {dn:+.3f} | 피팅 {len(dates)}날짜 {fit_s/60:.1f}분", flush=True)
    return dict(ok=True, tiles=n, post=d_post.isoformat(), dnbr_med=dn,
                n_dates=len(dates), fit_min=round(fit_s / 60, 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    cands = KB.candidates() if hasattr(KB, "candidates") else None
    if cands is None:                                  # ko_build_refl.main 과 같은 선별을 여기서 재현
        ds = ogr.Open(KB.SHP); ly = ds.GetLayer(); cands = []
        for f in ly:
            if f.GetField("OCCU_YEAR") not in KB.YEARS: continue
            g = f.GetGeometryRef()
            if not g: continue
            s = osr.SpatialReference(); s.ImportFromEPSG(5186); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            d = osr.SpatialReference(); d.ImportFromEPSG(KB.UTM); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            gu = g.Clone(); gu.Transform(osr.CoordinateTransformation(s, d)); ha = gu.GetArea() / 1e4
            if not (KB.MIN_HA <= ha <= KB.MAX_HA): continue
            od = f.GetField("OCCU_DATE")
            try: fdate = dt.date(int(od[:4]), int(od[4:6]), int(od[6:8]))
            except Exception: continue
            cands.append((ha, fdate, gu.Clone()))
        cands.sort(key=lambda c: c[0], reverse=True)
    sel = cands[:N_FIRES]
    man_p = os.path.join(PROJ, "data", "ko_harm_manifest.json")
    man = json.load(open(man_p, encoding="utf-8")) if os.path.exists(man_p) else {}
    print(f"후보 {len(cands)} · 대상 {len(sel)}" + (f" · FIRES 지정 {len(ONLY)}" if ONLY else ""), flush=True)
    for i, (ha, fdate, gu) in enumerate(sel):
        name = f"kf_{fdate.isoformat().replace('-','')}_{int(ha)}ha"
        if ONLY and name not in ONLY: continue
        if glob.glob(os.path.join(OUT, name, "*.npz")):
            print(f"[{name}] skip(있음)", flush=True); continue
        try:
            r = build_one(name, gu, fdate)
        except Exception as ex:
            r = dict(ok=False, why=f"{type(ex).__name__}: {str(ex)[:100]}")
        if not r.get("ok"): print(f"[{name}] 제외 — {r['why']}", flush=True)
        man[name] = dict(fire_date=fdate.isoformat(), ha=round(ha), **r)
        json.dump(man, open(man_p, "w"), ensure_ascii=False, indent=1)
    okn = [v for v in man.values() if v.get("ok")]
    print(f"\n완성 {len(okn)}화재 · 타일 {sum(v['tiles'] for v in okn)}"
          f" · 폴리곤 dNBR 중앙값의 중앙 {np.median([v['dnbr_med'] for v in okn]):+.3f}"
          if okn else "\n완성 0", flush=True)
    print(f"매니페스트: data/ko_harm_manifest.json", flush=True)


if __name__ == "__main__":
    main()
