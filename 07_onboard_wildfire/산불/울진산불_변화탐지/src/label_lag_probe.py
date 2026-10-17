"""ko_refl 라벨이 안 갈리는 이유가 **라벨**인가 **씬 지연**인가 (§19 후속).

§19.5 에서 61화재 전부 dNBR AUC < 0.7(중앙 0.480)이 나왔다. 두 가설이 있다.
  H1 라벨: 산림청 폴리곤이 화소 단위로 소각을 표시하지 않는다.
  H2 지연: post 씬이 화재 +15~100일 중 **최저구름**으로 뽑히니 봄 화재는 재성장기에 걸린다
          — 초본이 주변 산림보다 빨리 푸르러지면 폴리곤 안 dNBR 이 오히려 낮아진다.
둘은 **지연과 AUC 의 상관**으로 갈린다. H2 라면 짧은 지연에서 AUC 가 올라야 한다.

날짜는 픽셀을 받지 않고 STAC **메타데이터만** 조회해 재현한다(빌더와 같은 질의).
사용: conda run -n geo python src/label_lag_probe.py
"""
import os, sys, json, math, datetime as dt, glob
os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
import numpy as np
from osgeo import ogr, osr
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import ko_build_refl as KB


def pick(bbox, a, b):
    its = list(KB.CAT.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=KB.rfc(a, b),
                             query={"eo:cloud_cover": {"lt": 30}},
                             sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
                             max_items=1).items())
    return its[0].datetime.date() if its else None


def main():
    want = {os.path.basename(d) for d in glob.glob(os.path.join(PROJ, "ko_refl", "*")) if os.path.isdir(d)}
    ds = ogr.Open(KB.SHP); ly = ds.GetLayer(); out = {}
    s = osr.SpatialReference(); s.ImportFromEPSG(5186); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    u = osr.SpatialReference(); u.ImportFromEPSG(KB.UTM); u.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    w = osr.SpatialReference(); w.ImportFromEPSG(4326); w.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    for f in ly:
        g = f.GetGeometryRef()
        if not g: continue
        gu = g.Clone(); gu.Transform(osr.CoordinateTransformation(s, u)); ha = gu.GetArea() / 1e4
        od = f.GetField("OCCU_DATE")
        try: fd = dt.date(int(od[:4]), int(od[4:6]), int(od[6:8]))
        except Exception: continue
        name = f"kf_{fd.isoformat().replace('-','')}_{int(ha)}ha"
        if name not in want or name in out: continue
        e = gu.GetEnvelope()
        ct = osr.CoordinateTransformation(u, w)
        lo1, la1, _ = ct.TransformPoint(e[0], e[2]); lo2, la2, _ = ct.TransformPoint(e[1], e[3])
        bb = [min(lo1, lo2) - 0.02, min(la1, la2) - 0.02, max(lo1, lo2) + 0.02, max(la1, la2) + 0.02]
        pc = fd.replace(year=fd.year - 1)
        pre = pick(bb, pc - dt.timedelta(days=30), pc + dt.timedelta(days=45))
        if pre is None:
            pre = pick(bb, fd - dt.timedelta(days=60), fd - dt.timedelta(days=8))
        post = pick(bb, fd + dt.timedelta(days=15), fd + dt.timedelta(days=100))
        if post is None: continue
        lag = (post - fd).days
        ddoy = None
        if pre is not None:
            d1 = pre.timetuple().tm_yday; d2 = post.timetuple().tm_yday
            ddoy = min(abs(d1 - d2), 365 - abs(d1 - d2))
        out[name] = dict(fire=fd.isoformat(), pre=pre.isoformat() if pre else None,
                         post=post.isoformat(), lag=lag, ddoy=ddoy)
        print(f"{name:<24} 화재 {fd} · pre {pre} · post {post} · 지연 {lag:>3}일 · ΔDOY {ddoy}", flush=True)
    json.dump(out, open(os.path.join(PROJ, "data", "ko_refl_dates.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n{len(out)}화재 → data/ko_refl_dates.json")


if __name__ == "__main__":
    main()
