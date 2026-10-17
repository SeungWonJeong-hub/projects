"""확률표본용 — **RapidEye 씬 footprint 전체**에 대한 S2 쌍을 만든다.

왜: 기존 `nifos_cache/{site}_doy` 의 범위는 **GT 래스터의 bbox**(= 부분 라벨의 경계상자)라
강릉이 6.1x3.2km(1,977ha)뿐이다. 그 안에서 표본을 뽑으면 소각 비중이 33% 인 **화재 중심 크롭**을
다시 재게 된다(함정 12). 판독이 가능한 범위는 RapidEye 영상이 덮는 **10.1x6.2km(6,290ha)** 이고,
표본의 모집단은 그쪽이어야 한다.

씬 날짜는 기존 `_doy` 캐시가 쓴 것과 **똑같이 고정**한다(같은 쌍, 더 넓은 크롭).

산출: nifos_cache/{site}_refoot/{scene_20m.npz, ute_20m.npy, scene_dates_20m.npy}
사용: conda run -n geo python src/re_footprint_prep.py
"""
import os, sys, datetime as dt
os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
import numpy as np
from osgeo import gdal, osr
gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import ko_build_refl as KB

GSD = 20.0
SITES = {"gangneung": ("2019_63", "2018-04-20", "2019-04-20"),
         "goseong":   ("2019_74", "2018-05-10", "2019-05-10"),
         "yangyang":  ("2018_33", "2017-05-03", "2018-05-05")}


def footprint(re_id):
    d = gdal.Open(os.path.join(PROJ, "..", "nifos_burn_gt", "ext", "re_img", f"{re_id}.tif"))
    g = d.GetGeoTransform()
    x0, y1 = g[0], g[3]; x1 = x0 + g[1] * d.RasterXSize; y0 = y1 + g[5] * d.RasterYSize
    return [np.floor(x0 / GSD) * GSD, np.floor(y0 / GSD) * GSD,
            np.ceil(x1 / GSD) * GSD, np.ceil(y1 / GSD) * GSD]


def main():
    for site, (re_id, pre_d, post_d) in SITES.items():
        out = os.path.join(PROJ, "nifos_cache", f"{site}_refoot")
        if os.path.exists(os.path.join(out, "scene_20m.npz")):
            print(f"[{site}] skip(있음)"); continue
        ute = footprint(re_id)
        u = osr.SpatialReference(); u.ImportFromEPSG(KB.UTM); u.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        w = osr.SpatialReference(); w.ImportFromEPSG(4326); w.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(u, w)
        lo1, la1, _ = ct.TransformPoint(ute[0], ute[1]); lo2, la2, _ = ct.TransformPoint(ute[2], ute[3])
        bb = [min(lo1, lo2) - 0.02, min(la1, la2) - 0.02, max(lo1, lo2) + 0.02, max(la1, la2) + 0.02]
        def get(ds):
            d0 = dt.date.fromisoformat(ds)
            return KB.stac_refl(bb, KB.rfc(d0 - dt.timedelta(days=1), d0 + dt.timedelta(days=1)), ute)
        pre, post = get(pre_d), get(post_d)
        assert pre is not None and post is not None, f"{site} 씬 취득 실패"
        H = min(pre['B02'].shape[0], post['B02'].shape[0]); W = min(pre['B02'].shape[1], post['B02'].shape[1])
        x = np.stack([pre[k][:H, :W] for k in KB.ORDER] + [post[k][:H, :W] for k in KB.ORDER], 0).astype(np.int16)
        scl = np.stack([pre['SCL'][:H, :W], post['SCL'][:H, :W]], 0).astype(np.uint8)
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, "scene_20m.npz"), x=x, scl=scl)
        np.save(os.path.join(out, "ute_20m.npy"), np.array(ute))
        np.save(os.path.join(out, "scene_dates_20m.npy"), np.array([pre_d, post_d]))
        print(f"[{site}] {re_id} · {W}x{H}px · {W*H*GSD*GSD/1e4:,.0f}ha · pre {pre_d} / post {post_d} → {out}", flush=True)


if __name__ == "__main__":
    main()
