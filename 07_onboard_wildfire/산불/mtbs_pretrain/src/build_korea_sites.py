"""한국 파인튜닝/검증 GT 확대 — 명명 화재(양구·군위·밀양)의 등급 GT + S2 쌍 생성.

기존 안동·울진 2건뿐이던 한국 GT를 5건으로. NIFoS s2_labels 폴리곤(심/중/경)을 20m 래스터로
굽고, **정확한 화재일**(국토정보플랫폼 CIR 날짜명 교차확인)로 S2 pre/post 를 취득해
nifos_cache 포맷(scene_20m.npz 'x' 12ch int16 + ute_20m.npy)으로 저장 → mtbs_validate_korea
SITES 에 그대로 물린다.

★날짜 정확성 필수(2025영상 버그 재발 방지): 날짜는 CIR 폴더명에서 확정한 것만 쓴다.
사용: python src/build_korea_sites.py
"""
import os, sys
import numpy as np
from osgeo import gdal, ogr
gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "..", "..", "웹앱"))
from core import acquire  # noqa: E402

EXT = os.path.expanduser("~/Steven/nifos_burn_gt/ext")
NC = os.path.join(ROOT, "..", "울진산불_변화탐지", "nifos_cache")
GT_OUT = os.path.expanduser("~/Steven/nifos_burn_gt/ext")
EPSG = 32652
BANDS = {"B02": "blue", "B03": "green", "B04": "red", "B8A": "nir08", "B11": "swir16", "B12": "swir22"}
# 날짜=국토정보플랫폼 CIR 폴더명서 확정 (양구양구읍 220410 · 군위의흥면 220411 · 밀양부북면 220531)
FIRES = {
    # s2_labels(명명 화재): 필드 gridcode/Severity · 날짜=CIR 폴더명
    "양구": dict(shp="s2_labels/Yanggu_b.shp",  fire="2022-04-10", attr="gridcode", sevf="Severity"),
    "군위": dict(shp="s2_labels/Gunwi_b.shp",   fire="2022-04-11", attr="gridcode", sevf="Severity"),
    "밀양": dict(shp="s2_labels/Miryang_b.shp", fire="2022-05-31", attr="gridcode", sevf="Severity"),
    "봉화": dict(shp="s2_labels/Bonghwa_b.shp", fire="2022-04-22", attr="gridcode", sevf="Severity"),  # 웹확인 봉화읍
    # re_labels(RapidEye): 필드 Dam_class/degree · 날짜=웹확인(2019 강원 동시다발 04-04·2018 고성 03-28)
    "re2019_63": dict(shp="re_labels/2019_63.shp",   fire="2019-04-04", attr="Dam_class", sevf="degree"),
    "re2019_71": dict(shp="re_labels/2019_71_1.shp", fire="2019-04-04", attr="Dam_class", sevf="degree"),
    "re2019_74": dict(shp="re_labels/2019_74.shp",   fire="2019-04-04", attr="Dam_class", sevf="degree"),
    "re2018_33": dict(shp="re_labels/2018_33.shp",   fire="2018-03-28", attr="Dam_class", sevf="degree"),
}


def rasterize_gt(shp, out_tif, attr="gridcode", sevf="Severity"):
    """폴리곤 attr(gridcode|Dam_class) 를 20m 래스터로. attr→'심/중/경' 매핑도 도출."""
    ds = ogr.Open(shp); L = ds.GetLayer()
    x0, x1, y0, y1 = L.GetExtent()          # (minx,maxx,miny,maxy)
    # 등급 단어 매핑 도출
    g2sev = {}
    L.ResetReading()
    for f in L:
        g2sev[f.GetField(attr)] = f.GetField(sevf)
    # 경1중2심3 canonical
    name2c = {"경": 1, "중": 2, "심": 3}
    gtmap = {int(g): name2c[s] for g, s in g2sev.items() if s in name2c}
    # 20m 격자(버퍼 400m)
    b = 400
    xmin = np.floor((x0 - b) / 20) * 20; ymin = np.floor((y0 - b) / 20) * 20
    xmax = np.ceil((x1 + b) / 20) * 20; ymax = np.ceil((y1 + b) / 20) * 20
    W = int((xmax - xmin) / 20); H = int((ymax - ymin) / 20)
    drv = gdal.GetDriverByName("GTiff")
    o = drv.Create(out_tif, W, H, 1, gdal.GDT_Byte, ["COMPRESS=DEFLATE"])
    o.SetGeoTransform((xmin, 20, 0, ymax, 0, -20))
    srs = gdal.osr.SpatialReference(); srs.ImportFromEPSG(EPSG); o.SetProjection(srs.ExportToWkt())
    o.GetRasterBand(1).Fill(0)
    gdal.RasterizeLayer(o, [1], L, options=[f"ATTRIBUTE={attr}"])
    o = None
    return [xmin, ymin, xmax, ymax], (W, H), gtmap


def pick(lon, lat, d0, d1, target):
    f = acquire.search([lon-0.2, lat-0.18, lon+0.2, lat+0.18], d0, d1, max_cloud=40, limit=40)
    if not f:
        return None
    tgt = np.datetime64(target)
    f.sort(key=lambda x: (abs((np.datetime64(x["datetime"])-tgt).astype("timedelta64[D]").astype(int)), x["cloud"]))
    best = np.datetime64(f[0]["datetime"])
    return [s for s in f if abs((np.datetime64(s["datetime"])-best).astype("timedelta64[D]").astype(int)) <= 3]


def fetch_stack(scenes, bounds):
    """씬그룹 모자이크 → 12ch(pre 없음, 단일시점) int16 DN. 여기선 6밴드만 반환."""
    out = {}
    for b, key in BANDS.items():
        hrefs = ["/vsicurl/"+s["assets"][key]["href"] for s in scenes if s["assets"].get(key)]
        if not hrefs:
            return None
        a = gdal.Warp("", hrefs, format="MEM", outputBounds=bounds, dstSRS=f"EPSG:{EPSG}",
                      xRes=20, yRes=20, resampleAlg="average", outputType=gdal.GDT_Int16).ReadAsArray()
        out[b] = a
    return out


def main():
    from datetime import date, timedelta
    os.makedirs(NC, exist_ok=True)
    added = []
    _only = os.environ.get("MTBS_ONLY","").split(",") if os.environ.get("MTBS_ONLY") else None
    for name, cfg in FIRES.items():
        if _only and name not in _only: continue
        shp = os.path.join(EXT, cfg["shp"])
        gt_tif = os.path.join(GT_OUT, f"{name}_severity_20m.tif")
        bounds, (W, H), gtmap = rasterize_gt(shp, gt_tif, cfg.get("attr","gridcode"), cfg.get("sevf","Severity"))
        # 중심 lon/lat
        s = gdal.osr.SpatialReference(); s.ImportFromEPSG(EPSG)
        t = gdal.osr.SpatialReference(); t.ImportFromEPSG(4326); t.SetAxisMappingStrategy(gdal.osr.OAMS_TRADITIONAL_GIS_ORDER)
        lon, lat, _ = gdal.osr.CoordinateTransformation(s, t).TransformPoint((bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2)
        fdt = date(*map(int, cfg["fire"].split("-")))
        sp = pick(lon, lat, str(fdt-timedelta(days=25)), str(fdt-timedelta(days=2)), str(fdt-timedelta(days=12)))
        sa = pick(lon, lat, str(fdt+timedelta(days=3)), str(fdt+timedelta(days=35)), str(fdt+timedelta(days=18)))
        if not sp or not sa:
            print(f"[{name}] ⚠️ S2 부족 pre={bool(sp)} post={bool(sa)}"); continue
        pre = fetch_stack(sp, bounds); post = fetch_stack(sa, bounds)
        if pre is None or post is None:
            print(f"[{name}] ⚠️ 밴드 취득 실패"); continue
        x = np.stack([pre[b] for b in BANDS] + [post[b] for b in BANDS], 0).astype("int16")
        cdir = os.path.join(NC, name); os.makedirs(cdir, exist_ok=True)
        np.savez_compressed(os.path.join(cdir, "scene_20m.npz"), x=x)
        np.save(os.path.join(cdir, "ute_20m.npy"), np.array(bounds, float))
        # dNBR 신호 점검(소각 신호 있는지)
        gtA = gdal.Open(gt_tif).ReadAsArray()
        def nbr(n, sw): d = n.astype("f4")+sw; return np.where(d > 0, (n-sw)/np.maximum(d, 1), 0)
        dnbr = nbr(x[3], x[5]) - nbr(x[9], x[11]); m = gtA > 0
        print(f"[{name}] 화재 {cfg['fire']} · pre {sp[0]['datetime']}/post {sa[0]['datetime']} · "
              f"소각화소 {int(m.sum())} · dNBR중앙 {np.median(dnbr[m]):+.3f} · gtmap {gtmap} · 격자 {W}x{H}"
              + ("  ★신호없음(날짜확인)" if np.median(dnbr[m]) <= 0 else "  ✓신호"))
        added.append((name, cfg["fire"], gtmap))
    print("\n=== SITES 추가용 ===")
    for name, fire, gtmap in added:
        print(f'    "{name}": dict(gt=os.path.expanduser("~/Steven/nifos_burn_gt/ext/{name}_severity_20m.tif"), '
              f'cache="{name}", fire="{fire}", gtmap={gtmap}),')


if __name__ == "__main__":
    main()
