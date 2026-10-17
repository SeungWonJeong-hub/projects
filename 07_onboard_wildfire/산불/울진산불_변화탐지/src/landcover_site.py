"""환경부 EGIS 토지피복 중분류(L2)를 **표본 씬 격자**에 얹는다 — 판독 보조자료.

왜: QC 게이트 실패의 원인이 **피복 오판**이었다(§23.1) — "탄화 = 어둡다" 로 봤는데 놓친 16건이
전부 경작지·촌락·나지·능선이었다. 초지·경작지·촌락이 타면 재와 노출 토양이 남아 **밝아진다**.
화소의 **원래 피복**을 알면 기대 신호가 정해진다: 산림 소각 = 암록/청록, 농경·초지 소각 = 담갈/회.

자료는 젯슨 미러에 있다(`~/jetson_data/.../data/landcover_egis/extracted/shp/lv2_*.shp`, 289도엽).
기존 `egis_cls.vrt` 는 3클래스(식생/물/기타)로 뭉개져 **산림과 농경지가 안 갈린다** — 그래서
원본 L2_CODE 에서 **대분류(L1)** 로 다시 뽑는다.

★커버리지 실측: EGIS 도엽이 흩어져 있어 **강릉 41%(1도엽) · 고성 0% · 양양 0%** 다.
그래서 **ESA WorldCover 10m(2021)** 를 기본 자료로 쓰고 EGIS 는 강릉 교차확인용으로만 둔다.
⚠ WorldCover 2021 은 **화재 이후**다 — 심하게 탄 산림이 관목·나대지로 재분류됐을 수 있다.
  그래서 **소각 여부의 근거로 쓰지 않는다.** 쓰는 곳은 "이 화소가 원래 농경지·시가지·수역인가"
  라는 **비산림 판별**뿐이고, 그 세 부류는 불로 바뀌지 않는다(§9 와 같은 주의).

L1: 1 시가화건조 · 2 농업 · 3 산림 · 4 초지 · 5 습지 · 6 나지 · 7 수역 (0/255 = 자료 없음)
출력: nifos_cache/{site}_refoot/lc_20m.npy (uint8)
사용: conda run -n geo python src/landcover_site.py
"""
import os, sys, glob, json
os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
import numpy as np
from osgeo import gdal, ogr, osr
gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
GSD = 20.0
SHP = os.path.expanduser("~/jetson_data/syntax_inference/Steven/산불/울진산불_변화탐지"
                         "/data/landcover_egis/extracted/shp")
SITES = ["gangneung", "goseong", "yangyang"]
L1 = {1: "시가화건조", 2: "농업", 3: "산림", 4: "초지", 5: "습지", 6: "나지", 7: "수역"}
WC_URL = ("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_{}_Map.tif")
WC = {10: "수목", 20: "관목", 30: "초지", 40: "농경지", 50: "시가지", 60: "나지",
      70: "설빙", 80: "수역", 90: "습지", 95: "맹그로브", 100: "이끼"}


def worldcover(site, xmin, ymax, W, H, utm):
    """ESA WorldCover 10m → 씬 격자(uint8, WC 코드). 3°×3° 타일을 골라 warp."""
    import math
    wgs = osr.SpatialReference(); wgs.ImportFromEPSG(4326)
    wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(utm, wgs)
    cor = [tr.TransformPoint(xmin + i * W * GSD, ymax - j * H * GSD)[:2]
           for i in (0, 1) for j in (0, 1)]
    tiles = sorted({(int(math.floor(la / 3) * 3), int(math.floor(lo / 3) * 3)) for lo, la in cor})
    urls = [WC_URL.format(f"N{la:02d}E{lo:03d}") for la, lo in tiles]
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    dst = gdal.GetDriverByName("MEM").Create("", W, H, 1, gdal.GDT_Byte)
    dst.SetGeoTransform((xmin, GSD, 0.0, ymax, 0.0, -GSD)); dst.SetProjection(utm.ExportToWkt())
    gdal.Warp(dst, urls, options=gdal.WarpOptions(resampleAlg="mode", srcNodata=0, dstNodata=0))
    return dst.ReadAsArray()


def main():
    utm = osr.SpatialReference(); utm.ImportFromEPSG(32652)
    utm.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    shps = sorted(glob.glob(os.path.join(SHP, "lv2_*.shp")))
    assert shps, SHP
    # 도엽 범위를 한 번만 읽어 캐시
    ext = []
    for s in shps:
        ds = ogr.Open(s); ly = ds.GetLayer(); e = ly.GetExtent()   # (xmin,xmax,ymin,ymax) in 5186
        ext.append((s, e, ly.GetSpatialRef().ExportToWkt()))
    print(f"EGIS 도엽 {len(ext)}개")

    for site in SITES:
        C = os.path.join(PROJ, "nifos_cache", f"{site}_refoot")
        u = np.load(os.path.join(C, "ute_20m.npy"))        # [xmin, ymin, xmax, ymax] (UTM52N)
        x = np.load(os.path.join(C, "scene_20m.npz"))["x"]
        H, W = x.shape[1], x.shape[2]
        xmin, ymax = float(u[0]), float(u[3])
        # 씬 네 귀퉁이를 5186 으로 → 겹치는 도엽 고르기
        k5186 = osr.SpatialReference(); k5186.ImportFromWkt(ext[0][2])
        k5186.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tr = osr.CoordinateTransformation(utm, k5186)
        cor = [tr.TransformPoint(xmin + i * W * GSD, ymax - j * H * GSD)[:2] for i in (0, 1) for j in (0, 1)]
        bx = (min(c[0] for c in cor), max(c[0] for c in cor),
              min(c[1] for c in cor), max(c[1] for c in cor))
        hit = [s for s, e, _ in ext if not (e[1] < bx[0] or e[0] > bx[1] or e[3] < bx[2] or e[2] > bx[3])]
        wc = worldcover(site, xmin, ymax, W, H, utm)
        np.save(os.path.join(C, "wc_20m.npy"), wc)
        h = {WC.get(int(v), str(v)): int(n) for v, n in zip(*np.unique(wc[wc > 0], return_counts=True))}
        t2 = max(sum(h.values()), 1)
        print(f"[{site}] WorldCover 유효 {(wc>0).mean():.0%} · "
              + " ".join(f"{k} {n/t2:.0%}" for k, n in sorted(h.items(), key=lambda t: -t[1])[:5]))
        if not hit:
            print(f"[{site}] EGIS 겹치는 도엽 0 — 범위 밖"); continue
        # 5186 에서 10m 로 래스터화한 뒤 UTM 20m 격자로 최근접 워프
        mem = gdal.GetDriverByName("MEM")
        vrtp = os.path.join(PROJ, "data", f"_lc_{site}.vrt")
        codes = []
        for s in hit:
            name = os.path.splitext(os.path.basename(s))[0]
            t = os.path.join(PROJ, "data", f"_code_{name}.tif")
            if not os.path.exists(t):
                import subprocess
                r = subprocess.run(["gdal_rasterize", "-q", "-sql",
                                    f"SELECT CAST(L2_CODE AS integer) AS c FROM {name}",
                                    "-a", "c", "-tr", "10", "10", "-tap", "-a_nodata", "0",
                                    "-ot", "UInt16", s, t], capture_output=True, text=True)
                if r.returncode != 0 or not os.path.exists(t):
                    print(f"  [{name}] rasterize 실패: {r.stderr[:80]}"); continue
            codes.append(t)
        gdal.BuildVRT(vrtp, codes, options=gdal.BuildVRTOptions(srcNodata=0, VRTNodata=0))
        dst = mem.Create("", W, H, 1, gdal.GDT_UInt16)
        dst.SetGeoTransform((xmin, GSD, 0.0, ymax, 0.0, -GSD))
        dst.SetProjection(utm.ExportToWkt())
        gdal.Warp(dst, vrtp, options=gdal.WarpOptions(resampleAlg="near", srcNodata=0, dstNodata=0))
        code = dst.ReadAsArray()
        lc = (code // 100).astype(np.uint8)                 # L2 → L1 대분류
        np.save(os.path.join(C, "lc_20m.npy"), lc)
        cov = (lc > 0).mean()
        hist = {L1.get(int(v), str(v)): int(n) for v, n in zip(*np.unique(lc[lc > 0], return_counts=True))}
        tot = max(sum(hist.values()), 1)
        print(f"[{site}] 도엽 {len(hit)} · 유효 {cov:.0%} · "
              + " ".join(f"{k} {n/tot:.0%}" for k, n in sorted(hist.items(), key=lambda t: -t[1])))
        for t in codes: pass
    for f in glob.glob(os.path.join(PROJ, "data", "_lc_*.vrt")): os.remove(f)


if __name__ == "__main__":
    main()
