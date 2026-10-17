"""미학습 화재 before/after 데모 씬 내보내기 (웹앱 업로드용 20m·10m).

웹앱 파일 업로드는 멀티밴드 GeoTIFF [B02,B03,B04,NIR,B11,B12] 를 받는다(io_bands.BAND_ORDER).
20m 모델은 NIR=B8A, 10m 모델은 NIR=B08 이므로 해상도별로 파일을 따로 만든다.

earth-search STAC(무인증)에서 화재 AOI 로 크롭해 받으므로 파일이 작다(수 MB).
심각도(dNBR)는 B12 가 있어 20m·10m 둘 다 동작한다.

사용: python src/export_demo_scenes.py   (기본=안동 2025, 미학습 독립 자)
출력: <OUT>/{before,after}_{20m,10m}.tif · {before,after}_rgb.png · gt_20m.tif · README.md
"""
import os, sys, json
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "웹앱"))
from core import acquire  # noqa: E402

# ── 대상 화재(미학습): 안동 2025 (NIFoS 독립 자) ────────────────────────────
FIRE = dict(
    name="안동길안면_2025",
    epsg=32652,
    bbox_utm=[465420, 4033620, 473780, 4039320],   # [minx,miny,maxx,maxy] EPSG:32652
    pre=("2025-03-15", "2025-03-27", "2025-03-22"),  # (검색시작, 검색끝, 목표일)
    post=("2025-04-05", "2025-04-16", "2025-04-11"),
)
BANDS_20 = ["B02", "B03", "B04", "B8A", "B11", "B12"]   # 20m: NIR=B8A
BANDS_10 = ["B02", "B03", "B04", "B08", "B11", "B12"]   # 10m: NIR=B08
OUT = os.path.join(os.path.dirname(__file__), "..", "..", f"미학습_데모_{FIRE['name']}")


def pick_scene(win, lon, lat):
    d0, d1, tgt = win
    feats = acquire.search([lon - 0.15, lat - 0.12, lon + 0.15, lat + 0.12], d0, d1, max_cloud=40)
    if not feats:
        raise SystemExit(f"씬 없음: {win}")
    feats.sort(key=lambda f: (abs((np.datetime64(f["datetime"]) - np.datetime64(tgt)).astype(int)), f["cloud"]))
    print(f"  후보 {len(feats)}개 → 선택 {feats[0]['id']} ({feats[0]['datetime']}, ☁{feats[0]['cloud']}%)")
    return feats[0]


def fetch_band(href, dest, gsd, bbox):
    ulx, uly, lrx, lry = bbox[0], bbox[3], bbox[2], bbox[1]   # projWin=[ulx,uly,lrx,lry]
    opts = gdal.TranslateOptions(xRes=gsd, yRes=gsd, resampleAlg="average",
        projWin=[ulx, uly, lrx, lry], projWinSRS=f"EPSG:{FIRE['epsg']}",
        format="GTiff", creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])
    gdal.Translate(dest, "/vsicurl/" + href, options=opts)


def stack(scene_assets, bands, gsd, out_tif, tmp):
    paths = []
    for b in bands:
        key = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir",
               "B8A": "nir08", "B11": "swir16", "B12": "swir22"}[b]
        a = scene_assets.get(key)
        if not a:
            raise SystemExit(f"자산 없음: {b}")
        p = os.path.join(tmp, f"{b}_{gsd:g}.tif")
        fetch_band(a["href"], p, gsd, FIRE["bbox_utm"])
        paths.append(p)
    vrt = gdal.BuildVRT(out_tif + ".vrt", paths, separate=True)
    vrt = None
    gdal.Translate(out_tif, out_tif + ".vrt",
                   creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])
    os.remove(out_tif + ".vrt")
    return out_tif


def rgb_png(tif, png):
    ds = gdal.Open(tif)
    r = ds.GetRasterBand(3).ReadAsArray().astype("float32")   # B04
    g = ds.GetRasterBand(2).ReadAsArray().astype("float32")   # B03
    b = ds.GetRasterBand(1).ReadAsArray().astype("float32")   # B02
    rgb = np.stack([r, g, b], -1) / 10000.0
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    try:
        from PIL import Image
        Image.fromarray((rgb * 255).astype("uint8")).save(png)
    except Exception as e:
        print(f"  PNG 스킵({e})")


def main():
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, "_tmp"); os.makedirs(tmp, exist_ok=True)
    # AOI 중심 → lon/lat (검색용)
    from osgeo import osr
    s = osr.SpatialReference(); s.ImportFromEPSG(FIRE["epsg"])
    t = osr.SpatialReference(); t.ImportFromEPSG(4326); t.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(s, t)
    cx = (FIRE["bbox_utm"][0] + FIRE["bbox_utm"][2]) / 2
    cy = (FIRE["bbox_utm"][1] + FIRE["bbox_utm"][3]) / 2
    lon, lat, _ = ct.TransformPoint(cx, cy)
    print(f"AOI 중심 lon {lon:.3f} lat {lat:.3f}")

    print("[before] pre-fire 씬 검색"); sb = pick_scene(FIRE["pre"], lon, lat)
    print("[after]  post-fire 씬 검색"); sa = pick_scene(FIRE["post"], lon, lat)

    for tag, sc in (("before", sb), ("after", sa)):
        for gsd, bands in ((20, BANDS_20), (10, BANDS_10)):
            out = os.path.join(OUT, f"{tag}_{gsd:g}m.tif")
            print(f"  {tag} {gsd:g}m → {os.path.basename(out)}")
            stack(sc["assets"], bands, gsd, out, tmp)
        rgb_png(os.path.join(OUT, f"{tag}_20m.tif"), os.path.join(OUT, f"{tag}_rgb.png"))

    # GT (20m 격자) — gt_stac.npz 의 gt 를 안동 bbox 20m 로 기록
    cird = os.path.join(os.path.dirname(__file__), "..", "..", "산불데이터",
                        "국토정보플랫폼 데이터셋", "CIR_실측라벨", "fires", "250324_안동길안면", "gt_stac.npz")
    if os.path.exists(cird):
        z = np.load(cird, allow_pickle=True); gt = z["gt"].astype("uint8")
        ref = gdal.Open(os.path.join(OUT, "after_20m.tif"))
        drv = gdal.GetDriverByName("GTiff")
        gp = os.path.join(OUT, "gt_20m.tif")
        if gt.shape == (ref.RasterYSize, ref.RasterXSize):
            o = drv.Create(gp, ref.RasterXSize, ref.RasterYSize, 1, gdal.GDT_Byte,
                           ["COMPRESS=DEFLATE"])
            o.SetGeoTransform(ref.GetGeoTransform()); o.SetProjection(ref.GetProjection())
            o.GetRasterBand(1).WriteArray(gt); o = None
            print(f"  GT → gt_20m.tif (소각 {int(gt.sum())}px)")
        else:
            print(f"  ⚠️ GT shape {gt.shape} ≠ after_20m {(ref.RasterYSize, ref.RasterXSize)} — GT 스킵")

    # 정리
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print(f"\n완료 → {OUT}")
    for f in sorted(os.listdir(OUT)):
        print("  ", f, f"{os.path.getsize(os.path.join(OUT,f))//1024} KB")


if __name__ == "__main__":
    main()
