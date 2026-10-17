"""NGII 국토정보플랫폼 화재 이벤트 번들(CIR pre/post + 수치지도 벡터) → 지오레퍼런싱된 소각 GT.

국토정보플랫폼 산불 이벤트 zip은 다음을 담는다(안동 풍천 2020-04-24 실측):
  - CompResult_SEN_MMDD_CIR_s.tif  (Sentinel-2 10m CIR 위색, 8-bit 3밴드, 좌표 없음)
      * CIR 밴드순서: band1=NIR, band2=Red, band3=Green
      * 파일명 날짜(MMDD) 2개 = 화재 전/후 브래킷 쌍
  - (하천/도로/농경지레이어)N3A_*.shp  (EPSG:5179 Korea2000, 화재 AOI로 클립)
      * 세 레이어 extent 동일 = CIR 이미지의 실제 지상범위 → 이걸로 CIR 지오레퍼런싱

방법: 벡터 extent를 CIR 픽셀격자에 매핑(geotransform) → NDVI=(NIR-Red)/(NIR+Red)의
pre/post 차분(dNDVI)으로 소각 후보 → 정리 → EPSG:5179 폴리곤·래스터 저장.

한계(정직): 8-bit 스트레치라 절대반사율/NBR 불가(상대 NDVI만), SWIR 없음.
소각 가시성은 pre/post 15일 브래킷이라 봄에도 유효(개엽 변화 최소). 자동결과는 "초벌 GT"—
QGIS 손보정 전제. Sentinel 10m = 우리 모델 S2격자와 동급이라 rasterize 정합 깔끔.

사용: python ngii_cir_gt.py <이벤트폴더> [--dthr -0.12] [--postmax 0.35] [--minha 10]
산출: <폴더>/burn_<name>.shp, burn_mask_<name>.tif, *_geo.tif, qa_<name>.png
"""
import os, sys, glob, argparse
import numpy as np
from osgeo import gdal, ogr, osr
gdal.UseExceptions()
try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def find_inputs(folder):
    tifs = sorted(glob.glob(os.path.join(folder, "*CIR*_s.tif")) +
                  glob.glob(os.path.join(folder, "*CIR*.tif")))
    tifs = [t for t in tifs if "_geo" not in t and "mask" not in t.lower()]
    shps = glob.glob(os.path.join(folder, "*.shp"))
    shps = [s for s in shps if "burn" not in os.path.basename(s).lower()]
    if len(tifs) < 2:
        raise SystemExit(f"CIR tif 2장 필요, 발견 {len(tifs)}: {tifs}")
    if not shps:
        raise SystemExit("좌표 기준 벡터(.shp) 없음 → 지오레퍼런싱 불가")
    # 날짜(MMDD)로 pre/post 정렬
    def mmdd(p):
        import re
        m = re.search(r"_(\d{4})_CIR", os.path.basename(p))
        return m.group(1) if m else os.path.basename(p)
    tifs = sorted(tifs, key=mmdd)
    return tifs[0], tifs[-1], shps


def vector_extent(shps):
    """모든 벡터 공통 extent와 EPSG 반환(첫 유효 레이어 기준, 나머지는 검증)."""
    exts, srs_wkt = [], None
    for s in shps:
        ds = ogr.Open(s)
        if ds is None:
            continue
        ly = ds.GetLayer()
        x0, x1, y0, y1 = ly.GetExtent()
        exts.append((x0, y0, x1, y1))
        if srs_wkt is None and ly.GetSpatialRef():
            srs_wkt = ly.GetSpatialRef().ExportToWkt()
    if not exts:
        raise SystemExit("벡터 extent 추출 실패")
    # 교집합 아닌 합집합(동일해야 정상); 불일치 시 경고
    x0 = min(e[0] for e in exts); y0 = min(e[1] for e in exts)
    x1 = max(e[2] for e in exts); y1 = max(e[3] for e in exts)
    span = max(x1 - x0, y1 - y0)
    for e in exts:
        if abs(e[0]-x0) > 0.02*span or abs(e[2]-x1) > 0.02*span:
            print(f"  ⚠ 벡터 extent 불일치: {e} vs 통합 {(x0,y0,x1,y1)} — 정합 오차 가능")
            break
    return (x0, y0, x1, y1), srs_wkt


def ndvi(a):  # a=(3,H,W), band1=NIR band2=Red
    return (a[0] - a[1]) / (a[0] + a[1] + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--dthr", type=float, default=-0.12, help="dNDVI 소각 임계(이하)")
    ap.add_argument("--postmax", type=float, default=0.35, help="post NDVI 상한")
    ap.add_argument("--minha", type=float, default=10.0, help="최소 소각 덩어리(ha)")
    a = ap.parse_args()
    folder = a.folder.rstrip("/")
    name = os.path.basename(folder)

    pre_t, post_t, shps = find_inputs(folder)
    print(f"[{name}] pre={os.path.basename(pre_t)} post={os.path.basename(post_t)} 벡터 {len(shps)}개")
    (X0, Y0, X1, Y1), srs_wkt = vector_extent(shps)

    src = gdal.Open(post_t); W, H = src.RasterXSize, src.RasterYSize
    px = (X1 - X0) / W; py = (Y1 - Y0) / H
    gt = (X0, px, 0, Y1, 0, -py)
    print(f"  격자 {W}x{H}, 픽셀 {px:.3f}x{py:.3f} m (정사각 아니면 정합 의심)")
    srs = osr.SpatialReference()
    if srs_wkt:
        srs.ImportFromWkt(srs_wkt)
    else:
        srs.ImportFromEPSG(5179)

    pre = gdal.Open(pre_t).ReadAsArray().astype(np.float32)
    post = gdal.Open(post_t).ReadAsArray().astype(np.float32)
    dnd = ndvi(post) - ndvi(pre)
    burn = (dnd < a.dthr) & (ndvi(post) < a.postmax)

    if HAVE_SCIPY:
        burn = ndi.binary_opening(burn, iterations=2)
        burn = ndi.binary_closing(burn, iterations=3)
        burn = ndi.binary_fill_holes(burn)
        lab, n = ndi.label(burn)
        if n:
            sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            minpx = a.minha * 10000 / (px * py)
            keep = [i + 1 for i, s in enumerate(sizes) if s >= minpx]
            burn = np.isin(lab, keep)
    else:
        print("  ⚠ scipy 없음 → 형태학 정리 생략(노이즈 많을 수 있음)")
    area_ha = float(burn.sum()) * px * py / 10000
    print(f"  소각 {int(burn.sum())}px ~ {area_ha:.0f} ha")

    # 지오레퍼런싱 CIR + 마스크 저장
    drv = gdal.GetDriverByName("GTiff")
    for t, tag in [(pre_t, "pre"), (post_t, "post")]:
        arr = gdal.Open(t).ReadAsArray()
        ds = drv.Create(t.replace(".tif", "_geo.tif"), W, H, 3, gdal.GDT_Byte)
        ds.SetGeoTransform(gt); ds.SetProjection(srs.ExportToWkt())
        for b in range(3):
            ds.GetRasterBand(b + 1).WriteArray(arr[b])
        ds = None
    mpath = os.path.join(folder, f"burn_mask_{name}.tif")
    mds = drv.Create(mpath, W, H, 1, gdal.GDT_Byte)
    mds.SetGeoTransform(gt); mds.SetProjection(srs.ExportToWkt())
    mds.GetRasterBand(1).WriteArray(burn.astype(np.uint8))
    mb = mds.GetRasterBand(1)

    spath = os.path.join(folder, f"burn_{name}.shp")
    vdrv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(spath):
        vdrv.DeleteDataSource(spath)
    vds = vdrv.CreateDataSource(spath)
    vly = vds.CreateLayer("burn", srs=srs, geom_type=ogr.wkbPolygon)
    vly.CreateField(ogr.FieldDefn("val", ogr.OFTInteger))
    gdal.Polygonize(mb, mb, vly, 0, [])
    vds = None; mds = None
    print(f"  → {spath}")

    # QA 그림
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cir = np.dstack([post[0], post[1], post[2]]).astype(np.uint8)
    ext = [X0, X1, Y0, Y1]
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].imshow(cir, extent=ext); ax[0].set_title(f"{name} CIR post")
    ax[1].imshow(cir, extent=ext)
    ax[1].imshow(np.ma.masked_where(~burn, burn.astype(float)), extent=ext, cmap="autumn", alpha=0.45)
    ax[1].set_title(f"AUTO burn draft ~{area_ha:.0f} ha")
    qpath = os.path.join(folder, f"qa_{name}.png")
    plt.tight_layout(); plt.savefig(qpath, dpi=90, bbox_inches="tight")
    print(f"  → {qpath}")


if __name__ == "__main__":
    main()
