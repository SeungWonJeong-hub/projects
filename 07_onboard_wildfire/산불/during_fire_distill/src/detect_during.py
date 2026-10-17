"""온보드 during-fire 실시간 탐지 실행부 — 위성(Jetson)에서 도는 추론 경로.

학습된 student(광학 6밴드→FIRMS 활성화재 마스크)를 during 씬 전체에 슬라이딩윈도로
추론 → 활성화재 확률/마스크 → 벡터화(폴리곤+면적) + 지연·다운링크용량 산출.
[[unified-cd-pipeline-spec]] "온보드 seg-CD → 벡터화·압축 → 다운링크" 골격의 during-fire판.

컴퓨트 분담(2026-09-16): 이 실시간 탐지 추론은 **Jetson(온보드)** 역할. 결과 활용·
상황판은 B200. → device-agnostic이되 온보드 검증은 Jetson GPU에서.

단일 인터프리터: /usr/bin/python3 (torch 2.11 cuda · gdal 3.4.1 · numpy)

사용:  /usr/bin/python3 src/detect_during.py <scene_dir> [ckpt]
env:
  CKPT=runs/student/best.pt   기본 모델
  TILE=256 OVERLAP=64         슬라이딩윈도(겹침 평균으로 이음매 제거)
  THR=0.5                     이진화 임계
  MIN_AREA_HA=1.0             폴리곤 최소면적(온보드 잡음 제거)
출력(씬 디렉토리 옆 detect/):
  <scene>_prob.tif  <scene>_mask.tif  <scene>_fire.geojson  <scene>_detect.json
"""
import os, sys, json, time
import numpy as np
from osgeo import gdal, ogr, osr
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_student import UNet, BANDS  # 모델·밴드정의 재사용

TILE    = int(os.environ.get("TILE", 256))
OVERLAP = int(os.environ.get("OVERLAP", 64))
THR     = float(os.environ.get("THR", 0.5))
MIN_AREA_HA = float(os.environ.get("MIN_AREA_HA", 1.0))
DEV     = "cuda" if torch.cuda.is_available() else "cpu"


def load_bands(scene_dir):
    """6밴드 스택 + geo. refl/1e4, all-zero=무자료 valid 마스크."""
    arrs = []
    ref = None
    for b in BANDS:
        p = os.path.join(scene_dir, f"{b}.tif")
        if not os.path.exists(p):
            raise SystemExit(f"밴드 없음: {p}")
        ds = gdal.Open(p)
        if ref is None:
            ref = {"gt": ds.GetGeoTransform(), "proj": ds.GetProjection(),
                   "W": ds.RasterXSize, "H": ds.RasterYSize}
        arrs.append(ds.ReadAsArray().astype(np.float32) / 1e4)
    H = min(a.shape[0] for a in arrs); W = min(a.shape[1] for a in arrs)
    x = np.clip(np.stack([a[:H, :W] for a in arrs], 0), 0, 1)
    valid = (x.sum(0) > 0)
    ref["H"], ref["W"] = H, W
    return x, valid, ref


@torch.no_grad()
def infer(model, x):
    """슬라이딩윈도 추론 → 전 씬 확률맵(겹침 평균). 타일 지연도 계측."""
    C, H, W = x.shape
    step = TILE - OVERLAP
    prob = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    rows = list(range(0, max(1, H - TILE + 1), step)) + ([H - TILE] if H > TILE else [0])
    cols = list(range(0, max(1, W - TILE + 1), step)) + ([W - TILE] if W > TILE else [0])
    rows = sorted(set(max(0, r) for r in rows)); cols = sorted(set(max(0, c) for c in cols))
    n_tiles = 0; t_infer = 0.0
    for r0 in rows:
        for c0 in cols:
            r1 = min(r0 + TILE, H); c1 = min(c0 + TILE, W)
            r0c, c0c = r1 - TILE if r1 - r0 < TILE else r0, c1 - TILE if c1 - c0 < TILE else c0
            r0c, c0c = max(0, r0c), max(0, c0c)
            tile = x[:, r0c:r0c + TILE, c0c:c0c + TILE]
            th, tw = tile.shape[1], tile.shape[2]
            t = torch.from_numpy(tile[None]).to(DEV)
            if DEV == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            p = torch.sigmoid(model(t))[0, 0]
            if DEV == "cuda":
                torch.cuda.synchronize()
            t_infer += time.perf_counter() - t0; n_tiles += 1
            p = p.cpu().numpy()[:th, :tw]
            prob[r0c:r0c + th, c0c:c0c + tw] += p
            cnt[r0c:r0c + th, c0c:c0c + tw] += 1
    prob /= np.maximum(cnt, 1)
    return prob, n_tiles, t_infer


def save_tif(path, arr, ref, dtype, nodata=None):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, ref["W"], ref["H"], 1, dtype,
                    options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(ref["gt"]); ds.SetProjection(ref["proj"])
    ds.GetRasterBand(1).WriteArray(arr)
    if nodata is not None:
        ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds = None


def vectorize(mask_path, geojson_path, ref, min_area_ha):
    """마스크 → 폴리곤(GeoJSON, WGS84) + 면적. 온보드 다운링크용 압축표현."""
    px_m = abs(ref["gt"][1]); px_area_ha = (px_m * px_m) / 1e4
    src = gdal.Open(mask_path)
    band = src.GetRasterBand(1)
    srs = osr.SpatialReference(); srs.ImportFromWkt(ref["proj"])
    mem = ogr.GetDriverByName("Memory").CreateDataSource("m")
    lyr = mem.CreateLayer("fire", srs=srs, geom_type=ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("val", ogr.OFTInteger))
    gdal.Polygonize(band, band, lyr, 0, [], callback=None)  # band 자신을 마스크로(1만)
    # WGS84 변환기
    wgs = osr.SpatialReference(); wgs.ImportFromEPSG(4326)
    wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_wgs = osr.CoordinateTransformation(srs, wgs)
    out = ogr.GetDriverByName("GeoJSON")
    if os.path.exists(geojson_path):
        os.remove(geojson_path)
    ods = out.CreateDataSource(geojson_path)
    olyr = ods.CreateLayer("fire", srs=wgs, geom_type=ogr.wkbPolygon)
    olyr.CreateField(ogr.FieldDefn("area_ha", ogr.OFTReal))
    n = 0; total_ha = 0.0
    for feat in lyr:
        if feat.GetField("val") != 1:
            continue
        g = feat.GetGeometryRef()
        area_ha = g.GetArea() / 1e4                      # UTM m² → ha
        if area_ha < min_area_ha:
            continue
        gg = g.Clone(); gg.Transform(to_wgs)
        of = ogr.Feature(olyr.GetLayerDefn())
        of.SetField("area_ha", round(area_ha, 3)); of.SetGeometry(gg)
        olyr.CreateFeature(of); of = None
        n += 1; total_ha += area_ha
    ods = None; src = None
    return n, total_ha


def main():
    if len(sys.argv) < 2:
        sys.exit("사용: detect_during.py <scene_dir> [ckpt]")
    scene_dir = sys.argv[1].rstrip("/")
    ckpt = sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
        "CKPT", os.path.join(os.path.dirname(HERE), "runs", "student", "best.pt"))
    scene = os.path.basename(scene_dir)
    outdir = os.path.join(os.path.dirname(scene_dir), "detect"); os.makedirs(outdir, exist_ok=True)

    t_load0 = time.perf_counter()
    x, valid, ref = load_bands(scene_dir)
    t_load = time.perf_counter() - t_load0

    model = UNet().to(DEV).eval()
    st = torch.load(ckpt, map_location=DEV)
    model.load_state_dict(st["model"])
    print(f"device={DEV}  scene={scene}  {ref['W']}x{ref['H']}  ckpt_f1={st.get('f1')}", flush=True)

    prob, n_tiles, t_infer = infer(model, x)
    prob[~valid] = 0.0
    mask = ((prob > THR) & valid).astype(np.uint8)

    prob_p = os.path.join(outdir, f"{scene}_prob.tif")
    mask_p = os.path.join(outdir, f"{scene}_mask.tif")
    gj_p   = os.path.join(outdir, f"{scene}_fire.geojson")
    save_tif(prob_p, (prob * 255).astype(np.uint8), ref, gdal.GDT_Byte, nodata=None)
    save_tif(mask_p, mask, ref, gdal.GDT_Byte, nodata=None)
    n_poly, total_ha = vectorize(mask_p, gj_p, ref, MIN_AREA_HA)

    import gzip as _gz
    raster_bytes = os.path.getsize(mask_p)                 # DEFLATE-압축 래스터 마스크
    vec_bytes = os.path.getsize(gj_p)                      # raw GeoJSON(장황한 텍스트)
    vec_gz_bytes = len(_gz.compress(open(gj_p, "rb").read(), 9))  # 공정비교용 gzip
    dl_min = min(raster_bytes, vec_gz_bytes)
    fire_px = int(mask.sum())
    px_m = abs(ref["gt"][1])
    rep = {
        "scene": scene, "device": DEV, "grid": [ref["W"], ref["H"]],
        "gsd_m": px_m, "threshold": THR,
        "fire_px": fire_px, "fire_area_ha": round(fire_px * (px_m * px_m) / 1e4, 2),
        "n_polygons": n_poly, "vector_area_ha": round(total_ha, 2),
        "latency": {"load_s": round(t_load, 3), "infer_s": round(t_infer, 3),
                    "tiles": n_tiles, "ms_per_tile": round(1e3 * t_infer / max(1, n_tiles), 2),
                    "mpix_per_s": round(ref["W"] * ref["H"] / 1e6 / max(1e-6, t_infer), 2)},
        "downlink": {"raster_mask_deflate_bytes": raster_bytes,
                     "vector_geojson_raw_bytes": vec_bytes,
                     "vector_geojson_gzip_bytes": vec_gz_bytes,
                     "smallest": "raster" if raster_bytes <= vec_gz_bytes else "vector_gz",
                     "min_bytes": dl_min,
                     "vector_vs_raster_x": round(vec_gz_bytes / max(1, raster_bytes), 2)},
        "outputs": {"prob": prob_p, "mask": mask_p, "geojson": gj_p},
    }
    json.dump(rep, open(os.path.join(outdir, f"{scene}_detect.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"  화재 {rep['fire_area_ha']} ha · 폴리곤 {n_poly}개 · "
          f"추론 {t_infer:.2f}s({rep['latency']['mpix_per_s']} Mpix/s, "
          f"{rep['latency']['ms_per_tile']} ms/tile) · "
          f"다운링크 최소 {dl_min//1024}KB({rep['downlink']['smallest']}): "
          f"래스터(DEFLATE) {raster_bytes//1024}KB vs 벡터(gzip) {vec_gz_bytes//1024}KB "
          f"= 벡터가 {rep['downlink']['vector_vs_raster_x']}× 크기", flush=True)
    print(f"  → {outdir}/{scene}_*.{{tif,geojson,json}}", flush=True)


if __name__ == "__main__":
    main()
