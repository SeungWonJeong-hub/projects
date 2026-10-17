"""teacher 라벨 래스터화 — during 씬 그리드에 같은 날 FIRMS 활성화재 footprint를 굽는다.

acquire_during.py 가 받은 during 씬(광학 입력)마다, 그 씬의 촬영일(±DATE_PAD)에
관측된 FIRMS 활성화재 탐지점을 씬 CRS(UTM)로 재투영해 센서 footprint 원반으로
래스터화 → active-fire 마스크(teacher label). 결과 = distill 학습셋의 (입력=optical
6밴드, 라벨=teacher 화재마스크) 쌍.

daily 집계 json 에는 개별 점 좌표가 없으므로 FIRMS Area API 를 during 창으로 재질의
(firms_teacher.fetch 재사용). 소스별 픽셀크기: VIIRS 375 m, MODIS 1 km.

인터프리터: /usr/bin/python3 (gdal 3.4.1 · numpy · scipy · pyproj · requests)

env:
  EVENTS=all | camp_2018_paradise_us,...     매니페스트 중 처리할 이벤트(기본 all)
  DATE_PAD=0                                  씬 날짜 ± 며칠의 FIRMS 점을 포함
  VIIRS_M=200  MODIS_M=560                    footprint 반경(m) — 픽셀 반폭 근사
  FORCE=1                                     기존 마스크 덮어쓰기
출력:
  data/s2_during/<event>/<scene>/FIRMS_MASK.tif   (uint8 0/1, 밴드와 동일 그리드)
  data/teacher_labels_summary.json                (씬별 n_points·fire_px·fire_frac)
"""
import os, sys, json
import datetime as dt
import numpy as np
from osgeo import gdal, osr
from pyproj import Transformer
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import firms_teacher as ft  # fetch / map_key / conf_ok 재사용

MANIFEST = os.path.join(ROOT, "data", "s2_during_manifest.json")
OUTDIR   = os.path.join(ROOT, "data", "s2_during")
SUMMARY  = os.path.join(ROOT, "data", "teacher_labels_summary.json")

DATE_PAD = int(os.environ.get("DATE_PAD", "0"))
VIIRS_M  = float(os.environ.get("VIIRS_M", "200"))   # 375 m 픽셀 → 반경≈187, 여유 200
MODIS_M  = float(os.environ.get("MODIS_M", "560"))   # 1 km 픽셀 → 반경≈500, 여유 560
MASK_NAME = os.environ.get("MASK_NAME", "FIRMS_MASK.tif")  # ② footprint sweep용 출력명
FORCE    = os.environ.get("FORCE", "") == "1"


def disk(radius_px):
    r = int(round(radius_px))
    if r < 1:
        return np.ones((1, 1), bool)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def rasterize_scene(scene_dir, scene_date, pts):
    """pts=[(lat,lon,date,frp,src)] → 씬 그리드에 footprint 원반 래스터화."""
    band = os.path.join(scene_dir, "B12.tif")
    if not os.path.exists(band):
        return None
    ds = gdal.Open(band)
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    proj = ds.GetProjection()
    srs = osr.SpatialReference(); srs.ImportFromWkt(proj)
    ds = None

    px = abs(gt[1])
    lo = (dt.date.fromisoformat(scene_date) - dt.timedelta(days=DATE_PAD)).isoformat()
    hi = (dt.date.fromisoformat(scene_date) + dt.timedelta(days=DATE_PAD)).isoformat()

    # WGS84 → 씬 CRS (pyproj 는 EPSG 코드로, 없으면 WKT 로)
    src_crs = "EPSG:4326"
    dst = srs.GetAuthorityCode(None)
    dst_crs = f"EPSG:{dst}" if dst else proj
    tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    centers = {"viirs": [], "modis": []}   # (row, col)
    n_in = 0
    for lat, lon, d, frp, src in pts:
        if not (lo <= d <= hi):
            continue
        X, Y = tf.transform(lon, lat)
        col = (X - gt[0]) / gt[1]
        row = (Y - gt[3]) / gt[5]
        if not (0 <= col < W and 0 <= row < H):
            continue
        key = "modis" if src.startswith("MODIS") else "viirs"
        centers[key].append((int(row), int(col)))
        n_in += 1

    mask = np.zeros((H, W), bool)
    for key, radius_m in (("viirs", VIIRS_M), ("modis", MODIS_M)):
        cs = centers[key]
        if not cs:
            continue
        seed = np.zeros((H, W), bool)
        rr = np.array([r for r, _ in cs]); cc = np.array([c for _, c in cs])
        seed[rr, cc] = True
        mask |= ndimage.binary_dilation(seed, structure=disk(radius_m / px))

    # SCL 로 무효(무자료) 픽셀 확인용은 학습단계에서 처리 — 여기선 순수 teacher 마스크
    out = os.path.join(scene_dir, MASK_NAME)
    drv = gdal.GetDriverByName("GTiff")
    dst_ds = drv.Create(out, W, H, 1, gdal.GDT_Byte,
                        options=["COMPRESS=DEFLATE", "TILED=YES"])
    dst_ds.SetGeoTransform(gt); dst_ds.SetProjection(proj)
    dst_ds.GetRasterBand(1).WriteArray(mask.astype(np.uint8))
    dst_ds.GetRasterBand(1).SetNoDataValue(255)
    dst_ds = None
    return {"n_points_in_grid": n_in, "fire_px": int(mask.sum()),
            "fire_frac": round(float(mask.mean()), 6), "path": out}


def main():
    man = json.load(open(MANIFEST, encoding="utf-8"))
    want = os.environ.get("EVENTS", "all")
    ids = list(man) if want == "all" else [i.strip() for i in want.split(",") if i.strip()]
    key = ft.map_key()
    summary = {}
    if os.path.exists(SUMMARY):
        try:
            summary = json.load(open(SUMMARY, encoding="utf-8"))
        except Exception:
            pass

    print(f"teacher 라벨 래스터화: {len(ids)}개 이벤트 "
          f"(date_pad={DATE_PAD}, viirs={VIIRS_M}m, modis={MODIS_M}m)", flush=True)

    for i, eid in enumerate(ids, 1):
        ev = man.get(eid)
        if not ev:
            print(f"[{i}/{len(ids)}] {eid} — 매니페스트에 없음", flush=True); continue
        aoi = ev["aoi"]                                   # [minlon,minlat,maxlon,maxlat]
        during = [s for s in ev["scenes"] if s.get("kind") == "during"]
        if not during:
            print(f"[{i}/{len(ids)}] {eid} — during 씬 없음", flush=True); continue
        dates = sorted({s["date"] for s in during})
        d0 = dt.date.fromisoformat(dates[0]) - dt.timedelta(days=DATE_PAD)
        d1 = dt.date.fromisoformat(dates[-1]) + dt.timedelta(days=DATE_PAD)
        aoi_str = f"{aoi[0]:.3f},{aoi[1]:.3f},{aoi[2]:.3f},{aoi[3]:.3f}"
        print(f"[{i}/{len(ids)}] {eid}  씬날짜 {dates}  FIRMS 재질의 {d0}~{d1}", flush=True)
        pts = ft.fetch(key, aoi_str, d0, d1, lambda m: print(m, flush=True))
        print(f"    FIRMS {len(pts)}점 수신", flush=True)

        for s in during:
            scene_dir = os.path.join(OUTDIR, eid, s["id"])
            out = os.path.join(scene_dir, MASK_NAME)
            if os.path.exists(out) and not FORCE:
                print(f"    {s['id']} — 마스크 존재, skip", flush=True); continue
            r = rasterize_scene(scene_dir, s["date"], pts)
            if r is None:
                print(f"    {s['id']} — B12 없음, skip", flush=True); continue
            summary.setdefault(eid, {})[s["id"]] = {"date": s["date"], **r}
            print(f"    {s['id']} {s['date']} → 점 {r['n_points_in_grid']} · "
                  f"fire_px {r['fire_px']} ({r['fire_frac']*100:.3f}%)", flush=True)
            json.dump(summary, open(SUMMARY, "w"), ensure_ascii=False, indent=1)

    print(f"\n완료 → {SUMMARY}\n"
          f"마스크: data/s2_during/<event>/<scene>/FIRMS_MASK.tif\n"
          f"다음: optical 6밴드 + FIRMS_MASK 로 during-fire student distill 학습", flush=True)


if __name__ == "__main__":
    main()
