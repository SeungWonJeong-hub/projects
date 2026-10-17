"""PRE 다중합성 — 전년(2021) 봄 개엽기 여러 씬을 픽셀별 중앙값 합성.

동기: 단일 pre(2021-04-19)는 대기·잔여구름·센서 노이즈가 dNBR에 그대로 실림.
여러 leaf-on 씬을 구름마스킹 후 median 합성 → 노이즈↓, dNBR 안정화 → 울진 성능 개선 시도.

- 대상: 52SEG 타일(AOI 커버), 2021-04-01~05-25 개엽기, cloud<25%.
- 씬별 마스킹: SCL 드롭 + 단순 밝은구름(blue 반사율>0.20) — 단일씬이라 mono.
- 합성: 각 밴드 픽셀별로 valid 씬만 모아 nanmedian; SCL은 합성결과에 대해
  valid≥1 → 4(vegetation, 청천), valid=0 → 9(cloud) 로 기록해 다운스트림 valid 로직 호환.
- post(2022-04-09)와 개엽기 맞춘 leaf-on baseline.  다운스트림은 build_dataset.py(MT_CLOUD=1).

출력: data/raw/PRE_COMPOSITE_2021spring/{B02,B03,B04,B08,B11,B12,SCL}.tif (단일 pre와 동일 포맷·그리드).
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stac import stac_search, download_band, AOI_BBOX, WANT_BANDS, WANT_MASK

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(RAW, "PRE_COMPOSITE_2021spring")
TMP = os.path.join(RAW, "_pre_tmp")
WIN = "2021-04-01T00:00:00Z/2021-04-24T23:59:59Z"  # post(4/9) 개엽기와 일치하는 4월만 (5월=고식생 → dNBR 반전)
TILE = "52SEG"
MAX_CLOUD = 25
SCL_DROP = (0, 1, 3, 8, 9, 10, 11)
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def pick_scenes():
    feats = stac_search(AOI_BBOX, WIN, max_cloud=MAX_CLOUD, limit=50)
    sel = []
    for f in feats:
        p = f["properties"]
        grid = (p.get("grid:code") or p.get("s2:mgrs_tile") or "").replace("MGRS-", "")
        if TILE in grid:
            sel.append(f)
    # 씬id에서 날짜 중복(같은 날 여러 처리본) 제거 — 날짜당 최저구름 1개
    by_date = {}
    for f in sel:
        d = f["properties"]["datetime"][:10]
        if d not in by_date or f["properties"]["eo:cloud_cover"] < by_date[d]["properties"]["eo:cloud_cover"]:
            by_date[d] = f
    return list(by_date.values())


def load_scene_dir(d):
    bd = {}
    for b in BANDS + ["SCL"]:
        bd[b] = gdal.Open(os.path.join(d, f"{b}.tif")).ReadAsArray().astype(np.float32)
    return bd


def mono_cloud_bad(bd):
    """단일씬 구름 bad: SCL 드롭 ∪ 밝은구름(blue>0.25).
    그림자룰 제외 — 바다/어두운 육상을 과제거해 커버리지 반토막 → SCL(class3=구름그림자)에 위임.
    합성은 여러 씬 median이라 잔여 오염은 통계적으로 흡수됨."""
    scl = np.isin(bd["SCL"].astype(np.int16), SCL_DROP)
    bright = (bd["B02"] / 1e4) > 0.25
    return scl | bright


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    feats = pick_scenes()
    print(f"합성 대상 {len(feats)}개 씬 ({TILE}, {WIN[:10]}~):", flush=True)
    dirs = []
    ref_geo = None
    for f in sorted(feats, key=lambda x: x["properties"]["datetime"]):
        sid = f["id"]; dt = f["properties"]["datetime"][:10]; cc = f["properties"]["eo:cloud_cover"]
        sdir = os.path.join(TMP, sid); os.makedirs(sdir, exist_ok=True)
        print(f"  - {dt} cloud {cc:.1f}%  ({sid})", flush=True)
        ok = True
        for key, name in {**WANT_BANDS, **WANT_MASK}.items():
            dest = os.path.join(sdir, f"{name}.tif")
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                continue
            href = f["assets"].get(key, {}).get("href")
            if not href:
                ok = False; break
            alg = "nearest" if name == "SCL" else "average"
            if download_band(href, dest, resample=alg) == 0:
                ok = False; break
        if ok:
            dirs.append(sdir)
            if ref_geo is None:
                ds = gdal.Open(os.path.join(sdir, "B02.tif"))
                ref_geo = (ds.GetGeoTransform(), ds.GetProjection(), ds.RasterXSize, ds.RasterYSize)

    if len(dirs) < 2:
        print(f"[중단] 합성 가능한 씬 {len(dirs)}개 (<2)"); return
    gt, proj, W, H = ref_geo
    print(f"합성 {len(dirs)}씬 → 그리드 {H}x{W}", flush=True)

    # 밴드별 픽셀 중앙값(valid 씬만)
    stacks = {b: [] for b in BANDS}
    nvalid = np.zeros((H, W), np.int16)
    for d in dirs:
        bd = load_scene_dir(d)
        if bd["B02"].shape != (H, W):  # 그리드 불일치 방어(리샘플로 동일하지만 안전)
            print(f"  ! 그리드 불일치 스킵 {os.path.basename(d)} {bd['B02'].shape}"); continue
        bad = mono_cloud_bad(bd)
        good = ~bad & (bd["B02"] > 0)
        nvalid += good.astype(np.int16)
        for b in BANDS:
            v = bd[b].copy(); v[~good] = np.nan; stacks[b].append(v)

    comp = {}
    for b in BANDS:
        arr = np.stack(stacks[b], 0)
        with np.errstate(all="ignore"):
            med = np.nanmedian(arr, axis=0)
        med[np.isnan(med)] = 0
        comp[b] = med.astype(np.float32)
    # 합성 SCL: valid≥1 → 4(청천 vegetation), valid=0 → 9(cloud)
    scl = np.where(nvalid >= 1, 4, 9).astype(np.float32)

    def write(name, arr):
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(os.path.join(OUT, f"{name}.tif"), W, H, 1, gdal.GDT_Float32,
                        ["COMPRESS=DEFLATE", "TILED=YES"])
        ds.SetGeoTransform(gt); ds.SetProjection(proj); ds.GetRasterBand(1).WriteArray(arr); ds = None
    for b in BANDS:
        write(b, comp[b])
    write("SCL", scl)
    json.dump({"id": "PRE_COMPOSITE_2021spring", "n_scenes": len(dirs),
               "dates": [os.path.basename(d).split("_")[2] for d in dirs],
               "tile": TILE, "valid_pct": float(100 * (nvalid >= 1).mean()),
               "mean_obs_per_px": float(nvalid[nvalid > 0].mean())},
              open(os.path.join(OUT, "metadata.json"), "w"), indent=2, ensure_ascii=False)
    print(f"저장: {OUT}  valid {100*(nvalid>=1).mean():.2f}%  평균관측 {nvalid[nvalid>0].mean():.2f}씬/px", flush=True)


if __name__ == "__main__":
    main()
