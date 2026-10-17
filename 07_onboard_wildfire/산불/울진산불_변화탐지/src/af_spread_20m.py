"""실시간 산불 퍼짐 — 열적외 FIRMS를 S2 20m 격자에 정합 렌더.
역할분담(확정): 실시간 퍼짐=열적외(FIRMS/GK-2A), 정밀=S2 20m. 여기선 둘을 20m GSD로 결합.

  · FIRMS 활성화재점(VIIRS 375m/MODIS 1km)을 S2 20m 격자에 풋프린트(≈375m disk)로 rasterize
  · acq_date별 binning → 픽셀별 '최초 탐지일'(first-detection day) = 퍼짐 전파 가시화
  · 배경 = S2 20m RGB (정밀 맥락).  출력 격자 = S2 20m (프로젝트 표준 정합)

출력: active_fire/<site>/spread_20m.png  +  active_fire/<site>/spread_20m.tif(최초탐지일, 20m)
      active_fire/spread_20m.json (일별 신규/누적 면적 20m 기준)
사용: python af_spread_20m.py
"""
import os, csv, json, datetime as dt
import numpy as np
from osgeo import gdal, osr
from scipy import ndimage as ndi
gdal.UseExceptions()
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm, colors

HERE = os.path.dirname(os.path.abspath(__file__))
AF = os.path.join(HERE, "..", "active_fire")
FIRMS_CSV = os.path.join(HERE, "..", "..", "산불데이터", "firms_korea_bbox_2022_2025.csv")
OFF = 1000.0
FOOT_M = 187.5           # VIIRS 375m 풋프린트 반경(≈9px @20m)
# 화재별 기준격자(첫 날짜 마스크 tif) + FIRMS 기간
FIRES = {
    "uljin":     dict(ref="2022-03-05", d0="2022-03-04", d1="2022-03-13"),
    "sancheong": dict(ref="2025-03-22", d0="2025-03-21", d1="2025-03-30"),
    "uiseong":   dict(ref="2025-03-24", d0="2025-03-22", d1="2025-04-01"),
}

def _r(p):
    ds = gdal.Open(p); a = ds.ReadAsArray().astype(np.float32); ds = None; return a

def load_firms():
    out = []
    for r in csv.DictReader(open(FIRMS_CSV, encoding="utf-8-sig")):
        try:
            if (r.get("confidence") or "").strip().lower() in ("l", "low"): continue
            out.append((float(r["latitude"]), float(r["longitude"]), r["acq_date"].strip()))
        except (KeyError, ValueError): continue
    return out

def render(site, cfg, firms):
    sd = os.path.join(AF, site, cfg["ref"])
    ref = gdal.Open(os.path.join(sd, "active_fire_mask.tif"))
    gt = ref.GetGeoTransform(); proj = ref.GetProjection(); H, W = ref.ReadAsArray().shape
    px = abs(gt[1]); foot_px = int(round(FOOT_M / px))
    dst = osr.SpatialReference(); dst.ImportFromWkt(proj)
    src = osr.SpatialReference(); src.ImportFromEPSG(4326); src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    d0 = dt.date.fromisoformat(cfg["d0"]); d1 = dt.date.fromisoformat(cfg["d1"])
    days = [(d0 + dt.timedelta(n)) for n in range((d1 - d0).days + 1)]
    day_idx = {d: i for i, d in enumerate(days)}
    firstday = np.full((H, W), -1, np.int16)   # 최초 탐지일 인덱스
    perday = []
    ys, xs = np.ogrid[-foot_px:foot_px + 1, -foot_px:foot_px + 1]
    disk = (xs * xs + ys * ys) <= foot_px * foot_px
    for d in days:
        layer = np.zeros((H, W), bool)
        for la, lo, ad in firms:
            try: fd = dt.date.fromisoformat(ad)
            except ValueError: continue
            if fd != d: continue
            x, y, _ = ct.TransformPoint(lo, la); c = int((x - gt[0]) / gt[1]); rr = int((y - gt[3]) / gt[5])
            if not (0 <= rr < H and 0 <= c < W): continue
            r0, r1 = max(0, rr - foot_px), min(H, rr + foot_px + 1)
            c0, c1 = max(0, c - foot_px), min(W, c + foot_px + 1)
            dr0, dr1 = r0 - (rr - foot_px), disk.shape[0] - ((rr + foot_px + 1) - r1)
            dc0, dc1 = c0 - (c - foot_px), disk.shape[1] - ((c + foot_px + 1) - c1)
            layer[r0:r1, c0:c1] |= disk[dr0:dr1, dc0:dc1]
        i = day_idx[d]
        new = layer & (firstday < 0)
        firstday[new] = i
        cum = (firstday >= 0)
        perday.append(dict(date=str(d), new_ha=round(int(new.sum()) * (px*px)/1e4, 1),
                           cum_ha=round(int(cum.sum()) * (px*px)/1e4, 1)))
    # GeoTIFF (최초탐지일, 20m)
    drv = gdal.GetDriverByName("GTiff")
    ods = drv.Create(os.path.join(AF, site, "spread_20m.tif"), W, H, 1, gdal.GDT_Int16,
                     ["COMPRESS=DEFLATE"])
    ods.SetGeoTransform(gt); ods.SetProjection(proj); ods.GetRasterBand(1).WriteArray(firstday)
    ods.GetRasterBand(1).SetNoDataValue(-1); ods = None
    # 시각화: S2 20m RGB 배경 + 최초탐지일 컬러
    rgb = np.dstack([np.clip((_r(os.path.join(sd, b)) - OFF) / 3500, 0, 1) for b in ["B04.tif", "B03.tif", "B02.tif"]])
    fig, ax = plt.subplots(figsize=(10, 10 * H / W))
    ax.imshow(rgb)
    fm = np.ma.masked_where(firstday < 0, firstday)
    im = ax.imshow(fm, cmap="autumn_r", alpha=0.75, vmin=0, vmax=len(days) - 1)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=range(0, len(days), max(1, len(days)//6)))
    cb.ax.set_yticklabels([days[t].strftime("%m-%d") for t in range(0, len(days), max(1, len(days)//6))])
    cb.set_label("최초 탐지일 (퍼짐 전파)")
    ax.set_title(f"{site} — 실시간 산불 퍼짐 (열적외 FIRMS → S2 20m 격자)\n"
                 f"누적 {perday[-1]['cum_ha']:.0f} ha, 20m GSD", fontsize=11)
    ax.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(AF, site, "spread_20m.png"), dpi=95, bbox_inches="tight"); plt.close()
    return perday, (H, W, px)

def main():
    firms = load_firms()
    out = {}
    print(f"FIRMS {len(firms)}점 로드. 20m 격자 정합 렌더...\n")
    for site, cfg in FIRES.items():
        perday, (H, W, px) = render(site, cfg, firms)
        out[site] = dict(grid=f"{W}x{H} @ {px:.0f}m GSD", perday=perday,
                         total_ha=perday[-1]["cum_ha"])
        print(f"== {site} ({W}x{H}, {px:.0f}m GSD) ==")
        for r in perday:
            if r["new_ha"] > 0:
                print(f"  {r['date']}: 신규 {r['new_ha']:7.1f} ha  누적 {r['cum_ha']:7.1f} ha")
        print(f"  → 총 {perday[-1]['cum_ha']:.0f} ha\n")
    json.dump(out, open(os.path.join(AF, "spread_20m.json"), "w"), ensure_ascii=False, indent=1)
    print("saved active_fire/<site>/spread_20m.{png,tif} + spread_20m.json")

if __name__ == "__main__":
    main()
