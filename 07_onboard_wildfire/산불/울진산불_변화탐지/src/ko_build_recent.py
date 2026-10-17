"""최신 대형 산불(2022~2026) S2 취득 + pseudo-label 생성.
MTfire.shp가 2021에서 끊겨 최신 대형화재가 없음. 산불통계.xlsx 목록에서 대형 13건을
읍면 지오코딩(Nominatim)→S2 취득→|dNBR| '공식 면적맞춤 임계'로 pseudo 경계 생성.
품질=pseudo(진짜 폴리곤 아님, 봄 반전은 |dNBR|로 완화). v7 포맷(공동정규화·중첩) 타일.

주의: 이건 실측 경계가 아니라 dNBR 유도 pseudo-label. 대형 봄산불 '다양성' 확보용.
"""
from __future__ import annotations
import os, sys, json, time, math, urllib.parse, urllib.request
import numpy as np
from osgeo import gdal, ogr, osr
from scipy import ndimage
gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datetime as dt
from ko_build_multi import stac_bands, s6_joint, rfc, ratio, idx, UTM
from mt_cloud import mt_cloud_shadow_bad

OUT = os.path.join(HERE, "..", "ko_recent")
T = 256; STRIDE = 128
SCL_DROP = [0, 1, 3, 8, 9, 10, 11]

# (지오코딩 질의, 발생일, 공식 피해면적 ha, 이름)
FIRES = [
    ("의성군 안계면", "2025-03-22", 52707, "rf_20250322_uiseong_ang"),
    ("의성군 안평면", "2025-03-22", 46575, "rf_20250322_uiseong_anp"),
    ("울진군 북면",   "2022-03-04", 16302, "rf_20220304_uljin"),
    ("강릉시 옥계면", "2022-03-05",  4190, "rf_20220305_gangneung"),
    ("산청군 시천면", "2025-03-21",  3398, "rf_20250321_sancheong"),
    ("홍성군 서부면", "2023-04-02",  1337, "rf_20230402_hongseong"),
    ("울주군 온양읍", "2025-03-22",  1117, "rf_20250322_ulju"),
    ("금산군 복수면", "2023-04-02",   889, "rf_20230402_geumsan"),
    ("합천군 율곡면", "2022-02-28",   814, "rf_20220228_hapcheon"),
    ("양구군 양구읍", "2022-04-10",   716, "rf_20220410_yanggu"),
    ("함평군 대동면", "2023-04-03",   682, "rf_20230403_hampyeong"),
    ("밀양시 부북면", "2022-05-31",   661, "rf_20220531_miryang"),
    ("영덕군 지품면", "2022-02-15",   406, "rf_20220215_yeongdeok"),
]


def geocode(q):
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": q, "format": "json", "limit": 1, "countrycodes": "kr"})
    req = urllib.request.Request(url, headers={"User-Agent": "wildfire-research/1.0"})
    for _ in range(3):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=25))
            if d: return float(d[0]["lat"]), float(d[0]["lon"])
        except Exception as e:
            print("  geocode retry:", e); time.sleep(2)
    return None


def to_utm(lon, lat):
    s = osr.SpatialReference(); s.ImportFromEPSG(4326); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    d = osr.SpatialReference(); d.ImportFromEPSG(UTM); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    x, y, _ = osr.CoordinateTransformation(s, d).TransformPoint(lon, lat)
    return x, y


def pseudo_label(dnbr, valid, target_px, cx_px, cy_px):
    """|dNBR| 상위를 공식면적(target_px)에 맞춰 임계 → 형태학 정리 → 중심부 큰 연결요소."""
    ev = np.abs(dnbr); ev[~valid] = -1
    flat = ev[valid]
    if flat.size < target_px * 0.5 or target_px < 50:
        thr = np.percentile(flat, 90) if flat.size else 1
    else:
        k = min(int(target_px), flat.size - 1)
        thr = np.partition(flat, -k)[-k]           # 상위 target_px번째 값
    m = (ev >= thr) & valid
    m = ndimage.binary_closing(m, iterations=2)
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n == 0: return m.astype(np.uint8)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    # 충분히 큰 성분만 유지(target의 3% 이상), 중심 근처 우선
    keep = np.zeros_like(m)
    for i in np.argsort(sizes)[::-1]:
        if sizes[i] < max(target_px * 0.03, 200): break
        keep |= (lab == i)
    keep = ndimage.binary_fill_holes(keep)
    return keep.astype(np.uint8)


def build_one(q, date_s, ha, name):
    ll = geocode(q)
    if not ll: print(f"[{name}] 지오코딩 실패 ({q})", flush=True); return 0
    lat, lon = ll; cx, cy = to_utm(lon, lat)
    r = math.sqrt(ha * 1e4 / math.pi)             # 원등가 반경(m)
    half = max(r * 1.6, 2650)                       # 여유 1.6x, 최소 5.3km
    ute = [math.floor(cx - half), math.floor(cy - half), math.ceil(cx + half), math.ceil(cy + half)]
    fdate = dt.date(int(date_s[:4]), int(date_s[5:7]), int(date_s[8:10]))
    # bbox lon/lat
    d = osr.SpatialReference(); d.ImportFromEPSG(4326); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    s = osr.SpatialReference(); s.ImportFromEPSG(UTM); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(s, d)
    lo1, la1, _ = ct.TransformPoint(ute[0], ute[1]); lo2, la2, _ = ct.TransformPoint(ute[2], ute[3])
    bbox = [min(lo1, lo2) - 0.02, min(la1, la2) - 0.02, max(lo1, lo2) + 0.02, max(la1, la2) + 0.02]
    pc = fdate.replace(year=fdate.year - 1)
    pre = stac_bands(bbox, rfc(pc - dt.timedelta(30), pc + dt.timedelta(45)), ute)
    post = stac_bands(bbox, rfc(fdate + dt.timedelta(15), fdate + dt.timedelta(100)), ute)
    if pre is None or post is None:
        print(f"[{name}] 장면없음 pre={bool(pre)} post={bool(post)}", flush=True); return 0
    H = min(pre['B02'].shape[0], post['B02'].shape[0]); W = min(pre['B02'].shape[1], post['B02'].shape[1])
    for bd in (pre, post):
        for k in bd: bd[k] = bd[k][:H, :W]
    valid = (pre['B02'] > 0) & (pre['B08'] > 0) & (pre['B12'] > 0) & (post['B02'] > 0) & (post['B08'] > 0) & (post['B12'] > 0)
    valid &= ~np.isin(pre['SCL'], SCL_DROP) & ~np.isin(post['SCL'], SCL_DROP)
    valid &= ~mt_cloud_shadow_bad(pre, post)
    dnbr = np.clip(ratio(pre['B08'], pre['B12']) - ratio(post['B08'], post['B12']), -1, 1)
    target_px = ha * 1e4 / (20 * 20)
    y = pseudo_label(dnbr, valid, target_px, W // 2, H // 2)
    pre6, post6 = s6_joint(pre, post, valid)
    x = np.concatenate([pre6, post6, idx(dnbr)[None]], 0).astype(np.float32)
    od = os.path.join(OUT, name); os.makedirs(od, exist_ok=True); n = 0
    if H < T or W < T:
        print(f"[{name}] 타일0 (크롭<{T})", flush=True); os.rmdir(od); return 0
    ys = sorted(set(min(yy, H - T) for yy in range(0, H - T + 1, STRIDE)))
    xs = sorted(set(min(xx, W - T) for xx in range(0, W - T + 1, STRIDE)))
    for yy in ys:
        for xx in xs:
            v = valid[yy:yy + T, xx:xx + T]
            if v.mean() < 0.4: continue
            np.savez_compressed(os.path.join(od, f"{name}_{yy}_{xx}.npz"),
                                x=x[:, yy:yy + T, xx:xx + T], y=(y[yy:yy + T, xx:xx + T] & v).astype(np.uint8)); n += 1
    if n == 0: os.rmdir(od)
    bp = 100 * y[valid].mean()
    print(f"[{name}] {date_s} {ha}ha | 타일 {n} | pseudo불탄 {bp:.1f}% (그리드 {H}x{W})", flush=True)
    return n


def main():
    os.makedirs(OUT, exist_ok=True)
    only = os.environ.get("ONLY")  # 특정 이름만(테스트)
    tot = 0
    for q, ds, ha, name in FIRES:
        if only and only not in name: continue
        try:
            tot += build_one(q, ds, ha, name)
        except Exception as e:
            print(f"[{name}] 실패 {e}", flush=True)
        time.sleep(1)  # Nominatim 예의
    import glob
    print(f"완료: {len(glob.glob(os.path.join(OUT,'*','*.npz')))}타일 / {len(glob.glob(os.path.join(OUT,'*/')))}화재", flush=True)


if __name__ == "__main__":
    main()
