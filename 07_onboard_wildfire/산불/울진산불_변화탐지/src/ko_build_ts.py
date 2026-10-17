"""Stage1 — 시계열(다시점) 빌드. 봄 dNBR 반전을 '회복 궤적'으로 해소하기 위한 데이터.
화재별 pre 1장 + 불후 5개 시간창(구름최저)에서 씬 취득 → [T=6,6밴드,H,W] 스택 저장.
라벨=산림청 경계폴리곤 래스터화(ko_build_multi와 동일). 이름 kf_YYYYMMDD_XXXXha (고정홀드아웃 호환).
저장: ts[6,6,H,W](t0=pre,1-5=post창) · days[6] · tvalid[6] · dnbr[H,W](echo용) · y[H,W].
"""
from __future__ import annotations
import os, sys, math, glob, datetime as dt
import numpy as np
from osgeo import gdal, ogr, osr
gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ko_build_multi import (SHP, CAT, UTM, YEARS, MIN_HA, MAX_HA, CROP_M, STRIDE,
                            rfc, ratio, idx, stac_bands, _sab)

OUT = os.environ.get("TS_OUT", os.path.join(HERE, "..", "ko_ts"))
N_FIRES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
T = 256
# 불후 시간창(일) — 즉시연소→회복 궤적을 훑음
POST_WINS = [(10, 45), (45, 95), (95, 155), (155, 225), (225, 330)]
SCL_DROP = [0, 1, 3, 8, 9, 10, 11]


def acquire(bbox_ll, ute, fdate):
    """pre 1장 + POST_WINS별 구름최저 씬 → 밴드딕 리스트와 day-offset."""
    scenes = []; days = []
    pc = fdate.replace(year=fdate.year - 1)
    pre = stac_bands(bbox_ll, rfc(pc - dt.timedelta(30), pc + dt.timedelta(45)), ute)
    if pre is None: return None, None
    scenes.append(pre); days.append(-365)
    for a, b in POST_WINS:
        s = stac_bands(bbox_ll, rfc(fdate + dt.timedelta(a), fdate + dt.timedelta(b)), ute)
        scenes.append(s); days.append((a + b) // 2)
    return scenes, days


def norm_series(scenes, valid):
    """전 시점 공동 백분위로 B12/B08/B02 정규화(절대 변화 보존). 비율은 스케일불변."""
    ab = {}
    for k in ("B12", "B08", "B02"):
        vals = np.concatenate([s[k][valid] for s in scenes if s is not None]) if valid.any() else np.array([0., 1.])
        ab[k] = np.percentile(vals, [2, 98]) if vals.size else (0, 1)
    def b6(bd):
        return np.stack([_sab(bd['B12'], *ab['B12']), _sab(bd['B08'], *ab['B08']), _sab(bd['B02'], *ab['B02']),
                         idx(ratio(bd['B08'], bd['B12'])), idx(ratio(bd['B08'], bd['B04'])), idx(ratio(bd['B03'], bd['B08']))], 0)
    return b6


def build_one(name, gu, fdate):
    e = gu.GetEnvelope(); cx = (e[0] + e[1]) / 2; cy = (e[2] + e[3]) / 2
    hx = max((e[1] - e[0]) / 2 + 500, CROP_M); hy = max((e[3] - e[2]) / 2 + 500, CROP_M)
    ute = [math.floor(cx - hx), math.floor(cy - hy), math.ceil(cx + hx), math.ceil(cy + hy)]
    dst = osr.SpatialReference(); dst.ImportFromEPSG(4326); dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src = osr.SpatialReference(); src.ImportFromEPSG(UTM); src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    lo1, la1, _ = ct.TransformPoint(e[0], e[2]); lo2, la2, _ = ct.TransformPoint(e[1], e[3])
    bbox_ll = [min(lo1, lo2) - 0.02, min(la1, la2) - 0.02, max(lo1, lo2) + 0.02, max(la1, la2) + 0.02]
    scenes, days = acquire(bbox_ll, ute, fdate)
    if scenes is None: print(f"[{name}] pre 없음", flush=True); return 0
    nvalid_post = sum(s is not None for s in scenes[1:])
    if nvalid_post < 3: print(f"[{name}] 유효 post {nvalid_post}<3 → skip", flush=True); return 0
    # 공통 H,W
    shp = [s['B02'].shape for s in scenes if s is not None]
    H = min(s[0] for s in shp); W = min(s[1] for s in shp)
    for s in scenes:
        if s is not None:
            for k in s: s[k] = s[k][:H, :W]
    # 결측 post는 인접 유효 post로 채움(tvalid=0)
    tvalid = np.array([1 if s is not None else 0 for s in scenes], dtype=np.uint8)
    last = scenes[0]
    for i in range(1, len(scenes)):
        if scenes[i] is None: scenes[i] = last
        else: last = scenes[i]
    # valid 마스크(전 유효시점 교집합)
    valid = np.ones((H, W), bool)
    for i, s in enumerate(scenes):
        if tvalid[i]:
            valid &= (s['B02'] > 0) & (s['B08'] > 0) & (s['B12'] > 0) & ~np.isin(s['SCL'], SCL_DROP)
    # 라벨 래스터화
    mem = ogr.GetDriverByName("Memory").CreateDataSource("m"); srs = osr.SpatialReference(); srs.ImportFromEPSG(UTM)
    lyr = mem.CreateLayer("f", srs, ogr.wkbPolygon); ft = ogr.Feature(lyr.GetLayerDefn()); ft.SetGeometry(gu); lyr.CreateFeature(ft)
    rast = gdal.GetDriverByName("MEM").Create("", W, H, 1, gdal.GDT_Byte)
    rast.SetGeoTransform((ute[0], 20, 0, ute[3], 0, -20)); rast.SetProjection(srs.ExportToWkt())
    gdal.RasterizeLayer(rast, [1], lyr, burn_values=[1]); mask = rast.ReadAsArray()[:H, :W].astype(np.uint8)
    # 정규화 + 스택
    b6 = norm_series(scenes, valid)
    ts = np.stack([b6(s) for s in scenes], 0).astype(np.float32)  # [6,6,H,W]
    # 첫 유효 post로 dNBR(echo/호환)
    p0 = scenes[0]; p1 = next(scenes[i] for i in range(1, len(scenes)) if tvalid[i])
    dnbr = np.clip(ratio(p0['B08'], p0['B12']) - ratio(p1['B08'], p1['B12']), -1, 1).astype(np.float32)
    y = (mask > 0).astype(np.uint8)
    od = os.path.join(OUT, name); os.makedirs(od, exist_ok=True); n = 0
    if H < T or W < T: print(f"[{name}] 크롭<{T} → skip", flush=True); os.rmdir(od); return 0
    ys = sorted(set(min(yy, H - T) for yy in range(0, H - T + 1, STRIDE)))
    xs = sorted(set(min(xx, W - T) for xx in range(0, W - T + 1, STRIDE)))
    for yy in ys:
        for xx in xs:
            v = valid[yy:yy + T, xx:xx + T]
            if v.mean() < 0.3: continue   # 6시점 교집합이라 완화(화재 보존)
            np.savez_compressed(os.path.join(od, f"{name}_{yy}_{xx}.npz"),
                                ts=ts[:, :, yy:yy + T, xx:xx + T], days=np.array(days, np.int16),
                                tvalid=tvalid, dnbr=dnbr[yy:yy + T, xx:xx + T],
                                y=(y[yy:yy + T, xx:xx + T] & v).astype(np.uint8)); n += 1
    if n == 0: os.rmdir(od)
    print(f"[{name}] {fdate} | 타일 {n} | post {nvalid_post}/5 | 불탄 {y[valid].mean()*100:.1f}%", flush=True)
    return n


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = ogr.Open(SHP); ly = ds.GetLayer(); cand = []
    for f in ly:
        if f.GetField("OCCU_YEAR") not in YEARS: continue
        g = f.GetGeometryRef()
        if not g: continue
        s = osr.SpatialReference(); s.ImportFromEPSG(5186); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        d = osr.SpatialReference(); d.ImportFromEPSG(UTM); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        gu = g.Clone(); gu.Transform(osr.CoordinateTransformation(s, d)); ha = gu.GetArea() / 1e4
        if not (MIN_HA <= ha <= MAX_HA): continue
        od = f.GetField("OCCU_DATE")
        try: fdate = dt.date(int(od[:4]), int(od[4:6]), int(od[6:8]))
        except: continue
        cand.append((ha, fdate, gu.Clone()))
    cand.sort(key=lambda c: c[0], reverse=True)
    print(f"후보 {len(cand)}건, 상위 {N_FIRES} 빌드 (시계열 T=6, MIN_HA={MIN_HA})", flush=True)
    tot = built = 0
    for i, (ha, fdate, gu) in enumerate(cand[:N_FIRES]):
        name = f"kf_{fdate.isoformat().replace('-','')}_{int(ha)}ha"
        if glob.glob(os.path.join(OUT, name, "*.npz")):
            print(f"[{name}] 이미 존재 → skip", flush=True); built += 1; continue
        try:
            n = build_one(name, gu, fdate); tot += n; built += (n > 0)
        except Exception as ex:
            print(f"[{name}] 실패 {ex}", flush=True)
        if (i + 1) % 5 == 0: print(f"  진행 {i+1}/{min(N_FIRES,len(cand))} | 누적타일 {tot} 화재 {built}", flush=True)
    print(f"완성 화재 {built} | 총 타일 {len(glob.glob(os.path.join(OUT,'*','*.npz')))}", flush=True)


if __name__ == "__main__":
    main()
