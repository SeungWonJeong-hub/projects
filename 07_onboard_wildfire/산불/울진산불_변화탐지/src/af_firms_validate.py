"""active-fire(NHI) FIRMS 골드검증 — 오프라인(캐시 FIRMS, 라이브API 불필요).
캐시: Steven/산불/산불데이터/firms_korea_bbox_2022_2025.csv (VIIRS/MODIS 열탐지점).

시점별(same-day) FIRMS 열탐지점 vs NHI 마스크로 2가지:
  1) containment(=recall): same-day FIRMS점이 NHI 불화소 ±버퍼 안에 드는 비율.
      높으면 NHI가 열로 확인된 불을 놓치지 않음.
  2) 클러스터 corroboration(=오탐 판정): NHI 연결성분(불 덩어리) 중 근처(±버퍼)에
      FIRMS점이 있는 덩어리 비율.  '흩어진 클러스터가 진짜 불인가'를 결판.
      낮으면 흩어진 게 오탐, 높으면 실제 화선(주변부+스팟).
버퍼 200m (VIIRS 375m/MODIS 1km 대비 관대).  neighbor confidence≥nominal.

사용: python af_firms_validate.py
"""
import os, csv, json, datetime as dt
import numpy as np
from osgeo import gdal, osr
from scipy import ndimage as ndi
gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
AF = os.path.join(HERE, "..", "active_fire")
FIRMS_CSV = os.path.join(HERE, "..", "..", "산불데이터", "firms_korea_bbox_2022_2025.csv")
SITES = ["uljin", "sancheong", "uiseong"]
BUF_M = 200.0
DAY_TOL = 1          # same-day ±1일 허용(S2 10:30 vs VIIRS 01:30/13:30)

def load_firms():
    rows = []
    for r in csv.DictReader(open(FIRMS_CSV, encoding="utf-8-sig")):
        try:
            conf = (r.get("confidence") or "").strip().lower()
            if conf in ("l", "low"): continue           # 저신뢰 제외
            rows.append((float(r["latitude"]), float(r["longitude"]),
                         r["acq_date"].strip()))
        except (KeyError, ValueError):
            continue
    return rows

def dates_of(site):
    d = os.path.join(AF, site)
    return sorted([x for x in os.listdir(d) if x[:2].isdigit() and os.path.isdir(os.path.join(d, x))])

def validate_date(site, date, firms):
    sd = os.path.join(AF, site, date)
    mds = gdal.Open(os.path.join(sd, "active_fire_mask.tif"))
    mask = mds.ReadAsArray() > 0
    gt = mds.GetGeoTransform(); proj = mds.GetProjection()
    H, W = mask.shape
    # lon/lat -> mask UTM 픽셀
    dst = osr.SpatialReference(); dst.ImportFromWkt(proj)
    src = osr.SpatialReference(); src.ImportFromEPSG(4326)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    buf_px = int(round(BUF_M / abs(gt[1])))
    # NHI 마스크를 버퍼만큼 팽창(점-마스크 근접 판정용)
    if mask.any():
        dil = ndi.binary_dilation(mask, iterations=buf_px)
    else:
        dil = mask
    d0 = dt.date.fromisoformat(date)
    # same-day(±TOL) FIRMS점을 픽셀좌표로
    pts_px = []
    for la, lo, ad in firms:
        try:
            fd = dt.date.fromisoformat(ad)
        except ValueError:
            continue
        if abs((fd - d0).days) > DAY_TOL: continue
        x, y, _ = ct.TransformPoint(lo, la)
        col = int((x - gt[0]) / gt[1]); row = int((y - gt[3]) / gt[5])
        if 0 <= row < H and 0 <= col < W:
            pts_px.append((row, col))
    nf = len(pts_px)
    # 1) containment(recall): FIRMS점이 팽창마스크 안에?
    hit = sum(1 for r, c in pts_px if dil[r, c])
    recall = hit / nf if nf else None
    # 2) 클러스터 corroboration: NHI 덩어리 중 근처 FIRMS점 있는 비율
    lab, nlab = ndi.label(mask)
    corrob = None
    if nlab and nf:
        # 각 FIRMS점 주변 buf_px 안의 라벨 수집
        hit_labels = set()
        for r, c in pts_px:
            r0, r1 = max(0, r - buf_px), min(H, r + buf_px + 1)
            c0, c1 = max(0, c - buf_px), min(W, c + buf_px + 1)
            sub = lab[r0:r1, c0:c1]
            for L in np.unique(sub):
                if L: hit_labels.add(int(L))
        corrob = len(hit_labels) / nlab
    return dict(date=date, fire_px=int(mask.sum()), n_clusters=int(nlab),
                firms_sameday=nf, firms_in_nhi=hit,
                recall=round(recall, 3) if recall is not None else None,
                cluster_corrob=round(corrob, 3) if corrob is not None else None)

def main():
    if not os.path.exists(FIRMS_CSV):
        print("FIRMS 캐시 없음:", FIRMS_CSV); return
    firms = load_firms()
    print(f"FIRMS 캐시 로드: {len(firms)}점 (저신뢰 제외)\n")
    print(f"{'site/date':22s} {'불화소':>6} {'덩어리':>5} {'FIRMS(당일)':>10} {'∈NHI':>6} {'recall':>7} {'클러스터확증':>10}")
    print("-" * 82)
    out = {}
    agg_r, agg_c = [], []
    for site in SITES:
        rows = [validate_date(site, d, firms) for d in dates_of(site)]
        out[site] = rows
        for r in rows:
            rc = f"{r['recall']}" if r['recall'] is not None else "-"
            cc = f"{r['cluster_corrob']}" if r['cluster_corrob'] is not None else "-"
            print(f"{site+'/'+r['date']:22s} {r['fire_px']:6d} {r['n_clusters']:5d} {r['firms_sameday']:10d} {r['firms_in_nhi']:6d} {rc:>7} {cc:>10}")
            if r['recall'] is not None: agg_r.append(r['recall'])
            if r['cluster_corrob'] is not None: agg_c.append(r['cluster_corrob'])
    summ = dict(scenes_scored=len(agg_r),
                recall_mean=round(float(np.mean(agg_r)), 3) if agg_r else None,
                cluster_corrob_mean=round(float(np.mean(agg_c)), 3) if agg_c else None,
                buffer_m=BUF_M, day_tol=DAY_TOL)
    out["_summary"] = summ
    json.dump(out, open(os.path.join(AF, "af_firms_validation.json"), "w"), ensure_ascii=False, indent=1)
    print("-" * 82)
    print(f"평균 recall(FIRMS 포착) {summ['recall_mean']}  |  클러스터 확증(오탐아님) {summ['cluster_corrob_mean']}  (n={summ['scenes_scored']})")
    print("saved active_fire/af_firms_validation.json")

if __name__ == "__main__":
    main()
