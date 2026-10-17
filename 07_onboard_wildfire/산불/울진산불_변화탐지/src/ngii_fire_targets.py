"""국토정보플랫폼 커버리지 조회용 화재 타깃 목록 추출.
ee_alphaearth_build.py 의 화재선정 로직을 GEE 없이 그대로 재현.
산출: fire_id, occu_date, area_ha, season, centroid(lon,lat), bbox(WGS84), 그리고
      화재 직후 필요영상 윈도우(fire_date ~ +90d). → data/ngii_targets.csv
"""
import os, csv, datetime as dt
from osgeo import ogr, osr

HERE = os.path.dirname(os.path.abspath(__file__))
SHP = os.path.join(HERE, "..", "..", "산불데이터", "korea", "mtfire", "MTfire.shp")
OUT = os.path.join(HERE, "..", "data", "ngii_targets.csv")
MIN_HA = float(os.environ.get("MIN_HA", "500"))
MAX_HA = float(os.environ.get("MAX_HA", "8000"))
YEARS = set(os.environ.get("YEARS", "2018,2019,2020,2021").split(","))
UTM = 32652  # 면적계산용 (빌드와 동일)

def season(m):
    return {12:"겨울",1:"겨울",2:"겨울",3:"봄",4:"봄",5:"봄",
            6:"여름",7:"여름",8:"여름",9:"가을",10:"가을",11:"가을"}[m]

def tr(src_epsg, dst_epsg):
    s = osr.SpatialReference(); s.ImportFromEPSG(src_epsg); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    d = osr.SpatialReference(); d.ImportFromEPSG(dst_epsg); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(s, d)

def main():
    ds = ogr.Open(SHP); ly = ds.GetLayer()
    to_utm = tr(5186, UTM)
    to_wgs = tr(5186, 4326)
    rows = []
    for f in ly:
        if f.GetField("OCCU_YEAR") not in YEARS:
            continue
        g = f.GetGeometryRef()
        if not g:
            continue
        gu = g.Clone(); gu.Transform(to_utm); ha = gu.GetArea() / 1e4
        if not (MIN_HA <= ha <= MAX_HA):
            continue
        od = f.GetField("OCCU_DATE")
        try:
            fdate = dt.date(int(od[:4]), int(od[4:6]), int(od[6:8]))
        except Exception:
            continue
        gw = g.Clone(); gw.Transform(to_wgs)
        xmin, xmax, ymin, ymax = gw.GetEnvelope()  # lon min/max, lat min/max
        c = gw.Centroid()
        rows.append({
            "fire_id": f"kf_{fdate.isoformat().replace('-','')}_{int(ha)}ha",
            "occu_date": fdate.isoformat(),
            "area_ha": round(ha, 1),
            "season": season(fdate.month),
            "cen_lon": round(c.GetX(), 6),
            "cen_lat": round(c.GetY(), 6),
            "lon_min": round(xmin, 6), "lat_min": round(ymin, 6),
            "lon_max": round(xmax, 6), "lat_max": round(ymax, 6),
            "img_window_start": fdate.isoformat(),
            "img_window_end": (fdate + dt.timedelta(days=90)).isoformat(),
        })
    rows.sort(key=lambda r: r["area_ha"], reverse=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"화재 {len(rows)}개 → {OUT}")
    from collections import Counter
    print("계절분포:", dict(Counter(r["season"] for r in rows)))
    print(f"{'fire_id':28} {'date':11} {'ha':>7} {'season':4} {'lon':>9} {'lat':>8}")
    for r in rows:
        print(f"{r['fire_id']:28} {r['occu_date']:11} {r['area_ha']:>7} {r['season']:4} {r['cen_lon']:>9} {r['cen_lat']:>8}")

if __name__ == "__main__":
    main()
