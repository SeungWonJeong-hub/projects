"""국토정보플랫폼 항공정사영상(mapprime:air_YYYY, WMTS/NGIS_AIR/EPSG:5179) 커버리지 체커.
각 화재 중심좌표에서 연도별 타일을 받아 '실제 영상 vs 공백(미촬영)'을 판정 → 커버리지 행렬.
키 승인 불필요(사이트 공개 apikey 사용). go/no-go 산출물.
사용: python ngii_coverage_check.py [YEAR_START YEAR_END] [LEVEL]  (기본 2015 2025 16)
"""
import os, sys, csv, io, ssl, time, urllib.request
from osgeo import ogr, osr

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_IN = os.path.join(HERE, "..", "data", "ngii_targets.csv")
CSV_OUT = os.path.join(HERE, "..", "data", "ngii_coverage.csv")
APIKEY = os.environ.get("NGII_APIKEY", "")  # 사이트 공개키(승인키 있으면 교체)
WMTS = "https://map.ngii.go.kr/airmapprime/map/wmts"
Y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
Y1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
LEVEL = int(sys.argv[3]) if len(sys.argv) > 3 else 18  # 원본해상도(~0.25m)에서 판정해야 실제 촬영 커버리지
# NGIS_AIR 타일격자 (GetCapabilities에서 추출)
SD0 = 2.3873828571428573e8   # level0 ScaleDenominator
TLX, TLY = -200000.0, 4000000.0  # TopLeftCorner: (easting, northing) in EPSG:5179
TILE = 256
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

def res(L):  # m/px (OGC 0.28mm/px)
    return (SD0 / (2 ** L)) * 0.00028

def to_5179():
    s = osr.SpatialReference(); s.ImportFromEPSG(4326); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    d = osr.SpatialReference(); d.ImportFromEPSG(5179); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(s, d)

def tile_of(x, y, L):
    span = res(L) * TILE
    col = int((x - TLX) / span)
    row = int((TLY - y) / span)
    return col, row

def fetch(year, col, row, L):
    url = (f"{WMTS}?SERVICE=WMTS&VERSION=1.0.0&REQUEST=GetTile&LAYER=mapprime:air_{year}"
           f"&STYLE=korean&TILEMATRIXSET=NGIS_AIR&TILEMATRIX={L}&TILEROW={row}&TILECOL={col}"
           f"&FORMAT=image/jpeg&apikey={APIKEY}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                               "Referer": "https://map.ngii.go.kr/ms/map/NlipMap.do"})
    try:
        d = urllib.request.urlopen(req, timeout=30, context=CTX).read()
    except Exception as e:
        return None, f"ERR:{str(e)[:40]}"
    return d, None

def classify(d):
    """실제영상=covered / 공백타일=empty. 공백은 균일색이라 JPEG 크기가 매우 작음."""
    if not d or len(d) < 200:
        return "empty", len(d) if d else 0
    if d[:2] != b'\xff\xd8':  # not JPEG (error/xml)
        return "nonimg", len(d)
    # 공백(단색) JPEG는 보통 <1.5KB, 실제 영상은 수~수십KB
    return ("covered" if len(d) >= 1800 else "empty"), len(d)

def main():
    tr = to_5179()
    with open(CSV_IN) as fp:
        fires = list(csv.DictReader(fp))
    if os.environ.get("SMOKE"):
        fires = fires[:int(os.environ["SMOKE"])]
    years = list(range(Y0, Y1 + 1))
    print(f"화재 {len(fires)}개 × 연도 {Y0}-{Y1} (레벨 {LEVEL}, {res(LEVEL):.2f} m/px, 타일 {res(LEVEL)*TILE:.0f} m)")
    out = []
    for fr in fires:
        lon, lat = float(fr["cen_lon"]), float(fr["cen_lat"])
        x, y, _ = tr.TransformPoint(lon, lat)
        col, row = tile_of(x, y, LEVEL)
        fyear = int(fr["occu_date"][:4])
        cells, cov_years = {}, []
        for yr in years:
            d, err = fetch(yr, col, row, LEVEL)
            if err:
                cells[yr] = "E"
            else:
                cls, sz = classify(d)
                cells[yr] = {"covered": "O", "empty": ".", "nonimg": "x"}[cls]
                if cls == "covered":
                    cov_years.append(yr)
            time.sleep(0.15)
        # 화재직후(당해~+2년) 커버 여부: 화재년 이후 첫 커버연도
        post = [yr for yr in cov_years if yr >= fyear]
        row_str = "".join(cells[yr] for yr in years)
        first_post = min(post) if post else None
        out.append({**{k: fr[k] for k in ("fire_id", "occu_date", "season", "area_ha")},
                    "fire_year": fyear, "coverage": row_str,
                    "cov_years": ";".join(map(str, cov_years)),
                    "first_post_fire_year": first_post if first_post else "",
                    "post_fire_gap": (first_post - fyear) if first_post else ""})
        print(f"{fr['fire_id']:26} {fr['season']:3} fire{fyear}  [{Y0}]{row_str}[{Y1}]  "
              f"post→{first_post if first_post else '없음'}")
    with open(CSV_OUT, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    n_post = sum(1 for r in out if r["first_post_fire_year"] != "")
    print(f"\n→ {CSV_OUT}")
    print(f"화재직후(당해년 이후) 커버 화재: {n_post}/{len(out)}")

if __name__ == "__main__":
    main()
