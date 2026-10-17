"""gold 후보 화재의 NGII 항공정사영상 chip 생성 (소각 육안확인용).
각 화재: pre(화재전년)/화재년/post(화재+1년) 3개년 모자이크 + 산림청 폴리곤(빨강) 오버레이.
공개키 사용. → data/chips/{fire_id}.png
사용: python ngii_chip.py kf_20190826_3047ha kf_20181004_2666ha ...
"""
import os, sys, ssl, io, time, math, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from osgeo import ogr, osr

HERE = os.path.dirname(os.path.abspath(__file__))
SHP = os.path.join(HERE, "..", "..", "산불데이터", "korea", "mtfire", "MTfire.shp")
OUTD = os.path.join(HERE, "..", "data", "chips"); os.makedirs(OUTD, exist_ok=True)
K = os.environ.get("NGII_APIKEY", "")
WMTS = "https://map.ngii.go.kr/airmapprime/map/wmts"
SD0 = 2.3873828571428573e8; TLX, TLY = -200000.0, 4000000.0; TILE = 256
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
TARGET_PX = 1400

def res(L): return (SD0 / 2 ** L) * 0.00028

def trans(a, b):
    s = osr.SpatialReference(); s.ImportFromEPSG(a); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    d = osr.SpatialReference(); d.ImportFromEPSG(b); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(s, d)

def fetch_tile(year, col, row, L):
    u = (f"{WMTS}?SERVICE=WMTS&VERSION=1.0.0&REQUEST=GetTile&LAYER=mapprime:air_{year}"
         f"&STYLE=korean&TILEMATRIXSET=NGIS_AIR&TILEMATRIX={L}&TILEROW={row}&TILECOL={col}"
         f"&FORMAT=image/jpeg&apikey={K}")
    r = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://map.ngii.go.kr/ms/map/NlipMap.do"})
    try:
        d = urllib.request.urlopen(r, timeout=30, context=CTX).read()
        if len(d) < 200 or d[:2] != b'\xff\xd8':
            return None
        return Image.open(io.BytesIO(d)).convert("RGB")
    except Exception:
        return None

def build_year(year, xmin, ymin, xmax, ymax, L):
    """EPSG:5179 bbox를 덮는 타일 모자이크 (PIL Image, origin_x, origin_y)."""
    span = res(L) * TILE
    c0 = int((xmin - TLX) / span); c1 = int((xmax - TLX) / span)
    r0 = int((TLY - ymax) / span); r1 = int((TLY - ymin) / span)
    W = (c1 - c0 + 1) * TILE; H = (r1 - r0 + 1) * TILE
    canvas = Image.new("RGB", (W, H), (40, 40, 40)); got = 0
    for ri, row in enumerate(range(r0, r1 + 1)):
        for ci, col in enumerate(range(c0, c1 + 1)):
            im = fetch_tile(year, col, row, L)
            if im is not None:
                canvas.paste(im, (ci * TILE, ri * TILE)); got += 1
            time.sleep(0.05)
    origin_x = TLX + c0 * span; origin_y = TLY - r0 * span
    return canvas, origin_x, origin_y, got, (c1 - c0 + 1) * (r1 - r0 + 1)

def draw_poly(img, geom5179, origin_x, origin_y, L, color=(255, 40, 40)):
    dr = ImageDraw.Draw(img); r = res(L)
    def ring_px(ring):
        pts = []
        for i in range(ring.GetPointCount()):
            X, Y, *_ = ring.GetPoint(i)
            pts.append(((X - origin_x) / r, (origin_y - Y) / r))
        return pts
    def walk(g):
        gt = g.GetGeometryName()
        if gt == "POLYGON":
            for i in range(g.GetGeometryCount()):
                pts = ring_px(g.GetGeometryRef(i))
                if len(pts) > 1: dr.line(pts + [pts[0]], fill=color, width=4)
        elif gt in ("MULTIPOLYGON", "GEOMETRYCOLLECTION"):
            for i in range(g.GetGeometryCount()): walk(g.GetGeometryRef(i))
    walk(geom5179)

def label(img, text):
    dr = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    except Exception: font = ImageFont.load_default()
    dr.rectangle([0, 0, 360, 46], fill=(0, 0, 0)); dr.text((8, 6), text, fill=(255, 255, 0), font=font)

def load_fire_polys(fire_ids):
    """fire_id(kf_date_XXha) → EPSG:5179 geom, 매칭: OCCU_DATE + int(면적ha)."""
    want = {}
    for fid in fire_ids:
        _, d, ha = fid.split("_"); want[(d, int(ha[:-2]))] = fid
    ds = ogr.Open(SHP); ly = ds.GetLayer()
    to_utm = trans(5186, 32652); to_5179 = trans(5186, 5179)
    found = {}
    for f in ly:
        od = f.GetField("OCCU_DATE")
        if not od: continue
        key_date = od
        g = f.GetGeometryRef()
        if not g: continue
        gu = g.Clone(); gu.Transform(to_utm); ha = int(gu.GetArea() / 1e4)
        for dha, fid in want.items():
            if dha[0] == key_date and abs(dha[1] - ha) <= 1:
                g5 = g.Clone(); g5.Transform(to_5179); found[fid] = g5.Clone()
    return found

def pick_level(width_m):
    need = width_m / TARGET_PX
    L = int(math.floor(math.log2(SD0 * 0.00028 / need)))
    return max(13, min(17, L))

def main():
    fids = sys.argv[1:]
    polys = load_fire_polys(fids)
    for fid in fids:
        g = polys.get(fid)
        if g is None:
            print(f"[{fid}] 폴리곤 매칭 실패"); continue
        env = g.GetEnvelope()  # xmin,xmax,ymin,ymax in 5179
        xmin, xmax, ymin, ymax = env
        mx = (xmax - xmin) * 0.25; my = (ymax - ymin) * 0.25
        xmin -= mx; xmax += mx; ymin -= my; ymax += my
        L = pick_level(xmax - xmin)
        fyear = int(fid.split("_")[1][:4])
        years = [fyear - 1, fyear, fyear + 1]
        panels = []
        for yr in years:
            canvas, ox, oy, got, tot = build_year(yr, xmin, ymin, xmax, ymax, L)
            draw_poly(canvas, g, ox, oy, L)
            tag = {years[0]: "pre", years[1]: "FIRE-yr", years[2]: "post"}[yr]
            label(canvas, f"air_{yr} ({tag})  {got}/{tot} tiles")
            # 다운샘플 통일 높이
            h = 900; w = int(canvas.width * h / canvas.height)
            panels.append(canvas.resize((w, h)))
            print(f"[{fid}] air_{yr} L{L} ({res(L):.2f}m) tiles {got}/{tot} size {canvas.size}")
        W = sum(p.width for p in panels) + 20 * (len(panels) - 1)
        combo = Image.new("RGB", (W, 900), (255, 255, 255)); x = 0
        for p in panels:
            combo.paste(p, (x, 0)); x += p.width + 20
        out = os.path.join(OUTD, f"{fid}.png"); combo.save(out)
        print(f"[{fid}] → {out}\n")

if __name__ == "__main__":
    main()
