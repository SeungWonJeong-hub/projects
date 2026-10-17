"""MTBS 선학습 데이터셋 빌드 — S2(pre/post) ↔ MTBS 등급라벨 페어링.

목적: 미 전역 MTBS 화재(분석가 보정 등급, 3만건)로 심각도 분류기를 선학습해
'화재별 보정 없이 전이되는' 절대 등급 분류를 얻는다(→ 등급 이미지·면적 동시 신뢰).

경로(완전 자율, GEE 불필요):
  ① 화재리스트  data/mtbs_pts.zip  (IG_DATE·lat/lon·ACRES·PRE_ID/POST_ID·화재별 임계)
  ② 등급라벨    data/mtbs_CONUS_<YEAR>.tif  (30m Albers, 0배경1미소각~저2저3중4고5증녹6비매핑)
  ③ 영상        Sentinel-2 L2A  earth-search STAC (무인증)
각 화재를 공통 UTM 20m 격자로 warp → (S2 pre/post 6밴드 ×2) + 등급 라벨 패치.

B200 대량용: build_fire() 를 9,953개 S2-era 화재에 돌린다(재개가능·화재별 파일).
젯슨: 1건 실증 + 파이프라인 검증.

사용: python src/mtbs_pretrain_build.py --fire_id AZ3340811121020190608 --year 2019
"""
import os, sys, argparse
import numpy as np
# /vsicurl COG 읽기 가속 + HTTP 재시도(젯슨 병목=지연·네트워크). gdal import 전에 세팅.
for _k, _v in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_USE_HEAD": "NO",
               "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "2",
               "VSI_CACHE": "TRUE", "VSI_CACHE_SIZE": "50000000", "GDAL_NUM_THREADS": "2"}.items():
    os.environ.setdefault(_k, _v)
from osgeo import gdal, ogr, osr
gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "..", "..", "웹앱"))
from core import acquire  # noqa: E402

DATA = os.path.join(ROOT, "data"); OUT = os.path.join(ROOT, "patches"); os.makedirs(OUT, exist_ok=True)
PTS = os.path.join(DATA, "mtbs_pts.zip")
BANDS = {"B02": "blue", "B03": "green", "B04": "red", "B8A": "nir08", "B11": "swir16", "B12": "swir22"}
# SCL(Scene Classification) 에서 버릴 값: 0무자료 1결함 3구름그림자 8구름중 9구름고 10권운 11눈얼음.
# (2어두운·5비식생·6물·7미분류는 유지 — 탄자국이 어둡게 잡히므로 2·5 절대 제외금지)
DROP_SCL = frozenset({0, 1, 3, 8, 9, 10, 11})
MIN_BURN = 50                                  # 격자 내 유효 소각화소 최소치(미달=저장 안 함)


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) / 6) + 1


def get_fire(fire_id):
    ds = ogr.Open("/vsizip/" + PTS) if PTS.endswith(".zip") else ogr.Open(PTS)
    L = ds.GetLayer()
    L.SetAttributeFilter(f"FIRE_ID='{fire_id}'")
    f = L.GetNextFeature()
    if f is None:
        raise SystemExit(f"화재 없음: {fire_id}")
    return {k: f.GetField(k) for k in ("FIRE_ID", "FIRE_NAME", "IG_DATE", "LATITUDE",
                                       "LONGITUDE", "ACRES", "PRE_ID", "POST_ID", "FIRE_TYPE")}


def target_grid(lat, lon, acres):
    """화재 중심+면적으로 AOI bbox → UTM 20m 출력격자(outputBounds, epsg)."""
    epsg = utm_epsg(lon, lat)
    s = osr.SpatialReference(); s.ImportFromEPSG(4326); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    t = osr.SpatialReference(); t.ImportFromEPSG(epsg)
    ct = osr.CoordinateTransformation(s, t)
    cx, cy, _ = ct.TransformPoint(lon, lat)
    r = np.sqrt(max(acres, 100) * 4046.86 / np.pi) * 1.6      # 원면적 반경 ×1.6 버퍼
    r = min(max(r, 3000), 30000)                              # 3~30km 클램프(초대형화재 OOM 방지, 학습은 타일링)
    xmin, ymin, xmax, ymax = cx - r, cy - r, cx + r, cy + r
    xmin = np.floor(xmin / 20) * 20; ymin = np.floor(ymin / 20) * 20
    xmax = np.ceil(xmax / 20) * 20; ymax = np.ceil(ymax / 20) * 20
    return [xmin, ymin, xmax, ymax], epsg


def warp(src, bounds, epsg, res, alg):
    return gdal.Warp("", src, format="MEM", outputBounds=bounds, dstSRS=f"EPSG:{epsg}",
                     xRes=res, yRes=res, resampleAlg=alg, outputType=gdal.GDT_Float32
                     if alg != "near" else gdal.GDT_Byte).ReadAsArray()


def aoi_lonlat(bounds, epsg):
    """출력 격자(UTM bounds) → 경위도 다각형(ogr). 씬 풋프린트 덮임 계산용."""
    s = osr.SpatialReference(); s.ImportFromEPSG(epsg)
    t = osr.SpatialReference(); t.ImportFromEPSG(4326); t.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(s, t)
    x0, y0, x1, y1 = bounds
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in pts:
        lo, la, _ = ct.TransformPoint(x, y); ring.AddPoint_2D(lo, la)
    g = ogr.Geometry(ogr.wkbPolygon); g.AddGeometry(ring)
    return g


def _foot(s):
    import json as _json
    return ogr.CreateGeometryFromJson(_json.dumps(s["geometry"])) if s.get("geometry") else None


def _cover(scenes, aoi):
    """씬 풋프린트 합집합이 AOI 를 덮는 비율(0–1). 풋프린트가 없는 씬이 있으면 1 로 친다(판단 불가)."""
    u = None
    for s in scenes:
        g = _foot(s)
        if g is None:
            return 1.0
        g = g.Intersection(aoi)
        u = g if u is None else u.Union(g)
    return (u.GetArea() / aoi.GetArea()) if u is not None and aoi.GetArea() > 0 else 0.0


def pick_s2(lon, lat, d0, d1, target, aoi=None):
    """타깃일에 가깝고 맑으면서 **AOI 를 실제로 덮는** 통과(±3일 인접 타일 묶음)를 고른다.
    AOI 가 단일 MGRS 타일보다 크거나 타일경계를 걸쳐도 모자이크로 완전히 덮기 위함.

    2026-09-11 B200: 검색 박스(중심 ±0.2°)와 겹치기만 하면 골랐더니, 궤도 가장자리라 자료가 AOI 밖에만
    있는 씬이 뽑혀 격자 전체가 SCL 무자료(0) → '구름 100 %' 로 스킵됐다(재빌드 구름잠식 64건 중 30건이
    정확히 100 %, 개별 재현 8화재 중 3). 옛 단일타일 빌드의 빈 영상 패치(1,920/3,089)도 같은 원인으로 본다.
    이제 STAC 풋프린트로 덮임을 재서 99 % 이상 덮는 통과 중 가깝고 맑은 것, 없으면 가장 많이 덮는 것(50 % 미만은 포기)."""
    f = acquire.search([lon - 0.2, lat - 0.18, lon + 0.2, lat + 0.18], d0, d1, max_cloud=35, limit=40)
    # 일부 아이템은 자산이 s3://sentinel-s2-l2a/…jp2(요청자부담 버킷)라 /vsicurl 로 못 연다.
    # 모자이크는 타일을 여러 장 묶으므로 하나만 섞여도 화재 전체가 실패했다(B200 2026-09-11:
    # 표본 8화재 중 2 · 로그 65건 중 5). 필요한 밴드·SCL 이 전부 https 인 씬만 쓴다.
    need = list(BANDS.values()) + ["scl"]
    f = [s for s in (f or []) if all(str(s["assets"].get(k, {}).get("href", "")).startswith("http") for k in need)]
    if not f:
        return None
    tgt = np.datetime64(target)
    dd = lambda s, t=tgt: abs((np.datetime64(s["datetime"]) - t).astype("timedelta64[D]").astype(int))
    order = sorted(f, key=lambda x: (dd(x), x["cloud"]))
    grp = lambda s0: [s for s in order if dd(s, np.datetime64(s0["datetime"])) <= 3]
    if aoi is None:                                  # 예전 동작(덮임 모름)
        return grp(order[0])
    cands, seen = [], set()
    for s0 in order:                                 # 날짜마다 ±3일 묶음 — AOI 와 겹치는 타일만
        if s0["datetime"] in seen:
            continue
        seen.add(s0["datetime"])
        g = [s for s in grp(s0) if _foot(s) is None or _foot(s).Intersects(aoi)]
        if g:
            cands.append((_cover(g, aoi), dd(s0), s0["cloud"], g))
    if not cands:
        return None
    full = [c for c in cands if c[0] >= 0.99]
    best = min(full, key=lambda c: (c[1], c[2])) if full else max(cands, key=lambda c: (c[0], -c[1]))
    if best[0] < 0.5:                                # AOI 절반도 못 덮으면 포기
        return None
    best[3][0]["_cov"] = round(best[0], 3)
    return best[3]


def fetch_s2_stack(scenes, bounds, epsg):
    """씬 그룹(인접 타일들)을 밴드별로 모자이크 → AOI 완전커버. 하나라도 밴드 없으면 None."""
    out = {}
    for b, key in BANDS.items():
        hrefs = ["/vsicurl/" + s["assets"][key]["href"] for s in scenes if s["assets"].get(key)]
        if not hrefs:
            return None
        out[b] = warp(hrefs, bounds, epsg, 20, "average") / 10000.0
    return out


def fetch_scl_mask(scenes, bounds, epsg, shape):
    """씬 그룹 SCL 을 모자이크 → 격자상 '유효(비-구름/그림자/눈, 자료있음)' bool 마스크."""
    hrefs = ["/vsicurl/" + s["assets"]["scl"]["href"] for s in scenes if s["assets"].get("scl")]
    if not hrefs:
        return np.ones(shape, bool)
    scl = warp(hrefs, bounds, epsg, 20, "near")
    return ~np.isin(scl, list(DROP_SCL))


def build_fire(fire, year):
    lat, lon, acres = fire["LATITUDE"], fire["LONGITUDE"], fire["ACRES"] or 1000
    ig = str(fire["IG_DATE"]).replace("/", "-")
    post_d = f"{str(fire['POST_ID'])[-8:][:4]}-{str(fire['POST_ID'])[-8:][4:6]}-{str(fire['POST_ID'])[-8:][6:]}" \
        if fire.get("POST_ID") and str(fire["POST_ID"])[-8:].isdigit() else None
    igy, igm, igd = ig.split("-")
    bounds, epsg = target_grid(lat, lon, acres)
    print(f"  {fire['FIRE_NAME']} {fire['FIRE_ID']} · {acres:,.0f}ac · IG {ig} · UTM{epsg} · {int((bounds[2]-bounds[0])/20)}x{int((bounds[3]-bounds[1])/20)}px")

    # 라벨: MTBS 연 mosaic → 타깃 격자(near, 범주)
    mtbs = os.path.join(DATA, f"mtbs_CONUS_{year}.tif")
    sev = warp(mtbs, bounds, epsg, 20, "near")
    sev = sev.astype("uint8")
    burn = int(np.isin(sev, [1, 2, 3, 4]).sum())
    print(f"  MTBS 등급 화소: " + " ".join(f"{k}={int((sev==v).sum())}" for k, v in
          (("배경", 0), ("미저", 1), ("저", 2), ("중", 3), ("고", 4), ("증녹", 5), ("비맵", 6))) + f"  (소각1-4={burn})")
    if burn < MIN_BURN:                          # 격자에 소각footprint 사실상 없음 → S2 취득 낭비 말고 스킵
        print(f"  ⚠️ 소각화소 {burn}<{MIN_BURN} — bbox/연도 불일치, 스킵"); return None

    # 영상: pre(IG 이전 60일), post(POST_ID 근방 또는 IG+90일)
    pre_target = f"{igy}-{igm}-{max(1,int(igd)-15):02d}"
    from datetime import date, timedelta
    ig_dt = date(int(igy), int(igm), int(igd))
    aoi = aoi_lonlat(bounds, epsg)                  # 씬이 이 격자를 실제로 덮는지 재기 위해
    sp = pick_s2(lon, lat, str(ig_dt - timedelta(days=75)), str(ig_dt - timedelta(days=2)), pre_target, aoi=aoi)
    post_center = post_d or str(ig_dt + timedelta(days=60))
    pc = date(*map(int, post_center.split("-")))
    # post 창은 발화 다음날부터 — POST_ID 가 발화 직후면 pc-30 이 발화 전으로 넘어가고, 덮임 우선 선택이
    # 그 안의 발화 전 씬을 post 로 고를 수 있다(2026-09-11 로그: pre·post 가 같은 날짜인 쌍).
    p0 = max(pc - timedelta(days=30), ig_dt + timedelta(days=1))
    sa = pick_s2(lon, lat, str(p0), str(pc + timedelta(days=45)), post_center, aoi=aoi)
    if not sp or not sa:
        print(f"  ⚠️ S2 씬 부족·덮임<50% (pre={bool(sp)} post={bool(sa)})"); return None
    if not (sp[0]["datetime"] < str(ig_dt) < sa[0]["datetime"]):
        print(f"  ⚠️ 날짜 역전 pre {sp[0]['datetime']} · IG {ig_dt} · post {sa[0]['datetime']} — 스킵"); return None
    print(f"  S2 pre {sp[0]['datetime']}({len(sp)}타일 ☁{sp[0]['cloud']} 덮임{sp[0].get('_cov', '?')}) · "
          f"post {sa[0]['datetime']}({len(sa)}타일 ☁{sa[0]['cloud']} 덮임{sa[0].get('_cov', '?')})")
    pre = fetch_s2_stack(sp, bounds, epsg); post = fetch_s2_stack(sa, bounds, epsg)
    if pre is None or post is None:
        print("  ⚠️ 밴드 취득 실패"); return None

    # 구름/그림자 마스킹: pre·post 둘 다 clear 인 화소만 유효. 오염화소 라벨→6(비맵=학습 ignore).
    valid = fetch_scl_mask(sp, bounds, epsg, sev.shape) & fetch_scl_mask(sa, bounds, epsg, sev.shape)
    sev[~valid] = 6
    burn_clear = int(np.isin(sev, [1, 2, 3, 4]).sum())
    cloud_pct = round(100.0 * (1 - valid.mean()), 1)
    print(f"  구름마스킹: 유효소각 {burn_clear}/{burn} (구름 {cloud_pct}%)")
    if burn_clear < MIN_BURN:                     # 구름이 화재를 잠식 → 학습가치 없음, 저장 안 함
        print(f"  ⚠️ 구름잠식 유효소각 {burn_clear}<{MIN_BURN} — 스킵"); return None

    X = np.stack([pre[b] for b in BANDS] + [post[b] for b in BANDS], 0).astype("float16")  # 12ch, fp16(용량·메모리 절반)
    op = os.path.join(OUT, f"{fire['FIRE_ID']}.npz")
    np.savez_compressed(op, X=X, sev=sev, bands=list(BANDS) * 2,
                        epsg=epsg, bounds=np.array(bounds), fire_id=fire["FIRE_ID"],
                        pre=sp[0]["datetime"], post=sa[0]["datetime"], cloud_pct=cloud_pct)
    print(f"  ✅ 패치 저장 {op}  X{X.shape} · 크기 {os.path.getsize(op)//1024}KB")
    return op


def ensure_fire_list():
    """화재리스트(mtbs_pts.zip)가 없으면 FSGeodata 에서 자동 다운(B200 자가공급)."""
    if os.path.exists(PTS):
        return
    import urllib.request
    print("화재리스트 다운로드(MTBS FIRE_OCCURRENCE_PT)…")
    url = "https://data.fs.usda.gov/geodata/edw/edw_resources/shp/S_USA.MTBS_FIRE_OCCURRENCE_PT.zip"
    urllib.request.urlretrieve(url, PTS)


def list_fires(year, min_acres=1000, wildfire_only=True):
    """해당 연도 화재 리스트(dict) — S2 페어링 대상. IG_DATE 로 연도 필터."""
    ds = ogr.Open("/vsizip/" + PTS); L = ds.GetLayer()
    out = []; f = L.GetNextFeature()
    while f:
        ig = str(f.GetField("IG_DATE"))
        if ig[:4] == str(year) and (f.GetField("ACRES") or 0) >= min_acres and \
           (not wildfire_only or f.GetField("FIRE_TYPE") == "Wildfire"):
            lon = f.GetField("LONGITUDE")
            if -125 < lon < -66:                       # CONUS 만(mosaic 이 CONUS)
                out.append({k: f.GetField(k) for k in ("FIRE_ID", "FIRE_NAME", "IG_DATE",
                            "LATITUDE", "LONGITUDE", "ACRES", "PRE_ID", "POST_ID", "FIRE_TYPE")})
        f = L.GetNextFeature()
    return out


def download_mosaic(year):
    """ScienceBase 에서 해당 연도 CONUS thematic mosaic 다운로드(캐시). 완전 자율·재시도."""
    import urllib.request, json, zipfile, io, time
    dst = os.path.join(DATA, f"mtbs_CONUS_{year}.tif")
    if os.path.exists(dst):
        return dst
    last = None
    for attempt in range(4):                       # ScienceBase 간헐 실패 → 연도 전체 죽는 것 방지
        try:
            idx = json.load(urllib.request.urlopen(
                f"https://www.sciencebase.gov/catalog/items?parentId=5e91dee782ce172707f02cdd&format=json&max=60&fields=title", timeout=120))
            item = next(i for i in idx["items"] if f"for {year} " in i["title"])
            meta = json.load(urllib.request.urlopen(
                f"https://www.sciencebase.gov/catalog/item/{item['id']}?format=json&fields=files", timeout=120))
            url = next(fl["url"] for fl in meta["files"] if fl["name"] == f"mtbs_CONUS_{year}.zip")
            z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=600).read()))
            z.extract(f"mtbs_CONUS_{year}.tif", DATA)
            return dst
        except Exception as e:
            last = e; print(f"  ⚠️ mosaic {year} 다운 실패({attempt+1}/4): {str(e)[:80]} — 재시도"); time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"mosaic {year} 다운로드 4회 실패: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire_id", default="AZ3340811121020190608")   # WOODBURY 2019
    ap.add_argument("--year", default="2019")
    ap.add_argument("--batch", action="store_true", help="해당 연도 전 화재 빌드(재개가능)")
    ap.add_argument("--n", type=int, default=0, help="배치 상한(0=전부)")
    ap.add_argument("--min_acres", type=float, default=1000)
    ap.add_argument("--workers", type=int, default=1, help="화재 동시 취득 수(B200=12~24 권장)")
    a = ap.parse_args()
    if a.batch:
        ensure_fire_list()
        download_mosaic(a.year)
        fires = list_fires(a.year, a.min_acres)
        if a.n:
            fires = fires[:a.n]
        todo = [fr for fr in fires if not os.path.exists(os.path.join(OUT, f"{fr['FIRE_ID']}.npz"))]
        print(f"=== {a.year} 배치: {len(fires)} 화재 (미완 {len(todo)}) · workers {a.workers} ===")
        from threading import Lock
        cnt = {"ok": 0, "done": len(fires) - len(todo)}
        lk = Lock()

        def work(fr):
            try:
                r = build_fire(fr, a.year)
            except Exception as e:
                print(f"  ⚠️ 실패 {fr['FIRE_ID']}: {str(e)[:80]}"); r = None
            with lk:
                cnt["done"] += 1
                if r:
                    cnt["ok"] += 1
                if cnt["done"] % 25 == 0:
                    print(f"  ...진행 {cnt['done']}/{len(fires)} (성공 {cnt['ok']})")

        if a.workers <= 1:
            for fr in todo:
                work(fr)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                list(ex.map(work, todo))
        print(f"=== {a.year} 완료: 신규성공 {cnt['ok']} / 대상 {len(todo)} ===")
    else:
        build_fire(get_fire(a.fire_id), a.year)


if __name__ == "__main__":
    main()
