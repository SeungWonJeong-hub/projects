"""나라단위 소각영역 수집 ② — 이벤트 카탈로그 -> S2 pre/post 취득 -> GT 타일 + 매니페스트.

collect_catalog.py 가 만든 CSV 를 읽어 화재 하나씩:
  ① 이벤트 UTM 격자 확정(화재마다 존이 다르다) ② STAC 최저구름 pre/post ③ 권위 폴리곤 래스터화
  ④ valid(반사율>0 & SCL) ⑤ 타일 저장 ⑥ 매니페스트 1행 추가

**학습/서빙 계약을 그대로 따른다**(ko_build_refl.py 와 동일):
  x=(12,T,T) int16 raw DN [pre B02,B03,B04,<NIR>,B11,B12 | post 동일], y=(T,T) uint8, valid=(T,T)
  20m -> NIR=B8A·T=256 / 10m -> NIR=B08·T=512 (지상 시야 T*GSD 를 5.12km 로 고정)

재개 가능: 매니페스트에 있는 event_id 는 건너뛴다. 샤딩으로 여러 프로세스 병렬.

사용:
    collect_events.py US                                  # 20m 기본
    GSD=10 NIR=B08 T=512 collect_events.py US --limit 50
    SHARD_N=8 SHARD_I=3 collect_events.py US &            # 8분할 병렬
"""
from __future__ import annotations
import os, sys, csv, math, glob, json, argparse, datetime as dt, socket

for _v, _sub in (("PROJ_DATA", "share/proj"), ("GDAL_DATA", "share/gdal")):
    if not os.environ.get(_v):                       # conda activate 없이 부를 때의 좌표변환 사망 방지
        _p = os.path.join(sys.prefix, _sub)
        if os.path.isdir(_p):
            os.environ[_v] = _p
import numpy as np                                                   # noqa: E402
from osgeo import gdal, ogr, osr                                     # noqa: E402
from pystac_client import Client                                     # noqa: E402
gdal.UseExceptions()
socket.setdefaulttimeout(90)
for _k, _v2 in {"GDAL_HTTP_TIMEOUT": "40", "GDAL_HTTP_CONNECTTIMEOUT": "15",
                "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "3",
                "CPL_VSIL_CURL_USE_HEAD": "NO", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
                "AWS_NO_SIGN_REQUEST": "YES"}.items():
    gdal.SetConfigOption(_k, _v2)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
COLLECT = os.environ.get("COLLECT_DIR", os.path.join(ROOT, "collect"))

GSD = float(os.environ.get("GSD", "20"))
NIR = os.environ.get("NIR", "B8A")                    # 20m=B8A(865nm) / 10m=B08(842nm)
NIR_ASSET = {"B8A": "nir08", "B08": "nir"}[NIR]
T = int(os.environ.get("T", "256"))                   # 지상 시야 = T*GSD
STRIDE = int(os.environ.get("STRIDE", str(T)))
CROP_M = float(os.environ.get("CROP_M", "2650"))      # 최소 반크롭(작은 화재도 문맥 확보)
MAX_SPAN_M = float(os.environ.get("MAX_SPAN_M", "60000"))   # 초대형 화재 크롭 상한(한 변)
MIN_VALID = float(os.environ.get("MIN_VALID", "0.4"))
SHARD_N = int(os.environ.get("SHARD_N", "1")); SHARD_I = int(os.environ.get("SHARD_I", "0"))
BANDS = [("B02", "blue"), ("B03", "green"), ("B04", "red"), (NIR, NIR_ASSET),
         ("B11", "swir16"), ("B12", "swir22")]
ORDER = ["B02", "B03", "B04", NIR, "B11", "B12"]
SCL_BAD = [0, 1, 3, 8, 9, 10, 11]
CAT = Client.open("https://earth-search.aws.element84.com/v1")

MANIFEST_COLS = ["event_id", "country", "source", "name", "ig_date", "epsg", "gsd", "tile",
                 "pre_date", "post_date", "pre_scenes", "post_scenes", "n_tiles",
                 "gt_ha", "grid_ha", "burn_frac", "valid_frac", "status", "note"]


def rfc(a, b):
    return f"{a}T00:00:00Z/{b}T23:59:59Z"


def vsi(u):
    return "/vsis3/" + u[5:] if u.startswith("s3://") else "/vsicurl/" + u


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def _srs(epsg):
    s = osr.SpatialReference(); s.ImportFromEPSG(epsg)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


# ------------------------------------------------------------ 폴리곤 공급자
def mtbs_geoms(ids):
    """event_id -> UTM 이전 원본 폴리곤. 30k 피처를 한 번만 훑는다(건별 필터는 O(n) 반복이라 느리다)."""
    shp = os.environ.get("MTBS_SHP", os.path.join(ROOT, "mtbs", "mtbs_perims_DD.shp"))
    ds = ogr.Open(shp); ly = ds.GetLayer()
    epsg = int(ly.GetSpatialRef().GetAuthorityCode(None) or 4269)
    want, out = set(ids), {}
    for f in ly:
        e = f.GetField("event_id")
        if e in want:
            g = f.GetGeometryRef()
            if g is not None:
                out[e] = g.Clone()
    return out, epsg


GEOMS = {"MTBS": mtbs_geoms}


# ------------------------------------------------------------------- 씬 취득
def stac_pick(bbox_ll, rng, ute, epsg):
    """최저구름 1장 + 같은 날짜 이웃타일 모자이크 -> (밴드dict, 날짜, 씬id들). 없으면 None."""
    its = list(CAT.search(collections=["sentinel-2-l2a"], bbox=bbox_ll, datetime=rng,
                          query={"eo:cloud_cover": {"lt": 30}},
                          sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
                          max_items=1).items())
    if not its:
        return None
    d = its[0].datetime.date()
    sib = list(CAT.search(collections=["sentinel-2-l2a"], bbox=bbox_ll,
                          datetime=rfc(d - dt.timedelta(days=2), d + dt.timedelta(days=2)),
                          query={"eo:cloud_cover": {"lt": 40}}, max_items=10).items()) or its
    b = {}
    for k, a in BANDS + [("SCL", "scl")]:
        hrefs = [(x.assets[a].href if a in x.assets else x.assets[k].href)
                 for x in sib if (a in x.assets or k in x.assets)]
        if not hrefs:
            return None
        tmp = f"/tmp/_ce_{os.getpid()}_{k}.tif"
        gdal.Warp(tmp, [vsi(h) for h in hrefs], dstSRS=f"EPSG:{epsg}", xRes=GSD, yRes=GSD,
                  outputBounds=ute, resampleAlg="average", format="GTiff", srcNodata=0)
        b[k] = gdal.Open(tmp).ReadAsArray().astype(np.float32)
        os.remove(tmp)
    return b, d, [x.id for x in sib]


# ------------------------------------------------------------------ 이벤트 1건
def build_one(rec, geom, geom_epsg, outdir):
    lon, lat = float(rec["lon"]), float(rec["lat"])
    epsg = utm_epsg(lon, lat)
    gu = geom.Clone(); gu.Transform(osr.CoordinateTransformation(_srs(geom_epsg), _srs(epsg)))
    e = gu.GetEnvelope()
    cx, cy = (e[0] + e[1]) / 2, (e[2] + e[3]) / 2
    hx = min(max((e[1] - e[0]) / 2 + 500, CROP_M), MAX_SPAN_M / 2)
    hy = min(max((e[3] - e[2]) / 2 + 500, CROP_M), MAX_SPAN_M / 2)
    ute = [math.floor(cx - hx), math.floor(cy - hy), math.ceil(cx + hx), math.ceil(cy + hy)]
    ct = osr.CoordinateTransformation(_srs(epsg), _srs(4326))
    lo1, la1, _ = ct.TransformPoint(ute[0], ute[1]); lo2, la2, _ = ct.TransformPoint(ute[2], ute[3])
    bbox_ll = [min(lo1, lo2) - 0.02, min(la1, la2) - 0.02, max(lo1, lo2) + 0.02, max(la1, la2) + 0.02]

    fdate = dt.date.fromisoformat(rec["ig_date"])
    pc = fdate.replace(year=fdate.year - 1)
    pre = stac_pick(bbox_ll, rfc(pc - dt.timedelta(days=30), pc + dt.timedelta(days=45)), ute, epsg)
    if pre is None:      # 전년 동기 공백 -> 같은 해 발화 직전
        pre = stac_pick(bbox_ll, rfc(fdate - dt.timedelta(days=60), fdate - dt.timedelta(days=8)), ute, epsg)
    if pre is None:
        return {"status": "no_pre"}
    post = stac_pick(bbox_ll, rfc(fdate + dt.timedelta(days=15), fdate + dt.timedelta(days=100)), ute, epsg)
    if post is None:
        return {"status": "no_post"}
    (pb, pd, pids), (qb, qd, qids) = pre, post

    H = min(pb["B02"].shape[0], qb["B02"].shape[0]); W = min(pb["B02"].shape[1], qb["B02"].shape[1])
    for d_ in (pb, qb):
        for k in d_:
            d_[k] = d_[k][:H, :W]
    srs = _srs(epsg)
    mem = ogr.GetDriverByName("Memory").CreateDataSource("m")
    lyr = mem.CreateLayer("f", srs, ogr.wkbPolygon)
    ft = ogr.Feature(lyr.GetLayerDefn()); ft.SetGeometry(gu); lyr.CreateFeature(ft)
    rast = gdal.GetDriverByName("MEM").Create("", W, H, 1, gdal.GDT_Byte)
    rast.SetGeoTransform((ute[0], GSD, 0, ute[3], 0, -GSD)); rast.SetProjection(srs.ExportToWkt())
    gdal.RasterizeLayer(rast, [1], lyr, burn_values=[1])
    y = (rast.ReadAsArray()[:H, :W] > 0).astype(np.uint8)

    valid = ((pb["B02"] > 0) & (pb[NIR] > 0) & (pb["B12"] > 0) &
             (qb["B02"] > 0) & (qb[NIR] > 0) & (qb["B12"] > 0))
    valid &= ~np.isin(pb["SCL"], SCL_BAD) & ~np.isin(qb["SCL"], SCL_BAD)
    x = np.stack([pb[k] for k in ORDER] + [qb[k] for k in ORDER], 0).astype(np.int16)

    out = {"epsg": epsg, "pre_date": pd, "post_date": qd,
           "pre_scenes": "|".join(pids), "post_scenes": "|".join(qids),
           "grid_ha": round(float(y.sum()) * GSD * GSD / 1e4, 1),
           "burn_frac": round(float(y[valid].mean()) if valid.any() else 0.0, 4),
           "valid_frac": round(float(valid.mean()), 4)}
    if H < T or W < T:
        return {**out, "status": "crop_small", "note": f"{H}x{W}<{T}"}

    os.makedirs(outdir, exist_ok=True)
    ys = sorted(set(min(v, H - T) for v in range(0, H - T + 1, STRIDE)))
    xs = sorted(set(min(v, W - T) for v in range(0, W - T + 1, STRIDE)))
    n = 0
    for yy in ys:
        for xx in xs:
            v = valid[yy:yy + T, xx:xx + T]
            if v.mean() < MIN_VALID:
                continue
            np.savez_compressed(os.path.join(outdir, f"{os.path.basename(outdir)}_{yy}_{xx}.npz"),
                                x=x[:, yy:yy + T, xx:xx + T],
                                y=(y[yy:yy + T, xx:xx + T] & v).astype(np.uint8),
                                valid=v.astype(np.uint8))
            n += 1
    if n == 0:
        os.rmdir(outdir)
        return {**out, "status": "no_tiles"}
    return {**out, "n_tiles": n, "status": "ok"}


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country")
    ap.add_argument("--catalog")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N건만(스모크 테스트용)")
    ap.add_argument("--include-huge", action="store_true", help="qc=huge 도 시도")
    ap.add_argument("--merge", action="store_true", help="샤드 매니페스트를 하나로 합치고 종료")
    a = ap.parse_args()

    cat = a.catalog or os.path.join(COLLECT, f"catalog_{a.country}.csv")
    if not os.path.exists(cat):
        sys.exit(f"카탈로그 없음: {cat}  (collect_catalog.py {a.country} 먼저)")
    tag = f"{a.country}_{int(GSD)}m_{NIR.lower()}_t{T}"
    outroot = a.out or os.path.join(COLLECT, tag)
    # 샤드마다 매니페스트를 갈라 쓴다 — 한 파일에 동시 append 하면 행이 섞여 재개가 망가진다.
    # 합치기: collect_events.py US --merge  (또는 그냥 cat, 헤더 1줄만 남기면 된다)
    man = os.path.join(COLLECT, f"manifest_{tag}" + (f"_s{SHARD_I}" if SHARD_N > 1 else "") + ".csv")
    os.makedirs(outroot, exist_ok=True)

    if a.merge:
        parts = sorted(glob.glob(os.path.join(COLLECT, f"manifest_{tag}_s*.csv")))
        rows = [r for p_ in parts for r in csv.DictReader(open(p_))]
        rows.sort(key=lambda r: (r["ig_date"], r["event_id"]))
        dst = os.path.join(COLLECT, f"manifest_{tag}.csv")
        with open(dst, "w", newline="") as fh2:
            w2 = csv.DictWriter(fh2, fieldnames=MANIFEST_COLS); w2.writeheader(); w2.writerows(rows)
        tal = {}
        for r in rows:
            tal[r["status"]] = tal.get(r["status"], 0) + 1
        print(f"[병합] 샤드 {len(parts)} -> {dst} | {len(rows)}건 {tal}")
        print(f"  타일 합계 {sum(int(r['n_tiles'] or 0) for r in rows)}")
        return

    recs = [r for r in csv.DictReader(open(cat))]
    if not a.include_huge:
        recs = [r for r in recs if "huge" not in r["qc"]]
    if a.limit:
        recs = recs[:a.limit]
    recs = recs[SHARD_I::SHARD_N]

    done = set()
    if os.path.exists(man):
        done = {r["event_id"] for r in csv.DictReader(open(man))}
    todo = [r for r in recs if r["event_id"] not in done]
    print(f"[{tag}] 카탈로그 {len(recs)}건(샤드 {SHARD_I}/{SHARD_N}) | 완료 {len(recs)-len(todo)} | 남은 {len(todo)}", flush=True)
    if not todo:
        return

    geoms, gepsg = GEOMS[todo[0]["source"]]([r["event_id"] for r in todo])
    print(f"  폴리곤 {len(geoms)}/{len(todo)} 적재 (EPSG:{gepsg})", flush=True)

    new = not os.path.exists(man)
    fh = open(man, "a", newline=""); w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
    if new:
        w.writeheader(); fh.flush()
    tally = {}
    for i, r in enumerate(todo):
        name = f"{r['country'].lower()}_{r['ig_date'].replace('-', '')}_{r['event_id']}"
        row = {c: "" for c in MANIFEST_COLS}
        row.update({k: r[k] for k in ("event_id", "country", "source", "name", "ig_date")})
        row.update({"gsd": int(GSD), "tile": T, "gt_ha": r["area_ha"], "n_tiles": 0})
        g = geoms.get(r["event_id"])
        if g is None:
            row["status"] = "no_geom"
        elif glob.glob(os.path.join(outroot, name, "*.npz")):
            row["status"] = "ok"; row["n_tiles"] = len(glob.glob(os.path.join(outroot, name, "*.npz")))
        else:
            try:
                row.update(build_one(r, g, gepsg, os.path.join(outroot, name)))
            except Exception as ex:
                row["status"] = "error"; row["note"] = str(ex)[:200]
        w.writerow(row); fh.flush()
        tally[row["status"]] = tally.get(row["status"], 0) + 1
        print(f"[{i+1}/{len(todo)}] {name} {row['status']} 타일={row['n_tiles']} "
              f"불탄={row.get('burn_frac','')} valid={row.get('valid_frac','')}", flush=True)
    fh.close()
    print(f"[{tag}] 완료 {tally} -> {man}", flush=True)
    with open(man.replace(".csv", ".json"), "w") as f2:
        json.dump({"tag": tag, "gsd": GSD, "nir": NIR, "tile": T, "tally": tally}, f2, indent=1)


if __name__ == "__main__":
    main()
