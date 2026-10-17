"""나라단위 소각영역 수집 ① — 권위 산출물(Tier-1) -> 통일 이벤트 카탈로그 CSV.

계획문서 §2 의 이벤트 레코드 중 "영상 없이 정해지는 부분"만 만든다:
    {country, source, event_id, name, ig_date, lon, lat, bbox, area_ha, label, kind, qc}
pre/post 씬 id·소각마스크·최종 area_ha 는 다음 단계(collect_events.py)가 채운다.

나라 추가는 READERS 에 리더 하나만 붙이면 된다(US=MTBS 구현, EU/CA/AU 는 자리만).

사용:
    collect_catalog.py US                      # 기본 필터로 ../collect/catalog_US.csv
    collect_catalog.py US --min-ha 200 --from 2018 --kinds wildfire,wfu
    collect_catalog.py US --all                # 필터 없이 전량(qc 플래그만 기록)
"""
from __future__ import annotations
import os, sys, csv, math, json, argparse

# conda activate 없이 envs/geo/bin/python 을 직접 부르면 PROJ_DATA 가 비어 좌표변환이
# 조용히 죽는다(“proj_create_from_database: Open of ... failed”). 여기서 보정한다.
for _v, _sub in (("PROJ_DATA", "share/proj"), ("GDAL_DATA", "share/gdal")):
    if not os.environ.get(_v):
        _p = os.path.join(sys.prefix, _sub)
        if os.path.isdir(_p):
            os.environ[_v] = _p
from osgeo import ogr, osr                                            # noqa: E402
ogr.UseExceptions(); osr.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUTDIR = os.environ.get("COLLECT_DIR", os.path.join(ROOT, "collect"))

COLUMNS = ["country", "source", "event_id", "name", "ig_date", "lon", "lat",
           "xmin", "ymin", "xmax", "ymax", "area_ha", "area_ha_src", "label", "kind", "qc"]

# S2 L2A(earth-search) 아카이브가 pre 씬을 대주는 첫 해 — 나라마다 다르다.
# US: 실측(2026-09-08) CONUS 3지점 모두 2016년 0장 / 2017년 희박 / 2018년부터 안정.
#     발화년 N 은 전년(N-1) pre 를 먼저 노리므로 2018 부터 잡는다.
S2_FROM = {"US": 2018}
S2_FROM_DEFAULT = 2018
HUGE_HA = 50_000.0        # 타일 수가 폭발하는 초대형 — 제외가 아니라 플래그
MISMATCH = 0.25           # 폴리곤 면적 vs 기관 기재 면적 괴리


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def _tx(epsg_src, epsg_dst):
    s = osr.SpatialReference(); s.ImportFromEPSG(epsg_src)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    d = osr.SpatialReference(); d.ImportFromEPSG(epsg_dst)
    d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(s, d)


def poly_area_ha(geom, epsg_src, lon, lat):
    """실제 폴리곤 면적(ha). 중심 UTM 으로 옮겨 잰다 — 기관 기재 면적은 경계 외접값이라 다르다."""
    g = geom.Clone()
    g.Transform(_tx(epsg_src, utm_epsg(lon, lat)))
    return g.GetArea() / 10_000.0


# ---------------------------------------------------------------- US = MTBS
MTBS_SHP = os.environ.get("MTBS_SHP", os.path.join(ROOT, "mtbs", "mtbs_perims_DD.shp"))
MTBS_KIND = {"Wildfire": "wildfire", "Wildland Fire Use": "wfu",
             "Prescribed Fire": "prescribed", "Other": "other"}
ACRE_HA = 0.40468564224


def read_mtbs():
    """MTBS 30k+ 화재 경계 -> 이벤트 레코드 제너레이터. 폴리곤은 shp 에 두고 id 로 참조한다."""
    if not os.path.exists(MTBS_SHP):
        sys.exit(f"MTBS 경계 shp 없음: {MTBS_SHP}\n"
                 f"  mtbs_perimeter_data.zip 을 {os.path.dirname(MTBS_SHP)} 에 풀 것")
    ds = ogr.Open(MTBS_SHP)
    ly = ds.GetLayer()
    epsg_src = int(ly.GetSpatialRef().GetAuthorityCode(None) or 4269)
    for f in ly:
        g = f.GetGeometryRef()
        if g is None:
            continue
        lon, lat = float(f.GetField("burnbndlon")), float(f.GetField("burnbndlat"))
        x1, x2, y1, y2 = g.GetEnvelope()
        asmnt = f.GetField("asmnt_type") or ""
        qc = []
        if asmnt.startswith("Initial"):
            qc.append("initial")          # 진화 직후 판독 — 연장 연소를 놓쳤을 수 있음
        if "(SS)" in asmnt:
            qc.append("single_scene")
        yield {
            "country": "US", "source": "MTBS",
            "event_id": f.GetField("event_id"),
            "name": (f.GetField("incid_name") or "").strip(),
            "ig_date": (f.GetField("ig_date") or "")[:10],
            "lon": lon, "lat": lat,
            "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
            "area_ha": round(poly_area_ha(g, epsg_src, lon, lat), 1),
            "area_ha_src": round((f.GetField("burnbndac") or 0) * ACRE_HA, 1),
            "label": "perimeter",
            "kind": MTBS_KIND.get(f.GetField("incid_type"), "other"),
            "qc": qc,
        }


READERS = {"US": read_mtbs}   # EU=CEMS/EFFIS, CA=NBAC, AU=DEA/NAFI 는 같은 dict 를 반환하면 붙는다


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country", choices=sorted(READERS))
    ap.add_argument("--out")
    ap.add_argument("--min-ha", type=float, default=50.0)
    ap.add_argument("--from", dest="year_from", type=int, default=None)
    ap.add_argument("--to", dest="year_to", type=int, default=9999)
    ap.add_argument("--kinds", default="wildfire,wfu",
                    help="쉼표 구분. 'all' 이면 처방화입·기타까지 포함")
    ap.add_argument("--all", action="store_true", help="필터 없이 전량 기록(qc 만 남김)")
    a = ap.parse_args()

    s2from = S2_FROM.get(a.country, S2_FROM_DEFAULT)
    if a.year_from is None:
        a.year_from = s2from
    kinds = None if (a.all or a.kinds == "all") else set(a.kinds.split(","))
    os.makedirs(OUTDIR, exist_ok=True)
    out = a.out or os.path.join(OUTDIR, f"catalog_{a.country}.csv")

    rej = {"kind": 0, "year": 0, "small": 0, "no_date": 0}
    rows, n_src = [], 0
    for r in READERS[a.country]():
        n_src += 1
        y = int(r["ig_date"][:4]) if r["ig_date"][:4].isdigit() else 0
        if r["area_ha"] >= HUGE_HA:
            r["qc"].append("huge")
        src = r["area_ha_src"]
        if src > 0 and abs(r["area_ha"] - src) / src > MISMATCH:
            r["qc"].append("area_mismatch")
        if y < s2from:
            r["qc"].append("pre_s2")
        if not a.all:
            if kinds and r["kind"] not in kinds:
                rej["kind"] += 1; continue
            if not y:
                rej["no_date"] += 1; continue
            if y < a.year_from or y > a.year_to:
                rej["year"] += 1; continue
            if r["area_ha"] < a.min_ha:
                rej["small"] += 1; continue
        r["qc"] = "|".join(r["qc"])
        rows.append(r)

    rows.sort(key=lambda r: (r["ig_date"], r["event_id"]))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader(); w.writerows(rows)

    meta = {"country": a.country, "source": rows[0]["source"] if rows else None,
            "n_source": n_src, "n_kept": len(rows), "rejected": rej,
            "filters": {"min_ha": a.min_ha, "years": [a.year_from, a.year_to],
                        "kinds": sorted(kinds) if kinds else "all"},
            "years": {}, "total_ha": round(sum(r["area_ha"] for r in rows), 1)}
    for r in rows:
        meta["years"][r["ig_date"][:4]] = meta["years"].get(r["ig_date"][:4], 0) + 1
    with open(out.replace(".csv", ".json"), "w") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)

    print(f"[카탈로그] {out}")
    print(f"  원본 {n_src} -> 채택 {len(rows)} (제외 {rej})")
    print(f"  연도별 {meta['years']}")
    print(f"  소각면적 합계 {meta['total_ha']:,.0f} ha")
    qc = {}
    for r in rows:
        for t in filter(None, r["qc"].split("|")):
            qc[t] = qc.get(t, 0) + 1
    print(f"  QC 플래그 {qc}")


if __name__ == "__main__":
    main()
