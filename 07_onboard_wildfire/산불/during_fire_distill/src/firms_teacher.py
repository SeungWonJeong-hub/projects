"""during-fire FIRMS teacher — 근사 카탈로그(fires_global.json)를 실제 화재일로 좁힌다.

각 대형산불 이벤트에 대해 NASA FIRMS(VIIRS/MODIS) 활성화재 탐지를 질의해서:
  · 일별 탐지수 / FRP(화재복사출력) 시계열
  · 실제 활성창(임계 이상 첫날~끝날)·피크일  ← active_hint(뉴스근사)를 대체
  · 탐지 중심(FRP 가중) ← center(근사)를 정밀화
  · 대표 during-fire 날짜(피크 ±) ← 이후 S2 취득 대상일
를 산출한다. distill 학습셋의 "언제·어디가 지금 타는 중인가" 교사 라벨.

FIRMS Area API(SP 표준아카이브)만 사용 — GEE 인증 불필요, requests 만 있으면 됨.
(패턴 출처: ../../울진산불_변화탐지/src/validate_firms.py)

키:   env FIRMS_MAP_KEY  >  ~/.firms_map_key      무료: https://firms.modaps.eosdis.nasa.gov/api/map_key
사용:  python firms_teacher.py [event_id ...]      (인자 없으면 카탈로그 전체)
      PAD=3 MIN_FRAC=0.10 python firms_teacher.py dixie_2021_us
출력:  data/firms_teacher/{event_id}.json  +  fires_global_firms.json(정밀화 카탈로그)
"""
import os, sys, io, csv, json, math, time
import datetime as dt
from collections import defaultdict
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # during_fire_distill/
CATALOG = os.path.join(ROOT, "fires_global.json")
OUTDIR = os.path.join(ROOT, "data", "firms_teacher")
REFINED = os.path.join(ROOT, "fires_global_firms.json")

# FIRMS Area API: 요청당 최대 10일, bbox="W,S,E,N". SP=standard-processing 아카이브.
SOURCES = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP"]     # 위성별 커버(연도 자동 공백)
MAX_DAYS = 5                                                 # API 청크 상한
PAD = int(os.environ.get("PAD", "3"))                         # active_hint 앞뒤 여유일
MIN_FRAC = float(os.environ.get("MIN_FRAC", "0.10"))         # 피크대비 활성창 편입 하한
MIN_COUNT = int(os.environ.get("MIN_COUNT", "2"))            # 활성일 최소 탐지수(절대 하한)
RADIUS_CAP_DEG = 1.5                                          # bbox 반경 상한(과대질의 방지)
RADIUS_MIN_DEG = 0.25                                         # 소형화재도 375m 픽셀 여유


def map_key():
    k = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if k:
        return k
    p = os.path.expanduser("~/.firms_map_key")
    if os.path.exists(p):
        return open(p).read().strip()
    sys.exit("FIRMS_MAP_KEY 없음 — env 또는 ~/.firms_map_key 에 넣으세요 "
             "(무료: https://firms.modaps.eosdis.nasa.gov/api/map_key)")


def bbox_for(ev):
    """center + area_ha → FIRMS AOI 'W,S,E,N'. 큰 화재는 큰 bbox(반경 상한 적용)."""
    lat, lon = ev["center"]
    area_ha = ev.get("area_ha", 10000) or 10000
    r_m = math.sqrt(area_ha * 1e4 / math.pi)                  # 원면적 등가 반경(m)
    r_deg = r_m / 111000.0
    r_deg = min(RADIUS_CAP_DEG, max(RADIUS_MIN_DEG, r_deg * 1.6))  # 마진 60%
    dlon = r_deg / max(0.2, math.cos(math.radians(lat)))
    return f"{lon-dlon:.3f},{lat-r_deg:.3f},{lon+dlon:.3f},{lat+r_deg:.3f}"


def daterange_chunks(d0, d1):
    """[d0,d1] 을 ≤MAX_DAYS 청크 (start_iso, days) 로."""
    cur = d0
    while cur <= d1:
        days = min(MAX_DAYS, (d1 - cur).days + 1)
        yield cur.isoformat(), days
        cur += dt.timedelta(days=days)


def conf_ok(row):
    """저신뢰 제외. VIIRS: l/n/h → n,h.  MODIS: 0-100 → ≥30."""
    c = (row.get("confidence") or "").strip().lower()
    if c in ("l", "low"):
        return False
    if c.isdigit():
        return int(c) >= 30
    return True


def fetch(key, aoi, d0, d1, log):
    """이벤트 창 전체를 소스×청크로 질의 → [(lat,lon,date,frp,src), ...]."""
    pts = []
    for src in SOURCES:
        n = 0
        for start, days in daterange_chunks(d0, d1):
            url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                   f"{key}/{src}/{aoi}/{days}/{start}")
            r = None
            for attempt in range(4):
                try:
                    r = requests.get(url, timeout=120)
                    break
                except Exception as e:
                    log(f"    {src} {start}: 요청오류 {str(e)[:50]} (재시도 {attempt+1})")
                    time.sleep(3 * (attempt + 1))
            if r is None:
                continue
            txt = r.text
            low = txt.lstrip().lower()
            if r.status_code != 200 or low.startswith("invalid") or low.startswith("error"):
                log(f"    {src} {start}: HTTP {r.status_code} / {txt[:70]!r}")
                if "over the transaction limit" in low or r.status_code == 429:
                    log("    쿼터 한도 — 30초 대기"); time.sleep(30)
                continue
            for row in csv.DictReader(io.StringIO(txt)):
                if not conf_ok(row):
                    continue
                try:
                    frp = float(row.get("frp") or 0.0)
                    pts.append((float(row["latitude"]), float(row["longitude"]),
                                row.get("acq_date", "").strip(), frp, src))
                    n += 1
                except (KeyError, ValueError):
                    pass
        log(f"    {src}: {n}점")
    return pts


def summarize(ev_id, ev, pts):
    """탐지점 → 일별 시계열·활성창·피크·정밀중심·대표 during-fire 날짜."""
    by_day = defaultdict(lambda: {"count": 0, "frp_sum": 0.0, "frp_max": 0.0})
    lat_w = lon_w = wsum = 0.0
    for lat, lon, date, frp, src in pts:
        if not date:
            continue
        d = by_day[date]
        d["count"] += 1
        d["frp_sum"] += frp
        d["frp_max"] = max(d["frp_max"], frp)
        w = frp + 1.0                                         # FRP 가중(0 방지 +1)
        lat_w += lat * w; lon_w += lon * w; wsum += w
    if not by_day:
        return {"event": ev_id, "name": ev.get("name"), "n_points": 0,
                "status": "no_detections", "hint": ev.get("active_hint"),
                "daily": {}, "active_window": None, "peak_day": None,
                "center_hint": ev.get("center"), "center_firms": None,
                "during_fire_dates": []}

    days_sorted = sorted(by_day)
    peak_day = max(by_day, key=lambda k: by_day[k]["count"])
    peak_count = by_day[peak_day]["count"]
    thresh = max(MIN_COUNT, MIN_FRAC * peak_count)
    active = [d for d in days_sorted if by_day[d]["count"] >= thresh]
    window = [active[0], active[-1]] if active else [days_sorted[0], days_sorted[-1]]
    center_firms = [round(lat_w / wsum, 4), round(lon_w / wsum, 4)]
    # 대표 during-fire 취득 후보일 = 활성일 중 탐지수 상위(최대 6일), 날짜순
    top = sorted(active or days_sorted,
                 key=lambda d: by_day[d]["count"], reverse=True)[:6]
    during_dates = sorted(top)
    daily = {d: {"count": v["count"], "frp_sum": round(v["frp_sum"], 1),
                 "frp_max": round(v["frp_max"], 1)} for d, v in
             sorted(by_day.items())}
    return {
        "event": ev_id, "name": ev.get("name"), "country": ev.get("country"),
        "n_points": len(pts), "status": "ok",
        "hint": ev.get("active_hint"),
        "active_window": window,                              # ← 실제 화재일(FIRMS)
        "peak_day": peak_day, "peak_count": peak_count,
        "center_hint": ev.get("center"), "center_firms": center_firms,
        "during_fire_dates": during_dates,                   # ← S2 취득 대상일
        "n_active_days": len(active), "daily": daily,
    }


def main():
    cat = json.load(open(CATALOG, encoding="utf-8"))
    events = cat["events"]
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    ids = wanted or list(events)
    unknown = [i for i in ids if i not in events]
    if unknown:
        sys.exit(f"카탈로그에 없는 event_id: {unknown}\n가능: {list(events)}")

    os.makedirs(OUTDIR, exist_ok=True)
    key = map_key()
    force = os.environ.get("FORCE", "") == "1"
    # 재개: 기존 정밀카탈로그 이어받기(중단/세션종료 대비). FORCE=1 이면 전면 재질의.
    refined = {"_about": "firms_teacher.py 산출 — active_hint→FIRMS 실제 활성창, "
                         "center→FIRMS FRP가중 중심, during_fire_dates=S2 취득 대상일.",
               "events": {}}
    if os.path.exists(REFINED) and not force:
        try:
            refined["events"] = json.load(open(REFINED, encoding="utf-8")).get("events", {})
        except Exception:
            pass
    print(f"FIRMS teacher: {len(ids)}개 이벤트 (재개={'off' if force else 'on'})", flush=True)
    for i, ev_id in enumerate(ids, 1):
        ev = events[ev_id]
        # 이미 확정된 이벤트는 건너뜀(개별 json 존재 + status ok)
        done_p = os.path.join(OUTDIR, f"{ev_id}.json")
        if not force and os.path.exists(done_p):
            try:
                prev = json.load(open(done_p, encoding="utf-8"))
                if prev.get("status") == "ok":
                    # 건너뛰더라도 카탈로그에는 병합(누락 방지)
                    e2 = dict(ev)
                    e2["firms_status"] = prev.get("status")
                    e2["active_window"] = prev.get("active_window")
                    e2["peak_day"] = prev.get("peak_day")
                    e2["center_firms"] = prev.get("center_firms")
                    e2["during_fire_dates"] = prev.get("during_fire_dates")
                    refined["events"][ev_id] = e2
                    json.dump(refined, open(REFINED, "w"), ensure_ascii=False, indent=1)
                    print(f"[{i}/{len(ids)}] {ev_id} — 완료됨, 건너뜀(카탈로그 병합)", flush=True)
                    continue
            except Exception:
                pass
        d0 = dt.date.fromisoformat(ev["active_hint"][0]) - dt.timedelta(days=PAD)
        d1 = dt.date.fromisoformat(ev["active_hint"][1]) + dt.timedelta(days=PAD)
        aoi = bbox_for(ev)
        print(f"[{i}/{len(ids)}] {ev_id} {ev.get('name')}  {d0}~{d1}  aoi={aoi}",
              flush=True)
        log = lambda m: print(m, flush=True)
        pts = fetch(key, aoi, d0, d1, log)
        summ = summarize(ev_id, ev, pts)
        json.dump(summ, open(os.path.join(OUTDIR, f"{ev_id}.json"), "w"),
                  ensure_ascii=False, indent=1)
        if summ["status"] == "ok":
            print(f"    → 활성창 {summ['active_window'][0]}~{summ['active_window'][1]} "
                  f"(hint {ev['active_hint'][0]}~{ev['active_hint'][1]}), "
                  f"피크 {summ['peak_day']}({summ['peak_count']}), "
                  f"during {len(summ['during_fire_dates'])}일", flush=True)
        else:
            print("    → FIRMS 탐지 없음(창/좌표 재확인 필요)", flush=True)
        # 정밀화 카탈로그 갱신(증분저장 — 타임아웃/쿼터 대비)
        e2 = dict(ev)
        e2["firms_status"] = summ["status"]
        if summ["status"] == "ok":
            e2["active_window"] = summ["active_window"]
            e2["peak_day"] = summ["peak_day"]
            e2["center_firms"] = summ["center_firms"]
            e2["during_fire_dates"] = summ["during_fire_dates"]
        refined["events"][ev_id] = e2
        json.dump(refined, open(REFINED, "w"), ensure_ascii=False, indent=1)

    ok = sum(1 for e in refined["events"].values() if e.get("firms_status") == "ok")
    print(f"\n완료: {ok}/{len(ids)} 확정 → {REFINED}\n"
          f"개별 시계열 → {OUTDIR}/\n"
          f"다음: during_fire_dates 로 S2 씬 취득(stac.py) → distill 학습셋", flush=True)


if __name__ == "__main__":
    main()
