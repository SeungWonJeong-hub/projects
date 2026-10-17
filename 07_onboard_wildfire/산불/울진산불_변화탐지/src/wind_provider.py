"""현재 바람 provider — 기상청 초단기실황(KMA) / ERA5 폴백 통합.

진화 솔루션(헬기 물투하·확산)은 '현재 바람'이 핵심 입력이다.
  · 라이브(실시간 화재): 기상청 단기예보 조회서비스의 초단기실황 getUltraSrtNcst
    → 관측기반 '지금 이 순간' 풍향(VEC)·풍속(WSD)·습도(REH). 국내 최적밀도.
  · 아카이브(과거 back-test, 예 울진2022): KMA 초단기실황은 최근자료만 제공하므로
    2022 같은 과거는 ERA5 재분석 캐시(uljin_wind_era5.json)의 해당시각을 '현재 바람'의
    대역으로 쓴다. 라이브 배포 시 키만 있으면 KMA 경로가 자동으로 켜진다.

인증키: 환경변수 KMA_API_KEY 또는 ~/.kma_api_key (Decoding 일반 인증키). 둘 다 없으면 ERA5 폴백.
  ※ 키는 코드·커밋·외부전송에 절대 넣지 않는다(파일은 .gitignore).

반환 규약(전 경로 동일): dict(
    toward_deg  = 바람이 '향하는'(풍하) 나침반각 [0=북,90=동] — 확산/투하가 쓰는 각,
    from_deg    = 바람이 '불어오는' 각(기상 관례),
    speed_ms    = 풍속 m/s,
    rh_pct      = 상대습도 %,
    source      = "KMA-ncst" | "ERA5-archive" | "fallback-const",
    obs_time    = ISO 관측시각)
"""
import os, json, math, urllib.request, urllib.parse
from datetime import datetime, timedelta

HERE = __file__.rsplit("/", 2)[0]
ERA5_CACHE = HERE + "/data/uljin_route/uljin_wind_era5.json"      # 울진 데모용(있으면 최종 폴백)
WIND_CACHE_DIR = HERE + "/data/wind_cache"                        # 화재무관 ASOS 캐시(지점·기간 키)
VILAGE_ENDPOINT = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"   # 단기예보(초단기실황=현재바람)
ASOS_ENDPOINT = "https://apis.data.go.kr/1360000/AsosHourlyInfoService"         # 종관 시간자료(과거 실측)

# 전국 ASOS 주요 지점(stnId, lat, lon) — 화재 위치에서 최근접 선택. 산불다발 동부·남부 조밀.
ASOS_STATIONS = {
    90:(38.25,128.56),100:(37.68,128.72),101:(37.90,127.74),105:(37.75,128.89),108:(37.57,126.97),
    112:(37.48,126.62),114:(37.34,127.95),119:(37.27,126.99),129:(36.78,126.49),130:(36.99,129.41),
    131:(36.64,127.44),133:(36.37,127.37),135:(36.22,127.99),136:(36.57,128.71),138:(36.03,129.38),
    140:(35.99,126.76),143:(35.88,128.65),146:(35.84,127.12),152:(35.58,129.33),156:(35.17,126.89),
    159:(35.10,129.03),162:(34.84,128.44),165:(34.82,126.38),168:(34.74,127.74),
    201:(37.71,126.45),202:(37.49,127.49),211:(38.06,128.17),212:(37.68,127.88),216:(37.48,128.83),
    221:(37.16,128.19),226:(36.49,127.73),232:(36.76,127.12),235:(36.33,126.56),236:(36.27,126.92),
    238:(36.10,127.48),243:(35.73,126.72),244:(35.61,127.29),245:(35.56,126.84),247:(35.42,127.40),
    260:(34.69,126.92),261:(34.55,126.57),262:(34.62,127.28),271:(36.94,128.91),272:(36.87,128.52),
    273:(36.63,128.15),277:(36.53,129.41),278:(36.36,128.69),279:(36.13,128.32),281:(35.98,128.95),
    283:(35.84,129.22),284:(35.67,127.91),285:(35.57,128.17),288:(35.49,128.74),289:(35.41,127.88),
    294:(34.89,128.60),295:(34.82,127.93),
}


def nearest_station(lat, lon):
    return min(ASOS_STATIONS, key=lambda s: (ASOS_STATIONS[s][0]-lat)**2 + (ASOS_STATIONS[s][1]-lon)**2)


# ---------------------------------------------------------------- 인증키
def _load_key():
    k = os.environ.get("KMA_API_KEY")
    if k and k.strip():
        return k.strip()
    for p in (os.path.expanduser("~/.kma_api_key"),):
        if os.path.exists(p):
            s = open(p).read().strip()
            if s:
                return s
    return None


def _enc_key(key):
    """Encoding/Decoding 어느 쪽을 줘도 올바른 URL 인코딩 1회로 정규화(이중인코딩 방지)."""
    return urllib.parse.quote(urllib.parse.unquote(key), safe="")


def _get_json(endpoint, op, params, timeout=15):
    """serviceKey는 정규화해 직접 부착(urlencode 이중인코딩 회피), 나머지는 urlencode."""
    key = _enc_key(_load_key())
    qs = urllib.parse.urlencode({**params, "dataType": "JSON"})
    url = f"{endpoint}/{op}?serviceKey={key}&{qs}"
    raw = urllib.request.urlopen(url, timeout=timeout).read()
    return json.loads(raw)


# ---------------------------------------------------------------- KMA LCC 격자 변환
# 기상청 동네예보 격자(5km) 표준 변환 dfs_xy_conv (기상청 배포 알고리즘).
def latlon_to_grid(lat, lon):
    RE, GRID = 6371.00877, 5.0
    SLAT1, SLAT2, OLON, OLAT, XO, YO = 30.0, 60.0, 126.0, 38.0, 43, 136
    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1, slat2 = SLAT1 * DEGRAD, SLAT2 * DEGRAD
    olon, olat = OLON * DEGRAD, OLAT * DEGRAD
    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = (sf ** sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / (ro ** sn)
    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / (ra ** sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn
    nx = int(ra * math.sin(theta) + XO + 0.5)
    ny = int(ro - ra * math.cos(theta) + YO + 0.5)
    return nx, ny


def _ncst_base(now):
    """초단기실황 base_date/base_time: 매시 정시 생성·약 40분 후 제공 → 안전하게 40분 이전이면 직전 정시."""
    t = now - timedelta(minutes=40)
    return t.strftime("%Y%m%d"), t.strftime("%H00")


# ---------------------------------------------------------------- KMA 실황
def _kma_current(lat, lon, now=None, timeout=15):
    """초단기실황(현재 관측): 풍향 VEC·풍속 WSD·습도 REH. 라이브 화재용."""
    if not _load_key():
        return None
    now = now or datetime.now()
    base_date, base_time = _ncst_base(now)
    nx, ny = latlon_to_grid(lat, lon)
    try:
        j = _get_json(VILAGE_ENDPOINT, "getUltraSrtNcst", {
            "numOfRows": 100, "pageNo": 1,
            "base_date": base_date, "base_time": base_time, "nx": nx, "ny": ny,
        }, timeout=timeout)
        items = j["response"]["body"]["items"]["item"]
    except Exception as e:
        print(f"  [wind] KMA 실황 실패({str(e)[:90]}) → 폴백")
        return None
    vals = {it["category"]: float(it["obsrValue"]) for it in items if "obsrValue" in it}
    if "WSD" not in vals or "VEC" not in vals:
        print("  [wind] KMA 응답에 WSD/VEC 없음 → 폴백")
        return None
    from_deg = vals["VEC"]
    return dict(
        toward_deg=(from_deg + 180.0) % 360.0, from_deg=from_deg,
        speed_ms=vals["WSD"], rh_pct=vals.get("REH", float("nan")),
        source="KMA-ncst", obs_time=f"{base_date}T{base_time}",
    )


def _asos_hourly_series(start_dt, end_dt, stn, cache=None, timeout=30):
    """ASOS 종관 시간자료(과거 실측) → 시간별 (풍향from°, 풍속m/s, 습도%) 시리즈. 지점·기간 키로 캐시."""
    if cache is None:
        os.makedirs(WIND_CACHE_DIR, exist_ok=True)
        cache = f"{WIND_CACHE_DIR}/asos_{stn}_{start_dt}_{end_dt}.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    if not _load_key():
        return None
    try:
        rows = []
        for page in range(1, 20):
            j = _get_json(ASOS_ENDPOINT, "getWthrDataList", {
                "numOfRows": 700, "pageNo": page, "dataCd": "ASOS", "dateCd": "HR",
                "startDt": start_dt, "startHh": "00", "endDt": end_dt, "endHh": "23", "stnIds": stn,
            }, timeout=timeout)
            body = j["response"]["body"]; items = body["items"]["item"]
            if isinstance(items, dict):
                items = [items]
            rows += items
            if page * 700 >= int(body.get("totalCount", 0)):
                break
        ser = {"time": [], "wind_direction_10m": [], "wind_speed_10m": [], "relative_humidity_2m": []}
        for it in rows:
            tm = it.get("tm")           # "2022-03-04 04:00"
            wd = it.get("wd"); ws = it.get("ws"); hm = it.get("hm")
            if not tm or ws in (None, "") or wd in (None, ""):
                continue
            ser["time"].append(tm.replace(" ", "T"))
            ser["wind_direction_10m"].append(float(wd))
            ser["wind_speed_10m"].append(float(ws))
            ser["relative_humidity_2m"].append(float(hm) if hm not in (None, "") else float("nan"))
        if not ser["time"]:
            return None
        json.dump(ser, open(cache, "w"))
        print(f"  [wind] ASOS 지점{stn} 실측 {len(ser['time'])}시간 저장 → {os.path.basename(cache)}")
        return ser
    except Exception as e:
        print(f"  [wind] ASOS 조회 실패({str(e)[:90]})")
        return None


# ---------------------------------------------------------------- 과거(아카이브) 바람
def _series_at(ser, source, hours_after_ignition, ignition_iso=None):
    """시간 시리즈(ASOS/ERA5)에서 발화후 h에 가장 가까운 관측을 뽑는다."""
    if not ser or not ser.get("time"):
        return None
    wt = [datetime.fromisoformat(s) for s in ser["time"]]
    wfrom = ser["wind_direction_10m"]; wspd = ser["wind_speed_10m"]; wrh = ser["relative_humidity_2m"]
    if ignition_iso:
        tgt = datetime.fromisoformat(ignition_iso) + timedelta(hours=hours_after_ignition)
        idx = min(range(len(wt)), key=lambda i: abs((wt[i] - tgt).total_seconds()))
    else:
        idx = min(max(int(round(hours_after_ignition)), 0), len(wt) - 1)
    from_deg = float(wfrom[idx])
    return dict(
        toward_deg=(from_deg + 180.0) % 360.0, from_deg=from_deg,
        speed_ms=float(wspd[idx]), rh_pct=float(wrh[idx]),
        source=source, obs_time=wt[idx].isoformat(),
    )


def _era5_at(hours_after_ignition, ignition_iso=None, cache=ERA5_CACHE):
    if not os.path.exists(cache):
        return None
    return _series_at(json.load(open(cache))["hourly"], "ERA5-archive", hours_after_ignition, ignition_iso)


def archive_wind_series(ignition_iso, span_h, lat=None, lon=None, stn=None):
    """과거 back-test용 전 구간 시리즈: 화재위치 최근접 ASOS 실측 우선(국지성↑), 없으면 ERA5 캐시."""
    if stn is None and lat is not None and lon is not None:
        stn = nearest_station(lat, lon)
    if _load_key() and ignition_iso and stn is not None:
        t0 = datetime.fromisoformat(ignition_iso)
        ser = _asos_hourly_series(t0.strftime("%Y%m%d"), (t0 + timedelta(hours=span_h + 24)).strftime("%Y%m%d"), stn=stn)
        if ser:
            return ser, f"KMA-ASOS#{stn}"
    if os.path.exists(ERA5_CACHE):
        return json.load(open(ERA5_CACHE))["hourly"], "ERA5-archive"
    return None, None


# ---------------------------------------------------------------- 공개 API
def get_current_wind(lat, lon, now=None, hours_after_ignition=None, ignition_iso=None):
    """현재 바람. 라이브=KMA 초단기실황 우선(키 있을 때).
       불가(키없음/과거/오류) & hours_after_ignition 주어지면 과거 실측(ASOS>ERA5), 아니면 상수 폴백."""
    w = _kma_current(lat, lon, now=now)
    if w:
        return w
    if hours_after_ignition is not None:
        if ignition_iso and _load_key():
            stn = nearest_station(lat, lon)
            t0 = datetime.fromisoformat(ignition_iso)
            ser = _asos_hourly_series(t0.strftime("%Y%m%d"), (t0 + timedelta(hours=hours_after_ignition + 24)).strftime("%Y%m%d"), stn=stn)
            w = _series_at(ser, f"KMA-ASOS#{stn}", hours_after_ignition, ignition_iso)
            if w:
                return w
        w = _era5_at(hours_after_ignition, ignition_iso)
        if w:
            return w
    return dict(toward_deg=270.0, from_deg=90.0, speed_ms=4.0, rh_pct=40.0,
                source="fallback-const", obs_time=(now or datetime.now()).isoformat())


if __name__ == "__main__":
    import sys
    la, lo = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (37.06, 129.34)
    print("key:", "있음" if _load_key() else "없음")
    print("grid:", latlon_to_grid(la, lo))
    print("live :", get_current_wind(la, lo))
    print("arch :", get_current_wind(la, lo, hours_after_ignition=12, ignition_iso="2022-03-04T00:00:00"))
