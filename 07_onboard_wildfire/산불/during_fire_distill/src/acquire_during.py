"""during-fire S2 씬 취득 — FIRMS teacher 카탈로그(fires_global_firms.json) 기반.

teacher가 확정한 during_fire_dates(실제 연소일) 근방의 Sentinel-2 L2A 광학씬을
이벤트별 center_firms 중심 AOI로 취득한다. 이후 이 씬 + 같은 날 FIRMS 활성화재
footprint(teacher label)로 optical during-fire student distill 학습셋을 만든다.

재사용: 울진산불_변화탐지/src/stac.py 의 STAC 검색 + /vsicurl COG 리드(20 m).
AOI는 이벤트마다 다르므로 stac.AOI_BBOX(모듈 전역)를 이벤트별로 세팅해서 재사용.

device-agnostic·재개가능(이미 받은 밴드 skip). 스케일(35이벤트 전량)은 B200,
Jetson은 SMOKE=1로 소규모 검증.

env:
  EVENTS=all | camp_2018_paradise_us,mati_2018_gr,...   (기본 all)
  MAX_CLOUD=60            during 씬 최대 운량(%)
  MAX_SCENES=3           이벤트당 during 씬 상한(운량 낮은 순)
  PRE_SCENE=1            발화 전 baseline 씬 1장도 취득(before/after 문맥용)
  AOI_DEG=0.30           center_firms ± 반경(도). 대형화재도 코어만 샘플
  PAD_DAYS=3             during 검색창 = [첫 df-PAD, 끝 df+PAD]
  SMOKE=1                첫 2개 이벤트만(젯슨 검증용)
"""
import os, sys, json, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# stac.py 재사용
sys.path.insert(0, os.path.join(ROOT, "..", "울진산불_변화탐지", "src"))
import stac  # noqa: E402

CATALOG = os.path.join(ROOT, "fires_global_firms.json")
OUTDIR  = os.path.join(ROOT, "data", "s2_during")
MANIFEST = os.path.join(ROOT, "data", "s2_during_manifest.json")

MAX_CLOUD  = float(os.environ.get("MAX_CLOUD", 60))
MAX_SCENES = int(os.environ.get("MAX_SCENES", 3))
PRE_SCENE  = os.environ.get("PRE_SCENE", "1") == "1"
AOI_DEG    = float(os.environ.get("AOI_DEG", 0.30))
PAD_DAYS   = int(os.environ.get("PAD_DAYS", 3))
SMOKE      = os.environ.get("SMOKE", "0") == "1"


def bbox_from(center, deg):
    """center=[lat,lon] → [minlon,minlat,maxlon,maxlat] (경도는 위도보정)."""
    lat, lon = center
    import math
    dlon = deg / max(math.cos(math.radians(lat)), 0.2)
    return [round(lon - dlon, 4), round(lat - deg, 4),
            round(lon + dlon, 4), round(lat + deg, 4)]


def rfc(d):  # date → RFC3339 하루경계
    return d.strftime("%Y-%m-%dT00:00:00Z")


def pick_scenes(feats, lo, hi, max_cloud, n):
    """날짜 ∈ [lo,hi] & 운량<max_cloud 인 씬을 운량 오름차순 n개."""
    out = []
    for f in feats:
        p = f["properties"]
        d = p.get("datetime", "")[:10]
        if not d or not (lo <= d <= hi):
            continue
        cc = p.get("eo:cloud_cover", 100)
        if cc is None or cc >= max_cloud:
            continue
        out.append((cc, f))
    out.sort(key=lambda x: x[0])
    return [f for _, f in out[:n]]


def main():
    cat = json.load(open(CATALOG, encoding="utf-8"))["events"]
    want = os.environ.get("EVENTS", "all")
    ids = list(cat) if want == "all" else [i.strip() for i in want.split(",") if i.strip()]
    if SMOKE:
        ids = ids[:2]
    os.makedirs(OUTDIR, exist_ok=True)
    manifest = {}
    if os.path.exists(MANIFEST):
        try:
            manifest = json.load(open(MANIFEST, encoding="utf-8"))
        except Exception:
            pass

    print(f"during-fire S2 취득: {len(ids)}개 이벤트 "
          f"(cloud<{MAX_CLOUD}, ≤{MAX_SCENES}씬/이벤트, pre={PRE_SCENE}, smoke={SMOKE})",
          flush=True)

    for i, eid in enumerate(ids, 1):
        ev = cat.get(eid)
        if not ev or ev.get("firms_status") != "ok":
            print(f"[{i}/{len(ids)}] {eid} — teacher 미확정, 건너뜀", flush=True)
            continue
        center = ev.get("center_firms") or ev.get("center")
        df = ev.get("during_fire_dates") or []
        aw = ev.get("active_window") or [df[0], df[-1]] if df else None
        if not center or not df:
            print(f"[{i}/{len(ids)}] {eid} — center/df 없음, 건너뜀", flush=True)
            continue

        bbox = bbox_from(center, AOI_DEG)
        stac.AOI_BBOX = bbox  # download_band 이 참조하는 전역을 이벤트별로 교체

        lo, hi = df[0], df[-1]
        d0 = dt.date.fromisoformat(lo) - dt.timedelta(days=PAD_DAYS)
        d1 = dt.date.fromisoformat(hi) + dt.timedelta(days=PAD_DAYS)
        ev_dir = os.path.join(OUTDIR, eid)
        os.makedirs(ev_dir, exist_ok=True)
        print(f"[{i}/{len(ids)}] {eid}  df {lo}~{hi}  aoi={bbox}", flush=True)

        # during 씬
        feats = stac.stac_search(bbox, f"{rfc(d0)}/{rfc(d1)}", max_cloud=100, limit=100)
        during = pick_scenes(feats, lo, hi, MAX_CLOUD, MAX_SCENES)
        got = []
        for f in during:
            sd, tot = stac.fetch_scene(f, ev_dir)
            cc = f["properties"].get("eo:cloud_cover")
            print(f"    during {f['properties']['datetime'][:10]} cloud={cc:.0f}% "
                  f"→ {os.path.basename(sd)} ({tot/1e6:.1f} MB)", flush=True)
            got.append({"id": f["id"], "date": f["properties"]["datetime"][:10],
                        "cloud": cc, "kind": "during"})

        # pre-fire baseline 1장(발화 전 60일 창, 최저운량)
        pre = None
        if PRE_SCENE and aw:
            pstart = dt.date.fromisoformat(aw[0]) - dt.timedelta(days=60)
            pend   = dt.date.fromisoformat(aw[0]) - dt.timedelta(days=1)
            pf = stac.stac_search(bbox, f"{rfc(pstart)}/{rfc(pend)}", max_cloud=100, limit=100)
            pk = pick_scenes(pf, pstart.isoformat(), pend.isoformat(), MAX_CLOUD, 1)
            if pk:
                sd, tot = stac.fetch_scene(pk[0], ev_dir)
                cc = pk[0]["properties"].get("eo:cloud_cover")
                print(f"    pre     {pk[0]['properties']['datetime'][:10]} cloud={cc:.0f}% "
                      f"→ {os.path.basename(sd)} ({tot/1e6:.1f} MB)", flush=True)
                pre = {"id": pk[0]["id"], "date": pk[0]["properties"]["datetime"][:10],
                       "cloud": cc, "kind": "pre"}
                got.append(pre)

        if not during:
            print(f"    ⚠ during 씬 없음(운량<{MAX_CLOUD}% 부재) — MAX_CLOUD 상향 검토", flush=True)

        manifest[eid] = {"aoi": bbox, "center_firms": center,
                         "during_fire_dates": df, "active_window": aw,
                         "scenes": got}
        json.dump(manifest, open(MANIFEST, "w"), ensure_ascii=False, indent=1)

    n_sc = sum(len(v["scenes"]) for v in manifest.values())
    print(f"\n완료: {len(manifest)}개 이벤트 매니페스트, 총 {n_sc}씬 → {MANIFEST}\n"
          f"씬 데이터 → {OUTDIR}/<event>/<scene_id>/\n"
          f"다음: 같은 df날짜 FIRMS footprint 래스터화(teacher label) → distill 학습셋", flush=True)


if __name__ == "__main__":
    main()
