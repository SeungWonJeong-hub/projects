#!/usr/bin/env python3
"""during_fire.py — 산불 '진행 중(during-fire)' 관측 분석기.

파이프라인 빈칸 메우기: 지금까지 before↔after 단일 차분(seg-CD)만 있었다. 이 모듈은
사건 폴더에 이미 쌓인 다시점 S2 스택(발생 전 15개월~후 6개월, ~5일 재방문)을 써서
**소각이 시간에 따라 어떻게 번져 갔는지(burned-area growth)** 를 재방문 주기로 추적한다.

핵심 관측: 광학 S2로 '화염 픽셀' 직접 탐지는 불가(NHI recall 0.063, 열적외/천리안은 별도 축).
하지만 대형 산불은 며칠~몇 주에 걸쳐 타므로, 재방문마다 **소각 흉터가 커지는 것**은 광학으로 보인다.
그 증분이 곧 우리가 볼 수 있는 during 신호다.

산출:
 1) timeline: 장면별 누적 소각면적 burned_ha(t), 신규 소각 new_ha, 구름%, **생애단계(BEFORE/DURING/AFTER)**
 2) 진행맵 progression.tif/png: 픽셀별 '최초 소각 관측일'(발생일 기준 일수) = 관측된 확산 순서
 3) summary.md + timeline.png 그래프

'산불이 나는 상황을 구분하는 알고리즘'(사용자 요구) = classify_lifecycle():
   누적면적 궤적의 국소 기울기로 BEFORE(흉터 없음)/DURING(신선 흉터가 재방문 사이에 증가 중)/
   AFTER(흉터 안정) 를 가른다. 구름(연기 포함)으로 빈 장면은 SCL 로 걸러 carry-forward.

사용:
   during_fire.py --event ~/Desktop/s2_events/rf_20220215_yeongdeok --fire_date 20220215
"""
import os, sys, glob, json, argparse, datetime as dt
import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
except Exception as e:
    sys.exit(f"rasterio 필요: {e}")

# mt_cloud 의 SCL_DROP 재사용(동일 기준 유지)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from mt_cloud import SCL_DROP
except Exception:
    SCL_DROP = (0, 1, 3, 8, 9, 10, 11)  # nodata,defective,shadow,cloud-med,cloud-high,cirrus,snow

BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12", "SCL")
BIDX = {b: i for i, b in enumerate(BANDS)}
M2_PER_HA = 10000.0


def parse_date(fn):
    # S2A_52SEF_20220218_0_L2A.tif
    for tok in os.path.basename(fn).split("_"):
        if len(tok) == 8 and tok.isdigit():
            return tok
    return None


def load_scene(paths):
    """같은 날짜 타일을 구름 적은 픽셀 우선으로 합성. **메모리 절약: B8A·B12·valid 만 반환**
    (nbr·char 에 필요한 밴드만 유지 → 대형 AOI[의성 4145²] 도 8GB 안에서 처리)."""
    sum8 = sum12 = None; cnt = None; ref_shape = None
    for p in paths:
        with rasterio.open(p) as ds:
            b8a = ds.read(BIDX["B8A"] + 1).astype(np.float32) / 10000.0
            b12 = ds.read(BIDX["B12"] + 1).astype(np.float32) / 10000.0
            scl = ds.read(BIDX["SCL"] + 1).astype(np.int16)
        if ref_shape is None:
            ref_shape = b8a.shape
            sum8 = np.zeros(ref_shape, np.float32); sum12 = np.zeros(ref_shape, np.float32)
            cnt = np.zeros(ref_shape, np.float32)
        elif b8a.shape != ref_shape:
            continue
        v = (~np.isin(scl, SCL_DROP)) & ((b8a + b12) > 0)
        sum8 += np.where(v, b8a, 0); sum12 += np.where(v, b12, 0); cnt += v
    if ref_shape is None:
        return None
    vmask = cnt > 0
    b8a = np.where(vmask, sum8 / np.maximum(cnt, 1), np.nan).astype(np.float32)
    b12 = np.where(vmask, sum12 / np.maximum(cnt, 1), np.nan).astype(np.float32)
    return b8a, b12, vmask


def nbr_bands(b8a, b12):
    return (b8a - b12) / (b8a + b12 + 1e-6)


def char_bands(b8a_base, b12_base, b8a_t, b12_t):
    """봄 개엽으로 dNBR 이 반전할 때 우회: 소각 = SWIR(B12) 상승 + NIR(B8A) 하락(탄자국)."""
    return (b12_t - b12_base) + (b8a_base - b8a_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True, help="사건 폴더(다시점 S2 tif)")
    ap.add_argument("--fire_date", required=True, help="발생일 YYYYMMDD")
    ap.add_argument("--out", default=None, help="출력 폴더(기본: 사건폴더/during_fire_out)")
    ap.add_argument("--dnbr_thr", type=float, default=0.15, help="dNBR 소각 임계(여름 기준)")
    ap.add_argument("--char_thr", type=float, default=0.10, help="char 소각 임계(봄 개엽 우회)")
    ap.add_argument("--pre_window_days", type=int, default=120, help="기준선에 쓸 발생 전 구간(일)")
    ap.add_argument("--post_window_days", type=int, default=90, help="추적할 발생 후 구간(일)")
    ap.add_argument("--min_valid_frac", type=float, default=0.35, help="장면 채택 최소 유효(비구름) 비율")
    ap.add_argument("--max_base", type=int, default=6, help="기준선 median 에 쓸 발생전 clear 장면 최대수(메모리 캡)")
    a = ap.parse_args()

    ev = os.path.expanduser(a.event)
    out = a.out or os.path.join(ev, "during_fire_out")
    os.makedirs(out, exist_ok=True)
    fire = dt.datetime.strptime(a.fire_date, "%Y%m%d").date()

    # 1) 장면을 날짜별로 그룹
    tifs = sorted(glob.glob(os.path.join(ev, "S2*_L2A.tif")))
    bydate = {}
    for p in tifs:
        d = parse_date(p)
        if d:
            bydate.setdefault(d, []).append(p)
    dates = sorted(bydate)
    if not dates:
        sys.exit(f"장면 없음: {ev}")

    # 격자(geo) 참조
    with rasterio.open(bydate[dates[0]][0]) as ds:
        prof = ds.profile; H, W = ds.height, ds.width; transform = ds.transform; crs = ds.crs
    px_ha = abs(transform.a * transform.e) / M2_PER_HA   # 픽셀 면적(ha)

    # 2) 기준선 = 발생 전 pre_window 안 clear 장면들의 median. **메모리 캡**: 발생일에 가장 가까운
    #    clear 장면 max_base 장만 스택(대형 AOI 에서 전체 스택 시 OOM).
    pre_dates = [d for d in dates
                 if 0 < (fire - dt.datetime.strptime(d, "%Y%m%d").date()).days <= a.pre_window_days]
    b8s, b12s, nbs = [], [], []
    for d in reversed(pre_dates):     # 발생일에 가까운 것부터
        if len(nbs) >= a.max_base:
            break
        sc = load_scene(bydate[d])
        if sc is None:
            continue
        b8a, b12, vmask = sc
        if vmask.mean() < a.min_valid_frac:
            continue
        nb = nbr_bands(b8a, b12); nb[~vmask] = np.nan
        b8a[~vmask] = np.nan; b12[~vmask] = np.nan
        b8s.append(b8a); b12s.append(b12); nbs.append(nb)
    if not nbs:
        sys.exit("기준선(발생 전 clear) 장면 부족 — pre_window_days 를 늘리거나 폴더 확인")
    n_base = len(nbs)
    nbr_base = np.nanmedian(np.stack(nbs), axis=0)
    b8a_base = np.nanmedian(np.stack(b8s), axis=0)
    b12_base = np.nanmedian(np.stack(b12s), axis=0)
    del b8s, b12s, nbs
    base_valid = ~np.isnan(nbr_base)

    # 3) 발생일부터 추적: 누적 소각 마스크(단조) + 픽셀별 최초 소각일수
    track_dates = [d for d in dates
                   if -5 <= (dt.datetime.strptime(d, "%Y%m%d").date() - fire).days <= a.post_window_days]
    cum_burn = np.zeros((H, W), bool)
    first_off = np.full((H, W), np.nan, np.float32)   # 최초 소각 관측일(발생일 기준 일수)
    ever_valid = base_valid.copy()
    timeline = []
    for d in track_dates:
        off = (dt.datetime.strptime(d, "%Y%m%d").date() - fire).days
        sc = load_scene(bydate[d])
        if sc is None:
            continue
        b8a, b12, vmask = sc
        vv = vmask & base_valid
        cloud_pct = float(100 * (1 - vmask.mean()))
        if vmask.mean() < a.min_valid_frac:
            timeline.append(dict(date=d, off_days=off, cloud_pct=round(cloud_pct, 1),
                                 burned_ha=round(float(cum_burn.sum() * px_ha), 1), new_ha=0.0,
                                 skipped="구름/연기 과다"))
            continue
        dnbr = nbr_base - nbr_bands(b8a, b12)
        char = char_bands(b8a_base, b12_base, b8a, b12)
        burned_now = ((dnbr > a.dnbr_thr) | (char > a.char_thr)) & vv
        fresh = burned_now & (~cum_burn)      # 이번 재방문에 새로 관측된 소각
        first_off[fresh] = off
        cum_burn |= burned_now
        ever_valid |= vmask
        timeline.append(dict(date=d, off_days=off, cloud_pct=round(cloud_pct, 1),
                             burned_ha=round(float(cum_burn.sum() * px_ha), 1),
                             new_ha=round(float(fresh.sum() * px_ha), 1)))

    # 4) 생애단계 판별 — '산불 나는 상황을 구분하는 알고리즘'
    obs = [t for t in timeline if "skipped" not in t]
    final = max((t["burned_ha"] for t in obs), default=0.0)
    classify_lifecycle(timeline, final)

    # 5) 진행맵 저장
    prof.update(count=1, dtype="float32", nodata=-9999)
    prog_tif = os.path.join(out, "progression_days.tif")
    with rasterio.open(prog_tif, "w", **prof) as ds:
        ds.write(np.where(np.isnan(first_off), -9999, first_off).astype(np.float32), 1)

    # 6) 결과 dump
    result = dict(event=os.path.basename(ev), fire_date=a.fire_date, crs=str(crs),
                  px_ha=px_ha, final_burned_ha=round(final, 1),
                  n_pre_baseline=n_base, n_tracked=len(obs),
                  during_window=[t["date"] for t in timeline if t.get("state") == "DURING"],
                  timeline=timeline)
    with open(os.path.join(out, "timeline.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    _plots(timeline, first_off, fire, final, out)
    _summary_md(result, out)

    print(f"[during_fire] {os.path.basename(ev)} 발생 {a.fire_date}")
    print(f"  기준선 {n_base}장 · 추적 {len(obs)}장 · 최종 소각 {final:.0f} ha")
    dwin = result["during_window"]
    print(f"  DURING(진행중) 창: {dwin[0]}~{dwin[-1]}" if dwin else "  DURING 창: (단일 재방문에 완결 — 급속연소)")
    print(f"  → {out}/  (timeline.json · progression_days.tif · timeline.png · summary.md)")


def classify_lifecycle(timeline, final):
    """누적면적 궤적으로 각 장면을 BEFORE/DURING/AFTER 로 라벨.
    BEFORE: 누적 < 최종 5%.  DURING: 5~95% 이거나 직전 대비 신규소각 유의(급증).  AFTER: >=95% & 신규≈0."""
    if final <= 0:
        for t in timeline:
            t["state"] = "BEFORE"
        return
    new_thr = max(final * 0.03, 3.0)     # '유의한 신규 소각' = 최종의 3% 또는 3 ha
    for t in timeline:
        if "skipped" in t:
            t["state"] = "CLOUD"          # 연기/구름으로 못 봄(during 일 가능성 높음)
            continue
        frac = t["burned_ha"] / final
        if frac < 0.05:
            t["state"] = "BEFORE"
        elif frac >= 0.95 and t.get("new_ha", 0) < new_thr:
            t["state"] = "AFTER"
        else:
            t["state"] = "DURING"


def _plots(timeline, first_off, fire, final, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for fp in ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"):
            if os.path.exists(fp):
                font_manager.fontManager.addfont(fp)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
                break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        return
    obs = [t for t in timeline if "skipped" not in t]
    xs = [t["off_days"] for t in obs]; ys = [t["burned_ha"] for t in obs]
    col = {"BEFORE": "#7f8c8d", "DURING": "#e74c3c", "AFTER": "#2c3e50", "CLOUD": "#bdc3c7"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(xs, ys, "-", color="#34495e", lw=1.3, zorder=1)
    for t in obs:
        ax[0].scatter(t["off_days"], t["burned_ha"], c=col.get(t["state"], "#999"), s=54,
                      zorder=2, edgecolors="white", linewidths=0.6)
    ax[0].axvline(0, ls="--", c="#c0392b", lw=1, alpha=0.7)
    ax[0].set_xlabel("발생일 기준 일수"); ax[0].set_ylabel("누적 소각면적 (ha)")
    ax[0].set_title("소각 진행 timeline (관측)")
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=col[k], mec="w", label=k)
                          for k in ("BEFORE", "DURING", "AFTER", "CLOUD")], fontsize=8, loc="lower right")
    fo = np.where(np.isnan(first_off), np.nan, first_off)
    im = ax[1].imshow(fo, cmap="inferno")
    ax[1].set_title("관측된 확산 진행맵\n(픽셀별 최초 소각일수)"); ax[1].axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.046, label="발생일 기준 일수")
    plt.tight_layout(); plt.savefig(os.path.join(out, "timeline.png"), dpi=110); plt.close()


def _summary_md(r, out):
    lines = [f"# during-fire 관측 — {r['event']} (발생 {r['fire_date']})", "",
             f"- 기준선(발생전 clear): {r['n_pre_baseline']}장",
             f"- 추적 장면: {r['n_tracked']}장 · 최종 소각 **{r['final_burned_ha']:.0f} ha**",
             f"- DURING(진행중) 창: {', '.join(r['during_window']) if r['during_window'] else '단일 재방문 완결'}",
             "", "| 날짜 | 발생+일 | 단계 | 누적 ha | 신규 ha | 구름% |", "|---|---|---|---|---|---|"]
    for t in r["timeline"]:
        lines.append(f"| {t['date']} | {t['off_days']:+d} | {t.get('state','?')} | "
                     f"{t.get('burned_ha',0):.0f} | {t.get('new_ha',0):.0f} | {t.get('cloud_pct',0):.0f} |")
    lines += ["", "관측 한계: 재방문 주기(~5일)와 연기/구름으로 during 해상도가 제한됨.",
              "화염 픽셀 실시간 탐지가 아니라 재방문 사이 소각 흉터 증분(관측 가능한 during 신호)이다."]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
