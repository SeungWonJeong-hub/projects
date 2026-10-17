"""active-fire(NHI) 자체 신뢰성 진단 — FIRMS 키 없이 가능한 부분.
실시간 소방에 NHI 활성화재 탐지를 믿어도 되나? 를 3가지로 정량화(3화재).

  1) 공간 응집성  = 최대 연결성분 화소 / 전체 화소.  실제 화선은 연속체(→1에 근접),
                    센서잡음·오탐은 흩어짐(→낮음).  [scipy.ndimage.label]
  2) SWIR 분리도  = 마스크 내 NHI_SWNIR 중앙값 vs 배경 중앙값의 차 + AUC.
                    화염은 B11(1.6µm)이 B8A(0.86µm) 대비 급등 → 분리 뚜렷해야 신뢰.
  3) 구름 오염    = 마스크 화소 중 SCL 구름/그림자 비율.  낮아야(물리탐지가 구름 안 물림).
  + 최전선 이동  = 시점간 화재 중심 이동(km·방위·km/day).  소방 대응 산출물.

출력: active_fire/af_reliability.json  +  active_fire/<site>/reliability.png
사용: python af_reliability.py
"""
import os, sys, json, math
import numpy as np
from osgeo import gdal
from scipy import ndimage as ndi
gdal.UseExceptions()
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
AF = os.path.join(HERE, "..", "active_fire")
SITES = ["uljin", "sancheong", "uiseong"]
CLOUD_SCL = {3, 8, 9, 10}   # 그림자·구름중·구름높음·권운

def _r(p):
    ds = gdal.Open(p); a = ds.ReadAsArray().astype(np.float32); ds = None; return a

def auc(pos, neg, n=4000):
    if len(pos) < 20 or len(neg) < 20: return None
    rng = np.random.RandomState(0)
    p = pos if len(pos) <= n else rng.choice(pos, n, False)
    q = neg if len(neg) <= n else rng.choice(neg, n, False)
    return float((p[:, None] > q[None, :]).mean())

def dates_of(site):
    d = os.path.join(AF, site)
    return sorted([x for x in os.listdir(d) if x[:2].isdigit() and os.path.isdir(os.path.join(d, x))])

def analyze_date(site, date):
    sd = os.path.join(AF, site, date)
    mask = _r(os.path.join(sd, "active_fire_mask.tif")) > 0
    b8a, b11 = _r(os.path.join(sd, "B8A.tif")), _r(os.path.join(sd, "B11.tif"))
    scl = _r(os.path.join(sd, "SCL.tif"))
    ds = gdal.Open(os.path.join(sd, "active_fire_mask.tif")); gt = ds.GetGeoTransform(); ds = None
    npx = int(mask.sum())
    if npx == 0:
        return dict(date=date, fire_px=0, compact=None, swir_sep=None, swir_auc=None,
                    cloud_frac=None, cx=None, cy=None)
    # 1) 공간 응집성
    lab, nlab = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, nlab + 1))
    compact = float(sizes.max() / npx)
    # 2) SWIR 분리도: NHI_SWNIR = (B11-B8A)/(B11+B8A)
    e = 1e-6
    nhi = (b11 - b8a) / (b11 + b8a + e)
    valid = (b11 > 0) & (b8a > 0) & (~np.isin(scl, list(CLOUD_SCL)))
    inm = nhi[mask & valid]; bg = nhi[(~mask) & valid]
    swir_sep = float(np.median(inm) - np.median(bg)) if len(inm) and len(bg) else None
    swir_auc = auc(inm, bg)
    # 3) 구름 오염
    cloud_frac = float(np.isin(scl[mask], list(CLOUD_SCL)).mean())
    # 중심(UTM m)
    ys, xs = np.where(mask)
    cx = gt[0] + (xs.mean() + 0.5) * gt[1]
    cy = gt[3] + (ys.mean() + 0.5) * gt[5]
    return dict(date=date, fire_px=npx, fire_ha=round(npx * 0.04, 1),
                compact=round(compact, 3), swir_sep=round(swir_sep, 3) if swir_sep else None,
                swir_auc=round(swir_auc, 3) if swir_auc else None,
                cloud_frac=round(cloud_frac, 3), cx=cx, cy=cy)

def main():
    out = {}
    print(f"{'site/date':22s} {'화소':>6} {'응집성':>6} {'SWIR분리':>8} {'AUC':>5} {'구름%':>6} {'전진':>14}")
    print("-" * 82)
    for site in SITES:
        rows = [analyze_date(site, d) for d in dates_of(site)]
        # 최전선 이동(연속 시점, 둘 다 화재 있을 때)
        for i in range(1, len(rows)):
            a, b = rows[i - 1], rows[i]
            if a["cx"] is None or b["cx"] is None:
                b["advance"] = None; continue
            dx, dy = b["cx"] - a["cx"], b["cy"] - a["cy"]
            dist = math.hypot(dx, dy)
            days = (np.datetime64(b["date"]) - np.datetime64(a["date"])).astype(int)
            bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360   # 0=N,90=E
            b["advance"] = dict(km=round(dist / 1000, 2),
                                km_day=round(dist / 1000 / days, 3) if days else None,
                                bearing_deg=round(bearing, 0), days=int(days))
        out[site] = rows
        for r in rows:
            adv = r.get("advance")
            advs = f"{adv['km']}km {int(adv['bearing_deg'])}°" if adv else ""
            comp = f"{r['compact']}" if r['compact'] is not None else "-"
            sep = f"{r['swir_sep']}" if r['swir_sep'] is not None else "-"
            a_ = f"{r['swir_auc']}" if r['swir_auc'] is not None else "-"
            cf = f"{r['cloud_frac']*100:.0f}" if r['cloud_frac'] is not None else "-"
            print(f"{site+'/'+r['date']:22s} {r['fire_px']:6d} {comp:>6} {sep:>8} {a_:>5} {cf:>6} {advs:>14}")
        # 그림 — 시점별 마스크 + 전진 화살표
        fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 4))
        if len(rows) == 1: axes = [axes]
        for ax, d in zip(axes, dates_of(site)):
            sd = os.path.join(AF, site, d)
            m = _r(os.path.join(sd, "active_fire_mask.tif")) > 0
            rgb = np.dstack([np.clip(_r(os.path.join(sd, b)) / 4000, 0, 1) for b in ["B04.tif", "B03.tif", "B02.tif"]])
            ax.imshow(rgb); ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.9)
            ax.set_title(f"{d}\n{int(m.sum())}px ({m.sum()*0.04:.0f}ha)", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{site} — active-fire(NHI) 시점별 최전선", fontsize=11)
        plt.tight_layout(); plt.savefig(os.path.join(AF, site, "reliability.png"), dpi=90, bbox_inches="tight"); plt.close()
    # 요약 통계
    allc = [r["compact"] for s in out.values() for r in s if r["compact"] is not None]
    alla = [r["swir_auc"] for s in out.values() for r in s if r["swir_auc"] is not None]
    allcl = [r["cloud_frac"] for s in out.values() for r in s if r["cloud_frac"] is not None]
    summary = dict(n_scenes=len(allc),
                   compact_mean=round(float(np.mean(allc)), 3),
                   swir_auc_mean=round(float(np.mean(alla)), 3),
                   cloud_frac_mean=round(float(np.mean(allcl)), 3))
    out["_summary"] = summary
    json.dump(out, open(os.path.join(AF, "af_reliability.json"), "w"), ensure_ascii=False, indent=1)
    print("-" * 82)
    print(f"평균: 응집성 {summary['compact_mean']}  SWIR-AUC {summary['swir_auc_mean']}  구름오염 {summary['cloud_frac_mean']*100:.1f}%  (n={summary['n_scenes']})")
    print("saved active_fire/af_reliability.json + <site>/reliability.png")

if __name__ == "__main__":
    main()
