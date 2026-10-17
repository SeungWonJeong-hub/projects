"""화재 무관 물리 확산 CA — 일반화 검증(제2 화재). V1/V2를 재사용 가능 모듈로.

목적: 울진 전용이던 물리 확산 CA(V1/V2)가 '다른 화재/지역/연도'에도 이식되는지 검증.
seg-CD(학습 세그멘터)의 오랜 벽=일반화(한국 봄산불 도메인이동). 물리모델은 학습이 아니라
지형·연료·기상 방정식이라 이식에 유리(자유파라미터 R0만 화재별 스칼라 정규화) — 이 가설을 검증.

fire-agnostic 입력(전부 무-학습·공개):
  · burn_mask.tif (wbpk_zenodo, 검증 정답지)  · FIRMS 한국 bbox CSV (씨앗·타임라인, UTC)
  · Copernicus DEM 30m(경사)  · EGIS 토지피복(연료)  · Open-Meteo ERA5 바람·습도(UTC)
모델: V2와 동일(시변 ERA5 바람 MTT-Dijkstra, ROS=R0·연료·exp(K_S·경사)·exp(KW·U·cosΔ)·exp(-KM·RH)).

★타임존 수정: FIRMS acq는 UTC. ERA5도 timezone=GMT(UTC)로 받아 정합(이전 울진 V2는 ERA5를 KST로
받아 바람이 ~9h 어긋났음 — 여기서 바로잡음). 검증=도달순위 ρ(예측 vs FIRMS) + 최종범위 IoU(vs 흉터).
사용: FIRE=sancheong_2025 python src/firespread_general.py   (기본 둘 다)
출력: runs/gen_<fire>.png + runs/gen_<fire>.json
"""
import os, json, csv, heapq, numpy as np, rasterio, urllib.request
from datetime import datetime
from rasterio.warp import transform as warp_transform, transform_bounds
from rasterio.merge import merge
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    try: fm.fontManager.addfont(_fp)
    except Exception: pass
try: plt.rcParams["font.family"] = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name()
except Exception: pass
plt.rcParams["axes.unicode_minus"] = False

HERE = __file__.rsplit("/", 2)[0]
DATA = HERE + "/../산불데이터"
FIREDIR = DATA + "/wbpk_zenodo/data_extracted/wildfire-burn-probability-korea/data/fires"
FUEL = HERE + "/data/landcover_egis/egis_cls.vrt"
CRS = "EPSG:5186"; CELL = 100.0; PAD = 2000.0
K_S, KW, KM = 3.0, 0.20, 1.5
FUEL_MULT = {1: 1.0, 0: 0.15, 2: 0.0}; R0_INIT = 300.0

FIRES = {
  "uljin_2022": dict(burn=FIREDIR+"/fire_068_202234/burn_mask.tif", firms=DATA+"/firms_korea_bbox_2022.csv",
        date_lo="2022-03-01", date_hi="2022-03-16", dem=["N36E129","N37E129"], center=(37.04,129.30), label="울진 2022"),
  "sancheong_2025": dict(burn=FIREDIR+"/fire_142_2025321/burn_mask.tif", firms=DATA+"/firms_korea_bbox_2025.csv",
        date_lo="2025-03-15", date_hi="2025-04-05", dem=["N35E127"], center=(35.27,127.79), label="산청 2025"),
}

def get_dem(tiles):
    paths = []
    for t in tiles:
        p = HERE + f"/data/gen/cop_dem_{t}.tif"
        if not os.path.exists(p):
            url = f"https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t[:3]}_00_{t[3:]}_00_DEM/Copernicus_DSM_COG_10_{t[:3]}_00_{t[3:]}_00_DEM.tif"
            print("  DEM 다운로드", t); urllib.request.urlretrieve(url, p)
        paths.append(p)
    if len(paths) == 1: return paths[0]
    srcs = [rasterio.open(p) for p in paths]; mos, tr = merge(srcs)
    out = HERE + "/data/gen/_dem_merged.tif"; m = srcs[0].meta.copy()
    m.update(height=mos.shape[1], width=mos.shape[2], transform=tr)
    with rasterio.open(out, "w", **m) as d: d.write(mos)
    return out

def run(key):
    cfg = FIRES[key]; print(f"\n===== {cfg['label']} ({key}) =====")
    # 1) FIRMS: burn_mask footprint(±2km) + 날짜창, UTC
    s = rasterio.open(cfg["burn"]); bb = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
    mlon = 0.02 / max(np.cos(np.radians(cfg["center"][0])), 0.5); mlat = 0.018
    lo0, lo1, la0, la1 = bb[0]-mlon, bb[2]+mlon, bb[1]-mlat, bb[3]+mlat
    pts = []
    with open(cfg["firms"], encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try: la = float(r["latitude"]); lo = float(r["longitude"])
            except: continue
            if lo0 < lo < lo1 and la0 < la < la1 and cfg["date_lo"] <= r["acq_date"] <= cfg["date_hi"]:
                dt = datetime.strptime(r["acq_date"]+r["acq_time"].zfill(4), "%Y-%m-%d%H%M")  # UTC
                pts.append((la, lo, dt, float(r["frp"] or 0)))
    pts.sort(key=lambda p: p[2]); t0 = pts[0][2]
    flat = np.array([p[0] for p in pts]); flon = np.array([p[1] for p in pts])
    fh = np.array([(p[2]-t0).total_seconds()/3600 for p in pts])
    fx, fy = warp_transform("EPSG:4326", CRS, flon.tolist(), flat.tolist()); fx = np.array(fx); fy = np.array(fy)
    overpasses = np.array(sorted(set(fh.tolist())))
    print(f"FIRMS {len(pts)}점 | 씨앗 {(fh==0).sum()} | span {fh.max():.0f}h | 발화 {t0} UTC")

    # 2) 격자
    minx, maxx = fx.min()-PAD, fx.max()+PAD; miny, maxy = fy.min()-PAD, fy.max()+PAD
    W = int(np.ceil((maxx-minx)/CELL)); H = int(np.ceil((maxy-miny)/CELL))
    cx = minx+(np.arange(W)+0.5)*CELL; cy = maxy-(np.arange(H)+0.5)*CELL
    gx, gy = np.meshgrid(cx, cy)
    def samp(path):
        with rasterio.open(path) as ss:
            sx, sy = warp_transform(CRS, ss.crs.to_string(), gx.ravel().tolist(), gy.ravel().tolist())
            sx = np.array(sx); sy = np.array(sy); tt = ~ss.transform
            cols = (tt.a*sx+tt.b*sy+tt.c).astype(int); rows = (tt.d*sx+tt.e*sy+tt.f).astype(int)
            rows = np.clip(rows, 0, ss.height-1); cols = np.clip(cols, 0, ss.width-1)
            return ss.read(1)[rows, cols].reshape(H, W)
    fuel_cls = samp(FUEL); fuel_mult = np.vectorize(lambda c: FUEL_MULT.get(int(c), 0.15))(fuel_cls).astype("float32")
    elev = samp(get_dem(cfg["dem"])).astype("float32"); elev[elev < -100] = np.nan
    elev = np.where(np.isnan(elev), np.nanmedian(elev), elev)
    dzdx = np.gradient(elev, CELL, axis=1); dzdy = -np.gradient(elev, CELL, axis=0)
    burnable = fuel_mult > 0
    # 사면 향 수분(흉터 독립 물리선험): 북사면 습→ROS↓. northness=사면이 북을 향하는 정도(+1북/-1남),
    # 경사가 클수록 효과 큼(평지 중립). 한국 봄산불=건조 잎진기, 남사면만 타고 북사면 잔존. A_ASP·SLOPE_REF 고정.
    A_ASP, SLOPE_REF = 0.9, 0.30
    slope_mag = np.hypot(dzdx, dzdy)
    northness = np.where(slope_mag > 1e-6, -dzdy / np.maximum(slope_mag, 1e-6), 0.0)  # +1=북향 사면
    aspect_moist = np.exp(-A_ASP * np.clip(northness, 0, 1) * np.clip(slope_mag / SLOPE_REF, 0, 1))
    print(f"격자 {H}×{W} | 가연 {100*burnable.mean():.0f}% | 경사중앙 {np.degrees(np.arctan(np.hypot(dzdx,dzdy))).mean():.1f}° | 표고 {elev.min():.0f}~{elev.max():.0f}m")

    # 3) ERA5 시변바람 (UTC=GMT)
    wp = HERE + f"/data/gen/wind_{key}.json"
    if not os.path.exists(wp):
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={cfg['center'][0]}&longitude={cfg['center'][1]}"
               f"&start_date={cfg['date_lo']}&end_date={cfg['date_hi']}&hourly=wind_speed_10m,wind_direction_10m,"
               f"relative_humidity_2m&windspeed_unit=ms&timezone=GMT")
        urllib.request.urlretrieve(url, wp)
    w = json.load(open(wp))["hourly"]; wt = [datetime.fromisoformat(x) for x in w["time"]]
    wspd = np.array(w["wind_speed_10m"], float); wfrom = np.array(w["wind_direction_10m"], float); wrh = np.array(w["relative_humidity_2m"], float)
    off0 = (t0 - wt[0]).total_seconds()/3600.0
    toward = (wfrom+180.0) % 360.0; wang = np.radians((90-toward) % 360.0)
    WUX = np.cos(wang); WUY = np.sin(wang); WSTR = KW*wspd; WMOI = np.exp(-KM*wrh/100.0); NWt = len(wt)
    widx = lambda ta: min(NWt-1, max(0, int(round(off0+ta))))
    print(f"ERA5(UTC): 발화향 {toward[widx(0)]:.0f}° {wspd[widx(0)]:.1f}m/s RH{wrh[widx(0)]:.0f}% → 12h후 {toward[widx(12)]:.0f}° {wspd[widx(12)]:.1f}m/s")

    # 4) MTT Dijkstra
    NB = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    def dijkstra(R0, tv=True, cwu=None, aspect=False):
        am = aspect_moist if aspect else np.ones_like(aspect_moist)
        T = np.full((H, W), np.inf); pq = []
        for k in np.where(fh == 0)[0]:
            c = int((fx[k]-minx)/CELL); r = int((maxy-fy[k])/CELL)
            if 0 <= r < H and 0 <= c < W and burnable[r, c] and T[r, c] > 0:
                T[r, c] = 0.0; heapq.heappush(pq, (0.0, r, c))
        while pq:
            t, r, c = heapq.heappop(pq)
            if t > T[r, c]: continue
            if tv: i = widx(t); wux, wuy, wstr, wmoi = WUX[i], WUY[i], WSTR[i], WMOI[i]
            else: wux, wuy = cwu; wstr, wmoi = KW*float(np.median(wspd)), 1.0
            for dr, dc in NB:
                nr, nc = r+dr, c+dc
                if not (0 <= nr < H and 0 <= nc < W) or not burnable[nr, nc]: continue
                dist = CELL*(1.4142135 if dr and dc else 1.0); dl = np.hypot(dr, dc); ux, uy = dc/dl, -dr/dl
                ros = R0*fuel_mult[r, c]*am[nr, nc]*np.exp(K_S*(dzdx[r, c]*ux+dzdy[r, c]*uy))*np.exp(wstr*(wux*ux+wuy*uy))*wmoi
                nt = t + dist/max(ros, 1e-3)
                if nt < T[nr, nc]: T[nr, nc] = nt; heapq.heappush(pq, (nt, nr, nc))
        return T
    cc = ((fx-minx)/CELL).astype(int); rr = ((maxy-fy)/CELL).astype(int)
    ok = (rr >= 0)&(rr < H)&(cc >= 0)&(cc < W); frr, fcc, ffh = rr[ok], cc[ok], fh[ok]
    def norm_R0(tv, cwu=None, aspect=False):
        p1 = dijkstra(R0_INIT, tv, cwu, aspect)[frr, fcc]; f = np.isfinite(p1)&(ffh > 0)
        return R0_INIT*(np.median(p1[f])/max(np.median(ffh[ffh > 0]), 1e-6))
    scar = samp(cfg["burn"]) > 0
    def validate(T):
        pT = T[frr, fcc]; m = np.isfinite(pT)&(ffh > 0); rho = spearmanr(pT[m], ffh[m])[0]
        pb = np.isfinite(T)&(T <= ffh.max()+1e-6); un = (pb|scar).sum()
        iou = float((pb&scar).sum())/un if un else 0.0
        prec = float((pb&scar).sum())/max(pb.sum(), 1); rec = float((pb&scar).sum())/max(scar.sum(), 1)
        f1 = 2*prec*rec/max(prec+rec, 1e-9)
        r12 = None
        for tk in overpasses[1:]:
            seen = ffh <= tk+1e-6
            if seen.sum(): r12 = float((np.isfinite(pT)&(pT <= tk+1e-6)&seen).sum())/float(seen.sum()); break
        return dict(rho=float(rho), iou=iou, f1=f1, prec=prec, rec=rec, r_early=r12, pT=pT, m=m, pb=pb)

    R0v2 = norm_R0(True); Tv2 = dijkstra(R0v2, True); V2 = validate(Tv2)
    sx0, sy0 = fx[fh==0].mean(), fy[fh==0].mean(); far = int(np.argmax((fx-sx0)**2+(fy-sy0)**2))
    gwu = (np.cos(np.arctan2(fy[far]-sy0, fx[far]-sx0)), np.sin(np.arctan2(fy[far]-sy0, fx[far]-sx0)))
    R0v1 = norm_R0(False, gwu); Tv1 = dijkstra(R0v1, False, gwu); V1 = validate(Tv1)
    # 향-수분 강화판(precision 레버, 흉터 독립)
    R0e = norm_R0(True, aspect=True); Te = dijkstra(R0e, True, aspect=True); VE = validate(Te)
    print(f"[검증] 도달순위 ρ: V2(시변) {V2['rho']:+.3f}  V1(상수) {V1['rho']:+.3f} | 범위 IoU: V2 {V2['iou']:.3f} V1 {V1['iou']:.3f} | F1 {V2['f1']:.3f} | 초기recall {V2['r_early']*100:.0f}%")
    print(f"[precision 레버1=향수분 소프트] base IoU {V2['iou']:.3f} P {V2['prec']:.3f} R {V2['rec']:.3f} → IoU {VE['iou']:.3f} P {VE['prec']:.3f} R {VE['rec']:.3f} (ΔIoU {VE['iou']-V2['iou']:+.3f})")
    # precision 레버2: 소멸 임계(하드 배리어). 정적 연소적합도=연료×향수분 < thr 이면 비가연(전파정지).
    suit = fuel_mult * aspect_moist
    sweep = []
    if os.environ.get("SWEEP"):   # 소멸임계 하드배리어 진단(느림) — 결과=negative(recall 붕괴)
      print(f"[precision 레버2=소멸임계 하드배리어]  (연소적합도 thr↑ = 저적합셀 정지)")
      burnable0 = burnable.copy()
      for thr in [0.0, 0.15, 0.25, 0.35, 0.45]:
        globals_burn = suit >= thr
        # dijkstra는 모듈 지역 burnable을 참조 → 임시 교체
        nonlocal_mask = burnable0 & globals_burn
        _b = nonlocal_mask
        def dij_thr(R0):
            T = np.full((H, W), np.inf); pq = []
            for k in np.where(fh == 0)[0]:
                c = int((fx[k]-minx)/CELL); r = int((maxy-fy[k])/CELL)
                if 0 <= r < H and 0 <= c < W and _b[r, c] and T[r, c] > 0:
                    T[r, c] = 0.0; heapq.heappush(pq, (0.0, r, c))
            while pq:
                t, r, c = heapq.heappop(pq)
                if t > T[r, c]: continue
                i = widx(t); wux, wuy, wstr, wmoi = WUX[i], WUY[i], WSTR[i], WMOI[i]
                for dr, dc in NB:
                    nr, nc = r+dr, c+dc
                    if not (0 <= nr < H and 0 <= nc < W) or not _b[nr, nc]: continue
                    dist = CELL*(1.4142135 if dr and dc else 1.0); dl = np.hypot(dr, dc); ux, uy = dc/dl, -dr/dl
                    ros = R0*fuel_mult[r, c]*aspect_moist[nr, nc]*np.exp(K_S*(dzdx[r, c]*ux+dzdy[r, c]*uy))*np.exp(wstr*(wux*ux+wuy*uy))*wmoi
                    nt = t+dist/max(ros, 1e-3)
                    if nt < T[nr, nc]: T[nr, nc] = nt; heapq.heappush(pq, (nt, nr, nc))
            return T
        p1 = dij_thr(R0_INIT)[frr, fcc]; f = np.isfinite(p1)&(ffh > 0)
        R0t = R0_INIT*(np.median(p1[f])/max(np.median(ffh[ffh > 0]), 1e-6))
        Tt = dij_thr(R0t); vv = validate(Tt)
        sweep.append((thr, vv['iou'], vv['prec'], vv['rec'], vv['f1']))
        print(f"    thr {thr:.2f}: 가연 {100*_b.mean():4.0f}% | IoU {vv['iou']:.3f}  P {vv['prec']:.3f}  R {vv['rec']:.3f}  F1 {vv['f1']:.3f}")
      best = max(sweep, key=lambda s: s[1])
      print(f"    → best thr {best[0]:.2f}: IoU {best[1]:.3f} (base {V2['iou']:.3f}, Δ {best[1]-V2['iou']:+.3f}) R {best[3]:.3f}")

    # 5) 그림
    fig = plt.figure(figsize=(15, 6.5), dpi=120); ext = [minx, maxx, miny, maxy]
    axa = fig.add_subplot(1, 2, 1)
    Tvis = np.where(np.isfinite(Tv2)&(Tv2 <= ffh.max()*1.2), Tv2, np.nan)
    im = axa.imshow(Tvis, extent=ext, origin="upper", cmap="inferno")
    axa.imshow(np.where(~burnable, 0.0, np.nan), extent=ext, origin="upper", cmap=ListedColormap(["#3b6fb0"]), alpha=0.5)
    ys, xs = np.where(scar);
    if len(xs): axa.scatter(minx+(xs+0.5)*CELL, maxy-(ys+0.5)*CELL, s=1, c="#00e5ff", alpha=0.05, zorder=2)
    axa.scatter(fx, fy, c=fh, s=9, cmap="cool", edgecolors="k", linewidths=0.15, zorder=5)
    axa.scatter(fx[fh==0], fy[fh==0], marker="*", s=80, c="lime", edgecolors="k", linewidths=0.4, zorder=6)
    cb = fig.colorbar(im, ax=axa, fraction=0.045, pad=0.02); cb.set_label("예측 도달시각 (h)", fontsize=9)
    axa.set_title(f"{cfg['label']} — 예측 확산 (ρ={V2['rho']:.2f}, IoU={V2['iou']:.2f})", fontsize=11)
    axa.set_xlabel("X(m)"); axa.set_ylabel("Y(m)")
    axb = fig.add_subplot(1, 2, 2)
    axb.scatter(ffh[V2['m']], V2['pT'][V2['m']], s=12, alpha=0.5, c="#c0392b", edgecolors="none")
    lim = max(ffh[V2['m']].max(), V2['pT'][V2['m']].max())*1.05; axb.plot([0, lim], [0, lim], "k--", lw=1)
    axb.set_xlabel("관측 FIRMS 검출시각 (h)"); axb.set_ylabel("예측 도달시각 (h)")
    axb.set_title(f"도달시각 예측 vs 관측 (ρ={V2['rho']:.2f})", fontsize=11); axb.grid(alpha=0.3)
    plt.suptitle(f"generalize — {cfg['label']}: physical fire-spread on an unseen fire", fontsize=12)
    plt.tight_layout(); out = HERE + f"/runs/gen_{key}.png"; plt.savefig(out); plt.close()
    json.dump(dict(fire=cfg["label"], key=key, n_firms=len(pts), n_seed=int((fh==0).sum()), span_h=round(float(fh.max()),1),
        v2=dict(rho=round(V2['rho'],3), iou=round(V2['iou'],3), f1=round(V2['f1'],3), prec=round(V2['prec'],3), rec=round(V2['rec'],3), r_early=round(V2['r_early'],3)),
        v1_const=dict(rho=round(V1['rho'],3), iou=round(V1['iou'],3)),
        aspect_moist=dict(iou=round(VE['iou'],3), prec=round(VE['prec'],3), rec=round(VE['rec'],3), f1=round(VE['f1'],3), rho=round(VE['rho'],3)),
        R0_v2=round(float(R0v2),1)),
        open(HERE+f"/runs/gen_{key}.json", "w"), ensure_ascii=False, indent=1)
    print(f"saved {out} + gen_{key}.json")

    # 6a') 표준 tfire npz — 진화·복원 모듈(suppression_restoration)이 화재무관하게 소비.
    #   T장+격자geo+fuel/elev(재샘플 불필요)+발화/센터+소각면적(마스크 실측). uljin_tfire_v2.npz 확장 스키마.
    with rasterio.open(cfg["burn"]) as _bs:
        _ba = _bs.read(1); _pxha = abs(_bs.res[0]*_bs.res[1])/1e4
        burn_ha = float((_ba > 0).sum()) * _pxha
    np.savez(HERE + f"/runs/{key}_tfire.npz", T=Tv2.astype("float32"),
             minx=minx, maxy=maxy, cell=CELL, H=H, W=W, ignition=t0.isoformat(),
             R0=float(R0v2), span_h=float(fh.max()), source="firespread_general-ERA5tv",
             fuel_cls=fuel_cls.astype("int16"), elev=elev.astype("float32"), burn_ha=burn_ha,
             center_lat=float(cfg["center"][0]), center_lon=float(cfg["center"][1]), label=cfg["label"])
    print(f"saved runs/{key}_tfire.npz (진화·복원용 표준 필드, 소각 {burn_ha:,.0f}ha)")

    # 6b) 웹앱 번들(Steven/웹앱/data/fire_dispatch/<key>.json) — Plotly 렌더용 경량
    from rasterio.features import shapes as _shapes
    from rasterio.warp import transform_geom
    from pyproj import Transformer
    tr45 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lon1d, _ = tr45.transform(cx.tolist(), [float(cy[H//2])]*len(cx))
    _, lat1d = tr45.transform([float(cx[W//2])]*len(cy), cy.tolist())
    st_ = max(1, int(np.ceil(max(H, W)/90)))                      # 다운샘플(≤~90셀)
    Zt = np.where(np.isfinite(Tv2) & (Tv2 <= ffh.max()), Tv2, None)[::st_, ::st_]
    grid = dict(lon=[round(v, 5) for v in lon1d[::st_]], lat=[round(v, 5) for v in lat1d[::st_]],
                z=[[None if v is None else round(float(v), 1) for v in row] for row in Zt], span=round(float(ffh.max()), 1))
    firms = [[round(float(a), 5), round(float(b), 5), round(float(c), 1)] for a, b, c in zip(flon, flat, fh)]
    # 소각 경계(burn_mask → 4326 링, 다운샘플)
    with rasterio.open(cfg["burn"]) as bs:
        ff = max(1, min(bs.width, bs.height)//600); oh, ow = bs.height//ff, bs.width//ff
        ba = bs.read(1, out_shape=(oh, ow)).astype("uint8"); btr = bs.transform*bs.transform.scale(bs.width/ow, bs.height/oh)
        rings = []
        for g, v in _shapes(ba, mask=ba > 0, transform=btr):
            if v != 1: continue
            gg = transform_geom(bs.crs, "EPSG:4326", g)
            for ring in gg["coordinates"]:
                if len(ring) >= 4: rings.append([[round(x, 5), round(y, 5)] for x, y in ring])
    bundle = dict(name=cfg["label"], key=key, ignition=t0.isoformat(), span_h=round(float(fh.max()), 1),
                  n_firms=len(pts), n_seed=int((fh == 0).sum()),
                  validation=dict(rho=round(V2['rho'], 3), iou=round(V2['iou'], 3), f1=round(V2['f1'], 3),
                                  recall=round(V2['rec'], 3), precision=round(V2['prec'], 3)),
                  grid=grid, firms=firms, burn_rings=rings, dispatch=None)
    # 울진: 대피 dispatch 페이로드 병합(있으면)
    dp = HERE + "/runs/uljin_dispatch_payload.json"
    if key == "uljin_2022" and os.path.exists(dp):
        bundle["dispatch"] = json.load(open(dp))
    wdir = HERE + "/../../웹앱/data/fire_dispatch"; os.makedirs(wdir, exist_ok=True)
    json.dump(bundle, open(f"{wdir}/{key}.json", "w"), ensure_ascii=False)
    print(f"saved webapp bundle → 웹앱/data/fire_dispatch/{key}.json ({os.path.getsize(f'{wdir}/{key}.json')//1024}KB)")
    return dict(key=key, label=cfg["label"], **{k: V2[k] for k in ("rho","iou","f1","r_early")})

if __name__ == "__main__":
    sel = os.environ.get("FIRE")
    keys = [sel] if sel else list(FIRES)
    res = [run(k) for k in keys]
    if len(res) > 1:
        print("\n===== 일반화 요약 =====")
        print(f"{'화재':16s} {'도달순위ρ':>9s} {'범위IoU':>8s} {'F1':>6s} {'초기recall':>9s}")
        for r in res: print(f"{r['label']:16s} {r['rho']:+9.3f} {r['iou']:8.3f} {r['f1']:6.3f} {r['r_early']*100:8.0f}%")
