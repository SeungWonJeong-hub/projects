"""울진 2022 — V2: 실측 시변(時變) 기상 결합 확산 CA (ERA5 바람·습도).

V1(uljin_firespread.py)은 상수 풍향(씨앗→최원거리 기하추정)을 썼다. V1의 핵심 발견은
'도달순위 검증이 가정 풍향에 극히 민감'이었다 → 실측 바람을 넣으면 그 벽이 풀리는지 검증.

무엇이 바뀌나(오직 바람·습도):
  · 바람 = Open-Meteo ERA5 재분석의 발화지점 시간별 풍향(from)·풍속(runs 캐시 uljin_wind_era5.json).
    울진 2022: 발화 04시 서풍(from274°→동진) → 오후 남서풍(from~200°→북동진)으로 회전, 6m/s·RH15~20%
    (양간지풍 건조 강풍). 상수풍이 담을 수 없던 '회전하는 화선'을 시변 바람이 잡는다.
  · 시변 처리: MTT Dijkstra에서 셀이 도달시각 t에 '확정'될 때 그 시각의 바람으로 이웃 전파속도 계산
    (FIFO 시간의존 최단경로 — 비용 양수라 label-setting 유효). 물리적으로 '불이 그곳에 닿은 순간의
    바람으로 앞으로 번진다'.
  · ROS = R0 · 연료(EGIS) · exp(K_S·경사성분) · exp(KW·풍속(t)·풍향정렬(t)) · exp(-KM·RH(t))
    (풍속·습도까지 실측 반영. R0는 V1처럼 관측 중앙시각에 스칼라 정규화 → 공간패턴·ρ·IoU 불변).

정직한 비교: 같은 격자·연료·경사·씨앗에서 V1(상수 기하풍) vs V2(ERA5 시변풍)를 나란히 돌려
  도달순위 ρ / 최종범위 IoU / 시계열 recall 개선을 본다. 한계: ERA5 ~25km(국지 지형채널링 미반영),
  단일지점 바람을 전역 적용, RH도 단일지점. KMA ASOS 지점자료로 교체 시 국지성 향상 여지.
출력: runs/uljin_firespread_v2.png + runs/uljin_firespread_v2.json
"""
import json, heapq, numpy as np, rasterio
from datetime import datetime
from rasterio.warp import transform as warp_transform
from rasterio.enums import Resampling
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
FIRMS = HERE + "/data/uljin_route/uljin_firms_timeline.json"
DEM   = HERE + "/data/uljin_route/uljin_dem5186.tif"
FUEL  = HERE + "/data/landcover_egis/egis_cls.vrt"
BURN  = HERE + "/../산불데이터/wbpk_zenodo/data_extracted/wildfire-burn-probability-korea/data/fires/fire_068_202234/burn_mask.tif"
WIND  = HERE + "/data/uljin_route/uljin_wind_era5.json"
CRS   = "EPSG:5186"; CELL = 100.0; PAD = 2000.0
K_S   = 3.0                 # 경사 민감도
KW    = 0.20               # 풍속 민감도 (exp(KW·U·cosΔ), U=m/s)
KM    = 1.5                # 습도 민감도 (exp(-KM·RH_frac), 건조=빠름)
FUEL_MULT = {1: 1.0, 0: 0.15, 2: 0.0}
R0_INIT = 300.0

# ============ 1) FIRMS ============
fp = json.load(open(FIRMS))
flon = np.array([r["lon"] for r in fp]); flat = np.array([r["lat"] for r in fp])
ftt = np.array([datetime.fromisoformat(r["dt"]) for r in fp]); t0 = ftt.min()
fh = np.array([(t - t0).total_seconds() / 3600 for t in ftt])
fx, fy = warp_transform("EPSG:4326", CRS, flon.tolist(), flat.tolist()); fx = np.array(fx); fy = np.array(fy)
overpasses = np.array(sorted(set(fh.tolist())))
print(f"FIRMS {len(fp)}점 | 기간 {fh.max():.1f}h | 씨앗(첫패스) {(fh==0).sum()}점 | 발화 {t0}")

# ============ 2) 격자 ============
minx, maxx = fx.min()-PAD, fx.max()+PAD; miny, maxy = fy.min()-PAD, fy.max()+PAD
W = int(np.ceil((maxx-minx)/CELL)); H = int(np.ceil((maxy-miny)/CELL))
cx = minx + (np.arange(W)+0.5)*CELL; cy = maxy - (np.arange(H)+0.5)*CELL
gx, gy = np.meshgrid(cx, cy)
def sample_raster(path):
    with rasterio.open(path) as s:
        sx, sy = warp_transform(CRS, s.crs.to_string(), gx.ravel().tolist(), gy.ravel().tolist())
        sx = np.array(sx); sy = np.array(sy); t = ~s.transform
        cols = (t.a*sx+t.b*sy+t.c).astype(int); rows = (t.d*sx+t.e*sy+t.f).astype(int)
        rows = np.clip(rows,0,s.height-1); cols = np.clip(cols,0,s.width-1)
        return s.read(1)[rows, cols].reshape(H, W)
fuel_cls = sample_raster(FUEL)
fuel_mult = np.vectorize(lambda c: FUEL_MULT.get(int(c), 0.15))(fuel_cls).astype("float32")
elev = sample_raster(DEM).astype("float32"); elev[elev < -100] = np.nan
elev = np.where(np.isnan(elev), np.nanmedian(elev), elev)
dzdx = np.gradient(elev, CELL, axis=1); dzdy = -np.gradient(elev, CELL, axis=0)
burnable = fuel_mult > 0
print(f"격자 {H}×{W} = {H*W:,} @ {CELL:.0f}m | 가연 {100*burnable.mean():.0f}% | 경사중앙 {np.degrees(np.arctan(np.hypot(dzdx,dzdy))).mean():.1f}°")

# ============ 3) 실측 시변 바람·습도 (ERA5) → 시간별 배열 ============
# ★반드시 timezone=GMT(UTC) — FIRMS acq가 UTC라 KST로 받으면 바람이 9h 어긋남(정정 2026-09-06).
import os, urllib.request
if not os.path.exists(WIND):
    _u = ("https://archive-api.open-meteo.com/v1/archive?latitude=37.06&longitude=129.34"
          "&start_date=2022-03-04&end_date=2022-03-13&hourly=wind_speed_10m,wind_direction_10m,"
          "relative_humidity_2m,temperature_2m&windspeed_unit=ms&timezone=GMT")
    urllib.request.urlretrieve(_u, WIND); print("ERA5 바람 다운로드(UTC)")
w = json.load(open(WIND)); hh = w["hourly"]
wt = [datetime.fromisoformat(s) for s in hh["time"]]
wspd = np.array(hh["wind_speed_10m"], float); wfrom = np.array(hh["wind_direction_10m"], float)
wrh = np.array(hh["relative_humidity_2m"], float)
# 시리즈 시작 대비 발화시각 오프셋(h)
off0 = (t0 - wt[0]).total_seconds() / 3600.0
toward = (wfrom + 180.0) % 360.0                      # from → 향하는(풍하) 나침반각
wang = np.radians((90 - toward) % 360.0)              # 수학각
WUX = np.cos(wang); WUY = np.sin(wang)
WSTR = KW * wspd                                      # 시변 풍속강도
WMOI = np.exp(-KM * wrh / 100.0)                      # 시변 습도감쇠
NW = len(wt)
def widx(t_arr):                                      # 도달시각(h)→시리즈 인덱스
    return min(NW-1, max(0, int(round(off0 + t_arr))))
print(f"ERA5 시변바람: 발화향 {toward[widx(0)]:.0f}° {wspd[widx(0)]:.1f}m/s RH{wrh[widx(0)]:.0f}% "
      f"→ 12h후 {toward[widx(12)]:.0f}° {wspd[widx(12)]:.1f}m/s RH{wrh[widx(12)]:.0f}% (회전)")

# ============ 4) MTT Dijkstra ============
NB = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
SQ2 = 1.4142135
def dijkstra(R0, tv=True, const_wu=None):
    """tv=True: 실측 시변 바람(도달시각별). tv=False: 상수풍 const_wu=(ux,uy) (V1 비교용)."""
    T = np.full((H, W), np.inf); pq = []
    for k in np.where(fh == 0)[0]:
        c = int((fx[k]-minx)/CELL); r = int((maxy-fy[k])/CELL)
        if 0 <= r < H and 0 <= c < W and burnable[r, c] and T[r, c] > 0:
            T[r, c] = 0.0; heapq.heappush(pq, (0.0, r, c))
    while pq:
        t, r, c = heapq.heappop(pq)
        if t > T[r, c]: continue
        if tv:
            i = widx(t); wux, wuy, wstr, wmoi = WUX[i], WUY[i], WSTR[i], WMOI[i]
        else:
            wux, wuy = const_wu; wstr, wmoi = KW*4.0, 1.0    # 상수풍은 평균풍속·습도중립
        for dr, dc in NB:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < H and 0 <= nc < W) or not burnable[nr, nc]: continue
            dist = CELL*(SQ2 if dr and dc else 1.0)
            dl = np.hypot(dr, dc); ux, uy = dc/dl, -dr/dl
            slope_dir = dzdx[r, c]*ux + dzdy[r, c]*uy
            wind_al = wux*ux + wuy*uy
            ros = R0 * fuel_mult[r, c] * np.exp(K_S*slope_dir) * np.exp(wstr*wind_al) * wmoi
            ros = max(ros, 1e-3)
            nt = t + dist/ros
            if nt < T[nr, nc]:
                T[nr, nc] = nt; heapq.heappush(pq, (nt, nr, nc))
    return T

# FIRMS 셀 인덱스
cc = ((fx-minx)/CELL).astype(int); rr = ((maxy-fy)/CELL).astype(int)
ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
frr, fcc, ffh = rr[ok], cc[ok], fh[ok]
def normalize_R0(tv, const_wu=None):
    T1 = dijkstra(R0_INIT, tv, const_wu); p1 = T1[frr, fcc]; f = np.isfinite(p1) & (ffh > 0)
    scale = np.median(p1[f]) / max(np.median(ffh[ffh > 0]), 1e-6)
    return R0_INIT*scale

# 소각흉터
bcls = sample_raster(BURN); scar = bcls > 0
def validate(T):
    pT = T[frr, fcc]; m = np.isfinite(pT) & (ffh > 0)
    rho = spearmanr(pT[m], ffh[m])[0]
    pb = np.isfinite(T) & (T <= ffh.max()+1e-6); un = (pb | scar).sum()
    iou = float((pb & scar).sum())/un if un else 0.0
    prec = float((pb & scar).sum())/max(pb.sum(),1); rec = float((pb & scar).sum())/max(scar.sum(),1)
    f1 = 2*prec*rec/max(prec+rec,1e-9)
    tl = []
    for tk in overpasses[1:]:
        seen = ffh <= tk+1e-6
        if seen.sum() == 0: continue
        tl.append((float(tk), float((np.isfinite(pT)&(pT<=tk+1e-6)&seen).sum())/float(seen.sum())))
    return dict(rho=float(rho), iou=iou, prec=prec, rec=rec, f1=f1, pT=pT, m=m, pb=pb, tl=tl)

# ============ 5) V2(시변) & V1(상수 기하풍) ============
R0v2 = normalize_R0(True); Tv2 = dijkstra(R0v2, True); V2 = validate(Tv2)
# V1 상수 기하풍(씨앗→최원거리)
sx0, sy0 = fx[fh==0].mean(), fy[fh==0].mean(); far = int(np.argmax((fx-sx0)**2+(fy-sy0)**2))
gwd = np.arctan2(fy[far]-sy0, fx[far]-sx0); gwu = (np.cos(gwd), np.sin(gwd))
R0v1 = normalize_R0(False, gwu); Tv1 = dijkstra(R0v1, False, gwu); V1 = validate(Tv1)
print(f"\nR0 정규화: V2 {R0v2:.0f} · V1 {R0v1:.0f} m/h")
print(f"\n[검증  V2 실측시변풍  vs  V1 상수기하풍]")
print(f"  도달순위 Spearman ρ : V2 {V2['rho']:+.3f}   V1 {V1['rho']:+.3f}   (Δ {V2['rho']-V1['rho']:+.3f})")
print(f"  최종범위 IoU        : V2 {V2['iou']:.3f}   V1 {V1['iou']:.3f}   (Δ {V2['iou']-V1['iou']:+.3f})")
print(f"  최종범위 F1         : V2 {V2['f1']:.3f}   V1 {V1['f1']:.3f}")
print(f"  시계열 recall(초기12h): V2 {V2['tl'][0][1]*100:.0f}%   V1 {V1['tl'][0][1]*100:.0f}%")

# ============ 6) 그림 ============
fig = plt.figure(figsize=(17, 8.5), dpi=125); ext = [minx, maxx, miny, maxy]
axa = fig.add_subplot(1, 2, 1)
Tvis = np.where(np.isfinite(Tv2) & (Tv2 <= ffh.max()*1.2), Tv2, np.nan)
im = axa.imshow(Tvis, extent=ext, origin="upper", cmap="inferno")
axa.imshow(np.where(~burnable, 0.0, np.nan), extent=ext, origin="upper", cmap=ListedColormap(["#3b6fb0"]), alpha=0.5)
sc = axa.scatter(fx, fy, c=fh, s=8, cmap="cool", edgecolors="k", linewidths=0.15, zorder=5)
axa.scatter(fx[fh==0], fy[fh==0], marker="*", s=90, c="lime", edgecolors="k", linewidths=0.4, zorder=6, label="발화(첫 오버패스)")
cb = fig.colorbar(im, ax=axa, fraction=0.04, pad=0.02); cb.set_label("V2 예측 도달시각 (h, 실측 시변풍)", fontsize=9)
cb2 = fig.colorbar(sc, ax=axa, fraction=0.04, pad=0.09); cb2.set_label("관측 FIRMS 시각 (h)", fontsize=8)
axa.legend(loc="upper left", fontsize=8); axa.set_xlabel("EPSG:5186 X"); axa.set_ylabel("Y")
axa.set_title(f"울진 2022 — V2 실측 시변풍 확산 도달시각 (ρ={V2['rho']:.2f})", fontsize=12)

# 우상: 실측 바람 시계열(풍향 화살표축 + 풍속)
axw = fig.add_subplot(2, 2, 2)
hrs = np.array([(t - t0).total_seconds()/3600 for t in wt])
sel = (hrs >= 0) & (hrs <= fh.max())
axw.plot(hrs[sel], wspd[sel], "-", color="#c0392b", label="풍속 m/s")
ax2 = axw.twinx(); ax2.plot(hrs[sel], toward[sel], ".", color="#2c7fb8", ms=3, label="풍향(향하는°)")
ax2.set_ylabel("풍향(향하는, °)", color="#2c7fb8"); ax2.set_ylim(0, 360)
axw.set_xlabel("발화후 h"); axw.set_ylabel("풍속 m/s", color="#c0392b")
axw.set_title("ERA5 실측 바람: 서풍(동진)→남서풍(북동진) 회전", fontsize=10); axw.grid(alpha=0.3)

# 우하: V2 vs V1 지표 막대
axb = fig.add_subplot(2, 2, 4)
labels = ["도달순위 ρ", "범위 IoU", "범위 F1"]; xx = np.arange(3); wd = 0.38
axb.bar(xx-wd/2, [V2['rho'], V2['iou'], V2['f1']], wd, label="V2 실측 시변풍", color="#c0392b")
axb.bar(xx+wd/2, [V1['rho'], V1['iou'], V1['f1']], wd, label="V1 상수 기하풍", color="#4a90d9")
axb.axhline(0, color="k", lw=0.6); axb.set_xticks(xx); axb.set_xticklabels(labels, fontsize=9)
axb.set_title("V2 vs V1 검증지표", fontsize=10); axb.legend(fontsize=8); axb.grid(alpha=0.3, axis="y")
for i, (a, b) in enumerate(zip([V2['rho'],V2['iou'],V2['f1']], [V1['rho'],V1['iou'],V1['f1']])):
    axb.text(i-wd/2, a+0.01, f"{a:.2f}", ha="center", fontsize=7); axb.text(i+wd/2, b+0.01, f"{b:.2f}", ha="center", fontsize=7)

plt.suptitle("Uljin 2022 — V2 real time-varying weather (ERA5 wind/RH) fire-spread", fontsize=13)
plt.tight_layout(); plt.savefig(HERE + "/runs/uljin_firespread_v2.png"); print("\nsaved runs/uljin_firespread_v2.png")

# 예측 필드 저장(대피 forecast가 V2도 쓸 수 있게)
np.savez(HERE + "/runs/uljin_tfire_v2.npz", T=Tv2.astype("float32"), minx=minx, maxy=maxy,
         cell=CELL, H=H, W=W, ignition=t0.isoformat(), R0=float(R0v2), span_h=float(fh.max()), source="ERA5-timevarying")

json.dump(dict(
    model="V2 time-varying ERA5 wind/RH MTT-CA", cell_m=CELL, grid=[H, W],
    wind_source="open-meteo ERA5", ignition=t0.isoformat(), KW=KW, KM=KM, K_S=K_S,
    v2=dict(rho=round(V2['rho'],3), iou=round(V2['iou'],3), f1=round(V2['f1'],3), R0=round(R0v2,1)),
    v1_const=dict(rho=round(V1['rho'],3), iou=round(V1['iou'],3), f1=round(V1['f1'],3), R0=round(R0v1,1)),
    delta_rho=round(V2['rho']-V1['rho'],3), delta_iou=round(V2['iou']-V1['iou'],3),
    wind_toward_deg=dict(h0=round(float(toward[widx(0)])), h12=round(float(toward[widx(12)]))),
    note="실측 시변 바람(회전 화선)을 넣어 V1 상수풍 대비 도달순위/범위 개선 검증. ERA5 ~25km·단일지점 한계→KMA ASOS로 국지성 향상 여지. R0는 관측 중앙시각 정규화.",
), open(HERE + "/runs/uljin_firespread_v2.json", "w"), ensure_ascii=False, indent=1)
print("saved runs/uljin_firespread_v2.json + uljin_tfire_v2.npz")
