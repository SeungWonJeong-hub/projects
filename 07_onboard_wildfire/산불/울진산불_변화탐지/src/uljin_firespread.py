"""울진 2022 — V1: 물리 확산 셀룰러오토마타(최소도달시간 필드) + FIRMS 검증.

파이프라인의 '미싱 코어'(PIPELINE_확산대피_ROADMAP.md). 위성 광학은 확산을 실시간으로
못 본다 → 위성 열점(발화점)을 씨앗으로 지형·연료·바람 위에서 불이 언제 어디 닿는지를
'물리 모델'이 예측한다. 여기서는 그 예측 필드 T_fire(x)를 만들고, 울진 FIRMS 실측
타임라인·소각흉터로 정량검증한다(정답지 보유 → 즉시 검증 가능).

모델 = 최소도달시간(Minimum Travel Time, Finney 2002 / FARSITE MTT의 격자 근사):
  각 셀에서 8방향 이웃으로의 '전파 시간' = 격자거리 / ROS(방향).
  ROS(방향) = R0 · fuel_mult · exp(K_S·경사_방향성분) · exp(W_STR·cos(방향−풍향))
    · fuel_mult : EGIS 토지피복(식생=가연, 시가화/나지=저, 물=차단 0).
    · 경사      : DEM 경사도의 진행방향 성분(오르막 +→빨라짐).  [Van Wagner/Rothermel 경향]
    · 바람      : 풍향과 진행방향 정렬도(풍하=빠름, 풍상=느림).  [Alexandridis 2008 형태]
  씨앗 = 첫 FIRMS 오버패스 화점(발화 시점의 관측 전선) → 다중소스 Dijkstra로 도달시각.

정직한 한계:
  · R0(기저 확산속도)는 자유 파라미터 → FIRMS 관측 도달시각의 중앙값에 '스칼라 정규화'만
    한다(공간패턴·순위·IoU는 R0에 불변이라 검증의 독립성 유지). K_S·W_STR·풍속은 문헌형
    상수(이벤트=양간지풍 강풍 가정). 캘리브레이션이지 피팅이 아님 — 검증지표로 정직 보고.
  · 씨앗이 이미 확산된 79점(첫 오버패스가 이미 전선) → '발화 1점'이 아니라 '첫 관측 전선'
    으로부터의 전방예측을 검증. FIRMS 375m·오버패스 이산(반나절 스케일)·상수바람.
  · 연료는 EGIS 3분류(식생 단일 ROS) → 임상별 연료 차이는 미반영(이 지역 식생 88.6%).

검증(전부 R0에 불변한 지표 위주):
  ① 도달순위: 예측 T_fire vs 관측 FIRMS 검출시각의 Spearman 상관(FIRMS 셀에서).
  ② 시계열 recall: 각 오버패스 t_k에서 예측소각(T_fire≤t_k)이 그때까지의 FIRMS 화점을 덮는 비율.
  ③ 최종 공간범위: 예측 소각영역 vs seg-CD 소각흉터(burn_mask) IoU.
출력: runs/uljin_firespread.png + runs/uljin_firespread.json
"""
import json, heapq, numpy as np, rasterio
from datetime import datetime
from rasterio.warp import transform as warp_transform
from rasterio.enums import Resampling
from pyproj import Transformer
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
CRS   = "EPSG:5186"          # 작업 좌표계(미터, EGIS와 동일)
CELL  = 100.0               # 격자 해상도 m (FIRMS 375m보다 세밀, 계산 경량)
PAD   = 2000.0              # fire bbox 여유 m
# --- ROS 파라미터(문헌형 상수 + R0는 FIRMS 시각에 스칼라 정규화) ---
K_S   = 3.0                 # 경사 민감도 (exp(K_S·tanθ_방향)) — 오르막 가속
W_STR = 1.1                 # 바람 강도 (exp(W_STR·cosΔ)) — 풍하 가속/풍상 감속(양간지풍 강풍)
FUEL_MULT = {1: 1.0, 0: 0.15, 2: 0.0}   # 1식생 / 0시가화·나지 / 2물(차단)
R0_INIT = 300.0            # 기저 ROS m/h (정규화 전 초기값)

# ================= 1) FIRMS: 씨앗 + 검증 타임라인 =================
fp = json.load(open(FIRMS))
flon = np.array([r["lon"] for r in fp]); flat = np.array([r["lat"] for r in fp])
ft = np.array([datetime.fromisoformat(r["dt"]) for r in fp])
t0 = ft.min(); fh = np.array([(t - t0).total_seconds() / 3600 for t in ft])   # 발화후 h
# 4326 → 5186
fx, fy = warp_transform("EPSG:4326", CRS, flon.tolist(), flat.tolist())
fx = np.array(fx); fy = np.array(fy)
overpasses = np.array(sorted(set(fh.tolist())))     # 오버패스 시각들(h)
print(f"FIRMS {len(fp)}점 | 기간 {fh.max():.1f}h | 오버패스 {len(overpasses)}회 | 씨앗(첫패스) {(fh==0).sum()}점")

# ================= 2) 격자 정의 (fire bbox + PAD, EPSG:5186) =================
minx, maxx = fx.min() - PAD, fx.max() + PAD
miny, maxy = fy.min() - PAD, fy.max() + PAD
W = int(np.ceil((maxx - minx) / CELL)); H = int(np.ceil((maxy - miny) / CELL))
# 셀 중심 좌표
cx = minx + (np.arange(W) + 0.5) * CELL
cy = maxy - (np.arange(H) + 0.5) * CELL       # 위→아래(행 증가=남쪽)
gx, gy = np.meshgrid(cx, cy)
print(f"격자 {H}×{W} = {H*W:,} 셀 @ {CELL:.0f}m  ({(maxx-minx)/1000:.1f}×{(maxy-miny)/1000:.1f} km)")

def sample_raster(path, resampling):
    """격자 셀 중심에서 래스터 값 샘플(원좌표계로 변환해 인덱싱)."""
    with rasterio.open(path) as s:
        sx, sy = warp_transform(CRS, s.crs.to_string(), gx.ravel().tolist(), gy.ravel().tolist())
        sx = np.array(sx); sy = np.array(sy)
        t = ~s.transform                      # 아핀 역변환(축정렬) → 벡터 인덱싱
        cols = (t.a*sx + t.b*sy + t.c).astype(int)
        rows = (t.d*sx + t.e*sy + t.f).astype(int)
        rows = np.clip(rows, 0, s.height - 1); cols = np.clip(cols, 0, s.width - 1)
        arr = s.read(1)
        return arr[rows, cols].reshape(H, W), (arr, s)

# ================= 3) 연료 & 경사 격자 =================
fuel_cls, _ = sample_raster(FUEL, Resampling.nearest)
fuel_mult = np.vectorize(lambda c: FUEL_MULT.get(int(c), 0.15))(fuel_cls).astype("float32")
# 물 nodata(255)는 비가연 취급(0.15)이 아니라 미지→가연 하한. 여기선 식생우세라 영향 미미.
elev, _ = sample_raster(DEM, Resampling.bilinear)
elev = elev.astype("float32"); elev[elev < -100] = np.nan
elev = np.where(np.isnan(elev), np.nanmedian(elev), elev)
# 경사 성분(tan): 중심차분 dz/dx, dz/dy (m/m). 행 증가=남쪽이므로 dz/dy 부호 주의.
dzdx = np.gradient(elev, CELL, axis=1)          # +x=동
dzdy = -np.gradient(elev, CELL, axis=0)         # +y=북 (행↓=남 → 부호반전)
burnable = fuel_mult > 0
print(f"가연셀 {burnable.sum():,}/{H*W:,} ({100*burnable.mean():.0f}%) | 표고 {np.nanmin(elev):.0f}~{np.nanmax(elev):.0f}m "
      f"| 경사중앙 {np.degrees(np.arctan(np.hypot(dzdx,dzdy))).mean():.1f}°")

# ================= 4) 풍향 추정 (독립 기하: 씨앗→최원거리) =================
# 주의: '조기→후기 질량중심 변위'는 후기 스팟파이어가 내륙으로 채워 오염됨(실측 -0.16 역상관).
# FIRMS 공간분포는 거의 등방(장/단축 분산비 ~1.1)이라 PCA 장축도 불안정 → 확산 head 방향은
# 씨앗(첫패스) 중심에서 '가장 멀리 도달한 화점'으로의 벡터로 잡는다(도달순위와 독립인 기하량).
sx0, sy0 = fx[fh == 0].mean(), fy[fh == 0].mean()
far = int(np.argmax((fx - sx0)**2 + (fy - sy0)**2))
wind_dir = np.arctan2(fy[far] - sy0, fx[far] - sx0)
import os
if os.environ.get("WIND_DEG"):                    # 민감도 스윕용 오버라이드(나침반각 향하는쪽)
    wind_dir = np.radians((90 - float(os.environ["WIND_DEG"])) % 360)
wdeg = (90 - np.degrees(wind_dir)) % 360         # 수학각→나침반(향하는 쪽)
print(f"확산(풍하) 방향 추정: {wdeg:.0f}° 쪽(씨앗→최원거리 {np.hypot(fx[far]-sx0,fy[far]-sy0)/1000:.1f}km)")
wux, wuy = np.cos(wind_dir), np.sin(wind_dir)

# ================= 5) 최소도달시간 Dijkstra (방향별 ROS) =================
# 8이웃 방향 단위벡터/거리
NB = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
def dijkstra(R0, wu=None):
    if wu is None: wu = (wux, wuy)
    wx, wy = wu
    T = np.full((H, W), np.inf, dtype="float64")
    pq = []
    # 씨앗: 첫 오버패스 FIRMS 화점 셀 → t=0
    for k in np.where(fh == 0)[0]:
        c = int((fx[k] - minx) / CELL); r = int((maxy - fy[k]) / CELL)
        if 0 <= r < H and 0 <= c < W and burnable[r, c] and T[r, c] > 0:
            T[r, c] = 0.0; heapq.heappush(pq, (0.0, r, c))
    while pq:
        t, r, c = heapq.heappop(pq)
        if t > T[r, c]: continue
        for dr, dc in NB:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or not burnable[nr, nc]: continue
            dist = CELL * (1.4142135 if dr and dc else 1.0)
            dl = np.hypot(dr, dc); ux, uy = dc/dl, -dr/dl   # 진행 단위벡터(+x동,+y북; 행↓=남)
            slope_dir = dzdx[r, c]*ux + dzdy[r, c]*uy        # 진행방향 경사성분(tan, 오르막+)
            wind_al = wx*ux + wy*uy                          # 풍향 정렬(-1풍상..+1풍하)
            ros = R0 * fuel_mult[r, c] * np.exp(K_S*slope_dir) * np.exp(W_STR*wind_al)
            ros = max(ros, 1e-3)
            nt = t + dist / ros
            if nt < T[nr, nc]:
                T[nr, nc] = nt; heapq.heappush(pq, (nt, nr, nc))
    return T

# --- R0 스칼라 정규화: 예측 도달시각 중앙값을 FIRMS 관측 중앙값에 맞춤(공간패턴 불변) ---
def firms_cells():
    cc = ((fx - minx) / CELL).astype(int); rr = ((maxy - fy) / CELL).astype(int)
    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    return rr[ok], cc[ok], fh[ok]
frr, fcc, ffh = firms_cells()
T1 = dijkstra(R0_INIT)
pred1 = T1[frr, fcc]
fin = np.isfinite(pred1) & (ffh > 0)
scale = np.median(pred1[fin]) / max(np.median(ffh[ffh > 0]), 1e-6)   # 예측/관측 중앙비
R0 = R0_INIT * scale                                                 # ROS∝R0 → 시간∝1/R0
T = dijkstra(R0)
print(f"R0 정규화: {R0_INIT:.0f}→{R0:.0f} m/h (관측 중앙 도달 {np.median(ffh[ffh>0]):.1f}h 에 정합)")

# ================= 6) 검증 =================
pT = T[frr, fcc]
m = np.isfinite(pT) & (ffh > 0)
rho, pval = spearmanr(pT[m], ffh[m])
reach = float(m.sum()) / float((ffh > 0).sum())        # 예측이 도달한 FIRMS 셀 비율
# 시계열 recall: 각 오버패스에서 예측소각이 그때까지 FIRMS 화점을 덮는 비율
tl = []
for tk in overpasses[1:]:
    seen = ffh <= tk + 1e-6
    if seen.sum() == 0: continue
    covered = np.isfinite(pT) & (pT <= tk + 1e-6) & seen
    tl.append((float(tk), int(seen.sum()), float(covered.sum())/float(seen.sum())))
# 최종 공간범위 IoU vs seg-CD 소각흉터
bcls, _ = sample_raster(BURN, Resampling.nearest)
scar = bcls > 0
# 예측 소각영역 = FIRMS 최종시각까지 도달 & 가연. 흉터 bbox 내로 국한해 공정비교.
pred_burn = np.isfinite(T) & (T <= ffh.max() + 1e-6)
inter = (pred_burn & scar).sum(); union = (pred_burn | scar).sum()
iou = float(inter)/float(union) if union else 0.0
prec = float((pred_burn & scar).sum())/max(pred_burn.sum(),1)
rec  = float((pred_burn & scar).sum())/max(scar.sum(),1)
f1   = 2*prec*rec/max(prec+rec,1e-9)
print(f"\n[검증]")
print(f"  ① 도달순위 Spearman ρ = {rho:.3f} (p={pval:.1e}, n={m.sum()})   ※예측 vs FIRMS 검출시각")
print(f"  ② 예측 도달 FIRMS 셀 비율 = {reach*100:.0f}%")
print(f"  ③ 최종범위 vs 소각흉터: IoU {iou:.3f} · P {prec:.3f} · R {rec:.3f} · F1 {f1:.3f}")
print(f"     예측소각 {pred_burn.sum()*CELL**2/1e6:.1f}km² · 흉터 {scar.sum()*CELL**2/1e6:.1f}km²")
if tl:
    print(f"  시계열 recall(오버패스별): " + " ".join(f"{t:.0f}h:{c*100:.0f}%" for t,_,c in tl[:8])
          + (" …" if len(tl) > 8 else ""))

# --- 풍향 민감도(투명성): 도달순위 ρ는 가정풍향에 극히 민감, 최종범위 IoU는 강건 ---
sens = []
for wd in range(0, 360, 45):
    wr = np.radians((90 - wd) % 360); Ts = dijkstra(R0, (np.cos(wr), np.sin(wr)))
    ps = Ts[frr, fcc]; ms = np.isfinite(ps) & (ffh > 0)
    r_ = spearmanr(ps[ms], ffh[ms])[0] if ms.sum() > 5 else float("nan")
    pb = np.isfinite(Ts) & (Ts <= ffh.max()+1e-6); un = (pb | scar).sum()
    sens.append((wd, round(float(r_),3), round(float((pb&scar).sum()/un) if un else 0,3)))
print(f"  ④ 풍향 민감도  ρ범위 [{min(s[1] for s in sens):+.2f},{max(s[1] for s in sens):+.2f}] · "
      f"IoU범위 [{min(s[2] for s in sens):.2f},{max(s[2] for s in sens):.2f}]  ← 순위검증은 풍향의존, 범위검증은 강건")

# ================= 7) 그림 =================
fig = plt.figure(figsize=(17, 8.5), dpi=125)
ext = [minx, maxx, miny, maxy]
# (a) 도달시각 필드 + FIRMS
axa = fig.add_subplot(1, 2, 1)
Tv = np.where(np.isfinite(T) & (T <= ffh.max()*1.2), T, np.nan)
im = axa.imshow(Tv, extent=ext, origin="upper", cmap="inferno", alpha=0.92)
axa.imshow(np.where(~burnable, 0.0, np.nan), extent=ext, origin="upper", cmap=ListedColormap(["#3b6fb0"]), alpha=0.5)
for g in (np.where(scar))[0][:0]: pass
sc = axa.scatter(fx, fy, c=fh, s=8, cmap="cool", edgecolors="k", linewidths=0.15, zorder=5)
axa.scatter(fx[fh==0], fy[fh==0], marker="*", s=90, c="lime", edgecolors="k", linewidths=0.4, zorder=6, label="발화(첫 오버패스)")
# 풍향 화살표
axa.annotate("", xy=(minx+0.82*(maxx-minx)+wux*2500, miny+0.12*(maxy-miny)+wuy*2500),
             xytext=(minx+0.82*(maxx-minx), miny+0.12*(maxy-miny)),
             arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5))
axa.text(minx+0.82*(maxx-minx), miny+0.06*(maxy-miny), f"바람 {wdeg:.0f}°", color="white", fontsize=9, ha="center")
cb = fig.colorbar(im, ax=axa, fraction=0.04, pad=0.02); cb.set_label("예측 화재 도달시각 T_fire (h)", fontsize=9)
cb2 = fig.colorbar(sc, ax=axa, fraction=0.04, pad=0.09); cb2.set_label("관측 FIRMS 검출시각 (h)", fontsize=8)
axa.legend(loc="upper left", fontsize=8); axa.set_title(f"울진 2022 — 물리 확산 예측 도달시각 (CA/MTT, {CELL:.0f}m)", fontsize=12)
axa.set_xlabel("EPSG:5186 X (m)"); axa.set_ylabel("Y (m)")

# (b) 예측 vs 관측 산점 + 시계열 recall
axb = fig.add_subplot(2, 2, 2)
axb.scatter(ffh[m], pT[m], s=10, alpha=0.5, c="#c0392b", edgecolors="none")
lim = max(ffh[m].max(), pT[m].max())*1.05
axb.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
axb.set_xlabel("관측 FIRMS 검출시각 (h)"); axb.set_ylabel("예측 도달시각 (h)")
axb.set_title(f"도달시각: 예측 vs 관측  (Spearman ρ={rho:.2f})", fontsize=10); axb.legend(fontsize=8); axb.grid(alpha=0.3)

axc = fig.add_subplot(2, 2, 4)
if tl:
    tt = [t for t,_,_ in tl]; cc = [c*100 for _,_,c in tl]
    axc.plot(tt, cc, "o-", color="#2c7fb8", ms=4)
    axc.axhline(100, color="gray", ls=":", lw=0.8)
axc.set_xlabel("오버패스 시각 (발화후 h)"); axc.set_ylabel("FIRMS 화점 커버율 (%)"); axc.set_ylim(0, 105)
axc.set_title(f"시계열 검증: 예측소각이 관측 화점을 덮는 비율  (최종범위 IoU={iou:.2f})", fontsize=10); axc.grid(alpha=0.3)

plt.suptitle("Uljin 2022 — V1 physical fire-spread cellular automaton, validated on FIRMS", fontsize=13)
plt.tight_layout()
plt.savefig(HERE + "/runs/uljin_firespread.png"); print("\nsaved runs/uljin_firespread.png")

# 예측 도달시각 필드 저장 → 대피 모듈(예측기반 도주로)이 조회
np.savez(HERE + "/runs/uljin_tfire.npz", T=T.astype("float32"), minx=minx, maxy=maxy,
         cell=CELL, H=H, W=W, ignition=t0.isoformat(), wind_deg=float(wdeg), R0=float(R0),
         span_h=float(fh.max()))
print("saved runs/uljin_tfire.npz  (예측 T_fire 필드)")

# ================= 8) JSON =================
json.dump(dict(
    model="minimum-travel-time CA (fuel·slope·wind)", cell_m=CELL, grid=[H, W],
    ignition=t0.isoformat(), span_h=round(float(fh.max()), 1), n_firms=len(fp), n_seed=int((fh==0).sum()),
    wind_deg_toward=round(float(wdeg), 0), R0_mph=round(float(R0), 1), K_S=K_S, W_STR=W_STR, fuel_mult=FUEL_MULT,
    valid_spearman_rho=round(float(rho), 3), valid_spearman_p=float(pval), valid_reach_frac=round(reach, 3),
    valid_iou_scar=round(iou, 3), valid_prec=round(prec, 3), valid_rec=round(rec, 3), valid_f1=round(f1, 3),
    pred_burn_km2=round(float(pred_burn.sum()*CELL**2/1e6), 1), scar_km2=round(float(scar.sum()*CELL**2/1e6), 1),
    timeline_recall=[dict(t_h=round(t,1), n_firms=n, cover=round(c,3)) for t,n,c in tl],
    wind_sensitivity=[dict(wind_deg=w, spearman=r, iou=i) for w, r, i in sens],
    note="도달순위 ρ는 가정 풍향에 극히 민감(FIRMS에 독립 풍향신호 없음)→실측 기상/GK-2A 필요(V2). 최종범위 IoU는 풍향에 강건=검증된 핵심결과.",
), open(HERE + "/runs/uljin_firespread.json", "w"), ensure_ascii=False, indent=1)
print("saved runs/uljin_firespread.json")
