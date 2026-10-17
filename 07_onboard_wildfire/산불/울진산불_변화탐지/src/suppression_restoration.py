"""산불 진화·복원 솔루션 계산 엔진 (상황판 확장) — 화재무관(fire-agnostic).

한반도 Sentinel-2 어느 화재든 적용. during-fire 확산예측(firespread_general/v2의 T_fire장) 위에
'어떻게 막고 되살리나'를 얹는다. 울진2022는 앵커 데모일 뿐, 코드에 울진 고정 없음.
  ① 소방차·인력 최적배치   : 위협(위험건물 or ROS-가중 화선footprint)을 불 도달 전에 덮는 MCLP(방어반경).
  ② 헬기 물투하 최적좌표    : 현재 바람(기상청 최근접 ASOS/초단기실황)으로 풍하 화두 → 취수왕복·풍편류 스코어.
  ③ 산림 재조성 소요시간    : 심각도비율×임상 → 식재·수관울폐·생태성숙 타임라인.
  ④ 복원·재조성 비용        : 심각도별 복구 투입비(조림+사방+묘목+인건). 국가 + 민간 자부담(괄호).

입력: runs/{key}_tfire.npz (firespread_general 표준출력: T·격자·fuel·elev·center·burn_ha 포함).
      + (선택) 화재별 도로/대피/건물 payload(있으면 사용, 없으면 화선 footprint로 대체).
바람=wind_provider(KMA), 단가=산불_피해액환산/damage_valuation.py.
출력: runs/{key}_suppression_payload.json

사용: FIRE=sancheong_2025 python src/suppression_restoration.py [t*]   (기본 FIRE=uljin_2022)

정직성: 진화=상대 우선순위 forecast(절대보장X)·이동 직선근사 v1. 심각도 절대등급면적 cross-fire 불가
판정 존중(총면적=마스크실측, 등급=비율). 복원 단가/소요년=한국 소나무 산불 문헌 근사.
"""
import os, sys, json, math
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from scipy import ndimage
from pyproj import Transformer

HERE = __file__.rsplit("/", 2)[0]
sys.path.insert(0, HERE + "/src")
sys.path.insert(0, os.path.expanduser("~/syntax_inference/Steven/산불_피해액환산"))
import wind_provider as WP
try:
    import damage_valuation as DV
except Exception:
    DV = None

CRS = "EPSG:5186"
K_S, KW, KM = 3.0, 0.20, 1.5                             # 확산 상수(firespread 동일)
FUEL_MULT = {1: 1.0, 0: 0.15, 2: 0.0}                   # 1=산림 0=기타 2=물
V_TRUCK_KMH = 40.0; DETOUR = 1.4; SETUP_H = 0.25; DEF_RADIUS_KM = 2.0; STRUCT_PER_CREW = 8
V_HELI_KMH = 160.0; HELI_CYCLE_FIX_H = 3.0/60.0
N_STAGING = 8; N_DROPS = 8
OWNER_SHARE = 0.10
# 심각도별 회복년/단가(한국 소나무 산불 문헌 근사) — 화재무관 기본값
CANOPY_YEARS = {"severe": 30.0, "moderate": 18.0, "light": 10.0}
ECOSYS_YEARS = {"severe": 45.0, "moderate": 28.0, "light": 15.0}
PLANT_YEARS  = {"severe": 3.0,  "moderate": 1.5,  "light": 0.0}
COST_PER_HA  = {"severe": 12_000_000, "moderate": 5_500_000, "light": 1_000_000}

# 심각도 비율 프로파일: NIFoS 울진 실측(심72.3/중104.2/경58.6ha) & 전국 봄산불 기본 prior
SEV_NIFOS_ULJIN = {"severe": 0.308, "moderate": 0.443, "light": 0.249}
SEV_DEFAULT     = {"severe": 0.30,  "moderate": 0.45,  "light": 0.25}   # 근사 prior(실측 없을 때)

# 화재 레지스트리(firespread_general과 정합). tfire=표준 npz. payload/roads 없으면 옵션 폴백.
D = HERE + "/data"; R = HERE + "/runs"
FIRES = {
    "uljin_2022": dict(
        tfire_legacy=R + "/uljin_tfire_v2.npz", tfire=R + "/uljin_2022_tfire.npz",
        dem=D + "/uljin_route/uljin_dem5186.tif", fuel=D + "/landcover_egis/egis_cls.vrt",
        roads=D + "/uljin_route/uljin_roads.geojson", payload=R + "/uljin_dispatch_payload.json",
        center=(37.04, 129.30), conifer_frac=0.65, sev_frac=SEV_NIFOS_ULJIN, burn_real_ratio=0.78,
        label="울진 2022"),
    "sancheong_2025": dict(
        tfire=R + "/sancheong_2025_tfire.npz", roads=None, payload=None,
        center=(35.27, 127.79), conifer_frac=0.60, sev_frac=SEV_DEFAULT, burn_real_ratio=0.80,
        label="산청 2025"),
}


# ============================================================ 화재 로드(표준 npz 우선)
def load_fire(key):
    cfg = FIRES[key]
    std = cfg.get("tfire")
    if std and os.path.exists(std):
        z = np.load(std, allow_pickle=True)
        fuel_cls = z["fuel_cls"].astype(int); elev = z["elev"].astype("float32")
        burn_ha = float(z["burn_ha"]) if "burn_ha" in z.files else None
        center = (float(z["center_lat"]), float(z["center_lon"]))
    else:
        z = np.load(cfg["tfire_legacy"], allow_pickle=True)     # 레거시(uljin_tfire_v2): fuel/elev 재샘플
        fuel_cls = elev = None; burn_ha = None; center = cfg["center"]
    T = z["T"].astype("float32")
    g = dict(T=T, minx=float(z["minx"]), maxy=float(z["maxy"]), cell=float(z["cell"]),
             H=int(z["H"]), W=int(z["W"]), span=float(z["span_h"]), ignition=str(z["ignition"]),
             R0=float(z["R0"]), center=center, burn_ha=burn_ha, cfg=cfg)
    cx = g["minx"] + (np.arange(g["W"]) + 0.5) * g["cell"]
    cy = g["maxy"] - (np.arange(g["H"]) + 0.5) * g["cell"]
    g["gx"], g["gy"] = np.meshgrid(cx, cy)
    if fuel_cls is None:                                        # 레거시 재샘플
        fuel_cls = sample_raster(cfg["fuel"], g["gx"], g["gy"], g["H"], g["W"]).astype(int)
        elev = sample_raster(cfg["dem"], g["gx"], g["gy"], g["H"], g["W"]).astype("float32")
    g["fuel_cls"] = fuel_cls; g["elev"] = elev
    return g


def sample_raster(path, gx, gy, H, W):
    with rasterio.open(path) as s:
        sx, sy = warp_transform(CRS, s.crs.to_string(), gx.ravel().tolist(), gy.ravel().tolist())
        sx = np.array(sx); sy = np.array(sy); t = ~s.transform
        cols = (t.a*sx+t.b*sy+t.c).astype(int); rows = (t.d*sx+t.e*sy+t.f).astype(int)
        rows = np.clip(rows, 0, s.height-1); cols = np.clip(cols, 0, s.width-1)
        return s.read(1)[rows, cols].reshape(H, W)


def to_lonlat(x, y):
    tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lo, la = tr.transform(np.atleast_1d(x), np.atleast_1d(y))
    return np.array(lo), np.array(la)


# ============================================================ 메인
def main(key="uljin_2022", t_star=1.0):
    g = load_fire(key); cfg = g["cfg"]; T = g["T"]; H, W, cell = g["H"], g["W"], g["cell"]
    minx, maxy = g["minx"], g["maxy"]; R0 = g["R0"]; ign = g["ignition"]; span = g["span"]
    clat, clon = g["center"]
    fuel_cls = g["fuel_cls"]
    fuel_mult = np.vectorize(lambda c: FUEL_MULT.get(int(c), 0.15))(fuel_cls).astype("float32")
    elev = g["elev"].copy(); elev[elev < -100] = np.nan; elev = np.where(np.isnan(elev), np.nanmedian(elev), elev)
    dzdx = np.gradient(elev, cell, axis=1); dzdy = -np.gradient(elev, cell, axis=0)
    burnable = fuel_mult > 0

    # 현재 바람(t*): 과거=최근접 ASOS 실측, 라이브=KMA 초단기실황
    ser, wsrc = WP.archive_wind_series(ign, span, lat=clat, lon=clon)
    w = WP._series_at(ser, wsrc, t_star, ign) if ser else WP.get_current_wind(clat, clon, hours_after_ignition=t_star, ignition_iso=ign)
    toward = w["toward_deg"]; U = w["speed_ms"]; RH = w["rh_pct"]
    wang = math.radians((90 - toward) % 360.0); wux, wuy = math.cos(wang), math.sin(wang)
    wmoi = math.exp(-KM * RH / 100.0)
    print(f"[{key} t*={t_star}h] 바람({w['source']} {w['obs_time']}): 향함 {toward:.0f}° {U:.1f}m/s RH{RH:.0f}%")

    burned = np.isfinite(T) & (T <= t_star)
    unburned_future = np.isfinite(T) & (T > t_star)
    front = np.zeros((H, W), bool)
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        sh = np.zeros((H, W), bool)
        r0, r1 = max(0,dr), H+min(0,dr); c0, c1 = max(0,dc), W+min(0,dc)
        sh[r0:r1, c0:c1] = unburned_future[r0-dr:r1-dr, c0-dc:c1-dc] & burnable[r0-dr:r1-dr, c0-dc:c1-dc]
        front |= burned & sh
    fr, fc = np.where(front)
    print(f"불탄 {burned.sum():,}셀 · 활성화선 {front.sum()}셀 · 미연소 {unburned_future.sum():,}셀")

    Tg = np.where(np.isfinite(T), T, np.nanmax(T[np.isfinite(T)]) * 1.5)
    gTy, gTx = np.gradient(Tg, cell)
    def ros_at(r, c, sdx, sdy):
        n = math.hypot(sdx, sdy) or 1.0; ux, uy = sdx/n, sdy/n
        slope_dir = dzdx[r, c]*ux + dzdy[r, c]*uy; wind_al = wux*ux + wuy*uy
        return R0 * fuel_mult[r, c] * math.exp(K_S*slope_dir) * math.exp(KW*U*wind_al) * wmoi

    # --- 수요점: 건물 payload 있으면 건물, 없으면 ROS-가중 미연소 화선footprint ---
    dx, dy, dtf, demand_kind, bx, by, btf = build_demand(cfg, T, minx, maxy, cell, H, W,
                                                          unburned_future, burnable, t_star, gTx, gTy, ros_at)
    cand = road_candidates(cfg, minx, maxy, cell, H, W, T, t_star, burnable, unburned_future)
    print(f"수요({demand_kind}) {len(dx)} · 후보 배치지 {len(cand)}")
    staging = mclp(cand, dx, dy, dtf, t_star, k=N_STAGING)

    # --- ② 헬기 ---
    water = water_sources(fuel_cls, minx, maxy, cell, H, W)
    drops = heli_drops(fr, fc, gTx, gTy, ros_at, wux, wuy, U, water, minx, maxy, cell, bx, by, btf, t_star)

    # --- ③④ 복원 ---
    area_infl = float(g["burn_ha"]) if g["burn_ha"] else float(np.load(cfg.get("tfire_legacy", cfg["tfire"]), allow_pickle=True).get("burn_ha", 0) or predicted_area(T, span, cell, burnable))
    if cfg.get("payload") and os.path.exists(cfg["payload"]):
        area_infl = float(json.load(open(cfg["payload"])).get("burn_ha", area_infl))
    area_real = area_infl * cfg["burn_real_ratio"]
    restoration = restoration_plan(area_real, area_infl, cfg["sev_frac"], cfg["conifer_frac"])

    out = dict(
        meta=dict(fire=key, label=cfg["label"], t_star_h=t_star, ignition=ign, demand_kind=demand_kind,
                  wind=dict(source=w["source"], obs_time=w["obs_time"], toward_deg=round(toward,1),
                            speed_ms=round(U,1), rh_pct=round(RH,1)),
                  note="진화=상대 우선순위 forecast(절대보장X). 이동 직선근사 v1. 심각도=비율×총면적."),
        fire=dict(burned_cells=int(burned.sum()), front_cells=int(front.sum()),
                  future_cells=int(unburned_future.sum()), cell_m=cell),
        staging=staging, water_sources=water_out(water), heli_drops=drops, restoration=restoration,
    )
    outp = R + f"/{key}_suppression_payload.json"
    json.dump(out, open(outp, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {outp}")
    print(f"  ① 전진배치 {len(staging['sites'])}곳 · 수요 커버 {staging['coverage_pct']:.0f}%")
    print(f"  ② 헬기 투하점 {len(drops)}개 · 취수원 {len(water['bodies'])}곳")
    print(f"  ③ 실질녹화 {restoration['canopy_years_weighted']:.0f}년 · 생태성숙 {restoration['ecosys_years_weighted']:.0f}년")
    print(f"  ④ 복구비 국가 {restoration['cost_national_text']} (민간자부담 {restoration['cost_private_text']})")
    return out


def predicted_area(T, span, cell, burnable):
    return float((np.isfinite(T) & (T <= span) & burnable).sum()) * cell * cell / 1e4


# ============================================================ 수요점 구성(건물 or footprint)
def build_demand(cfg, T, minx, maxy, cell, H, W, unburned_future, burnable, t_star, gTx, gTy, ros_at):
    if cfg.get("payload") and os.path.exists(cfg["payload"]):
        blds = json.load(open(cfg["payload"])).get("buildings", [])
        if blds:
            tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)   # payload 좌표=lon/lat
            bx, by = tr.transform([b["x"] for b in blds], [b["y"] for b in blds])
            bx = np.array(bx); by = np.array(by)
            btf = np.array([(b["t_fire"] if b["t_fire"] is not None else np.nan) for b in blds], float)
            m = np.isfinite(btf) & (btf > t_star)
            return bx[m], by[m], btf[m], "위험건물", bx, by, btf
    # 폴백: 미연소 화선 footprint 셀을 수요로(도달시각=마감). ROS 높은 셀 우선(스트라이드 샘플).
    rr, cc = np.where(unburned_future & burnable)
    if len(rr) == 0:
        return np.array([]), np.array([]), np.array([]), "footprint", np.array([]), np.array([]), np.array([])
    stride = max(1, len(rr)//600)
    sel = np.arange(0, len(rr), stride)
    rr, cc = rr[sel], cc[sel]
    dx = minx + (cc + 0.5) * cell; dy = maxy - (rr + 0.5) * cell; dtf = T[rr, cc]
    return dx, dy, dtf, "화선footprint", dx, dy, dtf


# ============================================================ 도로 후보(없으면 안전격자)
def road_candidates(cfg, minx, maxy, cell, H, W, T, t_star, burnable, unburned_future, step=6):
    roads = cfg.get("roads")
    if roads and os.path.exists(roads):
        gj = json.load(open(roads)); tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
        pts = []
        for ft in gj.get("features", []):
            geom = ft.get("geometry") or {}; gt = geom.get("type"); co = geom.get("coordinates")
            lines = [co] if gt == "LineString" else (co if gt == "MultiLineString" else [])
            for ln in lines:
                for i in range(0, len(ln), step):
                    pts.append(ln[i])
        if pts:
            lon = np.array([p[0] for p in pts]); lat = np.array([p[1] for p in pts])
            X, Y = tr.transform(lon, lat); X = np.array(X); Y = np.array(Y)
            return _thin_safe(X, Y, minx, maxy, cell, H, W, T, t_star)
    # 폴백: 안전(미탄) 가연셀 코스격자 — 실도로 없을 때 접근가능 후보 대용
    rr, cc = np.where(unburned_future & burnable & (((np.arange(H)[:,None] % 4)==0) & ((np.arange(W)[None,:] % 4)==0)))
    X = minx + (cc + 0.5) * cell; Y = maxy - (rr + 0.5) * cell
    return _thin_safe(X, Y, minx, maxy, cell, H, W, T, t_star)


def _thin_safe(X, Y, minx, maxy, cell, H, W, T, t_star):
    cc = ((X - minx) / cell).astype(int); rr = ((maxy - Y) / cell).astype(int)
    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    seen = set(); out = []
    for x, y, r, c in zip(X[ok], Y[ok], rr[ok], cc[ok]):
        k = (r // 3, c // 3)
        if k in seen: continue
        if np.isfinite(T[r, c]) and T[r, c] <= t_star: continue   # 이미 탄 곳 배치불가
        seen.add(k); out.append((float(x), float(y)))
    return out


def mclp(cand, dx, dy, dtf, t_star, k=8):
    if not cand or len(dx) == 0:
        return dict(sites=[], coverage_pct=0.0, covered=0, demand=int(len(dx)))
    cand = np.array(cand)
    cov = []
    for sx, sy in cand:
        d_km = np.hypot(dx - sx, dy - sy) / 1000.0 * DETOUR; tt = d_km / V_TRUCK_KMH
        cov.append((d_km <= DEF_RADIUS_KM) & ((t_star + tt + SETUP_H) <= dtf))
    cov = np.array(cov); chosen = []; covered = np.zeros(len(dx), bool)
    for _ in range(min(k, len(cand))):
        gain = (cov & ~covered).sum(axis=1); bi = int(np.argmax(gain))
        if gain[bi] == 0: break
        chosen.append(bi); covered |= cov[bi]
    lons, lats = to_lonlat(cand[chosen, 0], cand[chosen, 1]) if chosen else (np.array([]), np.array([]))
    sites = []; prev = np.zeros(len(dx), bool)
    for rank, bi in enumerate(chosen):
        add = int((cov[bi] & ~prev).sum()); prev |= cov[bi]
        sites.append(dict(rank=rank+1, x=float(cand[bi,0]), y=float(cand[bi,1]),
                          lon=round(float(lons[rank]),5), lat=round(float(lats[rank]),5),
                          covers=int(cov[bi].sum()), covers_new=add,
                          crews=max(1, math.ceil(add/STRUCT_PER_CREW))))
    return dict(sites=sites, coverage_pct=100.0*covered.sum()/max(len(dx),1),
                covered=int(covered.sum()), demand=int(len(dx)))


def water_sources(fuel_cls, minx, maxy, cell, H, W, min_cells=4):
    water = (np.asarray(fuel_cls) == 2)
    lab, n = ndimage.label(water); bodies = []
    for i in range(1, n + 1):
        m = lab == i; sz = int(m.sum())
        if sz < min_cells: continue
        rr, cc = np.where(m); x = minx + (cc.mean()+0.5)*cell; y = maxy - (rr.mean()+0.5)*cell
        bodies.append(dict(x=float(x), y=float(y), cells=sz, ha=round(sz*cell*cell/1e4,1)))
    bodies.sort(key=lambda b: -b["cells"])
    return dict(bodies=bodies, mask_cells=int(water.sum()))


def water_out(water):
    out = []
    for b in water["bodies"]:
        lo, la = to_lonlat(b["x"], b["y"]); out.append(dict(lon=round(float(lo[0]),5), lat=round(float(la[0]),5), ha=b["ha"]))
    return out


def heli_drops(fr, fc, gTx, gTy, ros_at, wux, wuy, U, water, minx, maxy, cell, bx, by, btf, t_star,
               lead_m=250.0, topn=N_DROPS):
    if len(fr) == 0 or not water["bodies"]:
        return []
    wb = np.array([[b["x"], b["y"]] for b in water["bodies"]])
    has_b = len(bx) > 0
    cand = []
    for r, c in zip(fr, fc):
        sdx, sdy = gTx[r, c], gTy[r, c]; n = math.hypot(sdx, sdy)
        if n < 1e-6: continue
        ux, uy = sdx/n, sdy/n; align = ux*wux + uy*wuy; ros = ros_at(r, c, ux, uy)
        if ros <= 0: continue
        fx = minx + (c + 0.5)*cell; fy = maxy - (r + 0.5)*cell
        dxp = fx + wux*lead_m; dyp = fy + wuy*lead_m
        dists = np.hypot(wb[:,0]-dxp, wb[:,1]-dyp); wi = int(np.argmin(dists)); d_km = dists[wi]/1000.0
        cycle_h = 2*d_km/V_HELI_KMH + HELI_CYCLE_FIX_H; refill_eff = 1.0/cycle_h
        head_w = max(0.15, 0.5 + align); crosswind = abs(ux*(-wuy) + uy*wux)
        drift = 1.0/(1.0 + 0.06*U*crosswind)
        if has_b:
            asset_n = int(((np.hypot(bx-dxp, by-dyp) < 1500) & np.isfinite(btf) & (btf > t_star)).sum())
        else:
            asset_n = 0
        asset = 1.0 + min(asset_n, 20)/10.0
        score = ros*head_w*refill_eff*drift*asset
        cand.append(dict(x=dxp, y=dyp, score=score, ros=ros, align=align, water_i=wi, water_km=round(d_km,2),
                         sorties_hr=round(refill_eff,1), asset=asset_n,
                         kind=("화두" if align>0.5 else ("측면" if align>-0.3 else "후미"))))
    if not cand: return []
    cand.sort(key=lambda d: -d["score"]); picked = []
    for d in cand:
        if all(math.hypot(d["x"]-p["x"], d["y"]-p["y"]) > 600 for p in picked): picked.append(d)
        if len(picked) >= topn: break
    smax = picked[0]["score"]; lon, lat = to_lonlat([p["x"] for p in picked], [p["y"] for p in picked])
    out = []
    for i, d in enumerate(picked):
        wx, wy = water["bodies"][d["water_i"]]["x"], water["bodies"][d["water_i"]]["y"]
        wlo, wla = to_lonlat(wx, wy)
        out.append(dict(rank=i+1, lon=round(float(lon[i]),5), lat=round(float(lat[i]),5),
            priority=round(d["score"]/smax,2), front_type=d["kind"], ros_mh=round(d["ros"]),
            wind_align=round(d["align"],2), sorties_hr=d["sorties_hr"], assets_protected=d["asset"],
            water=dict(lon=round(float(wlo[0]),5), lat=round(float(wla[0]),5), dist_km=d["water_km"]),
            rationale=f"{d['kind']} 화선, ROS {round(d['ros'])}m/h, 취수 {d['water_km']}km(왕복 {d['sorties_hr']}회/h), 풍하 보호건물 {d['asset']}동"))
    return out


def restoration_plan(area_real, area_infl, sev_frac, conifer_frac):
    by_sev = {}; canopy_w = ecosys_w = cost_nat = 0.0
    conif_mult = 1.0 + 0.15*conifer_frac
    for s, frac in sev_frac.items():
        a = area_real*frac; cy = CANOPY_YEARS[s]*conif_mult; ey = ECOSYS_YEARS[s]*conif_mult
        cost = a*COST_PER_HA[s]; canopy_w += frac*cy; ecosys_w += frac*ey; cost_nat += cost
        by_sev[s] = dict(area_ha=round(a,1), frac=frac, plant_years=PLANT_YEARS[s],
                         canopy_years=round(cy,1), ecosys_years=round(ey,1), cost_national_krw=int(cost))
    cost_priv = cost_nat*OWNER_SHARE
    dmg = None
    if DV is not None:
        try:
            r = DV.estimate_damage(area_real, species="conifer", burn_fraction=0.85)
            dmg = dict(national_total_krw=int(r.national_total_krw), private_total_krw=int(r.private_total_krw))
        except Exception:
            pass
    return dict(area_infl_ha=round(area_infl), area_real_ha=round(area_real), conifer_frac=conifer_frac,
                by_severity=by_sev, canopy_years_weighted=round(canopy_w,1), ecosys_years_weighted=round(ecosys_w,1),
                plant_complete_years=round(PLANT_YEARS["severe"]*conif_mult,1),
                cost_national_krw=int(cost_nat), cost_private_krw=int(cost_priv),
                cost_national_text=won_text(cost_nat), cost_private_text=won_text(cost_priv), damage_loss=dmg,
                note="복구 '투입비'(forward)로 피해 '손실액'과 구분. 단가=조림+사방+묘목+인건 심각도 스케일.")


def won_text(v):
    v = float(v)
    if v >= 1e12: return f"{v/1e12:.2f}조원"
    if v >= 1e8:  return f"{v/1e8:,.0f}억원"
    if v >= 1e4:  return f"{v/1e4:,.0f}만원"
    return f"{v:,.0f}원"


if __name__ == "__main__":
    key = os.environ.get("FIRE", "uljin_2022")
    ts = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    main(key=key, t_star=ts)
