"""울진 2022 — 예측기반 대피(도주로). V0(FIRMS 사후재구성) → V1 물리예측 결합.

무엇이 바뀌나:
  V0(uljin_evacuation.py)는 화재 '도착시각'을 이미 지나간 FIRMS 관측의 최근접 보간으로 채웠다
  (=사후재구성, 실전에선 미래를 못 봄). 여기서는 그 도착시각을 V1 물리 확산 CA가 예측한
  도달시각 필드 T_fire(x)(runs/uljin_tfire.npz, uljin_firespread.py 산출)로 교체한다.
  → 발화 직후(첫 열점만 아는 시점)에 '앞으로 불이 언제 어디 닿을지'를 예측해 도주 마진을 계산
    = 진짜 forecast 기반 대피. 대피 라우팅(도로망·실제 안전목적지)은 V0와 동일.

정직한 비교: 예측 마진 vs FIRMS-사후 마진을 같은 건물에 대해 나란히 산출해
  (1) 갇힘/초박빙 판정이 얼마나 바뀌는지, (2) 최고위험 마을 순위가 바뀌는지 보인다.
  예측 T_fire는 씨앗=첫 FIRMS 오버패스(발화시점 관측 전선)에서의 전방예측이라, 그 시점 이후만
  '미래'다. 한계는 V1과 동일(풍향 민감·상수바람·연료 3분류·R0 정규화) — 이 지도는 상대위험.
출력: runs/uljin_evacuation_forecast.png + runs/uljin_evacuation_forecast.json
"""
import json, numpy as np, networkx as nx
from datetime import datetime
from rasterio.features import shapes
import rasterio
from shapely.geometry import shape, LineString, Point
from shapely.ops import unary_union, transform as shp_transform
from shapely.prepared import prep
from pyproj import Transformer, Geod
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import matplotlib.font_manager as fm
for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    try: fm.fontManager.addfont(_fp)
    except Exception: pass
try: plt.rcParams["font.family"] = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name()
except Exception: pass
plt.rcParams["axes.unicode_minus"] = False

geod = Geod(ellps="WGS84")
HERE = __file__.rsplit("/", 2)[0]
FIRE = HERE + "/../산불데이터/wbpk_zenodo/data_extracted/wildfire-burn-probability-korea/data/fires/fire_068_202234/burn_mask.tif"
ROADS = HERE + "/data/uljin_route/uljin_roads.geojson"
BLDS = HERE + "/data/uljin_route/uljin_buildings.geojson"
FIRMS = HERE + "/data/uljin_route/uljin_firms_timeline.json"
SHELTERS = HERE + "/data/uljin_route/uljin_shelters.geojson"
PLACES = HERE + "/data/uljin_route/uljin_places.geojson"
import os
TFIRE = HERE + "/runs/uljin_tfire_v2.npz" if os.path.exists(HERE + "/runs/uljin_tfire_v2.npz") else HERE + "/runs/uljin_tfire.npz"
V_EVAC = 30.0; FIRMS_R_KM = 2.0; SAFE_PAD_KM = 0.3
SHELTER_TYPES = {"town_hall", "school", "shelter", "hospital", "community_centre", "university"}

# ---------- 1) 소각 폴리곤 / 위험버퍼 ----------
with rasterio.open(FIRE) as s:
    f = 4; oh, ow = s.height // f, s.width // f
    a = s.read(1, out_shape=(oh, ow)).astype("uint8")
    tr = s.transform * s.transform.scale(s.width / ow, s.height / oh)
    polys = [shape(g) for g, v in shapes(a, mask=a > 0, transform=tr) if v == 1]
    to4326 = Transformer.from_crs(s.crs, "EPSG:4326", always_xy=True).transform
burn = unary_union([shp_transform(to4326, p) for p in polys]).buffer(0)
pburn = prep(burn)
danger = burn.buffer(SAFE_PAD_KM / 111.0); pdanger = prep(danger)
burn_ha = abs(geod.geometry_area_perimeter(burn)[0]) / 1e4

# ---------- 2) 도로 그래프(본토망) ----------
d = json.load(open(ROADS)); G = nx.Graph(); R = 6
nd = lambda x, y: (round(x, R), round(y, R))
for ft in d["features"]:
    g = ft["geometry"]
    if g["type"] != "LineString": continue
    cs = g["coordinates"]
    for (x1, y1), (x2, y2) in zip(cs, cs[1:]):
        n1, n2 = nd(x1, y1), nd(x2, y2)
        if n1 == n2: continue
        G.add_edge(n1, n2, length=geod.geometry_length(LineString([(x1, y1), (x2, y2)])),
                   geom=LineString([(x1, y1), (x2, y2)]))
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
geoms = [e[2]["geom"] for e in G.edges(data=True)]

# ---------- 3a) FIRMS 사후 도착시각(V0 방식) ----------
fp = json.load(open(FIRMS))
ft_xy = np.array([[r["lon"], r["lat"]] for r in fp])
ft_t = np.array([datetime.fromisoformat(r["dt"]) for r in fp]); t0 = ft_t.min()
ft_h = np.array([(t - t0).total_seconds() / 3600 for t in ft_t])
fkdt = cKDTree(ft_xy); FDEG = FIRMS_R_KM / 111.0
def fire_arrival_firms(x, y):
    near = fkdt.query_ball_point([x, y], FDEG)
    if near: return float(ft_h[near].min())
    _, j = fkdt.query([x, y]); return float(ft_h[j])

# ---------- 3b) V1 예측 도달시각 필드 T_fire (npz 조회) ----------
z = np.load(TFIRE, allow_pickle=True)
T = z["T"]; TMINX = float(z["minx"]); TMAXY = float(z["maxy"]); TCELL = float(z["cell"])
TH = int(z["H"]); TW = int(z["W"]); SPAN = float(z["span_h"])
WIND_DEG = float(z["wind_deg"]) if "wind_deg" in z.files else -1.0   # V2필드는 시변풍(단일각 없음)
to5186 = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True).transform
def fire_arrival_pred(lon, lat):
    """예측 T_fire 격자에서 3×3 최소값(건물이 도로/시가 셀 위여도 인근 가연셀 도달시각)."""
    x, y = to5186(lon, lat)
    c = int((x - TMINX) / TCELL); r = int((TMAXY - y) / TCELL)
    if not (0 <= r < TH and 0 <= c < TW): return np.inf
    sub = T[max(0, r-1):r+2, max(0, c-1):c+2]
    v = np.nanmin(sub) if np.isfinite(sub).any() else np.inf
    return float(v)

# ---------- 4) 실제 안전 목적지 → 탈출거리(V0 동일) ----------
nodes = list(G.nodes()); nkdt = cKDTree(np.array(nodes))
def load_dests(path, kind):
    out = []
    try: dd = json.load(open(path))
    except FileNotFoundError: return out
    for ft in dd["features"]:
        g = ft.get("geometry")
        if not g: continue
        c = shape(g).centroid; fc = (ft["properties"].get("fclass") or kind)
        if kind == "shelter" and fc not in SHELTER_TYPES: continue
        out.append((c.x, c.y, fc, kind))
    return out
dests_all = load_dests(SHELTERS, "shelter") + load_dests(PLACES, "town")
dests = [dd for dd in dests_all if not pdanger.contains(Point(dd[0], dd[1]))]
exit_set, dest_pts = set(), []
for (x, y, fc, kind) in dests:
    _, j = nkdt.query([x, y]); exit_set.add(nodes[j]); dest_pts.append((x, y, kind))
esc_m = nx.multi_source_dijkstra_path_length(G, exit_set, weight="length")
print(f"소각 {burn_ha:,.0f} ha | 본토노드 {len(nodes):,} | 안전목적지 {len(dests):,} | 예측필드 {TH}×{TW} 바람 {WIND_DEG:.0f}°")

# ---------- 5) 건물별 마진 (예측 vs FIRMS-사후) ----------
b = json.load(open(BLDS))
bxy = np.array([[shape(x["geometry"]).centroid.x, shape(x["geometry"]).centroid.y] for x in b["features"]])
_, bnode = nkdt.query(bxy)
rows = []
for k, (bx, by) in enumerate(bxy):
    if not pdanger.contains(Point(bx, by)): continue
    de = esc_m.get(nodes[bnode[k]], float("inf"))
    te = (de / 1000.0) / V_EVAC
    tf_p = fire_arrival_pred(bx, by); tf_f = fire_arrival_firms(bx, by)
    rows.append(dict(i=k, x=float(bx), y=float(by), esc_km=de/1000.0, t_esc=te,
                     t_fire_pred=tf_p, t_fire_firms=tf_f,
                     margin_pred=tf_p - te, margin_firms=tf_f - te,
                     burned=bool(pburn.contains(Point(bx, by)))))
mp = np.array([r["margin_pred"] for r in rows]); mf = np.array([r["margin_firms"] for r in rows])
finite = np.isfinite(mp) & np.isfinite(mf)
def bucket(m):
    m = m[np.isfinite(m)]
    return int((m < 0).sum()), int(((m >= 0) & (m < 1)).sum()), int((m >= 1).sum())
tp, tt, to = bucket(mp); fp_, ft_, fo_ = bucket(mf)
# 판정 변화(예측이 FIRMS-사후 대비 뒤집는 건물)
flip_worse = int(((mp < 0) & (mf >= 0) & finite).sum())   # 예측은 갇힘, 사후는 아님
flip_better = int(((mp >= 0) & (mf < 0) & finite).sum())
rho = spearmanr(mp[finite], mf[finite])[0] if finite.sum() > 5 else float("nan")
print(f"\n[갇힘/초박빙/여유]  (V_EVAC={V_EVAC:.0f})")
print(f"  예측(V1)     : 갇힘 {tp} · 초박빙 {tt} · 여유 {to}")
print(f"  FIRMS-사후(V0): 갇힘 {fp_} · 초박빙 {ft_} · 여유 {fo_}")
print(f"  판정변화: 예측만 갇힘 {flip_worse}동 · 사후만 갇힘 {flip_better}동 | 마진순위 상관 ρ={rho:.3f}")
print(f"  예측마진 중앙 {np.nanmedian(mp):+.1f}h · 사후마진 중앙 {np.nanmedian(mf):+.1f}h")

# ---------- 6) 예측기반 도주로 (예측 마진 최소 마을 12곳) ----------
cand = sorted([r for r in rows if r["esc_km"] < float("inf") and np.isfinite(r["margin_pred"])],
              key=lambda r: r["margin_pred"])
chosen, routes = [], []
G0 = G.copy(); G0.add_node("EXIT")
for e in exit_set: G0.add_edge("EXIT", e, length=0.0)
for r in cand:
    n = nodes[bnode[r["i"]]]
    if any(geod.inv(r["x"], r["y"], c[0], c[1])[2] < 1500 for c in chosen): continue
    try: path = nx.shortest_path(G0, "EXIT", n, weight="length")[1:]
    except nx.NetworkXNoPath: continue
    chosen.append((r["x"], r["y"])); routes.append((r, path))
    if len(routes) >= 12: break

# ---------- 7) 그림 ----------
fig = plt.figure(figsize=(16.5, 8), dpi=125)
axm = fig.add_subplot(1, 2, 1)
axm.add_collection(LineCollection([list(g.coords) for g in geoms], colors="#e8e8e8", linewidths=0.2, zorder=1))
for g in (burn.geoms if burn.geom_type == "MultiPolygon" else [burn]):
    xs, ys = g.exterior.xy; axm.fill(xs, ys, color="#ffb27a", alpha=0.35, zorder=2)
dp = np.array([(x, y) for (x, y, k) in dest_pts]); dk = np.array([k for (x, y, k) in dest_pts])
if len(dp):
    ms = dk == "shelter"; mt = dk == "town"
    if ms.any(): axm.scatter(dp[ms, 0], dp[ms, 1], marker="^", s=30, c="#0b8a0b", edgecolors="k", linewidths=0.3, zorder=8)
    if mt.any(): axm.scatter(dp[mt, 0], dp[mt, 1], marker="s", s=42, c="#1565c0", edgecolors="k", linewidths=0.4, zorder=9)
bx_ = np.array([r["x"] for r in rows]); by_ = np.array([r["y"] for r in rows])
mpp = np.where(np.isfinite(mp), mp, np.nan)
norm = TwoSlopeNorm(vmin=min(-3, float(np.nanmin(mpp))), vcenter=0, vmax=6)
sc = axm.scatter(bx_, by_, c=mpp, s=14, cmap="RdYlGn", norm=norm, zorder=4, edgecolors="none")
cb = fig.colorbar(sc, ax=axm, fraction=0.035, pad=0.02); cb.set_label("예측 도착-탈출 마진 (h)  ◀위험 | 여유▶", fontsize=9)
for (r, path) in routes:
    pts = np.array(path, dtype=float)
    axm.plot(pts[:, 0], pts[:, 1], "-", color="#1a1a1a", lw=1.4, zorder=6, alpha=0.8)
    axm.plot(r["x"], r["y"], "*", ms=11, mfc="#d62728", mec="k", mew=0.5, zorder=7)
axm.legend(handles=[
    Line2D([], [], color="#ffb27a", lw=6, alpha=0.5, label=f"소각 (~{burn_ha:,.0f} ha)"),
    Line2D([], [], marker="^", color="w", mfc="#0b8a0b", mec="k", ms=9, label="대피소(마을회관·학교)"),
    Line2D([], [], marker="s", color="w", mfc="#1565c0", mec="k", ms=9, label="읍·시"),
    Line2D([], [], marker="*", color="w", mfc="#d62728", mec="k", ms=12, label="최고위험(예측마진 최소)"),
    Line2D([], [], color="#1a1a1a", lw=1.4, label="예측기반 도주로"),
], loc="upper left", fontsize=8)
axm.set_title(f"울진 2022 — 예측기반 도주로 & 마진 (V1 T_fire, 바람 {WIND_DEG:.0f}°)", fontsize=12)
axm.set_xlabel("lon"); axm.set_ylabel("lat"); axm.set_aspect(1.24)

# 우상: 예측 vs FIRMS-사후 마진 산점
axs = fig.add_subplot(2, 2, 2)
axs.scatter(mf[finite], mp[finite], s=10, alpha=0.45, c="#8e44ad", edgecolors="none")
lo = min(np.nanmin(mf[finite]), np.nanmin(mp[finite])); hi = max(np.nanmax(mf[finite]), np.nanmax(mp[finite]))
axs.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
axs.axhline(0, color="#d62728", lw=0.8, ls=":"); axs.axvline(0, color="#d62728", lw=0.8, ls=":")
axs.set_xlabel("FIRMS-사후 마진 (h)"); axs.set_ylabel("V1 예측 마진 (h)")
axs.set_title(f"예측 vs 사후 마진  (ρ={rho:.2f})", fontsize=10); axs.legend(fontsize=8); axs.grid(alpha=0.3)

# 우하: 갇힘/초박빙/여유 막대 비교
axb = fig.add_subplot(2, 2, 4)
cats = ["갇힘(<0)", "초박빙(0~1h)", "여유(≥1h)"]; xx = np.arange(3); w = 0.38
axb.bar(xx - w/2, [tp, tt, to], w, label="V1 예측", color="#c0392b")
axb.bar(xx + w/2, [fp_, ft_, fo_], w, label="FIRMS-사후", color="#4a90d9")
axb.set_xticks(xx); axb.set_xticklabels(cats, fontsize=9); axb.set_ylabel("위험권 건물 수")
axb.set_title(f"판정 비교 (예측만 갇힘 {flip_worse}·사후만 갇힘 {flip_better})", fontsize=10)
axb.legend(fontsize=8); axb.grid(alpha=0.3, axis="y")

plt.suptitle("Uljin 2022 — forecast-based evacuation (V1 physical T_fire) vs post-hoc FIRMS", fontsize=13)
plt.tight_layout()
plt.savefig(HERE + "/runs/uljin_evacuation_forecast.png"); print("\nsaved runs/uljin_evacuation_forecast.png")

# ---------- 8) JSON ----------
json.dump(dict(
    method="forecast-based evacuation (V1 physical T_fire seeded from first FIRMS overpass)",
    v_evac_kmh=V_EVAC, wind_deg=WIND_DEG, burn_ha=round(burn_ha), n_risk_bld=len(rows),
    pred=dict(trapped=tp, tight=tt, ok=to, margin_median_h=round(float(np.nanmedian(mp)), 2)),
    firms_posthoc=dict(trapped=fp_, tight=ft_, ok=fo_, margin_median_h=round(float(np.nanmedian(mf)), 2)),
    flip_pred_only_trapped=flip_worse, flip_firms_only_trapped=flip_better, margin_rank_rho=round(float(rho), 3),
    top_risk_pred=[dict(x=round(r["x"], 5), y=round(r["y"], 5), margin_pred_h=round(r["margin_pred"], 2),
                        margin_firms_h=round(r["margin_firms"], 2), t_fire_pred_h=round(r["t_fire_pred"], 1),
                        esc_km=round(r["esc_km"], 2)) for (r, _p) in routes],
    note="V1 예측 도달시각으로 대피 마진 재계산=진짜 forecast. FIRMS-사후와 비교로 예측이 골든타임/갇힘 판정을 어떻게 바꾸는지 정량화. 한계=V1과 동일(풍향민감·상수바람·R0정규화).",
), open(HERE + "/runs/uljin_evacuation_forecast.json", "w"), ensure_ascii=False, indent=1)
print("saved runs/uljin_evacuation_forecast.json")

# ---------- 9) V3 송출 대시보드 페이로드(위험건물 전체 + 도주로 좌표 + 목적지) ----------
burn_rings = [list(zip(*g.exterior.xy)) for g in (burn.geoms if burn.geom_type == "MultiPolygon" else [burn])]
json.dump(dict(
    ignition=t0.isoformat(), v_evac_kmh=V_EVAC, wind_deg=round(WIND_DEG, 0), burn_ha=round(burn_ha),
    burn_rings=[[[round(x, 5), round(y, 5)] for x, y in r] for r in burn_rings],
    dests=[dict(x=round(x, 5), y=round(y, 5), kind=k) for (x, y, k) in dest_pts],
    buildings=[dict(x=round(r["x"], 5), y=round(r["y"], 5), margin=round(r["margin_pred"], 2),
                    t_fire=(round(r["t_fire_pred"], 1) if np.isfinite(r["t_fire_pred"]) else None),
                    esc_km=round(r["esc_km"], 2), burned=r["burned"]) for r in rows],
    routes=[dict(dest_margin=round(r["margin_pred"], 2), t_fire=(round(r["t_fire_pred"], 1) if np.isfinite(r["t_fire_pred"]) else None),
                 esc_km=round(r["esc_km"], 2), start=[round(r["x"], 5), round(r["y"], 5)],
                 path=[[round(px, 5), round(py, 5)] for (px, py) in path]) for (r, path) in routes],
), open(HERE + "/runs/uljin_dispatch_payload.json", "w"), ensure_ascii=False)
print("saved runs/uljin_dispatch_payload.json (V3 대시보드 입력)")
