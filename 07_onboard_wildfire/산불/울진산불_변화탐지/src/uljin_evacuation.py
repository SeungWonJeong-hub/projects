"""울진 2022 — C단계: 대피경로(도주로) + 도착·탈출 마진(골든타임).

왜 '연결성 고립'이 아니라 '시간 경쟁'인가(앞선 검증의 결론):
  울진 화재는 인구 밀집 계곡·해안 회랑을 통째로 태웠고 도로망은 조밀 → 특정 도로가
  타도 우회로가 남아 '위상적 완전차단' ≈ 0, '우회거리 폭발' ≈ 0 이었다(측정 확인).
  즉 이 화재의 대피 위험은 '길이 끊겨 갇힘'이 아니라 '불이 내가 빠져나가기 전에 도달'.

정식화(도착-탈출 마진):
  T_fire(건물)  = FIRMS 화점 보간으로 그 지점에 불이 닿는 시각(발화후 h).
  T_escape(건물)= 도로 최단거리로 '살아남은 실제 대피소/읍내' 도달까지 시간 = dist / V_EVAC.
                 안전목적지 = OSM 대피소POI(마을회관·학교·shelter·병원 등)∪읍/시 중 소각권 밖.
  margin        = T_fire − T_escape.  음수 = 빠져나가기 전에 불이 옴 = 위험/갇힘.
  골든타임(마을) = 그 지점에서 '지금 출발하면 산다'가 성립하는 마지막 시각 ≈ T_fire.

산출: 건물별 마진 지도(발산색) + 가장 위험한 마을의 실제 도주로 + 마진 분포.
※ 한계: FIRMS 375m·12h(도착시각 반나절 스케일), V_EVAC 가정, 도주로가 이미 탄
   방향으로 향할 수 있음(정적 최근접 exit) — 전부 명시. 절대수치 아닌 상대위험 지도.
출력: runs/uljin_evacuation.png + runs/uljin_evacuation.json
"""
import json, math, networkx as nx, rasterio, numpy as np
from datetime import datetime, timedelta
from rasterio.features import shapes
from shapely.geometry import shape, LineString, Point
from shapely.ops import unary_union, transform as shp_transform
from shapely.strtree import STRtree
from shapely.prepared import prep
from pyproj import Transformer, Geod
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.cm as cm
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
SHELTERS = HERE + "/data/uljin_route/uljin_shelters.geojson"   # OSM 대피소 POI(마을회관·학교·shelter 등)
PLACES = HERE + "/data/uljin_route/uljin_places.geojson"       # OSM 읍·시(피난처)
V_EVAC = 30.0                 # 대피 평균속도 km/h (야간·연기·정체 보수적 가정)
FIRMS_R_KM = 2.0             # 화점 보간 반경
SAFE_PAD_KM = 0.3            # 목적지가 소각경계서 이만큼 밖이어야 '살아남은 안전지'로 인정
SHELTER_TYPES = {"town_hall", "school", "shelter", "hospital", "community_centre", "university"}

# ---------- 1) 소각 폴리곤 ----------
with rasterio.open(FIRE) as s:
    f = 4; oh, ow = s.height // f, s.width // f
    a = s.read(1, out_shape=(oh, ow)).astype("uint8")
    tr = s.transform * s.transform.scale(s.width / ow, s.height / oh)
    polys = [shape(g) for g, v in shapes(a, mask=a > 0, transform=tr) if v == 1]
    to4326 = Transformer.from_crs(s.crs, "EPSG:4326", always_xy=True).transform
burn = unary_union([shp_transform(to4326, p) for p in polys]).buffer(0)
pburn = prep(burn)
danger = burn.buffer(SAFE_PAD_KM / 111.0); pdanger = prep(danger)   # 이 안 = 위험, 밖 = 안전목적지
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
el = list(G.edges(data=True)); geoms = [e[2]["geom"] for e in el]

# ---------- 3) FIRMS 화재 도착시각 필드 ----------
fp = json.load(open(FIRMS))
ft_xy = np.array([[r["lon"], r["lat"]] for r in fp])
ft_t = np.array([datetime.fromisoformat(r["dt"]) for r in fp])
t0 = ft_t.min()
ft_h = np.array([(t - t0).total_seconds() / 3600 for t in ft_t])
fkdt = cKDTree(ft_xy); FDEG = FIRMS_R_KM / 111.0
def fire_arrival(x, y):
    near = fkdt.query_ball_point([x, y], FDEG)
    if near: return float(ft_h[near].min())
    _, j = fkdt.query([x, y]); return float(ft_h[j])

# ---------- 4) 실제 안전 목적지(대피소·읍/시) → 도로망 스냅 → 최단탈출거리 ----------
nodes = list(G.nodes()); nkdt = cKDTree(np.array(nodes))
def load_dests(path, kind):
    out = []
    try: d = json.load(open(path))
    except FileNotFoundError: return out
    for ft in d["features"]:
        g = ft.get("geometry")
        if not g: continue
        p = shape(g); c = p.centroid
        fc = (ft["properties"].get("fclass") or kind)
        if kind == "shelter" and fc not in SHELTER_TYPES: continue
        out.append((c.x, c.y, fc, kind, ft["properties"].get("name")))
    return out
dests_all = load_dests(SHELTERS, "shelter") + load_dests(PLACES, "town")
# 소각권(위험 버퍼) 밖에 있는 = 화재를 피한 '살아남은 안전 목적지'만 채택
dests = [dd for dd in dests_all if not pdanger.contains(Point(dd[0], dd[1]))]
burned_dest = len(dests_all) - len(dests)
# 각 목적지를 최근접 도로노드에 스냅 → multi-source exit 집합
exit_set, dest_pts = set(), []
for (x, y, fc, kind, nm) in dests:
    _, j = nkdt.query([x, y]); exit_set.add(nodes[j]); dest_pts.append((x, y, kind))
esc_m = nx.multi_source_dijkstra_path_length(G, exit_set, weight="length")  # 각 노드→최근접 목적지 도로거리(m)
n_shelter = sum(1 for dd in dests if dd[3] == "shelter"); n_town = sum(1 for dd in dests if dd[3] == "town")
print(f"소각 {burn_ha:,.0f} ha | 본토노드 {len(nodes):,} | 안전목적지 채택 {len(dests):,}"
      f"(대피소 {n_shelter}·읍시 {n_town}, 소각권내 제외 {burned_dest}) | 스냅 exit노드 {len(exit_set):,}")

# ---------- 5) 건물별 도착-탈출 마진 ----------
b = json.load(open(BLDS))
bxy = np.array([[shape(x["geometry"]).centroid.x, shape(x["geometry"]).centroid.y] for x in b["features"]])
_, bnode = nkdt.query(bxy)
rows = []            # 위험권(소각 포함) 건물만
for k, (bx, by) in enumerate(bxy):
    if not pdanger.contains(Point(bx, by)):
        continue      # 안전지대 건물 제외
    tf = fire_arrival(bx, by)                 # 불 도착시각(h)
    de = esc_m.get(nodes[bnode[k]], float("inf"))  # 탈출거리(m)
    te = (de / 1000.0) / V_EVAC               # 탈출 소요시간(h)
    rows.append(dict(i=k, x=float(bx), y=float(by), t_fire=tf, esc_km=de / 1000.0,
                     t_esc=te, margin=tf - te, burned=bool(pburn.contains(Point(bx, by)))))
marg = np.array([r["margin"] for r in rows]) if rows else np.array([])
n_trap = int((marg < 0).sum()); n_tight = int(((marg >= 0) & (marg < 1)).sum()); n_ok = int((marg >= 1).sum())
print(f"\n[위험권 건물 {len(rows):,}동 — 도착·탈출 마진]  (V_EVAC={V_EVAC:.0f}km/h)")
print(f"  margin<0  갇힘(불이 먼저 도달)  : {n_trap:,}동")
print(f"  0≤margin<1h 초박빙            : {n_tight:,}동")
print(f"  margin≥1h  여유              : {n_ok:,}동")
if len(marg): print(f"  마진 중앙 {np.median(marg):+.1f}h | 최악 {marg.min():+.1f}h | 소각권 탈출거리 중앙 {np.median([r['esc_km'] for r in rows]):.1f}km")

# ---------- 6) 대표 도주로 (가장 위험한 마을 12곳: margin 최소) ----------
cand = sorted([r for r in rows if r["esc_km"] < float("inf")], key=lambda r: r["margin"])
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
fig = plt.figure(figsize=(16, 8), dpi=125)
axm = fig.add_subplot(1, 2, 1)
axm.add_collection(LineCollection([list(g.coords) for g in geoms], colors="#e8e8e8", linewidths=0.2, zorder=1))
for g in (burn.geoms if burn.geom_type == "MultiPolygon" else [burn]):
    xs, ys = g.exterior.xy; axm.fill(xs, ys, color="#ffb27a", alpha=0.35, zorder=2)
# 실제 안전 목적지 (대피소=녹색▲, 읍/시=파란■)
dp = np.array([(x, y) for (x, y, k) in dest_pts]) if dest_pts else np.empty((0, 2))
dk = np.array([k for (x, y, k) in dest_pts]) if dest_pts else np.array([])
if len(dp):
    ms = dk == "shelter"; mt = dk == "town"
    if ms.any(): axm.scatter(dp[ms, 0], dp[ms, 1], marker="^", s=34, c="#0b8a0b", edgecolors="k", linewidths=0.3, zorder=8)
    if mt.any(): axm.scatter(dp[mt, 0], dp[mt, 1], marker="s", s=46, c="#1565c0", edgecolors="k", linewidths=0.4, zorder=9)
# 건물 마진 산점
bx_ = np.array([r["x"] for r in rows]); by_ = np.array([r["y"] for r in rows])
norm = TwoSlopeNorm(vmin=min(-3, float(marg.min()) if len(marg) else -3), vcenter=0, vmax=6)
sc = axm.scatter(bx_, by_, c=marg, s=14, cmap="RdYlGn", norm=norm, zorder=4, edgecolors="none")
cb = fig.colorbar(sc, ax=axm, fraction=0.035, pad=0.02); cb.set_label("도착-탈출 마진 (h)  ◀위험 | 여유▶", fontsize=9)
# 도주로
for (r, path) in routes:
    pts = np.array(path, dtype=float)
    axm.plot(pts[:, 0], pts[:, 1], "-", color="#1a1a1a", lw=1.4, zorder=6, alpha=0.8)
    axm.plot(r["x"], r["y"], "*", ms=11, mfc="#d62728", mec="k", mew=0.5, zorder=7)
axm.legend(handles=[
    Line2D([], [], color="#ffb27a", lw=6, alpha=0.5, label=f"소각 (~{burn_ha:,.0f} ha)"),
    Line2D([], [], marker="^", color="w", mfc="#0b8a0b", mec="k", ms=9, label=f"대피소 {n_shelter}(마을회관·학교)"),
    Line2D([], [], marker="s", color="w", mfc="#1565c0", mec="k", ms=9, label=f"읍·시 {n_town}"),
    Line2D([], [], marker="*", color="w", mfc="#d62728", mec="k", ms=12, label="최고위험 마을(margin 최소)"),
    Line2D([], [], color="#1a1a1a", lw=1.4, label="실제 최단 도주로"),
], loc="upper left", fontsize=8)
axm.set_title(f"울진 2022 — 도주로 & 도착-탈출 마진 (V={V_EVAC:.0f}km/h)", fontsize=12)
axm.set_xlabel("lon"); axm.set_ylabel("lat"); axm.set_aspect(1.24)

# 오른쪽 상단: 마진 히스토그램
axh = fig.add_subplot(2, 2, 2)
if len(marg):
    hi = math.ceil(marg.max()) + 1
    axh.hist(marg, bins=np.arange(math.floor(marg.min()), hi + 1, 1.0), color="#4a90d9", edgecolor="w")
    axh.axvline(0, color="#d62728", lw=1.5, ls="--", label="갇힘 경계(0)")
    axh.axvline(np.median(marg), color="k", lw=1, ls=":", label=f"중앙 {np.median(marg):+.1f}h")
    axh.legend(fontsize=8)
axh.set_xlabel("도착-탈출 마진 (h)  ※FIRMS 12h주기→이산"); axh.set_ylabel("위험권 건물 수")
axh.set_title("마진 분포 (음수=불이 먼저 도달)", fontsize=10); axh.grid(alpha=0.3)

# 오른쪽 하단: 탈출거리 vs 불도착시각 산점(경쟁 구조)
axs = fig.add_subplot(2, 2, 4)
if rows:
    tf = np.array([r["t_fire"] for r in rows]); ek = np.array([r["esc_km"] for r in rows])
    axs.scatter(ek, tf, c=marg, s=10, cmap="RdYlGn", norm=norm, edgecolors="none")
    xx = np.linspace(0, max(ek.max(), 1), 50)
    axs.plot(xx, xx / V_EVAC, "k--", lw=1, label=f"margin=0 선 (V={V_EVAC:.0f})")
    axs.legend(fontsize=8)
axs.set_xlabel("탈출거리 (km)"); axs.set_ylabel("불 도착시각 (h)")
axs.set_title("탈출거리 vs 불 도착시각", fontsize=10); axs.grid(alpha=0.3)

plt.suptitle("Uljin 2022 — evacuation: fire-arrival vs escape-time margin", fontsize=13)
plt.tight_layout()
plt.savefig(HERE + "/runs/uljin_evacuation.png"); print("saved runs/uljin_evacuation.png")

# ---------- 8) JSON ----------
json.dump(dict(
    ignition=t0.isoformat(), v_evac_kmh=V_EVAC, safe_pad_km=SAFE_PAD_KM, burn_ha=round(burn_ha),
    n_dest_shelter=n_shelter, n_dest_town=n_town, n_dest_burned_out=burned_dest,
    n_risk_bld=len(rows), n_trapped=n_trap, n_tight=n_tight, n_ok=n_ok,
    margin_median_h=(round(float(np.median(marg)), 2) if len(marg) else None),
    margin_worst_h=(round(float(marg.min()), 2) if len(marg) else None),
    esc_km_median=(round(float(np.median([r["esc_km"] for r in rows])), 2) if rows else None),
    top_risk=[dict(x=round(r["x"], 5), y=round(r["y"], 5), margin_h=round(r["margin"], 2),
                   t_fire_h=round(r["t_fire"], 1), esc_km=round(r["esc_km"], 2))
              for (r, _p) in routes],
), open(HERE + "/runs/uljin_evacuation.json", "w"), ensure_ascii=False, indent=1)
print("saved runs/uljin_evacuation.json")
