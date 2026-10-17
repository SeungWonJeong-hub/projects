"""울진 2022 — B단계: 시간의존 도로차단·고립 시퀀스.
화재 도달시간 = 도로 세그먼트 근처 FIRMS 최초 화점 시각(반경 내 최소, 없으면 최근접).
각 12h 빈 T에서: 도달시각≤T인 차단도로 누적 제거 → 안전본토망 재계산
→ 그 시점 화재고립 노드/건물. 노드별 '고립시각'도 기록.
※ FIRMS 12h·375m → 도달시간은 반나절 스케일(분 단위 아님). 정직 명시.
출력: runs/uljin_time_seq.png(스냅샷4+곡선) + runs/uljin_time_seq.json
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

geod = Geod(ellps="WGS84")
HERE = __file__.rsplit("/", 2)[0]
FIRE = HERE + "/../산불데이터/wbpk_zenodo/data_extracted/wildfire-burn-probability-korea/data/fires/fire_068_202234/burn_mask.tif"
ROADS = HERE + "/data/uljin_route/uljin_roads.geojson"
BLDS = HERE + "/data/uljin_route/uljin_buildings.geojson"
FIRMS = HERE + "/data/uljin_route/uljin_firms_timeline.json"

# --- 소각 폴리곤 ---
with rasterio.open(FIRE) as s:
    f = 4; oh, ow = s.height // f, s.width // f
    a = s.read(1, out_shape=(oh, ow)).astype("uint8")
    tr = s.transform * s.transform.scale(s.width / ow, s.height / oh)
    polys = [shape(g) for g, v in shapes(a, mask=a > 0, transform=tr) if v == 1]
    to4326 = Transformer.from_crs(s.crs, "EPSG:4326", always_xy=True).transform
burn = unary_union([shp_transform(to4326, p) for p in polys]).buffer(0); pburn = prep(burn)

# --- 도로 그래프 ---
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
el = list(G.edges(data=True)); geoms = [e[2]["geom"] for e in el]; tree = STRtree(geoms)
cut_idx = [i for i in tree.query(burn) if geoms[i].intersects(burn)]

# --- FIRMS 도달시간 → 차단 엣지에 태깅 ---
fp = json.load(open(FIRMS))
ft_xy = np.array([[r["lon"], r["lat"]] for r in fp])
ft_t = np.array([datetime.fromisoformat(r["dt"]) for r in fp])
t0 = ft_t.min()
ft_h = np.array([(t - t0).total_seconds() / 3600 for t in ft_t])  # 발화후 시간(h)
kdt = cKDTree(ft_xy)
DEG = 1.5 / 111.0  # 반경 ~1.5km
arr_h = {}
for i in cut_idx:
    mx, my = geoms[i].interpolate(0.5, normalized=True).coords[0]
    near = kdt.query_ball_point([mx, my], DEG)
    if near:
        arr_h[i] = float(ft_h[near].min())           # 반경 내 최초 화점
    else:
        _, j = kdt.query([mx, my]); arr_h[i] = float(ft_h[j])  # 없으면 최근접
main_before = max(nx.connected_components(G), key=len)

# --- 건물 최근접 노드 매핑 ---
nodes = list(G.nodes()); narr = np.array(nodes); nkdt = cKDTree(narr)
b = json.load(open(BLDS)); bxy = np.array([[shape(x["geometry"]).centroid.x, shape(x["geometry"]).centroid.y] for x in b["features"]])
_, bnode_idx = nkdt.query(bxy); node_of_b = [nodes[i] for i in bnode_idx]

# --- 12h 빈별 누적 차단 → 고립 ---
maxh = math.ceil(ft_h.max()); bins = list(range(12, maxh + 13, 12))
seq = []
iso_time_node = {}     # 노드 → 고립시각(h)
for T in bins:
    cutE = [(el[i][0], el[i][1]) for i in cut_idx if arr_h[i] <= T]
    Gc = G.copy(); Gc.remove_edges_from(cutE)
    safe = max(nx.connected_components(Gc), key=len)
    burned_by_T = set()  # 이 시점 이미 탄 도로 노드(근사: 차단엣지 양끝)
    for (u, v) in cutE: burned_by_T.add(u); burned_by_T.add(v)
    iso = set(n for n in main_before if n not in safe and n not in burned_by_T)
    for n in iso:
        if n not in iso_time_node: iso_time_node[n] = T
    iso_bld = sum(1 for n in node_of_b if n in iso)
    burned_bld = sum(1 for n in node_of_b if n in burned_by_T)
    seq.append(dict(T=T, cut_km=sum(el[i][2]["length"] for i in cut_idx if arr_h[i] <= T) / 1000,
                    iso_nodes=len(iso), iso_bld=iso_bld, burned_bld=burned_bld, safe=len(safe)))
    print(f"T+{T:>3}h ({(t0+timedelta(hours=T)).strftime('%m-%d %H:%M')}): 차단 {seq[-1]['cut_km']:6.1f}km | "
          f"고립노드 {len(iso):6,} | 고립건물 {iso_bld:4} | 소각건물 {burned_bld:4}")

json.dump(dict(bins=seq, ignition=t0.isoformat()), open(HERE + "/runs/uljin_time_seq.json", "w"), ensure_ascii=False, indent=1)

# --- 그림: 스냅샷 4 + 곡선 ---
snap_T = [12, 24, 60, bins[-1]]
fig = plt.figure(figsize=(15, 10), dpi=120)
allsegs = [list(e[2]["geom"].coords) for e in el]
for k, T in enumerate(snap_T):
    ax = fig.add_subplot(2, 3, k + 1)
    ax.add_collection(LineCollection(allsegs, colors="#e2e2e2", linewidths=0.2, zorder=1))
    for g in (burn.geoms if burn.geom_type == "MultiPolygon" else [burn]):
        xs, ys = g.exterior.xy; ax.fill(xs, ys, color="#ffcaa0", alpha=0.5, zorder=2)
    cutseg = [list(geoms[i].coords) for i in cut_idx if arr_h[i] <= T]
    ax.add_collection(LineCollection(cutseg, colors="#d62728", linewidths=0.6, zorder=3))
    isonodes = [n for n, tt in iso_time_node.items() if tt <= T]
    if isonodes:
        xy = np.array(isonodes); ax.scatter(xy[:, 0], xy[:, 1], s=1, c="#1f77b4", zorder=4)
    ax.set_title(f"T+{T}h ({(t0+timedelta(hours=T)).strftime('%m-%d %Hh')})", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect(1.24)
# 곡선
ax = fig.add_subplot(2, 3, (5, 6))
Ts = [x["T"] for x in seq]
ax.plot(Ts, [x["cut_km"] for x in seq], "-o", color="#d62728", label="차단 도로 (km)")
ax.plot(Ts, [x["iso_bld"] for x in seq], "-s", color="#1f77b4", label="고립 건물 (동)")
ax.plot(Ts, [x["burned_bld"] for x in seq], "-^", color="#8b0000", label="소각 건물 (동)")
ax.set_xlabel("발화 후 경과시간 (h)"); ax.set_ylabel("누적"); ax.legend(fontsize=9)
ax.set_title("시간에 따른 차단·고립·소각 진행", fontsize=10); ax.grid(alpha=0.3)
plt.suptitle("Uljin 2022 — time-dependent road severance & isolation (FIRMS 12h scale)", fontsize=13)
plt.tight_layout(); plt.savefig(HERE + "/runs/uljin_time_seq.png"); print("saved runs/uljin_time_seq.png")
