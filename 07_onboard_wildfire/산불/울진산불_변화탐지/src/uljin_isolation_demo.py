"""울진 2022 산불 — 도로 고립/도주로 트랙 프로토타입 (GSD 자유 트랙).
입력: OSM 한국 도로(osm_korea) + wbpk 소각마스크(fire_068_202234).
출력: 소각으로 차단된 도로 + 진짜 화재고립 구역 지도 (runs/uljin_isolation.png).
정의: 화재고립 = ①차단 전 본토망 연결 ②차단 후 안전본토서 끊김 ③자신은 안 탐.
정적(최종 소각경계) 버전 — 시간의존(도달시간) 버전은 확산 envelope 얹으면 됨.
"""
import json, networkx as nx, rasterio
from rasterio.features import shapes
from shapely.geometry import shape, LineString, Point
from shapely.ops import unary_union, transform as shp_transform
from shapely.strtree import STRtree
from shapely.prepared import prep
from pyproj import Transformer, Geod
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

geod = Geod(ellps="WGS84")
HERE = __file__.rsplit("/", 2)[0]
BP = HERE + "/../산불데이터/wbpk_zenodo/data_extracted/wildfire-burn-probability-korea/data/fires/fire_068_202234/burn_mask.tif"
ROADS = HERE + "/data/uljin_route/uljin_roads.geojson"
OUT = HERE + "/runs/uljin_isolation.png"

# 1) 소각 폴리곤
with rasterio.open(BP) as s:
    f = 4; oh, ow = s.height // f, s.width // f
    a = s.read(1, out_shape=(oh, ow)).astype("uint8")
    tr = s.transform * s.transform.scale(s.width / ow, s.height / oh)
    polys = [shape(g) for g, v in shapes(a, mask=a > 0, transform=tr) if v == 1]
    to4326 = Transformer.from_crs(s.crs, "EPSG:4326", always_xy=True).transform
burn = unary_union([shp_transform(to4326, p) for p in polys]).buffer(0)
pburn = prep(burn)

# 2) 도로 그래프
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

# 3) 차단 엣지
el = list(G.edges(data=True)); geoms = [e[2]["geom"] for e in el]; tree = STRtree(geoms)
cut = [i for i in tree.query(burn) if geoms[i].intersects(burn)]
cutE = [(el[i][0], el[i][1]) for i in cut]
cut_len = sum(el[i][2]["length"] for i in cut)

# 4) 화재고립
main_before = max(nx.connected_components(G), key=len)
Gc = G.copy(); Gc.remove_edges_from(cutE)
safe = max(nx.connected_components(Gc), key=len)
fire_iso = [n for n in main_before if n not in safe and not pburn.contains(Point(*n))]
H = Gc.subgraph(fire_iso)
frags = sorted((c for c in nx.connected_components(H) if len(c) >= 3), key=len, reverse=True)

print(f"소각 area≈{abs(geod.geometry_area_perimeter(burn)[0])/1e4:,.0f} ha")
print(f"도로 그래프 노드 {G.number_of_nodes():,} 엣지 {G.number_of_edges():,}")
print(f"차단 도로 {len(cutE):,}세그 {cut_len/1000:.1f} km")
print(f"화재고립 {len(fire_iso):,}노드 / {len(frags)}조각")

# 5) 지도
fig, ax = plt.subplots(figsize=(11, 9), dpi=130)
segs = [list(e[2]["geom"].coords) for e in el]
ax.add_collection(LineCollection(segs, colors="#cfcfcf", linewidths=0.3, zorder=1))
ax.add_collection(LineCollection([list(el[i][2]["geom"].coords) for i in cut],
                                 colors="#d62728", linewidths=0.7, zorder=3))
for geom in (burn.geoms if burn.geom_type == "MultiPolygon" else [burn]):
    xs, ys = geom.exterior.xy
    ax.fill(xs, ys, color="#ff7f0e", alpha=0.25, zorder=2)
iso_segs = [list(e[2]["geom"].coords) for e in Gc.subgraph(fire_iso).edges(data=True)]
ax.add_collection(LineCollection(iso_segs, colors="#1f77b4", linewidths=0.6, zorder=4))
for c in frags[:8]:
    cx = sum(n[0] for n in c) / len(c); cy = sum(n[1] for n in c) / len(c)
    ax.plot(cx, cy, "o", ms=6, mfc="none", mec="#1f77b4", mew=1.5, zorder=5)
    ax.annotate(f"{len(c)}", (cx, cy), fontsize=7, color="#1f77b4")
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], color="#cfcfcf", lw=1.5, label="roads"),
    Line2D([], [], color="#ff7f0e", lw=6, alpha=0.3, label="burn scar (~19,100 ha)"),
    Line2D([], [], color="#d62728", lw=1.5, label=f"severed roads ({cut_len/1000:.0f} km)"),
    Line2D([], [], color="#1f77b4", lw=1.5, label=f"fire-isolated ({len(fire_iso):,} nodes)"),
], loc="upper right", fontsize=9)
ax.set_title("Uljin 2022 wildfire — road severance & fire-caused isolation", fontsize=12)
ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_aspect(1.24)
plt.tight_layout(); plt.savefig(OUT); print("saved", OUT)
