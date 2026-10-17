"""울진 2022 — 건물 노출 오버레이 (A단계).
각 건물 중심을 최근접 도로노드 상태(소각/화재고립/안전)로 분류.
※ OSM 농촌 건물 커버리지 희박(1,893동) → 정확한 노출엔 건물통합정보 필요(한계 명시).
출력: runs/uljin_exposure.png + 콘솔 카운트.
"""
import json, networkx as nx, rasterio, numpy as np
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

geod = Geod(ellps="WGS84")
HERE = __file__.rsplit("/", 2)[0]
BP = HERE + "/../산불데이터/wbpk_zenodo/data_extracted/wildfire-burn-probability-korea/data/fires/fire_068_202234/burn_mask.tif"
ROADS = HERE + "/data/uljin_route/uljin_roads.geojson"
BLDS = HERE + "/data/uljin_route/uljin_buildings.geojson"
OUT = HERE + "/runs/uljin_exposure.png"

with rasterio.open(BP) as s:
    f = 4; oh, ow = s.height // f, s.width // f
    a = s.read(1, out_shape=(oh, ow)).astype("uint8")
    tr = s.transform * s.transform.scale(s.width / ow, s.height / oh)
    polys = [shape(g) for g, v in shapes(a, mask=a > 0, transform=tr) if v == 1]
    to4326 = Transformer.from_crs(s.crs, "EPSG:4326", always_xy=True).transform
burn = unary_union([shp_transform(to4326, p) for p in polys]).buffer(0); pburn = prep(burn)

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
cut = [i for i in tree.query(burn) if geoms[i].intersects(burn)]
cutE = [(el[i][0], el[i][1]) for i in cut]
main_before = max(nx.connected_components(G), key=len)
Gc = G.copy(); Gc.remove_edges_from(cutE)
safe = max(nx.connected_components(Gc), key=len)
fire_iso = set(n for n in main_before if n not in safe and not pburn.contains(Point(*n)))

# 노드 분류 + KDTree
nodes = list(G.nodes()); arr = np.array(nodes)
def cls(n):
    if pburn.contains(Point(*n)): return "burned"
    if n in fire_iso: return "isolated"
    if n in safe: return "safe"
    return "other"
ncls = np.array([cls(n) for n in nodes])
kdt = cKDTree(arr)

# 건물 분류
b = json.load(open(BLDS)); bxy = []
for ft in b["features"]:
    g = shape(ft["geometry"]); c = g.centroid; bxy.append((c.x, c.y))
bxy = np.array(bxy)
_, idx = kdt.query(bxy)
bcls = ncls[idx]
from collections import Counter
cnt = Counter(bcls)
print(f"건물 총 {len(bxy):,}동 (OSM, 농촌 희박)")
for k in ["burned", "isolated", "safe", "other"]:
    print(f"  {k:9s}: {cnt.get(k,0):,}동")
print(f"★ 소각+고립 = 피해·위험 노출 {cnt.get('burned',0)+cnt.get('isolated',0):,}동")

# 지도
fig, ax = plt.subplots(figsize=(11, 9), dpi=130)
ax.add_collection(LineCollection([list(e[2]["geom"].coords) for e in el], colors="#dddddd", linewidths=0.3, zorder=1))
for geom in (burn.geoms if burn.geom_type == "MultiPolygon" else [burn]):
    xs, ys = geom.exterior.xy; ax.fill(xs, ys, color="#ff7f0e", alpha=0.22, zorder=2)
ax.add_collection(LineCollection([list(el[i][2]["geom"].coords) for i in cut], colors="#d62728", linewidths=0.6, zorder=3))
col = {"burned": "#8b0000", "isolated": "#1f77b4", "safe": "#2ca02c", "other": "#999999"}
for k in ["safe", "other", "isolated", "burned"]:
    m = bcls == k
    if m.any(): ax.scatter(bxy[m, 0], bxy[m, 1], s=10, c=col[k], zorder=5, label=f"{k} ({m.sum()})")
ax.set_title("Uljin 2022 — building exposure by road-access status", fontsize=12)
ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_aspect(1.24)
ax.legend(loc="upper right", fontsize=9, title="buildings (n)")
plt.tight_layout(); plt.savefig(OUT); print("saved", OUT)
