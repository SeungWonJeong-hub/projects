"""확률표본 판독 — 판독용 chip 과 기입표를 만든다.

각 표본점마다 PNG 한 장을 낸다:
  [RapidEye 5m CIR] [RapidEye 5m 자연색] [S2 pre 20m CIR] [S2 post 20m CIR]
가운데 20m 표본 화소를 노란 사각형으로 표시한다. **판정 대상은 그 화소 하나**다.

★판독 자료의 독립성: 1·2번 칸(RapidEye 5m)이 **판정의 근거**다 — S2 와 다른 센서라
모델 입력과 독립이다. 3·4번 칸(S2)은 **벌채·나대지와 소각을 가르기 위한 참고**로만 둔다
(소각은 pre→post 에서 바뀌고, 오래된 나대지는 pre 에도 있다). S2 를 근거로 삼으면 순환이므로
기입표의 `basis` 에 무엇을 보고 판정했는지 적는다.

산출: data/sample_chips/P####.png · sample/sample_form.csv(판독 기입표)
사용: conda run -n geo python src/sample_chips.py [--n 50]  (--n 은 시험 생성용)
"""
import os, sys, json, csv, argparse
os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
RE_DIR = os.path.join(PROJ, "..", "nifos_burn_gt", "ext", "re_img")
WIN_M = float(os.environ.get("WIN_M", "400"))       # chip 한 변(지상 m)
GSD = 20.0


def rgb(bands, rng, gamma=0.75):
    """★늘이는 구간은 **씬 전체에서 밴드별로** 미리 잡는다(rng).

    chip 안에서 밴드별로 늘이면 ①색 관계가 깨지고(식생이 자홍색이 된다) ②chip 마다 색이 달라져
    판독자가 기준을 못 잡는다. 밴드 공통 구간 하나로 늘이면 NIR 이 R·G 를 눌러 전부 새빨개진다.
    씬 단위 밴드별 구간이 둘 다 피한다."""
    out = []
    for a, (p1, p2) in zip(bands, rng):
        v = np.clip((a.astype(np.float32) - p1) / max(p2 - p1, 1e-6), 0, 1) ** gamma
        out.append((v * 255).astype(np.uint8))
    return np.dstack(out)


def band_ranges(arr, idx, lo=2, hi=98, step=4):
    """씬 전체(성기게 표본)에서 밴드별 (p_lo, p_hi).

    ★RapidEye 는 채움값이 **65535** 다 — 이걸 안 빼면 p98 이 65535 가 되고 chip 이 새까매진다.
    (2026-09-10 실제로 그랬다.)"""
    r = []
    for i in idx:
        a = arr[i][::step, ::step].astype(np.float32)
        a = a[(a > 0) & (a < 60000)]
        r.append(tuple(np.percentile(a, [lo, hi])) if a.size else (0.0, 1.0))
    return r


def re_chip(re_id, ux, uy, cache={}):
    """RapidEye 5m — (CIR, 자연색) 각 (H,W,3) uint8. 밴드 1=B 2=G 3=R 4=RE 5=NIR."""
    if re_id not in cache:
        d = gdal.Open(os.path.join(RE_DIR, f"{re_id}.tif"))
        full = d.ReadAsArray()
        cache[re_id] = (d, d.GetGeoTransform(),
                        band_ranges(full, [4, 2, 1]), band_ranges(full, [2, 1, 0]))
    d, g, rc, rn = cache[re_id]
    px = abs(g[1]); k = int(WIN_M / px / 2)
    cx = int((ux - g[0]) / g[1]); cy = int((uy - g[3]) / g[5])
    x0, y0 = max(cx - k, 0), max(cy - k, 0)
    w = min(2 * k, d.RasterXSize - x0); h = min(2 * k, d.RasterYSize - y0)
    if w <= 0 or h <= 0: return None, None, None
    a = d.ReadAsArray(x0, y0, w, h).astype(np.float32)
    cir = rgb([a[4], a[2], a[1]], rc)   # NIR,R,G — 식생=적색
    nat = rgb([a[2], a[1], a[0]], rn)   # R,G,B
    box = ((cx - x0) / w, (cy - y0) / h, GSD / px / w, GSD / px / h)
    return cir, nat, box


def s2_chip(site, row, col, cache={}):
    """★pre 와 post 는 **같은 늘이기 구간**을 쓴다.

    밴드별로 pre/post 를 따로 늘이면 두 칸이 서로 다른 자가 되어 **변화가 지워진다** — 소각으로
    반사율이 바뀌어도 각 칸이 제 분포로 정규화해 버리니 pre→post 색차가 실제 변화가 아니라
    스트레치 차이가 된다. 이 칸의 존재 이유가 "벌채·나대지는 pre 에도 있고 소각은 pre→post 에서
    바뀐다" 인데, 따로 늘이면 그 판별이 불가능하다(2026-09-10)."""
    if site not in cache:
        C = os.path.join(PROJ, "nifos_cache", f"{site}_refoot")
        arr = np.load(os.path.join(C, "scene_20m.npz"))["x"]
        # 밴드별로 pre·post 를 합쳐 하나의 구간을 잡는다(NIR,R,G = 3,2,1 / 9,8,7)
        both = np.concatenate([arr[[3, 2, 1]], arr[[9, 8, 7]]], axis=1)
        r = band_ranges(both, [0, 1, 2], step=2)
        cache[site] = (arr, r, r)
    x, rpre, rpost = cache[site]; k = int(WIN_M / GSD / 2)
    y0, x0 = max(row - k, 0), max(col - k, 0)
    y1, x1 = min(row + k, x.shape[1]), min(col + k, x.shape[2])
    out = []
    for off, rr in ((0, rpre), (6, rpost)):              # pre, post
        b = x[[off + 3, off + 2, off + 1], y0:y1, x0:x1].astype(np.float32)   # NIR,R,G
        out.append(rgb([b[0], b[1], b[2]], rr))
    box = ((col - x0) / (x1 - x0), (row - y0) / (y1 - y0), 1 / (x1 - x0), 1 / (y1 - y0))
    return out[0], out[1], box


def draw(panels, boxes, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.patches as mp
    f, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    names = ["RapidEye 5m CIR  [JUDGE ON THIS]", "RapidEye 5m natural color",
             "S2 pre 20m (context only)", "S2 post 20m (context only)"]
    for i, (a, b, nm) in enumerate(zip(panels, boxes, names)):
        ax[i].imshow(a); ax[i].set_title(nm, fontsize=8); ax[i].axis("off")
        if b:
            H, W = a.shape[:2]
            ax[i].add_patch(mp.Rectangle(((b[0] - b[2] / 2) * W, (b[1] - b[3] / 2) * H),
                                         max(b[2] * W, 3), max(b[3] * H, 3),
                                         fill=False, ec="yellow", lw=1.6))
    f.suptitle(title, fontsize=10)
    f.tight_layout(); f.savefig(path, dpi=110); plt.close(f)


WC = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up",
      60: "bare/sparse", 70: "snow/ice", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}


def lcname(site, row, col, cache={}):
    """★토지피복을 제목에 적는다 — §23.1 실패의 절반이 피복 오판이었다(초지·경작지 소각면은
    밝아진다). ⚠WorldCover 2021 은 화재 이후라 **소각 여부의 근거가 아니다** — 비산림 판별용."""
    if site not in cache:
        f = os.path.join(PROJ, "nifos_cache", f"{site}_refoot", "wc_20m.npy")
        cache[site] = np.load(f) if os.path.exists(f) else None
    a = cache[site]
    if a is None or not (0 <= row < a.shape[0] and 0 <= col < a.shape[1]): return "n/a"
    return WC.get(int(a[row, col]), f"code{int(a[row,col])}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    S = json.load(open(os.path.join(PROJ, "sample", "sample_points.json"), encoding="utf-8"))
    pts = S["points"][:a.n] if a.n else S["points"]
    od = os.path.join(PROJ, "data", "sample_chips"); os.makedirs(od, exist_ok=True)
    fp = os.path.join(PROJ, "sample", "sample_form.csv")
    with open(fp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "chip", "판정", "등급", "근거", "확신", "비고"])
        # ★4지선다. 소각/비소각 이분이면 **벌채를 판독 시점에 버린다** — §13 의 오탐 21~33% 를
        #   벌채와 순수 오탐으로 갈라야 하는데, 1,284장을 다시 볼 일은 없으므로 여기서 못 받으면
        #   끝이다. '판정불가' 가 없으면 애매한 화소에서 강제 선택이 일어나고 그 쏠림을 못 본다.
        w.writerow(["# 판정: 소각 / 벌채·개간 / 비변화 / 판정불가",
                    "# 등급: 소각일 때만 경/중/심",
                    "# 근거: RE(RapidEye만) / RE+S2",
                    "# 확신: 상/중/하",
                    "# ★탄화는 '어둡다'가 아니라 '같은 피복의 안 탄 이웃 대비 변했다'로 본다 — 초지·경작지·촌락 소각면은 재·노출토양이 남아 오히려 밝다(§23.1)",
                    "", ""])
        for i, p in enumerate(pts, 1):
            cir, nat, b1 = re_chip(p["re_scene"], p["utm_x"], p["utm_y"])
            if cir is None: continue
            pre, post, b2 = s2_chip(p["site"], p["row"], p["col"])
            png = os.path.join(od, f"{p['id']}.png")
            draw([cir, nat, pre, post], [b1, b1, b2, b2], png,
                 f"{p['id']}   {p['site']} / {p['re_scene']}   "
                 f"yellow box = the 20 m pixel to judge  ({int(WIN_M)} m window)"
                 f"   landcover(WorldCover 2021): {lcname(p['site'], p['row'], p['col'])}")
            w.writerow([p["id"], f"sample_chips/{p['id']}.png", "", "", "", "", ""])
            if i % 100 == 0: print(f"  … {i}/{len(pts)}", flush=True)
    print(f"chip {len(pts)}장 → data/sample_chips/ · 기입표 → sample/sample_form.csv")


if __name__ == "__main__":
    main()
