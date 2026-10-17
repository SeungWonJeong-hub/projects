"""등급별 소각면적 신뢰성 — 측정·보정 하니스 (자율 반복용).

독립 GT = NIFoS 안동 피해등급(심/중/경, 20m). S2 를 GT 격자에 맞춰 취득해 dNBR 등
지수·절단·학습으로 등급을 매기고, **등급별 면적오차**와 혼동행렬을 GT 대비로 잰다.

목표: 등급별 면적을 '믿을 수 있는' 수준으로. 라운드마다 문제→해법→적용→측정.
결과는 severity_calib_log.md 에 누적한다.

사용: LD_LIBRARY_PATH 불필요(torch 안 씀). python src/severity_calib.py [--round N]
"""
import os, sys, json, argparse
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(PROJ, "..", "..", "웹앱"))
NIFOS = os.path.expanduser("~/Steven/nifos_burn_gt/andong/andong_severity_20m.tif")
CACHE = os.path.join(PROJ, "runs", "severity_calib_cache")
os.makedirs(CACHE, exist_ok=True)
PXA_HA = 20.0 * 20.0 / 1e4   # 0.04 ha/px

# NIFoS 안동 fire 이벤트 (2025 봄) — GT 격자에 맞춰 취득
PRE_WIN = ("2025-03-15", "2025-03-27", "2025-03-22")
POST_WIN = ("2025-04-05", "2025-04-16", "2025-04-11")


def fetch_to_gt_grid():
    """GT 와 동일 격자(EPSG·bounds·20m)로 S2 pre/post B8A·B12·B04 를 취득(캐시)."""
    from core import acquire
    g = gdal.Open(NIFOS); gt = g.GetGeoTransform()
    W, H = g.RasterXSize, g.RasterYSize
    ulx, uly = gt[0], gt[3]; lrx = ulx + W * gt[1]; lry = uly + H * gt[5]
    proj = g.GetProjection()
    from osgeo import osr
    srs = osr.SpatialReference(); srs.ImportFromWkt(proj)
    epsg = int(srs.GetAuthorityCode(None))
    # AOI 중심 → lon/lat
    t = osr.SpatialReference(); t.ImportFromEPSG(4326); t.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(srs, t)
    lon, lat, _ = ct.TransformPoint((ulx + lrx) / 2, (uly + lry) / 2)

    def pick(win):
        f = acquire.search([lon - 0.15, lat - 0.12, lon + 0.15, lat + 0.12], win[0], win[1], max_cloud=40)
        f.sort(key=lambda x: (abs((np.datetime64(x["datetime"]) - np.datetime64(win[2])).astype(int)), x["cloud"]))
        return f[0]

    bands = {"B04": "red", "B8A": "nir08", "B12": "swir22"}
    out = {}
    for tag, win in (("pre", PRE_WIN), ("post", POST_WIN)):
        sc = pick(win)
        for b, key in bands.items():
            dest = os.path.join(CACHE, f"{tag}_{b}.tif")
            if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
                opts = gdal.TranslateOptions(
                    xRes=20, yRes=20, resampleAlg="average",
                    projWin=[ulx, uly, lrx, lry], projWinSRS=f"EPSG:{epsg}",
                    format="GTiff", creationOptions=["COMPRESS=DEFLATE"])
                gdal.Translate(dest, "/vsicurl/" + sc["assets"][key]["href"], options=opts)
            out[f"{tag}_{b}"] = gdal.Open(dest).ReadAsArray().astype("float32") / 10000.0
        out[f"{tag}_id"] = sc["id"]; out[f"{tag}_date"] = sc["datetime"]
    return out


def nbr(nir, swir):
    d = nir + swir
    return np.where(d > 1e-6, (nir - swir) / np.maximum(d, 1e-6), 0.0)


def load_gt():
    a = gdal.Open(NIFOS).ReadAsArray()
    # 값 1/2/3 → 면적으로 심/중/경 판별(SESSION: 심>중>경 면적). 0=배경.
    areas = {v: int((a == v).sum()) for v in (1, 2, 3)}
    order = sorted(areas, key=lambda v: -areas[v])   # 면적 큰 순 = 심,중,경
    # 심각도 순서: 큰 면적=심? SESSION 심926>중660>경432 → 면적순=심,중,경
    lab = {order[0]: "심", order[1]: "중", order[2]: "경"}
    grade = np.zeros_like(a, dtype="int8")            # 0=비소각,3=심,2=중,1=경
    sev_val = {"심": 3, "중": 2, "경": 1}
    for v, nm in lab.items():
        grade[a == v] = sev_val[nm]
    return grade, {nm: areas[v] * PXA_HA for v, nm in lab.items()}


def grade_area(pred_sev):
    return {nm: int((pred_sev == k).sum()) * PXA_HA for nm, k in (("심", 3), ("중", 2), ("경", 1))}


def assign(dnbr, burn, cuts):
    """burn 마스크 안에서 dNBR 절단(lo,hi)로 경(1)/중(2)/심(3)."""
    s = np.zeros_like(dnbr, dtype="int8")
    s[burn & (dnbr < cuts[0])] = 1
    s[burn & (dnbr >= cuts[0]) & (dnbr < cuts[1])] = 2
    s[burn & (dnbr >= cuts[1])] = 3
    return s


def report(name, pred_sev, gt_sev, gt_area):
    pa = grade_area(pred_sev)
    lines = [f"### {name}"]
    lines.append(f"| 등급 | GT(ha) | 예측(ha) | 면적오차 |")
    lines.append("|---|---|---|---|")
    for nm in ("심", "중", "경"):
        g = gt_area.get(nm, 0); p = pa[nm]
        err = (p - g) / max(g, 1e-9) * 100
        lines.append(f"| {nm} | {g:,.0f} | {p:,.0f} | {err:+.1f}% |")
    # 화소 정확도(소각 화소만)
    m = gt_sev > 0
    acc = float((pred_sev[m] == gt_sev[m]).mean()) if m.sum() else float("nan")
    # 평균 절대 면적오차
    mae = np.mean([abs(pa[nm] - gt_area.get(nm, 0)) / max(gt_area.get(nm, 0), 1e-9) for nm in ("심", "중", "경")]) * 100
    lines.append(f"\n- 등급 화소정확도(소각 내) **{acc:.3f}** · 평균 절대 면적오차 **{mae:.1f}%**")
    return "\n".join(lines), dict(acc=acc, mae=mae, area=pa)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--round", default="0"); a = ap.parse_args()
    print("데이터 취득(GT 격자 정렬)…")
    D = fetch_to_gt_grid()
    print(f"  pre {D['pre_id']} ({D['pre_date']}) · post {D['post_id']} ({D['post_date']})")
    gt_sev, gt_area = load_gt()
    print(f"  GT 등급면적(ha): " + " · ".join(f"{k} {v:,.0f}" for k, v in gt_area.items()))
    burn = gt_sev > 0                                  # 소각 마스크는 GT 로 고정(등급 문제만 분리)

    preN = nbr(D["pre_B8A"], D["pre_B12"]); postN = nbr(D["post_B8A"], D["post_B12"])
    dnbr = preN - postN
    np.savez(os.path.join(CACHE, "arrays.npz"), dnbr=dnbr, preN=preN, postN=postN,
             gt_sev=gt_sev, burn=burn)

    from core.analyze import SEV_CUTS
    out = [f"# 등급별 소각면적 신뢰성 — 측정 로그\n",
           f"GT=NIFoS 안동 피해등급 20m · S2 pre {D['pre_date']}/post {D['post_date']} · GT격자 정렬",
           f"GT 등급면적(ha): " + " · ".join(f"{k} {v:,.0f}" for k, v in gt_area.items()) + "\n"]
    # 베이스라인: 현재 배포 절단
    txt, m = report(f"R0 baseline · 배포 dNBR 절단 {SEV_CUTS}", assign(dnbr, burn, SEV_CUTS), gt_sev, gt_area)
    out.append(txt)
    print("\n" + txt)
    open(os.path.join(PROJ, "runs", "severity_calib_log.md"), "w").write("\n".join(out))
    print(f"\n로그 → runs/severity_calib_log.md")


if __name__ == "__main__":
    main()
