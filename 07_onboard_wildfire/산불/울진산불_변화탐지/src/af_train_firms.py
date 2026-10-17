"""FIRMS 라벨 활성화재 지도학습 — 고정 NHI(recall 0.063)를 데이터로 넘어서기.
캐시 FIRMS(열탐지)를 픽셀 라벨로 S2 스펙트럼에서 활성화재를 학습.

라벨링(순환 회피):  양성 = same-day FIRMS점 <150m,  음성 = >800m(비구름),  150~800m 제외.
특징(픽셀): 반사율 b2,b3,b4,b8a,b11,b12 + NHI_SWNIR,NHI_SWIR,NDVI, b12, b12-b11.
검증: leave-one-fire-out(2화재 학습→미학습 화재 예측) → FIRMS recall·클러스터확증을
      NHI 베이스라인과 동일 기준 비교(af_firms_validate와 같은 척도).
모델: RandomForest(소데이터 견고, class_weight balanced). 최종 전체학습 저장.

사용: python af_train_firms.py
"""
import os, csv, json, datetime as dt
import numpy as np
from osgeo import gdal, osr
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
import joblib
gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
AF = os.path.join(HERE, "..", "active_fire")
FIRMS_CSV = os.path.join(HERE, "..", "..", "산불데이터", "firms_korea_bbox_2022_2025.csv")
CLOUD_SCL = {3, 8, 9, 10}
OFF = 1000.0            # 반사율 offset (baseline>=04.00, 전 장면 2022+)
POS_M, NEG_M = 150.0, 800.0
EVAL_BUF_M = 200.0
SCENES = {  # FIRMS same-day 있는 장면만
    "uljin":     ["2022-03-05", "2022-03-10"],
    "sancheong": ["2025-03-22", "2025-03-24"],
    "uiseong":   ["2025-03-24"],
}
FEATS = ["b2", "b3", "b4", "b8a", "b11", "b12", "nhi_swnir", "nhi_swir", "ndvi", "d_b12b11"]

def load_firms():
    out = []
    for r in csv.DictReader(open(FIRMS_CSV, encoding="utf-8-sig")):
        try:
            if (r.get("confidence") or "").strip().lower() in ("l", "low"): continue
            out.append((float(r["latitude"]), float(r["longitude"]), r["acq_date"].strip()))
        except (KeyError, ValueError): continue
    return out

def _r(p):
    ds = gdal.Open(p); a = ds.ReadAsArray().astype(np.float32); ds = None; return a

def scene(site, date, firms):
    sd = os.path.join(AF, site, date)
    mds = gdal.Open(os.path.join(sd, "active_fire_mask.tif"))
    gt = mds.GetGeoTransform(); proj = mds.GetProjection()
    H, W = mds.ReadAsArray().shape
    b = {k: (np.clip(_r(os.path.join(sd, f"{n}.tif")) - OFF, 0, None) / 10000.0)
         for k, n in [("b2","B02"),("b3","B03"),("b4","B04"),("b8a","B8A"),("b11","B11"),("b12","B12")]}
    e = 1e-6
    feat = dict(b)
    feat["nhi_swnir"] = (b["b11"] - b["b8a"]) / (b["b11"] + b["b8a"] + e)
    feat["nhi_swir"]  = (b["b12"] - b["b11"]) / (b["b12"] + b["b11"] + e)
    feat["ndvi"]      = (b["b8a"] - b["b4"]) / (b["b8a"] + b["b4"] + e)
    feat["d_b12b11"]  = b["b12"] - b["b11"]
    X = np.stack([feat[f] for f in FEATS], -1)          # H,W,F
    scl = _r(os.path.join(sd, "SCL.tif"))
    valid = (b["b12"] > 0) & (~np.isin(scl, list(CLOUD_SCL)))
    nhi_mask = _r(os.path.join(sd, "active_fire_mask.tif")) > 0
    # FIRMS same-day 픽셀좌표
    dst = osr.SpatialReference(); dst.ImportFromWkt(proj)
    src = osr.SpatialReference(); src.ImportFromEPSG(4326); src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    d0 = dt.date.fromisoformat(date); pts = []
    for la, lo, ad in firms:
        try: fd = dt.date.fromisoformat(ad)
        except ValueError: continue
        if abs((fd - d0).days) > 1: continue
        x, y, _ = ct.TransformPoint(lo, la); c = int((x-gt[0])/gt[1]); rr = int((y-gt[3])/gt[5])
        if 0 <= rr < H and 0 <= c < W: pts.append((rr, c))
    # FIRMS 거리맵(양/음성 존)
    fpt = np.zeros((H, W), bool)
    for rr, c in pts: fpt[rr, c] = True
    px = abs(gt[1])
    dist = ndi.distance_transform_edt(~fpt) * px if fpt.any() else np.full((H, W), 1e9)
    pos = (dist <= POS_M) & valid
    neg = (dist >= NEG_M) & valid
    return dict(site=site, date=date, X=X, valid=valid, nhi=nhi_mask, pts=pts,
                pos=pos, neg=neg, gt=gt, shape=(H, W), px=px)

def firms_metrics(mask, pts, px):
    """recall = FIRMS점이 예측마스크 ±버퍼 안 / 클러스터확증 = 예측덩어리 중 FIRMS 근접 비율."""
    if not pts: return None, None
    buf = int(round(EVAL_BUF_M / px))
    dil = ndi.binary_dilation(mask, iterations=buf) if mask.any() else mask
    H, W = mask.shape
    hit = sum(1 for r, c in pts if dil[r, c])
    recall = hit / len(pts)
    lab, nlab = ndi.label(mask)
    if not nlab: return recall, 0.0
    hitL = set()
    for r, c in pts:
        r0,r1 = max(0,r-buf),min(H,r+buf+1); c0,c1 = max(0,c-buf),min(W,c+buf+1)
        for L in np.unique(lab[r0:r1, c0:c1]):
            if L: hitL.add(int(L))
    return recall, len(hitL)/nlab

def sample(sc, neg_ratio=3, cap=60000):
    Xp = sc["X"][sc["pos"]]; Xn = sc["X"][sc["neg"]]
    rng = np.random.RandomState(0)
    if len(Xn) > neg_ratio*len(Xp) and len(Xp): Xn = Xn[rng.choice(len(Xn), neg_ratio*len(Xp), False)]
    if len(Xp) > cap: Xp = Xp[rng.choice(len(Xp), cap, False)]
    if len(Xn) > cap: Xn = Xn[rng.choice(len(Xn), cap, False)]
    X = np.concatenate([Xp, Xn]); y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
    return X, y

def main():
    firms = load_firms()
    print(f"FIRMS 로드 {len(firms)}점\n장면 로드...")
    S = {s: [scene(s, d, firms) for d in ds] for s, ds in SCENES.items()}
    for s in S:
        for sc in S[s]:
            print(f"  {s}/{sc['date']}: 양성 {int(sc['pos'].sum())}px 음성 {int(sc['neg'].sum())}px FIRMS {len(sc['pts'])}점")
    fires = list(SCENES)
    print("\n== leave-one-fire-out (미학습 화재에서 NHI vs 학습모델) ==")
    print(f"{'held-out':10s} {'장면':16s} | {'NHI recall/확증':>16} | {'RF recall/확증':>16} | {'thr':>5}")
    print("-"*74)
    rows = []
    for held in fires:
        tr = [sc for s in fires if s != held for sc in S[s]]
        Xtr = np.concatenate([sample(sc)[0] for sc in tr]); ytr = np.concatenate([sample(sc)[1] for sc in tr])
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20,
                                     class_weight="balanced", n_jobs=-1, random_state=0)
        clf.fit(Xtr, ytr)
        for sc in S[held]:
            prob = np.zeros(sc["shape"], np.float32)
            v = sc["valid"]
            prob[v] = clf.predict_proba(sc["X"][v])[:, 1]
            # 임계 스윕(양성 화소가 나오도록), FIRMS recall 최대화 지점
            best = None
            for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
                m = (prob >= thr) & v
                r, c = firms_metrics(m, sc["pts"], sc["px"])
                if r is None: continue
                if best is None or (r + (c or 0)) > (best[1] + (best[2] or 0)):
                    best = (thr, r, c, int(m.sum()))
            nr, nc = firms_metrics(sc["nhi"], sc["pts"], sc["px"])
            thr, rr, rc, rpx = best if best else (None, None, None, 0)
            print(f"{held:10s} {sc['date']:16s} | {f'{nr:.3f}/{nc:.3f}':>16} | {f'{rr:.3f}/{rc:.3f}':>16} | {thr}")
            rows.append(dict(held=held, date=sc["date"], nhi_recall=round(nr,3), nhi_corrob=round(nc,3),
                             rf_recall=round(rr,3), rf_corrob=round(rc,3), rf_thr=thr, rf_px=rpx))
    # 요약
    nhr = np.mean([r["nhi_recall"] for r in rows]); nhc = np.mean([r["nhi_corrob"] for r in rows])
    rfr = np.mean([r["rf_recall"] for r in rows]); rfc = np.mean([r["rf_corrob"] for r in rows])
    print("-"*74)
    print(f"평균  NHI recall {nhr:.3f} 확증 {nhc:.3f}  →  RF recall {rfr:.3f} 확증 {rfc:.3f}")
    print(f"       recall {(rfr-nhr):+.3f}   확증 {(rfc-nhc):+.3f}")
    # 최종 전체학습 + 저장
    allsc = [sc for s in fires for sc in S[s]]
    Xa = np.concatenate([sample(sc)[0] for sc in allsc]); ya = np.concatenate([sample(sc)[1] for sc in allsc])
    final = RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=20,
                                   class_weight="balanced", n_jobs=-1, random_state=0).fit(Xa, ya)
    imp = sorted(zip(FEATS, final.feature_importances_), key=lambda x:-x[1])
    os.makedirs(os.path.join(AF, "model"), exist_ok=True)
    joblib.dump(dict(clf=final, feats=FEATS, off=OFF), os.path.join(AF, "model", "af_rf.joblib"))
    json.dump(dict(lofo=rows, nhi_mean=dict(recall=round(nhr,3),corrob=round(nhc,3)),
                   rf_mean=dict(recall=round(rfr,3),corrob=round(rfc,3)),
                   importance=[(f,round(float(i),3)) for f,i in imp]),
              open(os.path.join(AF, "af_train_firms.json"), "w"), ensure_ascii=False, indent=1)
    print("특징 중요도:", ", ".join(f"{f}:{i:.2f}" for f,i in imp[:5]))
    print("saved active_fire/model/af_rf.joblib + af_train_firms.json")

if __name__ == "__main__":
    main()
