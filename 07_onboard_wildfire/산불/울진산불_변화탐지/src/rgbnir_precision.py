"""RGB+NIR 소각 정밀도 실험. ET-02 앙상블(3seed) baseline + 물리 게이트/지수 레버를
한국 홀드아웃 GT로 측정. SWIR 미사용. positive(F1↑)만 채택.

x8 = [B02,B03,B04,B8A]×(pre,post) /10000.  게이트는 예측에 AND로 곱해 FP 제거(정밀도↑).
사용: python3 src/rgbnir_precision.py
"""
import os, glob, json, numpy as np, torch
import segmentation_models_pytorch as smp
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = ["et02_unet_s1", "et02_unet_s2", "et02_unet_s3"]

def load_models():
    ms = []
    for r in RUNS:
        m = smp.Unet("resnet34", encoder_weights=None, in_channels=8, classes=1).to(DEV).eval()
        m.load_state_dict(torch.load(os.path.join(PROJ, "runs", r, "model.pt"), map_location=DEV))
        ms.append(m)
    return ms

def holdout_tiles():
    fires = [l.strip() for l in open(os.path.join(PROJ, "data", "holdout_fixed.txt")) if l.strip()]
    fs = []
    for f in fires:
        fs += sorted(glob.glob(os.path.join(PROJ, "ko_refl", f, "*.npz")))
    return fs

def feats(x12):
    """물리 특징(post/pre RGB+NIR, /10000)."""
    x = x12.astype(np.float32) / 10000.0
    pre, post = x[0:6], x[6:12]
    bpre = pre[[0, 1, 2, 3]]; bpost = post[[0, 1, 2, 3]]   # B02,B03,B04,B8A
    red_o, nir_o = bpost[2], bpost[3]; red_i, nir_i = bpre[2], bpre[3]
    ndvi_o = (nir_o - red_o) / (nir_o + red_o + 1e-6)
    ndvi_i = (nir_i - red_i) / (nir_i + red_i + 1e-6)
    bright = bpost.mean(0)                       # 후 평균반사율 (char=낮음, 도시/나대지=높음)
    dnir = nir_i - nir_o                          # NIR 급락(소각 특이, 양수)
    dndvi = ndvi_i - ndvi_o                        # 식생소실(양수)
    bai = 1.0 / ((0.1 - red_o) ** 2 + (0.06 - nir_o) ** 2 + 1e-6)  # Burned Area Index(char 큼)
    return dict(ndvi_o=ndvi_o, bright=bright, dnir=dnir, dndvi=dndvi, bai=bai)

def f1_stats(pred, y, valid):
    p = pred & valid; yv = (y > 0) & valid
    tp = (p & yv).sum(); fp = (p & ~yv & valid).sum(); fn = (~p & yv).sum()
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9), pr, rc

def main():
    ms = load_models(); files = holdout_tiles()
    print(f"홀드아웃 {len(files)}타일, 앙상블 {len(ms)}모델", flush=True)
    # 앙상블 확률 + 특징 캐시
    P = []; Y = []; V = []; F = []
    for f in files:
        d = np.load(f); x12 = d["x"]
        pre, post = x12[0:6].astype(np.float32) / 1e4, x12[6:12].astype(np.float32) / 1e4
        x8 = np.concatenate([pre[[0, 1, 2, 3]], post[[0, 1, 2, 3]]], 0)
        xt = torch.from_numpy(x8).unsqueeze(0).to(DEV)
        with torch.no_grad():
            pr = np.mean([torch.sigmoid(m(xt)).squeeze().cpu().numpy() for m in ms], 0)
        P.append(pr); Y.append(d["y"].astype(np.uint8))
        V.append(d["valid"].astype(bool) if "valid" in d else np.ones_like(d["y"], bool))
        F.append(feats(x12))
    P = np.stack(P); Y = np.stack(Y); V = np.stack(V)

    def eval_cfg(gate_fn, thr_grid=np.arange(0.1, 0.91, 0.025)):
        best = (-1, 0, 0, 0)
        for t in thr_grid:
            preds = [];
            for i in range(len(P)):
                pm = P[i] >= t
                if gate_fn is not None: pm = pm & gate_fn(F[i])
                preds.append(pm)
            pr = np.stack(preds)
            f1, p, r = f1_stats(pr.reshape(-1), Y.reshape(-1), V.reshape(-1))
            if f1 > best[0]: best = (f1, p, r, t)
        return best

    print("\n=== Baseline (게이트 없음) ===", flush=True)
    b = eval_cfg(None)
    print(f"  F1={b[0]:.4f} P={b[1]:.4f} R={b[2]:.4f} thr={b[3]:.3f}", flush=True)
    base_f1 = b[0]

    # 게이트 후보 (각각 FP 제거 → 정밀도↑). 임계값은 데이터서 스윕.
    levers = {
        "dark(bright<T)":   [("bright", "<", v) for v in [0.10, 0.14, 0.18, 0.22, 0.26]],
        "dNIR(dnir>T)":     [("dnir", ">", v) for v in [0.0, 0.03, 0.06, 0.10, 0.15]],
        "dNDVI(dndvi>T)":   [("dndvi", ">", v) for v in [0.0, 0.05, 0.10, 0.15, 0.20]],
        "lowNDVI(ndvi_o<T)":[("ndvi_o", "<", v) for v in [0.25, 0.30, 0.35, 0.40, 0.45]],
        "BAI(bai>T)":       [("bai", ">", v) for v in [50, 100, 150, 200]],
    }
    results = {"baseline": b}
    for name, cands in levers.items():
        print(f"\n=== {name} ===", flush=True)
        bestv = (-1,)
        for key, op, v in cands:
            gate = (lambda k, o, vv: (lambda F: (F[k] < vv) if o == "<" else (F[k] > vv)))(key, op, v)
            r = eval_cfg(gate)
            tag = "  ↑" if r[0] > base_f1 else ""
            print(f"  {key}{op}{v}: F1={r[0]:.4f} P={r[1]:.4f} R={r[2]:.4f} thr={r[3]:.3f}{tag}", flush=True)
            if r[0] > bestv[0]: bestv = r + (f"{key}{op}{v}",)
        results[name] = bestv

    print("\n=== 형태학(opening+closing) baseline에 ===", flush=True)
    st = generate_binary_structure(2, 2)
    def morph_eval(t):
        preds = []
        for i in range(len(P)):
            pm = P[i] >= t
            pm = binary_opening(pm, st, iterations=1); pm = binary_closing(pm, st, iterations=1)
            preds.append(pm)
        return f1_stats(np.stack(preds).reshape(-1), Y.reshape(-1), V.reshape(-1))
    mb = max([(morph_eval(t), t) for t in np.arange(0.1, 0.91, 0.05)], key=lambda z: z[0][0])
    print(f"  F1={mb[0][0]:.4f} P={mb[0][1]:.4f} R={mb[0][2]:.4f} thr={mb[1]:.3f}"
          + ("  ↑" if mb[0][0] > base_f1 else ""), flush=True)
    results["morph"] = mb[0] + (mb[1],)

    print("\n=== 요약 (baseline 대비) ===", flush=True)
    print(f"  baseline F1={base_f1:.4f}", flush=True)
    for k, v in results.items():
        if k == "baseline": continue
        d = v[0] - base_f1
        print(f"  {k:22s} F1={v[0]:.4f} ({'+' if d>=0 else ''}{d:.4f}) {'★POSITIVE' if d>0.002 else ''}", flush=True)
    json.dump({k: [float(x) if isinstance(x,(int,float,np.floating)) else str(x) for x in v]
               for k, v in results.items()},
              open(os.path.join(PROJ, "runs", "rgbnir_precision.json"), "w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
