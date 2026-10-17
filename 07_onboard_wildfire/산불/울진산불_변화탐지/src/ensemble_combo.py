"""다양성 앙상블 최적 서브셋 탐색. 가용 et02 모델(원본3 + 다양성)을 각 아키텍처로 로드,
holdout 확률 계산 후 greedy forward selection으로 F1 최대 서브셋을 찾음. baseline 0.688 넘기기.
사용: python3 src/ensemble_combo.py
"""
import os, glob, json, numpy as np, torch
import segmentation_models_pytorch as smp
PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEC = {"unet": smp.Unet, "fpn": smp.FPN, "deeplabv3plus": smp.DeepLabV3Plus,
       "manet": smp.MAnet, "unetpp": smp.UnetPlusPlus}

def load(run):
    j = json.load(open(os.path.join(PROJ, "runs", run, "results.json")))
    dec = j.get("decoder", "unet"); enc = j.get("encoder", "resnet34")
    m = DEC[dec](enc, encoder_weights=None, in_channels=8, classes=1).to(DEV).eval()
    m.load_state_dict(torch.load(os.path.join(PROJ, "runs", run, "model.pt"), map_location=DEV))
    return m

def main():
    runs = [os.path.basename(os.path.dirname(p)) for p in
            sorted(glob.glob(os.path.join(PROJ, "runs", "et02_*", "model.pt")))]
    print("가용 모델:", runs, flush=True)
    fires = [l.strip() for l in open(os.path.join(PROJ, "data", "holdout_fixed.txt")) if l.strip()]
    files = [f for fr in fires for f in sorted(glob.glob(os.path.join(PROJ, "ko_refl", fr, "*.npz")))]
    # 각 모델 확률 (N_models, N_tiles) 텐서 목록
    Y = []; V = []
    per = {r: [] for r in runs}
    mods = {r: load(r) for r in runs}
    for f in files:
        d = np.load(f); x = d["x"].astype(np.float32) / 1e4
        x8 = np.concatenate([x[0:6][[0, 1, 2, 3]], x[6:12][[0, 1, 2, 3]]], 0)
        xt = torch.from_numpy(x8).unsqueeze(0).to(DEV)
        for r in runs:
            with torch.no_grad(): per[r].append(torch.sigmoid(mods[r](xt)).squeeze().cpu().numpy())
        Y.append(d["y"].astype(np.uint8))
        V.append(d["valid"].astype(bool) if "valid" in d else np.ones_like(d["y"], bool))
    per = {r: np.stack(per[r]) for r in runs}; Y = np.stack(Y); V = np.stack(V)
    yv = (Y > 0) & V

    def f1_of(prob):
        b = (-1, 0, 0, 0)
        for t in np.arange(0.1, 0.91, 0.025):
            p = (prob >= t) & V; tp = (p & yv).sum(); fp = (p & ~yv & V).sum(); fn = (~p & yv).sum()
            pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9); f1 = 2 * pr * rc / (pr + rc + 1e-9)
            if f1 > b[0]: b = (f1, pr, rc, t)
        return b

    print("\n=== 개별 모델 F1 ===", flush=True)
    for r in runs:
        b = f1_of(per[r]); print(f"  {r:20s} F1={b[0]:.4f} P={b[1]:.4f} R={b[2]:.4f}", flush=True)

    base = f1_of(np.mean([per[r] for r in ["et02_unet_s1", "et02_unet_s2", "et02_unet_s3"] if r in per], 0))
    print(f"\n원본3 앙상블 baseline: F1={base[0]:.4f} thr={base[3]:.3f}", flush=True)

    # greedy forward selection
    chosen = []; cur = None; best_f1 = -1
    while True:
        cand = None
        for r in runs:
            if r in chosen: continue
            avg = np.mean([per[c] for c in chosen + [r]], 0)
            f = f1_of(avg)[0]
            if f > best_f1: best_f1 = f; cand = r
        if cand is None: break
        chosen.append(cand)
        print(f"  + {cand:20s} → 앙상블 F1={best_f1:.4f} ({chosen})", flush=True)
    final = f1_of(np.mean([per[c] for c in chosen], 0))
    print(f"\n=== 최적 서브셋 {chosen} ===", flush=True)
    print(f"  F1={final[0]:.4f} P={final[1]:.4f} R={final[2]:.4f} thr={final[3]:.3f}", flush=True)
    print(f"  baseline(원본3)={base[0]:.4f} → {'★POSITIVE +%.4f' % (final[0]-base[0]) if final[0]>base[0]+0.002 else 'no gain'}", flush=True)
    json.dump({"chosen": chosen, "f1": float(final[0]), "p": float(final[1]), "r": float(final[2]),
               "thr": float(final[3]), "baseline3": float(base[0])},
              open(os.path.join(PROJ, "runs", "ensemble_combo.json"), "w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
