"""TTA(test-time augmentation) 빠른 확인 — v3 홀드아웃에서 flip·rot 8종 평균이 F1 올리나.
정직 기준: v3 val-thr(고정)에서 baseline vs TTA 비교. TTA-best-thr는 참고(낙관).
"""
import os, glob, json
import numpy as np, torch
import segmentation_models_pytorch as smp

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "runs", "ko_native")
DATA = os.path.join(HERE, "..", "ko_multi")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

res = json.load(open(os.path.join(RUN, "ko_native_results.json")))
HOLD = res["holdout"]; VTHR = res["test_valthr"]["thr"]
m = smp.Unet("resnet34", encoder_weights=None, in_channels=13, classes=1).to(DEV)
m.load_state_dict(torch.load(os.path.join(RUN, "ko_native_unet.pt"), map_location=DEV, weights_only=True))
m.eval()

# 8종 변환(정방향)과 역변환 쌍: identity, hflip, vflip, hvflip, rot90/180/270, transpose
def fwd(x, t):
    if t == 0: return x
    if t == 1: return x[:, :, ::-1]
    if t == 2: return x[:, ::-1, :]
    if t == 3: return x[:, ::-1, ::-1]
    if t == 4: return np.rot90(x, 1, axes=(1, 2))
    if t == 5: return np.rot90(x, 2, axes=(1, 2))
    if t == 6: return np.rot90(x, 3, axes=(1, 2))
    if t == 7: return np.transpose(x, (0, 2, 1))
def inv(p, t):  # p: (H,W)
    if t == 0: return p
    if t == 1: return p[:, ::-1]
    if t == 2: return p[::-1, :]
    if t == 3: return p[::-1, ::-1]
    if t == 4: return np.rot90(p, -1)
    if t == 5: return np.rot90(p, -2)
    if t == 6: return np.rot90(p, -3)
    if t == 7: return p.T

@torch.no_grad()
def predict(x):
    return torch.sigmoid(m(torch.from_numpy(x[None].copy()).to(DEV)))[0, 0].cpu().numpy()

def f1_at(P, Y, thr):
    pr = P >= thr; tp = (pr & (Y > 0)).sum(); fp = (pr & (Y == 0)).sum(); fn = ((~pr) & (Y > 0)).sum()
    pe = tp / (tp + fp + 1e-9); re = tp / (tp + fn + 1e-9)
    return 2 * pe * re / (pe + re + 1e-9), pe, re

Pb, Pt, Y = [], [], []
for fire in HOLD:
    for f in sorted(glob.glob(os.path.join(DATA, fire, "*.npz"))):
        d = np.load(f); x = d["x"].astype(np.float32); y = d["y"].astype(np.uint8)
        Pb.append(predict(x).reshape(-1))                     # baseline
        acc = np.zeros(x.shape[1:], np.float32)
        for t in range(8):
            acc += inv(predict(fwd(x, t)), t)
        Pt.append((acc / 8).reshape(-1)); Y.append(y.reshape(-1))
Pb = np.concatenate(Pb); Pt = np.concatenate(Pt); Y = np.concatenate(Y)

fb, pb, rb = f1_at(Pb, Y, VTHR)
ft, pt, rt = f1_at(Pt, Y, VTHR)
grid = np.linspace(0.1, 0.9, 33)
tb = max(grid, key=lambda t: f1_at(Pt, Y, t)[0]); ftb, ptb, rtb = f1_at(Pt, Y, tb)
print(f"v3 홀드아웃 7화재, val-thr={VTHR}")
print(f"  baseline   (val-thr): F1 {fb:.4f} P {pb:.3f} R {rb:.3f}")
print(f"  TTA x8     (val-thr): F1 {ft:.4f} P {pt:.3f} R {rt:.3f}  | Δ {ft-fb:+.4f}")
print(f"  TTA x8    (best-thr {tb:.3f}, 참고): F1 {ftb:.4f} P {ptb:.3f} R {rtb:.3f}")
