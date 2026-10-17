"""v3 오차 구조 분석 — FP(거짓경보)가 흩뿌려진 소면적 얼룩인지 확인.
소면적이면 min-area 후처리가 정밀도↑→F1↑에 유효. 큰 덩어리면 무효.
v3 모델(runs/ko_native/ko_native_unet.pt, resnet34 13ch)을 v3 홀드아웃 7화재에 적용.
"""
import os, glob, json
import numpy as np, torch
import segmentation_models_pytorch as smp
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "runs", "ko_native")
DATA = os.path.join(HERE, "..", "ko_multi")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

res = json.load(open(os.path.join(RUN, "ko_native_results.json")))
HOLD = res["holdout"]; THR = res["test_valthr"]["thr"]
print(f"v3 홀드아웃 {len(HOLD)}화재, thr={THR}", flush=True)

m = smp.Unet("resnet34", encoder_weights=None, in_channels=13, classes=1).to(DEV)
m.load_state_dict(torch.load(os.path.join(RUN, "ko_native_unet.pt"), map_location=DEV, weights_only=True))
m.eval()

fp_sizes, fn_sizes = [], []
tp_tot = fp_tot = fn_tot = 0
tiles_px = 0
for fire in HOLD:
    for f in sorted(glob.glob(os.path.join(DATA, fire, "*.npz"))):
        d = np.load(f); x = d["x"].astype(np.float32); y = d["y"].astype(np.uint8)
        with torch.no_grad():
            p = torch.sigmoid(m(torch.from_numpy(x[None]).to(DEV)))[0, 0].cpu().numpy()
        pred = (p >= THR).astype(np.uint8)
        tp = pred & (y > 0); fp = pred & (y == 0); fn = (~pred.astype(bool)) & (y > 0)
        tp_tot += int(tp.sum()); fp_tot += int(fp.sum()); fn_tot += int(fn.sum()); tiles_px += y.size
        # FP/FN 연결요소 크기
        for mask, store in [(fp, fp_sizes), (fn.astype(np.uint8), fn_sizes)]:
            lab, n = ndimage.label(mask)
            if n: store.extend(list(np.bincount(lab.ravel())[1:]))

fp_sizes = np.array(fp_sizes); fn_sizes = np.array(fn_sizes)
P = tp_tot / (tp_tot + fp_tot + 1e-9); R = tp_tot / (tp_tot + fn_tot + 1e-9)
print(f"\n픽셀 집계: TP {tp_tot} FP {fp_tot} FN {fn_tot} | P {P:.3f} R {R:.3f} F1 {2*P*R/(P+R):.3f}")


def summary(name, s):
    if not len(s):
        print(f"{name}: 없음"); return
    s = np.sort(s)[::-1]
    tot = s.sum()
    for cut in [10, 30, 100]:
        small = s[s <= cut]
        print(f"  {name} ≤{cut}px: 덩어리 {len(small)}/{len(s)}개, 이들이 전체 {name}픽셀의 {100*small.sum()/tot:.1f}%")
    print(f"  {name} 총 {len(s)}덩어리, 최대 {s[0]}px, 중앙값 {int(np.median(s))}px, 상위3 {s[:3].tolist()}")


print("\n=== FP(거짓경보) 크기 구조 ===")
summary("FP", fp_sizes)
print("\n=== FN(놓침) 크기 구조 ===")
summary("FN", fn_sizes)

# min-area 제거 시뮬레이션: FP 소덩어리 제거하면 정밀도/F1 어떻게 변하나 (FN엔 영향 없음 가정 X — 실제 재계산)
print("\n=== min-area 후처리 시뮬레이션 (FP만 제거, val 기준 아님·구조확인용) ===")
for A in [5, 10, 20, 50]:
    removed = int(fp_sizes[fp_sizes <= A].sum()) if len(fp_sizes) else 0
    fp2 = fp_tot - removed
    P2 = tp_tot / (tp_tot + fp2 + 1e-9); F2 = 2 * P2 * R / (P2 + R)
    print(f"  <{A}px FP 제거: FP {fp_tot}→{fp2} | P {P:.3f}→{P2:.3f} | F1 {2*P*R/(P+R):.3f}→{F2:.3f}")
