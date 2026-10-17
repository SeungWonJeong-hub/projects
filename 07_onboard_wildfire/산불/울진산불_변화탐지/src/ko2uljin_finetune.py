"""한국 다지역 사전학습 → 울진 파인튜닝 실험.

목적: 울진 단일산불 결과(TEST F1 0.772 / IoU 0.629, UNetResNet50+attn)를
      한국 다지역(ko_multi) 사전학습으로 끌어올릴 수 있는지 검증.

정직한 설계 — 사전학습 효과를 아키텍처 교란과 분리하기 위해 같은 arch(smp.Unet
resnet34, 13ch)로 두 조건을 모두 학습:
  A) 대조군  : imagenet 초기화 → 울진 train 학습          (resnet34 native 기준선)
  B) 실험군  : runs/ko_native/ko_native_unet.pt 초기화 → 울진 파인튜닝
비교: B가 A를 넘으면 '한국 다지역 사전학습이 울진에 도움' = 결론.
      원 baseline(UNetResNet50 0.772/0.629)도 참고로 병기.

데이터: data/cd/{train,val,test} 256타일(컬럼-블록 분할, 원 baseline과 동일 분할).
        build_dataset.py의 다시점 구름마스킹(MT_CLOUD=1)이 적용된 라벨을 사용.
"""
from __future__ import annotations
import os, glob, json, copy
import numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp

HERE = os.path.dirname(os.path.abspath(__file__))
CD = os.path.join(HERE, "..", "data", "cd")
KO_CKPT = os.path.join(HERE, "..", "runs", "ko_native", "ko_native_unet.pt")
OUT = os.path.join(HERE, "..", "runs", "ko2uljin"); os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", "40")); BATCH = 8
LR_A = 3e-4                       # 대조군: 표준 LR
LR_B = float(os.environ.get("LR_B", "1e-4"))  # 파인튜닝: 낮은 LR


class DS(torch.utils.data.Dataset):
    def __init__(s, split, aug=False):
        s.f = sorted(glob.glob(os.path.join(CD, split, "*.npz"))); s.aug = aug
    def __len__(s): return len(s.f)
    def __getitem__(s, i):
        d = np.load(s.f[i]); x = d["image"].astype(np.float32); y = d["mask"].astype(np.float32)
        if s.aug:
            import random
            if random.random() < 0.5: x = x[:, :, ::-1].copy(); y = y[:, ::-1].copy()
            if random.random() < 0.5: x = x[:, ::-1].copy(); y = y[::-1].copy()
            k = random.randint(0, 3)
            if k: x = np.rot90(x, k, axes=(1, 2)).copy(); y = np.rot90(y, k).copy()
        return torch.from_numpy(x), torch.from_numpy(y)


class DiceBCE(nn.Module):
    def __init__(s): super().__init__(); s.bce = nn.BCEWithLogitsLoss()
    def forward(s, p, y):
        y = y.unsqueeze(1); pb = torch.sigmoid(p); inter = (pb * y).sum((2, 3))
        dice = 1 - (2 * inter + 1) / (pb.sum((2, 3)) + y.sum((2, 3)) + 1)
        return s.bce(p, y) + dice.mean()


def mets(P, Y, thr=None):
    def at(t):
        pr = P >= t; tp = (pr & (Y > 0)).sum(); fp = (pr & (Y == 0)).sum(); fn = ((~pr) & (Y > 0)).sum()
        pe = tp / (tp + fp + 1e-9); re = tp / (tp + fn + 1e-9)
        return 2 * pe * re / (pe + re + 1e-9), tp / (tp + fp + fn + 1e-9), pe, re
    if thr is None: thr = float(max(np.linspace(0.1, 0.9, 33), key=lambda t: at(t)[0]))
    f1, iou, pe, re = at(thr)
    return dict(thr=round(float(thr), 3), f1=round(float(f1), 4), iou=round(float(iou), 4),
                precision=round(float(pe), 3), recall=round(float(re), 3))


def infer(m, dl):
    m.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            p = torch.sigmoid(m(x.to(DEV))).float()[:, 0].cpu().numpy()
            P.append(p.reshape(-1)); Y.append(y.numpy().reshape(-1))
    return np.concatenate(P), np.concatenate(Y)


def train(tag, init_ckpt, lr):
    print(f"\n===== [{tag}] init={'ko_native' if init_ckpt else 'imagenet'} lr={lr} =====", flush=True)
    tl = torch.utils.data.DataLoader(DS("train", aug=True), batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
    vl = torch.utils.data.DataLoader(DS("val"), batch_size=BATCH, num_workers=0)
    te = torch.utils.data.DataLoader(DS("test"), batch_size=BATCH, num_workers=0)
    enc_w = None if init_ckpt else "imagenet"
    m = smp.Unet("resnet34", encoder_weights=enc_w, in_channels=13, classes=1).to(DEV)
    if init_ckpt:
        m.load_state_dict(torch.load(init_ckpt, map_location=DEV, weights_only=True))
        print(f"  로드: {os.path.basename(init_ckpt)}")
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lf = DiceBCE(); sc = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda")); best = -1; bs = None
    for ep in range(EPOCHS):
        m.train(); run = 0
        for x, y in tl:
            x = x.to(DEV); y = y.to(DEV); opt.zero_grad()
            with torch.autocast(device_type=DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
                loss = lf(m(x), y)
            sc.scale(loss).backward(); sc.step(opt); sc.update(); run += loss.item()
        sch.step()
        Pv, Yv = infer(m, vl); vf = mets(Pv, Yv)["f1"]
        if vf > best: best = vf; bs = copy.deepcopy({k: v.cpu() for k, v in m.state_dict().items()})
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  [ep {ep:02d}] loss={run/max(len(tl),1):.3f} val_f1={vf:.3f}", flush=True)
    m.load_state_dict(bs)
    vm = mets(*infer(m, vl))
    Pt, Yt = infer(m, te)
    tm = mets(Pt, Yt, thr=vm["thr"]); tmb = mets(Pt, Yt)
    print(f"  VAL: {vm}\n  TEST(val-thr): {tm}\n  TEST(best-thr): {tmb}", flush=True)
    torch.save(bs, os.path.join(OUT, f"{tag}.pt"))
    return dict(tag=tag, init=("ko_native" if init_ckpt else "imagenet"), lr=lr,
                val=vm, test_valthr=tm, test_bestthr=tmb)


def main():
    res = {}
    res["A_resnet34_scratch"] = train("A_resnet34_scratch", None, LR_A)
    if os.path.exists(KO_CKPT):
        res["B_ko2uljin_finetune"] = train("B_ko2uljin_finetune", KO_CKPT, LR_B)
    else:
        print(f"[경고] {KO_CKPT} 없음 — ko_native 먼저 완료 필요. 실험군 건너뜀.")
    json.dump(res, open(os.path.join(OUT, "ko2uljin_results.json"), "w"), indent=2, ensure_ascii=False)
    print("\n========== 요약 (TEST val-thr) ==========")
    print(f"  원 baseline UNetResNet50+attn : F1 0.772 / IoU 0.629")
    for k, r in res.items():
        t = r["test_valthr"]; print(f"  {k:26s}: F1 {t['f1']} / IoU {t['iou']}")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
