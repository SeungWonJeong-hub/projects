"""Prithvi-EO 파운데이션 백본으로 한국 산불 일반화 뚫기.
frozen Prithvi-100M 인코더(HLS 6밴드, num_frames=3) + 경량 conv 디코더 파인튜닝.
입력 = [pre, post, post] 3프레임(bi-temporal 변화 맥락). 소데이터(45화재/54타일)에
파운데이션 few-shot 전이가 맞는 use-case. 고정홀드아웃 8화재로 v3(F1 0.616) 공정비교.

사용: python prithvi_ft.py [--smoke] [--unfreeze N] [--epochs E]
"""
import os, sys, glob, json, random, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PRI = os.path.join(HERE, "..", "prithvi"); sys.path.insert(0, PRI)
from prithvi_mae import PrithviViT

DATA = os.path.join(HERE, "..", os.environ.get("KO_DATA", "ko_multi"))
CKPT = os.path.join(PRI, "Prithvi_EO_V1_100M.pt")
HOLD = os.path.join(HERE, "..", "data", "holdout_fixed.txt")
RUN = os.path.join(HERE, "..", "runs", os.environ.get("RUN_NAME", "ko_prithvi")); os.makedirs(RUN, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# Prithvi HLS 정규화 (reflectance*10000 스케일). 우리 밴드 [B2,B3,B4,B8,B11,B12]≈HLS[B02,B03,B04,B05,B06,B07]
MEAN = torch.tensor([775.23, 1080.99, 1228.59, 2497.20, 2204.21, 1610.83]).view(1, 6, 1, 1, 1)
STD  = torch.tensor([1281.53, 1270.03, 1399.48, 1368.34, 1291.68, 1154.51]).view(1, 6, 1, 1, 1)
IMG = 224

def prep(x6_pre, x6_post):
    """0-1 반사율 → Prithvi 입력 [6,3,224,224]. 프레임=[pre,post,post]."""
    fr = np.stack([x6_pre, x6_post, x6_post], 1)          # [6,3,H,W]
    t = torch.from_numpy(fr).float().unsqueeze(0)          # [1,6,3,H,W]
    t = F.interpolate(t.reshape(1, 6*3, *fr.shape[2:]), size=(IMG, IMG), mode="bilinear", align_corners=False)
    t = t.reshape(1, 6, 3, IMG, IMG)
    t = (t * 10000.0 - MEAN) / STD
    return t[0]

class DS(torch.utils.data.Dataset):
    def __init__(s, files, aug=False): s.f = files; s.aug = aug
    def __len__(s): return len(s.f)
    def __getitem__(s, i):
        d = np.load(s.f[i]); x = d["x"].astype(np.float32) / 10000.0; y = d["y"].astype(np.float32)  # 반사율×10000 → 0-1 (prep가 ×10000 복원)
        pre, post = x[0:6], x[6:12]
        if s.aug and random.random() < 0.5: pre, post, y = pre[:, :, ::-1].copy(), post[:, :, ::-1].copy(), y[:, ::-1].copy()
        if s.aug and random.random() < 0.5: pre, post, y = pre[:, ::-1].copy(), post[:, ::-1].copy(), y[::-1].copy()
        return prep(pre, post), torch.from_numpy(y).float()

class PrithviSeg(nn.Module):
    def __init__(s, unfreeze=0):
        super().__init__()
        s.enc = PrithviViT(img_size=IMG, patch_size=(1, 16, 16), num_frames=3, in_chans=6,
                           embed_dim=768, depth=12, num_heads=12, encoder_only=True)
        sd = torch.load(CKPT, map_location="cpu")
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        miss, unexp = s.enc.load_state_dict(enc, strict=False)
        print(f"[Prithvi] loaded enc, missing {len(miss)} unexpected {len(unexp)}", flush=True)
        s.unfreeze = unfreeze
        for p in s.enc.parameters(): p.requires_grad = False
        if unfreeze > 0:
            for blk in s.enc.blocks[-unfreeze:]:
                for p in blk.parameters(): p.requires_grad = True
        C = 768 * 3
        s.dec = nn.Sequential(
            nn.Conv2d(C, 512, 3, padding=1), nn.BatchNorm2d(512), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),   # 28
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),   # 56
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),   # 112
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),   # 224
            nn.Conv2d(64, 1, 1))

    def forward(s, x):   # x [B,6,3,224,224]
        ctx = torch.no_grad() if s.unfreeze == 0 else torch.enable_grad()
        with ctx:
            feats = s.enc.forward_features(x)
        f = s.enc.prepare_features_for_image_model([feats[-1]])[0]   # [B, 3*768, 14,14]
        out = s.dec(f)                                                # [B,1,224,224]
        return F.interpolate(out, size=(256, 256), mode="bilinear", align_corners=False)

def tversky_bce(logit, y, a=0.7, b=0.3):
    p = torch.sigmoid(logit).squeeze(1); e = 1e-6
    tp = (p*y).sum((1,2)); fn = (y*(1-p)).sum((1,2)); fp = ((1-y)*p).sum((1,2))
    tv = 1 - ((tp+e)/(tp + a*fn + b*fp + e)).mean()
    bce = F.binary_cross_entropy_with_logits(logit.squeeze(1), y)
    return tv + bce

@torch.no_grad()
def predict(model, files):
    model.eval(); P, Y = [], []
    for f in files:
        d = np.load(f); x = d["x"].astype(np.float32) / 10000.0  # 반사율×10000 → 0-1 (prep가 ×10000 복원)
        t = prep(x[0:6], x[6:12]).unsqueeze(0).to(DEV)
        p = torch.sigmoid(model(t)).squeeze().cpu().numpy()
        P.append(p.reshape(-1)); Y.append(d["y"].astype(np.uint8).reshape(-1))
    return np.concatenate(P), np.concatenate(Y)

def stats(P, Y, thr):
    pred = P >= thr
    tp = (pred & (Y==1)).sum(); fp = (pred & (Y==0)).sum(); fn = (~pred & (Y==1)).sum()
    f1 = 2*tp/(2*tp+fp+fn+1e-9); iou = tp/(tp+fp+fn+1e-9)
    pr = tp/(tp+fp+1e-9); re = tp/(tp+fn+1e-9)
    return dict(thr=round(float(thr),3), f1=round(float(f1),3), iou=round(float(iou),3),
               precision=round(float(pr),3), recall=round(float(re),3))

def best_thr(P, Y):
    return max([stats(P,Y,t) for t in np.arange(0.1,0.9,0.05)], key=lambda d:d["f1"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--unfreeze", type=int, default=0); ap.add_argument("--epochs", type=int, default=40)
    A = ap.parse_args()
    random.seed(42); torch.manual_seed(42)
    fires = sorted([d for d in glob.glob(os.path.join(DATA, "*")) if glob.glob(os.path.join(d, "*.npz"))])
    hold = set(l.strip() for l in open(HOLD) if l.strip())
    te_fires = [d for d in fires if os.path.basename(d) in hold]
    tr_all = [d for d in fires if os.path.basename(d) not in hold]
    random.shuffle(tr_all); nval = max(3, len(tr_all)//6)
    va_fires, trn_fires = tr_all[:nval], tr_all[nval:]
    tf = lambda dirs: [f for d in dirs for f in glob.glob(os.path.join(d, "*.npz"))]
    trn, va, te = tf(trn_fires), tf(va_fires), tf(te_fires)
    print(f"화재 {len(fires)} → train {len(trn_fires)}화재/{len(trn)}타일, val {len(va_fires)}/{len(va)}, TEST(홀드아웃 {len(te_fires)}화재) {len(te)}타일", flush=True)

    model = PrithviSeg(unfreeze=A.unfreeze).to(DEV)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"학습 파라미터 {ntr/1e6:.2f}M (unfreeze={A.unfreeze})", flush=True)
    if A.smoke:
        xb, yb = next(iter(torch.utils.data.DataLoader(DS(trn), batch_size=2)))
        out = model(xb.to(DEV)); loss = tversky_bce(out, yb.to(DEV))
        print(f"[smoke] out {tuple(out.shape)} loss {loss.item():.4f} — OK", flush=True); return

    lr = 1e-3 if A.unfreeze == 0 else 3e-4   # 부분 파인튜닝은 낮은 lr
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    bs = 8 if A.unfreeze == 0 else 4          # 언프리즈 시 메모리 절약
    dl = torch.utils.data.DataLoader(DS(trn, aug=True), batch_size=bs, shuffle=True, num_workers=2)
    scaler = torch.cuda.amp.GradScaler()
    best = {"f1": -1}
    for ep in range(A.epochs):
        model.train(); tl = 0
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = tversky_bce(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tl += loss.item()
        Pv, Yv = predict(model, va); vb = best_thr(Pv, Yv)
        if vb["f1"] > best["f1"]:
            best = dict(vb); torch.save(model.state_dict(), os.path.join(RUN, "best.pt"))
        print(f"ep{ep:02d} loss {tl/len(dl):.4f} val_bestF1 {vb['f1']} thr {vb['thr']}", flush=True)
    # 평가 (best.pt, 고정 홀드아웃)
    model.load_state_dict(torch.load(os.path.join(RUN, "best.pt")))
    Pv, Yv = predict(model, va); vthr = best_thr(Pv, Yv)["thr"]
    Pt, Yt = predict(model, te)
    tm_val = stats(Pt, Yt, vthr); tm_best = best_thr(Pt, Yt)
    # dNBR echo 베이스라인 (홀드아웃) — ko_refl 12ch [B02,B03,B04,B8A,B11,B12]×(pre,post)서 직접 계산
    def _dnbr(x):
        x = x.astype(np.float32)                              # NBR=(B8A-B12)/(B8A+B12), 비율이라 ×10000 상쇄
        nbr_pre  = (x[3]-x[5])  / (x[3]+x[5]  + 1e-6)         # pre  NIR=idx3, SWIR2=idx5
        nbr_post = (x[9]-x[11]) / (x[9]+x[11] + 1e-6)         # post NIR=idx9, SWIR2=idx11
        return (nbr_pre - nbr_post).reshape(-1)               # 소실(burn)→ 양수 dNBR
    dn = np.concatenate([ _dnbr(np.load(f)["x"]) for f in te])
    echo = max([stats(dn, Yt, t) for t in np.arange(0.05,0.6,0.05)], key=lambda d:d["f1"])
    res = dict(model="Prithvi-100M frozen+dec", unfreeze=A.unfreeze, fires=len(fires), test_fires=len(te_fires),
               val_thr=vthr, test_valthr=tm_val, test_bestthr=tm_best, dnbr_echo=echo,
               vs_v3={"v3_f1":0.616, "prithvi_f1":tm_best["f1"], "delta":round(tm_best["f1"]-0.616,3)})
    json.dump(res, open(os.path.join(RUN, "prithvi_results.json"), "w"), ensure_ascii=False, indent=1)
    print("\n== 홀드아웃(미학습 화재) 결과 ==")
    print("TEST val-thr :", tm_val)
    print("TEST best-thr:", tm_best)
    print("dNBR echo    :", echo)
    print(f"vs v3(0.616): {tm_best['f1']} ({tm_best['f1']-0.616:+.3f})")

if __name__ == "__main__":
    main()
