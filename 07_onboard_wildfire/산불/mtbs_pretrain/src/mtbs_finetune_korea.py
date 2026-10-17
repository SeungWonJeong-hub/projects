"""MTBS 사전학습 모델 → 한국 소량 파인튜닝(도메인적응 ②).

장면 정규화(①)만으로 부족하면, 한국 GT 소량으로 미세조정한다. GT 가 안동·울진 2건뿐이라
**LORO(하나 학습 → 다른 하나 평가)** 로 누출 없이 파인튜닝 전/후 전이개선을 잰다.

Korea GT 3등급(경1중2심3) → 모델 출력공간 매핑 {경→1(저),중→2,고/심→3}. 배경=ignore.
입력은 장면 정규화(train.normalize_scene) — 사전학습과 동일.

사용: python src/mtbs_finetune_korea.py --ckpt ckpt/mtbs_sev_best.pt --epochs 40
"""
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from mtbs_pretrain_train import build_unet, normalize_scene, IGNORE  # noqa: E402
from mtbs_validate_korea import fetch_site, korea_gt3, run_model, SITES, MODEL2K  # noqa: E402

K2MODEL = {1: 1, 2: 2, 3: 3}          # 한국 경1중2심3 → 모델 저1중2고3 (미저0 미사용)


def korea_xy(name, s):
    """(정규화 X, 모델타깃 y) — y: 경→1,중→2,심→3, 배경=IGNORE."""
    X, gtA = fetch_site(name, s)

    def nbr(Z):
        n, ss = Z[9], Z[11]; d = n + ss; return np.where(d > 1e-6, (n - ss) / np.maximum(d, 1e-6), 0)
    k3 = korea_gt3(name, gtA, nbr(X))                 # 경1중2심3
    y = np.full(k3.shape, IGNORE, np.uint8)
    for kk, mv in K2MODEL.items():
        y[k3 == kk] = mv
    return normalize_scene(X), y, k3


def evaluate(net, dev, Xn, k3):
    """held-out 화재: 모델→한국3등급 매핑 후 등급정확도·면적오차."""
    pred_m = run_model(net, Xn, dev)
    pred_k = np.vectorize(lambda v: MODEL2K.get(int(v), 1))(pred_m).astype("int8")
    b = k3 > 0
    acc = float((pred_k[b] == k3[b]).mean()) if b.sum() else float("nan")
    ga = {k: int((k3 == k).sum()) * 0.04 for k in (1, 2, 3)}
    pa = {k: int((pred_k[b] == k).sum()) * 0.04 for k in (1, 2, 3)}
    mae = np.mean([abs(pa[k] - ga[k]) / max(ga[k], 1e-9) for k in (1, 2, 3)]) * 100
    return acc, mae, ga, pa


def finetune(ckpt, train_names, dev, epochs, lr, tile=256):
    import torch
    from mtbs_pretrain_train import IGNORE as IG
    net = build_unet(12, 4).to(dev)
    if os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location=dev))
    # 한국 학습화재 타일 수집
    tiles = []
    for nm in train_names:
        Xn, y, _ = korea_xy(nm, SITES[nm])
        C, H, W = Xn.shape
        for r in range(0, H - tile + 1, tile // 2):
            for c in range(0, W - tile + 1, tile // 2):
                yt = y[r:r+tile, c:c+tile]
                if (yt != IG).sum() > tile * tile * 0.02:
                    tiles.append((Xn[:, r:r+tile, c:c+tile], yt))
    if not tiles:                                     # 작은 화재면 전체 1타일
        for nm in train_names:
            Xn, y, _ = korea_xy(nm, SITES[nm]); tiles.append((Xn, y))
    w = torch.tensor([1., 1., 1.5, 2.5], device=dev)  # 심 희소 보정
    opt = torch.optim.AdamW(net.parameters(), lr)
    lossf = torch.nn.CrossEntropyLoss(weight=w, ignore_index=IG)
    net.train()
    for ep in range(epochs):
        np.random.shuffle(tiles); tl = 0
        for Xt, yt in tiles:
            X = torch.from_numpy(np.ascontiguousarray(Xt))[None].float().to(dev)
            Y = torch.from_numpy(yt.astype("int64"))[None].to(dev)
            # UNet 16배수 패딩
            _, _, H, W = X.shape; ph, pw = (16-H % 16) % 16, (16-W % 16) % 16
            X = torch.nn.functional.pad(X, (0, pw, 0, ph)); Y = torch.nn.functional.pad(Y, (0, pw, 0, ph), value=IG)
            opt.zero_grad(); l = lossf(net(X), Y); l.backward(); opt.step(); tl += l.item()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "ckpt", "mtbs_sev_best.pt"))
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--deploy", action="store_true", help="홀드아웃 없이 전 사이트로 1회 파인튜닝→배포모델 저장")
    ap.add_argument("--out", default=os.path.join(ROOT, "ckpt", "mtbs_sev_deploy.pt"))
    a = ap.parse_args()
    import torch
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    dev = "cuda" if (cvd != "" and torch.cuda.is_available()) else "cpu"
    names = list(SITES)
    if a.deploy:                                    # 배포용: 전 8화재로 파인튜닝, 단일 모델 저장
        print(f"배포 파인튜닝 — device {dev} · 학습 {names}")
        net = finetune(a.ckpt, names, dev, a.epochs, a.lr); net.eval()
        torch.save(net.state_dict(), a.out)
        print(f"✅ 배포모델 저장 {a.out} ({os.path.getsize(a.out)//1024}KB)")
        return
    print(f"LORO 파인튜닝 — device {dev} · ckpt {os.path.basename(a.ckpt)}")
    # 기준: 파인튜닝 전(사전학습만)
    base = build_unet(12, 4).to(dev)
    if os.path.exists(a.ckpt):
        base.load_state_dict(torch.load(a.ckpt, map_location=dev)); base.eval()
    for hold in names:
        tr = [n for n in names if n != hold]
        Xn, y, k3 = korea_xy(hold, SITES[hold])
        a0 = evaluate(base, dev, Xn, k3) if os.path.exists(a.ckpt) else (float("nan"),)*2+({},{})
        net = finetune(a.ckpt, tr, dev, a.epochs, a.lr); net.eval()
        a1 = evaluate(net, dev, Xn, k3)
        print(f"\n[{hold}] (학습={tr})  파인튜닝 전 acc {a0[0]:.3f}/면적{a0[1]:.0f}% → 후 acc {a1[0]:.3f}/면적{a1[1]:.0f}%")
        for k, nm in ((3, "심"), (2, "중"), (1, "경")):
            print(f"   {nm}: GT {a1[2][k]:,.0f}ha · 예측 {a1[3][k]:,.0f}ha")


if __name__ == "__main__":
    main()
