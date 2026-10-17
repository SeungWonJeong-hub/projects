#!/usr/bin/env python3
"""재브래킷 A/B 정직 평가 (2026-09-24).

산청 홀드아웃(250321_산청시천면)에서 baseline vs 재브래킷-라벨 모델을 비교한다.
- 같은 입력(재브래킷 tight 04-01 쌍)을 두 모델에 먹인다 → 학습라벨 효과만 분리.
- 헤드라인 지표 = **예측 소각면적 vs 공식 1,858 ha**(독립·비순환).
- 보조 = F1 (재브래킷-GT / 표준-GT 둘 다, 배포임계 0.25 + best-thr sweep).

사용:
  python src/eval_ab.py runs/ab_baseline/reh_deeplabv3plus_s1.pt runs/ab_treat/reh_deeplabv3plus_s1.pt
"""
import os, sys, numpy as np, torch
import segmentation_models_pytorch as smp

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, "..")
_ROOT = os.path.join(PROJ, "..", "..", "..")
CIRD = os.path.join(_ROOT, "Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires")
FIRE = os.environ.get("FIRE", "250321_산청시천면")
OFFICIAL_HA = float(os.environ.get("OFFICIAL_HA", "1858"))   # 미상이면 rb-GT를 기준으로 표기
GSD = 20.0
PIX_HA = GSD * GSD / 1e4
DEPLOY_THR = 0.25
DEV = "cuda" if torch.cuda.is_available() else "cpu"


_ARCH = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus, "fpn": smp.FPN}
ARCH = os.environ.get("ARCH", "unet")          # 배포 = 단일 U-Net s1 / resnet34


def load_model(ckpt):
    m = _ARCH[ARCH]("resnet34", encoder_weights=None, in_channels=8, classes=1).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV)); m.eval()
    return m


def infer(m, x, H, W, T=256):
    """타일 슬라이딩 추론 → 확률맵 (step4_rehearsal.infer 와 동일 규약, 8ch)."""
    prob = np.zeros((H, W), "float32"); cnt = np.zeros((H, W), "float32")
    st = int(T * 0.75)
    ys = sorted(set(min(a, H - T) for a in range(0, max(H - T, 0) + 1, st)) or [0])
    xs = sorted(set(min(b, W - T) for b in range(0, max(W - T, 0) + 1, st)) or [0])
    with torch.no_grad():
        for y in ys:
            for z in xs:
                xt = x[:, y:y+T, z:z+T]
                if xt.shape[1] < T or xt.shape[2] < T:
                    pad = np.zeros((x.shape[0], T, T), "float32")
                    pad[:, :xt.shape[1], :xt.shape[2]] = xt; xt = pad
                t = torch.from_numpy(xt[None]).to(DEV)
                p = torch.sigmoid(m(t))[0, 0].cpu().numpy()
                prob[y:y+T, z:z+T] += p[:min(T, H-y), :min(T, W-z)]
                cnt[y:y+T, z:z+T] += 1
    return prob / np.maximum(cnt, 1e-6)


def f1_at(prob, gt, vd, t):
    p = (prob >= t) & vd; g = (gt > 0) & vd
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return 2 * pr * rc / max(pr + rc, 1e-9), pr, rc


def best_f1(prob, gt, vd):
    best = (0, 0)
    for t in np.arange(0.05, 0.9, 0.05):
        f = f1_at(prob, gt, vd, t)[0]
        if f > best[0]: best = (f, t)
    return best


def evaluate(ckpt, x, vd, gt_rb, gt_std):
    m = load_model(ckpt)
    H, W = vd.shape
    prob = infer(m, x, H, W)
    pred = (prob >= DEPLOY_THR) & vd
    area = float(pred.sum()) * PIX_HA
    err = (area - OFFICIAL_HA) / OFFICIAL_HA
    f1_rb, pr_rb, rc_rb = f1_at(prob, gt_rb, vd, DEPLOY_THR)
    f1_std, _, _ = f1_at(prob, gt_std, vd, DEPLOY_THR)
    bf_rb, bt = best_f1(prob, gt_rb, vd)
    return dict(area=area, err=err, f1_rb=f1_rb, pr=pr_rb, rc=rc_rb,
                f1_std=f1_std, bestf1_rb=bf_rb, bestthr=bt)


def main():
    base_ck, treat_ck = sys.argv[1], sys.argv[2]
    rb = np.load(os.path.join(CIRD, FIRE, "gt_stac_rb.npz"))
    std = np.load(os.path.join(CIRD, FIRE, "gt_stac.npz"))
    x = rb["x"].astype("float32")            # 같은 입력(04-01 tight) — 두 모델 공통
    vd = rb["valid"].astype(bool)
    gt_rb = rb["gt"].astype(np.uint8)
    # 표준 GT는 격자가 다를 수 있음(post 04-11) → 크기 맞으면만 비교, 아니면 rb GT로 대체표기
    gt_std = std["gt"].astype(np.uint8) if std["gt"].shape == gt_rb.shape else gt_rb
    print(f"산청 입력 {x.shape}  valid {vd.mean():.2f}  "
          f"재브래킷-GT {int((gt_rb>0).sum())*PIX_HA:.0f}ha  공식 {OFFICIAL_HA:.0f}ha\n")
    hdr = f"{'모델':<10}{'예측면적(ha)':>13}{'면적오차':>10}{'F1@0.25(rb)':>13}{'정밀':>7}{'재현':>7}{'best-F1':>9}"
    print(hdr); print("-" * len(hdr))
    for name, ck in (("baseline", base_ck), ("treatment", treat_ck)):
        if not os.path.exists(ck):
            print(f"{name:<10} 체크포인트 없음: {ck}"); continue
        r = evaluate(ck, x, vd, gt_rb, gt_std)
        print(f"{name:<10}{r['area']:>13.0f}{r['err']:>+9.1%}{r['f1_rb']:>13.3f}"
              f"{r['pr']:>7.2f}{r['rc']:>7.2f}{r['bestf1_rb']:>9.3f}")
    print(f"\n헤드라인 = 예측면적이 공식 {OFFICIAL_HA:.0f}ha 에 가까울수록(면적오차 0에 가까울수록) 좋음.")


if __name__ == "__main__":
    main()
