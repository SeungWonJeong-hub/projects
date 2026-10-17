#!/usr/bin/env python3
"""treatment(재브래킷) unet-s1 의 추론 임계 스윕 — 3 홀드아웃 봄화재 면적/F1 중심 정렬 (2026-09-24).
재학습 없이 배포 임계만 바꿔 오버슛/언더슛을 잡을 수 있는지 본다."""
import os, numpy as np, torch
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ab import load_model, infer, f1_at, CIRD, PIX_HA

CKPT = os.environ.get("CKPT", "runs/ab_unet_treat/reh_unet_s1.pt")
# 화재: (폴더, rb-GT extent ha, 공식 ha or None)
FIRES = [("250321_산청시천면", 2077, 1858),
         ("250324_안동길안면", 2838, None),
         ("230402_홍성서부면",  911, None)]
THRS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def main():
    m = load_model(CKPT)
    print(f"CKPT={CKPT}\n")
    # 각 화재 prob 한번만 계산
    data = []
    for folder, rbha, offha in FIRES:
        z = np.load(os.path.join(CIRD, folder, "gt_stac_rb.npz"))
        x = z["x"].astype("float32"); vd = z["valid"].astype(bool); gt = z["gt"].astype(np.uint8)
        H, W = vd.shape; prob = infer(m, x, H, W)
        data.append((folder, rbha, offha, prob, vd, gt))
    # 임계별 표
    for folder, rbha, offha, prob, vd, gt in data:
        anchor = offha or rbha; alab = "공식" if offha else "rbGT"
        print(f"== {folder}  기준 {alab} {anchor}ha (rbGT {rbha}) ==")
        print(f"{'thr':>6}{'예측ha':>9}{'면적오차':>9}{'F1':>7}{'P':>6}{'R':>6}")
        for t in THRS:
            area = float(((prob >= t) & vd).sum()) * PIX_HA
            err = (area - anchor) / anchor
            fv, pr, rc = f1_at(prob, gt, vd, t)
            print(f"{t:>6.2f}{area:>9.0f}{err:>+8.0%}{fv:>7.3f}{pr:>6.2f}{rc:>6.2f}")
        print()
    # 평균 |면적오차| 와 평균 F1 을 임계별로 (중심 정렬 임계 선택)
    print("=== 임계별 3화재 평균 |면적오차| / 평균 F1 ===")
    print(f"{'thr':>6}{'평균|오차|':>11}{'평균F1':>9}")
    for t in THRS:
        errs, f1s = [], []
        for folder, rbha, offha, prob, vd, gt in data:
            anchor = offha or rbha
            area = float(((prob >= t) & vd).sum()) * PIX_HA
            errs.append(abs((area - anchor) / anchor))
            f1s.append(f1_at(prob, gt, vd, t)[0])
        print(f"{t:>6.2f}{np.mean(errs):>10.0%}{np.mean(f1s):>9.3f}")


if __name__ == "__main__":
    main()
