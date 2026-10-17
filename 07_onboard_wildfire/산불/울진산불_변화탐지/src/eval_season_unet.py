#!/usr/bin/env python3
"""배포 unet-s1/resnet34 의 산림청(ko_refl) 홀드아웃 계절별 F1 (2026-09-24).
eval_by_season.py 는 Prithvi 용이라, 실제 배포 모델로 다시 잰다.
CKPT=배포 et02_unet_s1.pt, ARCH=unet, 배포임계 0.25."""
import os, glob, numpy as np, torch
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step4_rehearsal as S

THR = float(os.environ.get("THR", "0.25"))
KO = os.path.join(S.PROJ, os.environ.get("KO_DIR", "ko_refl"))


def season(m): return "봄" if 3<=m<=5 else "여름" if 6<=m<=8 else "가을" if 9<=m<=11 else "겨울"


def f1(P, Y, t):
    p = P >= t; g = Y > 0.5
    tp=int((p&g).sum()); fp=int((p&~g).sum()); fn=int((~p&g).sum())
    pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); return 2*pr*rc/max(pr+rc,1e-9), pr, rc


def main():
    m = S.build()   # ARCH/ENCODER/CKPT env 로 배포 unet-s1 로드
    print(f"모델 ARCH={S.ARCH} ENCODER={S.ENCODER}\nCKPT={S.CKPT}\n배포임계 THR={THR}\n")
    from collections import defaultdict
    byseas = defaultdict(lambda: [[], []])
    print(f"{'화재':26s}{'계절':5s}{'F1':>7}{'P':>7}{'R':>7}")
    for fire in S.FOREST_HOLD:
        d = os.path.join(KO, fire if fire.startswith("kf_") else "kf_"+fire)
        cands = glob.glob(d) or glob.glob(os.path.join(KO, f"*{fire.replace('kf_','')}*"))
        if not cands: print(f"{fire:26s} (없음)"); continue
        d = cands[0]; mo = int(os.path.basename(d).split("_")[1][4:6])
        Ps, Ys = [], []
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f)
            x = np.clip(S.sel12(z["x"].astype(np.float32)/1e4), 0, 1).astype("float32")
            y = z["y"].astype("float32")
            v = z["valid"].astype(bool) if "valid" in z else np.ones(y.shape, bool)
            if v.mean() < 0.5: continue
            H, W = y.shape; prob = S.infer(m, x, H, W)
            Ps.append(prob[v]); Ys.append(y[v])
        if not Ps: continue
        P = np.concatenate(Ps); Y = np.concatenate(Ys)
        se = season(mo); fv, pr, rc = f1(P, Y, THR)
        print(f"{os.path.basename(d):26s}{se:5s}{fv:>7.3f}{pr:>7.3f}{rc:>7.3f}")
        byseas[se][0].append(P); byseas[se][1].append(Y)
    print("\n=== 계절별 통합 F1 (산림청 GT, 배포임계 0.25) ===")
    for se in ["봄", "여름", "가을", "겨울"]:
        if se not in byseas: continue
        P = np.concatenate(byseas[se][0]); Y = np.concatenate(byseas[se][1])
        fv, pr, rc = f1(P, Y, THR)
        print(f"{se}: F1={fv:.3f}  P={pr:.3f}  R={rc:.3f}  (화재 {len(byseas[se][0])})")


if __name__ == "__main__":
    main()
