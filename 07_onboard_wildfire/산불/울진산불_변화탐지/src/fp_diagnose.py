"""예측이 GT 와 어긋난 화소를 **GT 생성식으로 되짚어** 원인을 가른다.

왜 필요한가: `cir_gt_profile.py` 는 GT 를 만든 식과 **같은 조건**(dNDVI·postNDVI)으로
오염을 세기 때문에 **GT 생성식 자체의 결함은 원리적으로 못 잡는다.** 실제로
`220531_밀양부북면`(F1 0.460 · 면적 +36.7%)을 오염배수 0.02x 로 "깨끗함" 이라 판정했다.

여기서는 FP(예측했는데 GT 아님) 화소를 GT 생성식의 세 조건으로 다시 통과시켜 본다:
    burn = (dNBR>0.15) AND (dNDVI>0.08) AND (postNDVI<0.45) AND valid
FP 가 앞의 둘은 통과하는데 postNDVI 에서만 걸렸다면, 그건 오탐이 아니라 **GT 가 놓친
소각**일 가능성이 크다(늦봄·여름 화재는 green-up 때문에 postNDVI 가 잘 안 내려간다).

12밴드 판(`gt_stac_20m_raw12.npz`, BANDS12=1)이 필요하다 — dNBR 에 SWIR 이 든다.
사용: python fp_diagnose.py <화재명>:<fold> [...]
"""
import os, sys, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("GSD", "20"); os.environ.setdefault("TILE", "256")
os.environ["BANDS12"] = "1"
GSD = float(os.environ["GSD"]); PXA = GSD*GSD/1e4
import step4_rehearsal as S, nifos_eval as N
N.TILE = int(os.environ["TILE"])
from scipy.ndimage import binary_dilation, label

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
FIRES = os.path.join(PROJ, "..", "산불데이터", "국토정보플랫폼 데이터셋", "CIR_실측라벨", "fires")
THR_NBR, THR_NDVI, POSTNDVI = 0.15, 0.08, 0.45      # build_cir_gt_stac.py 와 같은 값

def idx(a, b): return (a-b)/(a+b+1e-6)

def diagnose(name, fold):
    z = np.load(os.path.join(FIRES, name, "gt_stac_20m_raw12.npz"))
    x = z["x"].astype(np.float32); gt = z["gt"].astype(bool); vd = z["valid"].astype(bool)
    assert x.shape[0] == 12, f"{name}: 12ch 판이 필요하다 (BANDS12=1 로 재생성)"
    pre, post = x[:6], x[6:]                        # B02,B03,B04,NIR,B11,B12
    dnbr  = idx(pre[3], pre[5]) - idx(post[3], post[5])
    dndvi = idx(pre[3], pre[2]) - idx(post[3], post[2])
    pndvi = idx(post[3], post[2])

    runs = sorted(glob.glob(os.path.join(PROJ, f"runs/cv_f{fold}_20m/*.pt")))
    x12 = np.zeros((12,)+x.shape[1:], np.float32)   # 8ch 모델이므로 rgbnir 자리만 채운다
    x12[[0,1,2,3]] = pre[:4]*1e4; x12[[6,7,8,9]] = post[:4]*1e4
    prob, _ = N.infer(x12, runs)
    def f1(t):
        p = (prob >= t) & vd; g = gt & vd
        tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
        pr = tp/(tp+fp+1e-9); rc = tp/(tp+fn+1e-9); return 2*pr*rc/(pr+rc+1e-9)
    t = max(np.arange(0.05, 0.95, 0.025), key=f1)
    pred = (prob >= t) & vd; g = gt & vd
    TP, FP, FN = pred & g, pred & ~g, ~pred & g

    print(f"\n===== {name}  (fold{fold} 홀드아웃, thr={t:.3f})")
    print(f"  씬: pre={z['pre']} post={z['post']}  | GT {g.sum()*PXA:.0f}ha  예측 {pred.sum()*PXA:.0f}ha"
          f"  ({(pred.sum()/max(g.sum(),1)-1)*100:+.1f}%)")
    print(f"  TP {TP.sum()*PXA:7.0f}ha   FP {FP.sum()*PXA:7.0f}ha   FN {FN.sum()*PXA:7.0f}ha")

    # ① FP 가 GT 경계에 붙은 후광인가, 떨어진 덩어리인가
    halo = FP & binary_dilation(g, iterations=2)      # GT 에서 40m 이내
    lab, n = label(FP & ~halo)
    sz = np.bincount(lab.ravel())[1:] if n else np.array([])
    big = np.sort(sz)[::-1][:3]*PXA if sz.size else []
    print(f"  ① FP 위치: GT 40m 이내 후광 {halo.sum()/max(FP.sum(),1)*100:5.1f}%  ·  "
          f"분리 덩어리 {n}개 (최대 {', '.join(f'{b:.0f}ha' for b in big) if len(big) else '-'})")

    # ② FP 를 GT 생성식으로 되짚는다
    c_nbr, c_ndvi, c_post = dnbr > THR_NBR, dndvi > THR_NDVI, pndvi < POSTNDVI
    fpn = max(int(FP.sum()), 1)
    both = c_nbr & c_ndvi
    print(f"  ② FP 를 GT 식으로 되짚기 (통과 비율)")
    print(f"     dNBR>{THR_NBR}  {int((FP&c_nbr).sum())/fpn*100:5.1f}%   "
          f"dNDVI>{THR_NDVI}  {int((FP&c_ndvi).sum())/fpn*100:5.1f}%   "
          f"postNDVI<{POSTNDVI}  {int((FP&c_post).sum())/fpn*100:5.1f}%")
    print(f"     소각신호는 있는데(dNBR·dNDVI 둘 다 통과) postNDVI 에서만 걸린 FP: "
          f"**{int((FP&both&~c_post).sum())/fpn*100:5.1f}%**  ({int((FP&both&~c_post).sum())*PXA:.0f}ha)")
    print(f"     세 조건 모두 통과한 FP(=GT 형태연산에서 탈락): {int((FP&both&c_post).sum())/fpn*100:5.1f}%")
    print(f"     소각신호 자체가 없는 FP(진짜 오탐): {int((FP&~both).sum())/fpn*100:5.1f}%")

    # ③ 화소 성질 비교
    print(f"  ③ 중앙값        {'dNBR':>8}{'dNDVI':>8}{'postNDVI':>10}")
    for nm, m in (("TP", TP), ("FP", FP), ("FN", FN), ("배경", vd & ~g & ~pred)):
        if m.sum() == 0: continue
        print(f"     {nm:<12}{np.median(dnbr[m]):>8.3f}{np.median(dndvi[m]):>8.3f}{np.median(pndvi[m]):>10.3f}")

for a in sys.argv[1:]:
    nm, fd = a.split(":"); diagnose(nm, int(fd))
