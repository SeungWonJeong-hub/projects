"""해외 유사라벨을 한국 폴리곤 정의(전체 scar·고밀도)로 하모나이즈 — 오프라인 relabel(재fetch無).

진단: 해외 burn 4-9% vs 한국 43% (화재 작음 + char코어만 라벨). 전이 실패 근본원인.
해결(사용자 제안): ①느슨한 dNBR + post-NBR ②모폴로지 팽창(코어→전체scar, 산림·valid내 제한)
③대형밀도 필터(burn%<MIN_BURN 타일 드랍) → 한국 43% 분포에 최대한 근접.
저장된 x·fmask·valid 재사용, jp_refl_harm / eastus_refl_harm 로 출력(원본 보존).

env: DNBR_THR=0.10 POST_NBR_MAX=0.35 DILATE=4 MIN_BURN=0.15 python3 harmonize_labels.py
"""
import os, glob, numpy as np
from scipy.ndimage import binary_dilation
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
DNBR_THR=float(os.environ.get("DNBR_THR","0.10"))
POST_NBR_MAX=float(os.environ.get("POST_NBR_MAX","0.35"))
DILATE=int(os.environ.get("DILATE","4"))
MIN_BURN=float(os.environ.get("MIN_BURN","0.15"))   # 이 밀도 미만 타일 드랍(대형 선별)
def nbr(x6): return (x6[3]-x6[5])/(x6[3]+x6[5]+1e-6)
def harm_label(x,fmask,valid):
    pre,post=x[0:6],x[6:12]
    dnbr=nbr(pre)-nbr(post); fg=valid&(fmask>0.5)
    med=np.median(dnbr[fg]) if fg.sum()>50 else 0.0
    core=((dnbr-med)>DNBR_THR)&(nbr(post)<POST_NBR_MAX)&(fmask>0.5)&valid   # 느슨 코어
    if DILATE>0:
        grown=binary_dilation(core,iterations=DILATE)&(fmask>0.5)&valid    # 팽창(산림·valid 제한)
    else: grown=core
    return grown.astype(np.uint8)
for region,src in [("japan","jp_refl"),("eastus","eastus_refl")]:
    fs=sorted(glob.glob(os.path.join(PROJ,src,"*","*.npz")))
    out=os.path.join(PROJ,f"{src}_harm"); os.makedirs(out,exist_ok=True)
    kept=0;bfs=[]
    for f in fs:
        z=np.load(f);x=z["x"].astype(np.float32);v=z["valid"].astype(bool)
        fm=z["fmask"] if "fmask" in z else np.ones_like(v)
        y=harm_label(x,fm,v); bf=y[v].mean() if v.sum() else 0
        if bf<MIN_BURN: continue                                            # 소형·저밀도 드랍
        name=os.path.basename(os.path.dirname(f)); od=os.path.join(out,name); os.makedirs(od,exist_ok=True)
        np.savez_compressed(os.path.join(od,os.path.basename(f)),x=z["x"],y=y,valid=z["valid"])
        kept+=1;bfs.append(bf)
    bfs=np.array(bfs)
    print(f"[{region}] {len(fs)}→ 유지 {kept}타일 (burn≥{MIN_BURN*100:.0f}%) | burn평균 {bfs.mean()*100:.1f}% 중앙 {np.median(bfs)*100:.1f}%" if kept else f"[{region}] 유지 0타일 — 기준 완화 필요",flush=True)
print("한국 참고 burn 43.3%. 다음: TRAIN=japan_harm,eastus_harm 로 전이 재시도")
