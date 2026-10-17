"""③ 기하 정합 민감도 — pre/post 오정합이 소각탐지를 얼마나 망가뜨리나.

온보드는 자체 정사보정이라 pre/post 사이에 서브픽셀~수픽셀 시프트가 생긴다.
"10m로 가면 두 배 민감해진다"는 주장을 실측으로 검증한다: post 영상만 인공 시프트시키고
같은 앙상블·같은 GT로 F1 하락을 잰다. 시프트는 **미터 고정**이라 10m/20m를 같은 물리
오정합으로 비교한다(px 고정이면 10m가 부당하게 유리해진다).

사용: GSD=10 TILE=512 python shift_sensitivity.py --ens "runs/cir_reh_ens_10m/*.pt"
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
GSD=float(os.environ.get("GSD","20")); TILE=int(os.environ.get("TILE","256"))
SHIFTS_M=[float(x) for x in os.environ.get("SHIFTS_M","0,5,10,20,40").split(",")]
import nifos_eval as N
N.TILE=TILE
import step4_rehearsal as S

def shift_post(x12,dx_px):
    """post 6밴드만 x축으로 dx_px 이동(서브픽셀은 선형보간). pre는 고정."""
    out=x12.copy()
    if dx_px==0: return out
    i=int(np.floor(dx_px)); f=dx_px-i
    for c in range(6,12):
        a=np.roll(x12[c],i,axis=1)
        b=np.roll(x12[c],i+1,axis=1)
        out[c]=(1-f)*a+f*b
    return out

def score(pred,gt,valid):
    p=pred&valid; g=gt&valid
    tp=int((p&g).sum()); fp=int((p&~g).sum()); fn=int((~p&g).sum())
    pr=tp/(tp+fp+1e-9); rc=tp/(tp+fn+1e-9)
    return 2*pr*rc/(pr+rc+1e-9)

def main():
    assert sys.argv[1]=="--ens"
    runs=[d for d in sorted(glob.glob(os.path.join(N.PROJ,sys.argv[2])))
          if d.endswith(".pt") or os.path.exists(os.path.join(d,"results.json"))]
    fires=S.CIR_HOLD
    print(f"앙상블 {len(runs)}모델 · GSD={GSD:.0f}m · 시프트는 미터 고정\n")
    hdr="".join(f"{m:>10.0f}m" for m in SHIFTS_M)
    print(f"{'화재':<20}{hdr}")
    print("-"*(20+11*len(SHIFTS_M)))
    tab=[]
    for name in fires:
        f=os.path.join(S.CIRD,name,S.CIR_NPZ)
        if not os.path.exists(f): continue
        z=np.load(f); x8=z["x"].astype(np.float32); gt=z["gt"].astype(bool); vd=z["valid"].astype(bool)
        if vd.mean()<0.2 or gt.sum()==0: continue
        x12=np.zeros((12,)+x8.shape[1:],np.float32)
        x12[[0,1,2,3]]=x8[:4]*1e4; x12[[6,7,8,9]]=x8[4:]*1e4
        row=[]
        for m in SHIFTS_M:
            prob,_=N.infer(shift_post(x12,m/GSD),runs)
            row.append(max(score(prob>=t,gt,vd) for t in np.arange(0.05,0.95,0.05)))
        tab.append(row)
        print(f"{name:<20}"+"".join(f"{v:>11.3f}" for v in row))
    if tab:
        mean=np.mean(tab,axis=0)
        print("-"*(20+11*len(SHIFTS_M)))
        print(f"{'평균':<20}"+"".join(f"{v:>11.3f}" for v in mean))
        print(f"{'0m 대비 손실':<20}"+"".join(f"{v-mean[0]:>+11.3f}" for v in mean))

if __name__=="__main__": main()
