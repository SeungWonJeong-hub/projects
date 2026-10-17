"""CIR 홀드아웃 화재로 일반화 확정 — 설계상 미학습인 4화재를 앙상블로 채점.

CIR_HOLD(step4_rehearsal.py:17) = 250324_안동길안면 · 250321_산청시천면 ·
230402_홍성서부면 · 220410_양구양구읍. 배포 앙상블(cir_reh_ens)이 학습때 못 본 화재다.
GT = 각 화재 폴더의 gt_stac[_10m].npz (x=8ch rgbnir 0-1, gt, valid).

사용: GSD=20 TILE=256 python cir_hold_eval.py --ens "runs/cir_reh_ens/*.pt"
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
GSD=float(os.environ.get("GSD","20")); TILE=int(os.environ.get("TILE","256"))
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
FIRES=os.path.join(PROJ,"..","산불데이터","국토정보플랫폼 데이터셋","CIR_실측라벨","fires")
# 홀드아웃 정의는 step4_rehearsal 단일 소스에서 가져온다(정의가 갈리면 누출을 못 잡는다)
import step4_rehearsal as S
GROUP=os.environ.get("GROUP","large")          # large | small | both
HOLD={"large":S.CIR_HOLD,"small":S.SMALL_HOLD,"both":S.CIR_HOLD+S.SMALL_HOLD}[GROUP]
NPZ=S.CIR_NPZ
import nifos_eval as N                                  # _load / infer 재사용
N.TILE=TILE

def score(pred,gt,valid):
    p=pred&valid; g=gt&valid
    tp=int((p&g).sum()); fp=int((p&~g).sum()); fn=int((~p&g).sum())
    pr=tp/(tp+fp+1e-9); rc=tp/(tp+fn+1e-9)
    return 2*pr*rc/(pr+rc+1e-9),pr,rc

def main():
    assert sys.argv[1]=="--ens"
    runs=[d for d in sorted(glob.glob(os.path.join(PROJ,sys.argv[2])))
          if d.endswith(".pt") or os.path.exists(os.path.join(d,"results.json"))]
    assert runs, "앙상블 대상 없음"
    print(f"앙상블 {len(runs)}모델 · GSD={GSD:.0f}m · TILE={TILE} · 홀드아웃={GROUP}({len(HOLD)}) · GT={NPZ}\n")
    print(f"{'화재':<20}{'F1':>8}{'P':>7}{'R':>7}{'GT ha':>9}{'예측 ha':>9}")
    print("-"*62)
    allP=[];allG=[];rows=[]
    for name in HOLD:
        f=os.path.join(FIRES,name,NPZ)
        if not os.path.exists(f): print(f"{name:<20}   GT 없음"); continue
        z=np.load(f); x8=z["x"].astype(np.float32); gt=z["gt"].astype(bool); vd=z["valid"].astype(bool)
        if vd.mean()<0.2 or gt.sum()==0:
            print(f"{name:<20}   무효(valid {vd.mean()*100:.0f}%, GT {gt.sum()}px)"); continue
        # nifos_eval.infer는 12ch에서 설정 채널을 뽑으므로, 8ch 판은 여기서 12ch로 되돌려 넘긴다.
        # 12ch 판(BANDS12=1)이면 그대로 — SWIR 자리를 0 으로 채워 넘기면 조용히 성능이 죽는다.
        if x8.shape[0]==12:
            x12=x8*1e4
        else:
            x12=np.zeros((12,)+x8.shape[1:],np.float32)
            x12[[0,1,2,3]]=x8[:4]*1e4; x12[[6,7,8,9]]=x8[4:]*1e4
        prob,_=N.infer(x12,runs)
        best=max(np.arange(0.05,0.95,0.025),key=lambda t:score(prob>=t,gt,vd)[0])
        f1,pr,rc=score(prob>=best,gt,vd)
        px=GSD*GSD/1e4
        print(f"{name:<20}{f1:>8.3f}{pr:>7.3f}{rc:>7.3f}{gt[vd].sum()*px:>9.0f}{(prob>=best)[vd].sum()*px:>9.0f}")
        allP.append(prob[vd]); allG.append(gt[vd]); rows.append(f1)
    if rows:
        P=np.concatenate(allP); G=np.concatenate(allG)
        b=max(np.arange(0.05,0.95,0.025),key=lambda t:score(P>=t,G,np.ones_like(G))[0])
        f1,pr,rc=score(P>=b,G,np.ones_like(G))
        print("-"*62)
        print(f"{'화재평균':<20}{np.mean(rows):>8.3f}")
        print(f"{'전체풀(pooled)':<20}{f1:>8.3f}{pr:>7.3f}{rc:>7.3f}")

if __name__=="__main__": main()
