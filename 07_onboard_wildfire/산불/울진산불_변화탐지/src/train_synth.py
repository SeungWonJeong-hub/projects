"""우리 sim2real 검증: 실측-분광 합성 소각(cir_synth)을 학습 positive로 추가.
recipe는 step4_rehearsal.main과 동일(동일 base·LR·EP·aug·손실) — 합성만 추가해 apples-to-apples.
baseline(합성 없음): 봄 0.792 / 산림청 0.647 / 음성오탐률 34.2% (SEED1 EP15).
델타는 cross-fire σ=0.068 기준. 사용: SYNTH_OS=1 CIR_RAW=0 SEED=1 EPOCHS=15 python train_synth.py
"""
from __future__ import annotations
import os, glob, random, json, numpy as np, torch, torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
import step4_rehearsal as S

SYNTH_DIR=os.path.join(S.PROJ,"cir_synth")
SYNTH_OS=int(os.environ.get("SYNTH_OS","1"))
random.seed(S.SEED); np.random.seed(S.SEED); torch.manual_seed(S.SEED)

def synth_tiles():
    out=[]
    for f in sorted(glob.glob(os.path.join(SYNTH_DIR,"*","gt_stac.npz"))):
        z=np.load(f); x=z["x"].astype(np.float32); gt=z["gt"].astype("float32"); vd=z["valid"].astype(bool)
        if vd.mean()>=0.5 and x.shape[1]==S.T and x.shape[2]==S.T:
            out.append((x.copy(),gt.copy()))
    return out

def main():
    cir=[]
    for n in S.CIR_TRAIN: cir+=S.cir_tiles(n)
    forest=S.forest_train_tiles()
    syn=synth_tiles()
    FOREST_OS=int(os.environ.get("FOREST_OS","1"))
    pos=forest*FOREST_OS+cir*S.CIR_OS+syn*SYNTH_OS
    tiles=pos
    print(f"학습타일: 산림청 {len(forest)}x{FOREST_OS} + CIR {len(cir)}x{S.CIR_OS} + 합성 {len(syn)}x{SYNTH_OS} = {len(tiles)}"
          f" | LR={S.LR} EP={S.EPOCHS} SEED={S.SEED}",flush=True)
    # 음성 홀드아웃(오탐률 측정용, 학습 미투입)
    _,neg_ho=S.neg_tiles()
    m=S.build()
    print(f"[baseline] 봄={S.eval_cir(m):.3f} 산림청={S.eval_forest(m):.3f} 음성오탐률={S.eval_neg_fpr(m,neg_ho):.1%}",flush=True)
    opt=torch.optim.AdamW(m.parameters(),lr=S.LR,weight_decay=1e-4); bce=nn.BCEWithLogitsLoss(); dice=DiceLoss(mode="binary"); BS=8
    for ep in range(S.EPOCHS):
        m.train(); random.shuffle(tiles); xs_,ys_=[],[]
        for xt,yt in tiles:
            k=random.randint(0,3); fl=random.random()<0.5
            xa=np.rot90(xt,k,axes=(1,2)); ya=np.rot90(yt,k)
            if fl: xa=xa[:,:,::-1]; ya=ya[:,::-1]
            xs_.append(xa.copy()); ys_.append(ya.copy())
        for i in range(0,len(xs_),BS):
            xb=torch.from_numpy(np.stack(xs_[i:i+BS])).to(S.DEV); yb=torch.from_numpy(np.stack(ys_[i:i+BS]))[:,None].to(S.DEV)
            if xb.shape[0]<2: continue
            opt.zero_grad(); out=m(xb); (bce(out,yb)+dice(out,yb)).backward(); opt.step()
        if (ep+1)%5==0: print(f"  ep{ep+1}: 봄={S.eval_cir(m):.3f} 산림청={S.eval_forest(m):.3f} 음성오탐률={S.eval_neg_fpr(m,neg_ho):.1%}",flush=True)
    c=S.eval_cir(m); f=S.eval_forest(m); nf=S.eval_neg_fpr(m,neg_ho)
    print(f"\n★ 합성증강 결과: 봄 cross-fire={c:.3f} (baseline 0.792) | 산림청={f:.3f} (baseline 0.647) | 음성오탐률={nf:.1%} (baseline 34.2%)",flush=True)
    _out=os.path.join(S.PROJ,"runs",os.environ.get("OUT_DIR","cir_synth_train"))
    os.makedirs(_out,exist_ok=True)
    torch.save(m.state_dict(),os.path.join(_out,"reh_deeplabv3plus_s1.pt"))
    json.dump(dict(cir=c,forest=f,neg_fpr=nf,n_synth=len(syn),synth_os=SYNTH_OS,
                   n_cir=len(cir),n_forest=len(forest),epochs=S.EPOCHS,seed=S.SEED),
              open(os.path.join(_out,"results.json"),"w"),ensure_ascii=False,indent=1)
    print(f"저장: runs/{os.path.basename(_out)}/",flush=True)

if __name__=="__main__": main()
