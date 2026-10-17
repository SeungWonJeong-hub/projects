"""#④ 봄 cross-fire 10폴드 LORO (고정임계): 4-홀드아웃이 운 아님 검증.
10개 사용가능 CIR화재 각각 leave-one-out: 나머지9 CIR + 산림청 forest replay로 단일 deeplab
rehearsal 학습 → held-out CIR화재 평가(고정 thr0.15 + best). 10폴드 평균으로 로버스트니스.
사용: EPOCHS=15 python step6_spring_loro.py
"""
import os, glob, random, numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import step4_rehearsal as S
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
WEIGHTS=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/웹앱/weights"); DEV=S.DEV
EPOCHS=int(os.environ.get("EPOCHS","15")); LR=3e-5; FIXED=0.15
random.seed(0); np.random.seed(0); torch.manual_seed(0)
USABLE=["250325_경북영덕청송","250322_의성안평면","220304_울진북면","250325_영덕축산면","250324_안동길안면",
        "250321_산청시천면","220410_양구양구읍","230402_홍성서부면","220528_울진근남면","230403_영주평은면"]
def f1_at(P,Y,t):
    pr=P>=t; tp=(pr&(Y==1)).sum(); fp=(pr&(Y==0)).sum(); fn=((~pr)&(Y==1)).sum()
    p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); return 2*p*r/(p+r+1e-9)
def train_fold(holdout):
    cir=[]
    for n in USABLE:
        if n==holdout: continue
        cir+=S.cir_tiles(n)
    forest=S.forest_train_tiles(); tiles=forest+cir*S.CIR_OS
    m=smp.DeepLabV3Plus("resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV)
    m.load_state_dict(torch.load(os.path.join(WEIGHTS,"et02_ens","et02_deeplabv3plus_s1.pt.bak_prespring"),map_location=DEV)); m.train()
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4); bce=nn.BCEWithLogitsLoss(); dice=DiceLoss(mode="binary"); BS=8
    for ep in range(EPOCHS):
        random.shuffle(tiles); xs_,ys_=[],[]
        for xt,yt in tiles:
            k=random.randint(0,3); fl=random.random()<0.5
            xa=np.rot90(xt,k,axes=(1,2)); ya=np.rot90(yt,k)
            if fl: xa=xa[:,:,::-1]; ya=ya[:,::-1]
            xs_.append(xa.copy()); ys_.append(ya.copy())
        for i in range(0,len(xs_),BS):
            xb=torch.from_numpy(np.stack(xs_[i:i+BS])).to(DEV); yb=torch.from_numpy(np.stack(ys_[i:i+BS]))[:,None].to(DEV)
            if xb.shape[0]<2: continue
            opt.zero_grad(); out=m(xb); (bce(out,yb)+dice(out,yb)).backward(); opt.step()
    m.eval(); return m
def main():
    print(f"봄 10폴드 LORO (고정thr={FIXED}) — 단일 deeplab rehearsal",flush=True)
    fx=[]; bs=[]
    for i,h in enumerate(USABLE):
        m=train_fold(h)
        x,gt,vd=S.cir_load(h); H,W=gt.shape; prob=S.infer(m,x,H,W)
        P=prob[vd]; Y=gt[vd].astype(int)
        fixed=f1_at(P,Y,FIXED); best=S.bestf1(P,Y)
        fx.append(fixed); bs.append(best)
        print(f"  [{i+1}/10] {h:20s} 고정F1={fixed:.3f} bestF1={best:.3f}",flush=True)
    print(f"\n★ 봄 LORO 10폴드: 고정thr{FIXED} 평균={np.mean(fx):.3f}±{np.std(fx):.3f} | best 평균={np.mean(bs):.3f}±{np.std(bs):.3f}",flush=True)
    print(f"  (4-홀드아웃 앙상블 0.764 대비 — 단일모델·10폴드로 로버스트니스 확인)",flush=True)
if __name__=="__main__": main()
