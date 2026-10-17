"""Step2: 다중화재 CIR-GT cross-fire 봄 파인튜닝 (4-홀드아웃, 1모델).
train 화재들 gt_stac.npz → 128타일 풀 → ET-02 단일모델(deeplabv3+) 파인튜닝 →
홀드아웃(미학습) 화재서 baseline vs FT의 F1 비교. 진짜 cross-fire 봄 일반화 측정.
사용: python step2_cir_finetune.py
"""
import os, glob, random, numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
FIRES=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires")
WEIGHTS=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/웹앱/weights")
DEV="cuda" if torch.cuda.is_available() else "cpu"
SEED=0; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
CKPT=os.path.join(WEIGHTS,"et02_ens","et02_deeplabv3plus_s1.pt")  # 단일 대표모델
HOLD=os.environ.get("HOLD","250324_안동길안면,250321_산청시천면,230402_홍성서부면,220410_양구양구읍").split(",")
TRAIN=["250325_경북영덕청송","250322_의성안평면","250325_영덕축산면","220304_울진북면","220528_울진근남면","230403_영주평은면"]
EPOCHS=int(os.environ.get("EPOCHS","30")); T=128

def load(name):
    f=os.path.join(FIRES,name,"gt_stac.npz")
    d=np.load(f); return d["x"].astype("float32"),d["gt"].astype(bool),d["valid"].astype(bool)
def tiles_from(name,stride=64,minvalid=0.5):
    x,gt,vd=load(name); C,H,W=x.shape; out=[]
    ys=sorted(set(min(y,H-T) for y in range(0,max(H-T,0)+1,stride))) or [0]
    xs=sorted(set(min(z,W-T) for z in range(0,max(W-T,0)+1,stride))) or [0]
    for y in ys:
        for z in xs:
            xt=x[:,y:y+T,z:z+T]; yt=gt[y:y+T,z:z+T]; vt=vd[y:y+T,z:z+T]
            if xt.shape[1]<T or xt.shape[2]<T: continue
            if vt.mean()<minvalid: continue
            out.append((xt.copy(),yt.copy()));
    return out
def build_model():
    m=smp.DeepLabV3Plus(encoder_name="resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV)
    m.load_state_dict(torch.load(CKPT,map_location=DEV)); return m
@torch.no_grad()
def infer(m,x,H,W):
    m.eval(); prob=np.zeros((H,W),"float32")
    for i in range(0,H,256):
        for j in range(0,W,256):
            p=x[:,i:i+256,j:j+256]; ph,pw=p.shape[1],p.shape[2]
            pad=np.zeros((8,256,256),"float32"); pad[:,:ph,:pw]=p
            o=torch.sigmoid(m(torch.from_numpy(pad[None]).to(DEV)))[0,0].cpu().numpy()
            prob[i:i+ph,j:j+pw]=o[:ph,:pw]
    return prob
def prf(pred,y,mm):
    pred=pred&mm; y=y&mm; tp=(pred&y).sum(); fp=(pred&~y).sum(); fn=(~pred&y).sum()
    p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); return 2*p*r/(p+r+1e-9),p,r
def bestf1(prob,gt,vd):
    b=None
    for t in np.arange(0.05,0.95,0.05):
        f,p,r=prf(prob>=t,gt,vd)
        if b is None or f>b[0]: b=(f,t,p,r)
    return b

def eval_holdout(m,tag):
    print(f"\n=== 홀드아웃 평가 ({tag}) ===",flush=True)
    tot=[]
    for h in HOLD:
        x,gt,vd=load(h); C,H,W=x.shape
        prob=infer(m,x,H,W)
        f,t,p,r=bestf1(prob,gt,vd); f2,p2,r2=prf(prob>=0.2,gt,vd)
        tot.append(f)
        print(f"  {h:20s} best F1={f:.3f}(P{p:.2f}/R{r:.2f},t{t:.2f})  thr0.2 F1={f2:.3f}",flush=True)
    print(f"  홀드아웃 평균 best-F1 = {np.mean(tot):.3f}",flush=True)
    return np.mean(tot)

def main():
    print(f"train {len(TRAIN)}화재 / holdout {len(HOLD)}화재",flush=True)
    tiles=[]
    for n in TRAIN: tiles+=tiles_from(n)
    print(f"훈련타일 {len(tiles)}개",flush=True)
    # baseline (파인튜닝 前)
    m=build_model(); base=eval_holdout(m,"baseline 파인튜닝前")
    # 파인튜닝
    m.train(); opt=torch.optim.AdamW(m.parameters(),lr=5e-5,weight_decay=1e-4)
    bce=nn.BCEWithLogitsLoss(); dice=DiceLoss(mode="binary"); BS=int(os.environ.get("BS","8"))
    print("\n파인튜닝...",flush=True)
    for ep in range(EPOCHS):
        random.shuffle(tiles); xs_,ys_=[],[]
        for xt,yt in tiles:
            k=random.randint(0,3); fl=random.random()<0.5
            xa=np.rot90(xt,k,axes=(1,2)); ya=np.rot90(yt,k)
            if fl: xa=xa[:,:,::-1]; ya=ya[:,::-1]
            xs_.append(xa.copy()); ys_.append(ya.astype("float32").copy())
        for i in range(0,len(xs_),BS):
            xb=torch.from_numpy(np.stack(xs_[i:i+BS])).to(DEV); yb=torch.from_numpy(np.stack(ys_[i:i+BS]))[:,None].to(DEV)
            if xb.shape[0]<2: continue
            opt.zero_grad(); loss=bce(m(xb),yb)+dice(m(xb),yb); loss.backward(); opt.step()
    ft=eval_holdout(m,"파인튜닝後")
    print(f"\n★ cross-fire 홀드아웃 best-F1: baseline {base:.3f} → 파인튜닝 {ft:.3f} ({ft-base:+.3f})",flush=True)
    os.makedirs(os.path.join(PROJ,"runs","cir_ft"),exist_ok=True)
    torch.save(m.state_dict(),os.path.join(PROJ,"runs","cir_ft","deeplab_ft.pt"))

if __name__=="__main__": main()
