"""Step2b: ET-02 9모델 앙상블 전체 CIR-GT 파인튜닝 → 앙상블 cross-fire 재평가.
6화재 학습→4미학습 홀드아웃. baseline 앙상블(원본9) vs 파인튜닝 앙상블(FT9) 비교.
FT 가중치 9개 저장(배포용). loss forward 1회(step2 double-forward 수정)·epoch 20.
사용: python step2b_ensemble_ft.py
"""
import os, glob, random, numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
FIRES=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires")
WEIGHTS=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/웹앱/weights")
OUT=os.path.join(PROJ,"runs","cir_ft_ens"); os.makedirs(OUT,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
SEED=0; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
ENS=[("unet","et02_unet_s1"),("unet","et02_unet_s2"),("unet","et02_unet_s3"),
     ("fpn","et02_fpn_s1"),("fpn","et02_fpn_s2"),("fpn","et02_fpn_s3"),
     ("deeplabv3plus","et02_deeplabv3plus_s1"),("deeplabv3plus","et02_deeplabv3plus_s2"),("deeplabv3plus","et02_deeplabv3plus_s3")]
ARCH={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}
HOLD=["250324_안동길안면","250321_산청시천면","230402_홍성서부면","220410_양구양구읍"]
TRAIN=["250325_경북영덕청송","250322_의성안평면","250325_영덕축산면","220304_울진북면","220528_울진근남면","230403_영주평은면"]
EPOCHS=int(os.environ.get("EPOCHS","20")); T=128

def load(name):
    d=np.load(os.path.join(FIRES,name,"gt_stac.npz")); return d["x"].astype("float32"),d["gt"].astype(bool),d["valid"].astype(bool)
def tiles_from(name,stride=64,minvalid=0.5):
    x,gt,vd=load(name); C,H,W=x.shape; out=[]
    ys=sorted(set(min(y,H-T) for y in range(0,max(H-T,0)+1,stride))) or [0]
    xs=sorted(set(min(z,W-T) for z in range(0,max(W-T,0)+1,stride))) or [0]
    for y in ys:
        for z in xs:
            xt=x[:,y:y+T,z:z+T]; yt=gt[y:y+T,z:z+T]; vt=vd[y:y+T,z:z+T]
            if xt.shape[1]<T or xt.shape[2]<T or vt.mean()<minvalid: continue
            out.append((xt.copy(),yt.copy()))
    return out
def build(dec,name):
    m=ARCH[dec](encoder_name="resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV)
    m.load_state_dict(torch.load(os.path.join(WEIGHTS,"et02_ens",name+".pt"),map_location=DEV)); return m
@torch.no_grad()
def ens_prob(models,x,H,W):
    for m in models: m.eval()
    prob=np.zeros((H,W),"float32")
    for i in range(0,H,256):
        for j in range(0,W,256):
            p=x[:,i:i+256,j:j+256]; ph,pw=p.shape[1],p.shape[2]
            pad=np.zeros((8,256,256),"float32"); pad[:,:ph,:pw]=p
            xt=torch.from_numpy(pad[None]).to(DEV)
            o=np.mean([torch.sigmoid(m(xt))[0,0].cpu().numpy() for m in models],0)
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
def eval_ens(models,tag):
    print(f"\n=== 앙상블 홀드아웃 ({tag}) ===",flush=True)
    tot=[]; tot2=[]
    for h in HOLD:
        x,gt,vd=load(h); C,H,W=x.shape; prob=ens_prob(models,x,H,W)
        f,t,p,r=bestf1(prob,gt,vd); f2,p2,r2=prf(prob>=0.2,gt,vd)
        tot.append(f); tot2.append(f2)
        print(f"  {h:20s} best F1={f:.3f}(P{p:.2f}/R{r:.2f},t{t:.2f}) thr0.2 F1={f2:.3f}",flush=True)
    print(f"  평균 best-F1={np.mean(tot):.3f}  thr0.2={np.mean(tot2):.3f}",flush=True)
    return np.mean(tot)

def finetune(m,tiles,epochs):
    m.train(); opt=torch.optim.AdamW(m.parameters(),lr=5e-5,weight_decay=1e-4)
    bce=nn.BCEWithLogitsLoss(); dice=DiceLoss(mode="binary"); BS=8
    for ep in range(epochs):
        random.shuffle(tiles); xs_,ys_=[],[]
        for xt,yt in tiles:
            k=random.randint(0,3); fl=random.random()<0.5
            xa=np.rot90(xt,k,axes=(1,2)); ya=np.rot90(yt,k)
            if fl: xa=xa[:,:,::-1]; ya=ya[:,::-1]
            xs_.append(xa.copy()); ys_.append(ya.astype("float32").copy())
        for i in range(0,len(xs_),BS):
            xb=torch.from_numpy(np.stack(xs_[i:i+BS])).to(DEV); yb=torch.from_numpy(np.stack(ys_[i:i+BS]))[:,None].to(DEV)
            if xb.shape[0]<2: continue
            opt.zero_grad(); out=m(xb); (bce(out,yb)+dice(out,yb)).backward(); opt.step()   # forward 1회
    m.eval(); return m

def main():
    tiles=[]
    for n in TRAIN: tiles+=tiles_from(n)
    print(f"train {len(TRAIN)}화재 {len(tiles)}타일 / holdout {len(HOLD)}",flush=True)
    base=[build(d,n) for d,n in ENS]
    b=eval_ens(base,"baseline 원본9")
    print("\n9모델 파인튜닝...",flush=True)
    ft=[]
    for idx,(d,n) in enumerate(ENS):
        m=finetune(build(d,n),tiles,EPOCHS)
        torch.save(m.state_dict(),os.path.join(OUT,n+"_ft.pt")); ft.append(m)
        print(f"  [{idx+1}/9] {n} 완료",flush=True)
    f=eval_ens(ft,"파인튜닝 FT9")
    print(f"\n★ 앙상블 cross-fire best-F1: baseline {b:.3f} → 파인튜닝 {f:.3f} ({f-b:+.3f})",flush=True)

if __name__=="__main__": main()
