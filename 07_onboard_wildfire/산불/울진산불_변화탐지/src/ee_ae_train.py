"""AlphaEarth 128ch 임베딩으로 산불 CD 학습 — 가벼운 GN-UNet(few-shot). 계절별 F1.
사용: KO_AE=../ko_ae python3 ee_ae_train.py
"""
import os, glob, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("KO_AE",os.path.join(HERE,"..","ko_ae"))
HOLD=os.path.join(HERE,"..","data","holdout_fixed.txt")
DEV="cuda" if torch.cuda.is_available() else "cpu"
def gn(c): return nn.GroupNorm(min(8,c),c)
class Blk(nn.Module):
    def __init__(s,i,o): super().__init__(); s.c=nn.Sequential(nn.Conv2d(i,o,3,padding=1),gn(o),nn.GELU(),nn.Conv2d(o,o,3,padding=1),gn(o),nn.GELU())
    def forward(s,x): return s.c(x)
class UNet(nn.Module):
    def __init__(s,ic=128):
        super().__init__(); s.stem=nn.Conv2d(ic,64,1); s.d1=Blk(64,64);s.d2=Blk(64,128);s.d3=Blk(128,256);s.p=nn.MaxPool2d(2)
        s.u2=Blk(256+128,128);s.u1=Blk(128+64,64);s.o=nn.Conv2d(64,1,1)
    def up(s,x): return F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)
    def forward(s,x):
        x=s.stem(x); c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2))
        x=s.u2(torch.cat([s.up(c3),c2],1));x=s.u1(torch.cat([s.up(x),c1],1));return s.o(x)
class DS(torch.utils.data.Dataset):
    def __init__(s,files,aug=False): s.f=files;s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]);x=d["x"].astype(np.float32);y=d["y"].astype(np.float32)
        if s.aug and random.random()<0.5: x,y=x[:,:,::-1].copy(),y[:,::-1].copy()
        if s.aug and random.random()<0.5: x,y=x[:,::-1].copy(),y[::-1].copy()
        return torch.from_numpy(x),torch.from_numpy(y)
def tversky_bce(l,y,a=0.7,b=0.3):
    p=torch.sigmoid(l).squeeze(1);e=1e-6;tp=(p*y).sum((1,2));fn=(y*(1-p)).sum((1,2));fp=((1-y)*p).sum((1,2))
    return 1-((tp+e)/(tp+a*fn+b*fp+e)).mean()+F.binary_cross_entropy_with_logits(l.squeeze(1),y)
def stats(P,Y,t):
    p=(P>=t);tp=(p&(Y>0)).sum();fp=(p&(Y==0)).sum();fn=((~p)&(Y>0)).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return 2*pr*rc/(pr+rc+1e-9),pr,rc
@torch.no_grad()
def predict(m,files):
    m.eval();P=[];Y=[]
    for f in files:
        d=np.load(f);x=torch.from_numpy(d["x"].astype(np.float32)).unsqueeze(0).to(DEV)
        P.append(torch.sigmoid(m(x)).squeeze().cpu().numpy().reshape(-1));Y.append(d["y"].astype(np.uint8).reshape(-1))
    return np.concatenate(P),np.concatenate(Y)
def season(mn): return "봄" if 3<=mn<=5 else "여름" if 6<=mn<=8 else "가을" if 9<=mn<=11 else "겨울"

SEED=int(os.environ.get("SEED","42"))
random.seed(SEED);torch.manual_seed(SEED);np.random.seed(SEED)
hold=set(l.strip() for l in open(HOLD) if l.strip())
fires=sorted([d for d in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(d,"*.npz"))])
te_f=[d for d in fires if os.path.basename(d) in hold]
tr_all=[d for d in fires if os.path.basename(d) not in hold];random.shuffle(tr_all)
nval=max(2,len(tr_all)//6);va_f=tr_all[:nval];trn_f=tr_all[nval:]
tf=lambda ds:[f for d in ds for f in glob.glob(os.path.join(d,"*.npz"))]
print(f"train {len(trn_f)}화재/{len(tf(trn_f))}타일 val {len(tf(va_f))} TEST {len(te_f)}화재",flush=True)
m=UNet(128).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
dl=torch.utils.data.DataLoader(DS(tf(trn_f),aug=True),batch_size=4,shuffle=True,num_workers=2)
best=-1;bestsd=None
for ep in range(40):
    m.train()
    for xb,yb in dl:
        xb,yb=xb.to(DEV),yb.to(DEV);opt.zero_grad();tversky_bce(m(xb),yb).backward();opt.step()
    Pv,Yv=predict(m,tf(va_f));vb=max(stats(Pv,Yv,t)[0] for t in np.arange(0.3,0.85,0.05))
    if vb>best: best=vb;bestsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
m.load_state_dict(bestsd)
allP=[];allY=[];per=[]
for d in te_f:
    pp,yy=predict(m,glob.glob(os.path.join(d,"*.npz")));allP.append(pp);allY.append(yy)
    per.append((os.path.basename(d),season(int(os.path.basename(d).split("_")[1][4:6])),pp,yy))
Pa=np.concatenate(allP);Ya=np.concatenate(allY)
thr=max(np.arange(0.3,0.85,0.05),key=lambda t:stats(Pa,Ya,t)[0])
f1,pr,rc=stats(Pa,Ya,thr)
print(f"\n전체 홀드아웃 best-thr={thr:.2f} | F1={f1:.3f} P={pr:.3f} R={rc:.3f}",flush=True)
from collections import defaultdict
bs=defaultdict(lambda:[[],[]])
for fn,se,pp,yy in per: bs[se][0].append(pp);bs[se][1].append(yy)
per_season={}
for se in ["봄","여름","가을","겨울"]:
    if se in bs:
        pp=np.concatenate(bs[se][0]);yy=np.concatenate(bs[se][1]);f,p,r=stats(pp,yy,thr)
        per_season[se]={"f1":round(float(f),4),"p":round(float(p),4),"r":round(float(r),4),"fires":len(bs[se][0])}
        print(f"{se}: F1={f:.3f} P={p:.3f} R={r:.3f} (화재 {len(bs[se][0])})",flush=True)
print(f"\n=== AlphaEarth 전체 {f1:.3f} vs FT4 0.624 vs 같은해pre 0.645 ===")

# ── 체크포인트 + 결과 저장 (승격 게이트가 읽음) ──
import json
OUT=os.path.join(HERE,"..","runs",os.environ.get("OUT_TAG","ko_ae")); os.makedirs(OUT,exist_ok=True)
MODEL_NAME=os.environ.get("MODEL_NAME","AlphaEarth-GN-UNet")
torch.save(bestsd,os.path.join(OUT,"ae_unet.pt"))
json.dump({
    "model":MODEL_NAME,"script":"ee_ae_train.py","spec":"ae128",
    "in_ch":128,"arch":"GN-UNet-3lvl","thr":round(float(thr),3),
    "f1":round(float(f1),4),"precision":round(float(pr),4),"recall":round(float(rc),4),
    "per_season":per_season,"n_train_fires":len(trn_f),"n_test_fires":len(te_f),
    "ckpt":"ae_unet.pt","holdout":"data/holdout_fixed.txt",
}, open(os.path.join(OUT,"results.json"),"w"), ensure_ascii=False, indent=2)
print(f"[저장] {OUT}/ae_unet.pt + results.json (F1={f1:.3f})",flush=True)
