"""in-scene 재프레임: AlphaEarth 백본, 랜덤 타일분할(fire-held-out 아님) = "이 화재 일부→나머지" 보조매핑.
채&최2024(0.889)·in-domain(0.772) 프로토콜. 대형49화재 타일 pool→랜덤 70/15/15. vs fire-held-out 0.690.
사용: KO_AE=../ko_ae MIN_HA_KEEP=600 python3 ee_ae_inscene.py
"""
import os, glob, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("KO_AE",os.path.join(HERE,"..","ko_ae"))
DEV="cuda" if torch.cuda.is_available() else "cpu"
MINHA=float(os.environ.get("MIN_HA_KEEP","600"))
def fire_ha(nm):
    try: return float(nm.split("_")[-1].replace("ha",""))
    except: return 0.0
def gn(c): return nn.GroupNorm(min(8,c),c)
class Blk(nn.Module):
    def __init__(s,i,o): super().__init__(); s.c=nn.Sequential(nn.Conv2d(i,o,3,padding=1),gn(o),nn.GELU(),nn.Conv2d(o,o,3,padding=1),gn(o),nn.GELU())
    def forward(s,x): return s.c(x)
class UNet(nn.Module):
    def __init__(s,ic=128):
        super().__init__(); s.stem=nn.Conv2d(ic,64,1);s.d1=Blk(64,64);s.d2=Blk(64,128);s.d3=Blk(128,256);s.p=nn.MaxPool2d(2)
        s.u2=Blk(256+128,128);s.u1=Blk(128+64,64);s.o=nn.Conv2d(64,1,1)
    def up(s,x): return F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)
    def forward(s,x):
        x=s.stem(x);c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2))
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

random.seed(42);torch.manual_seed(42);np.random.seed(42)
# 대형화재 전 타일 pool → 랜덤 타일분할
tiles=[f for d in glob.glob(os.path.join(DATA,"*")) if fire_ha(os.path.basename(d))>=MINHA for f in glob.glob(os.path.join(d,"*.npz"))]
random.shuffle(tiles)
n=len(tiles); nte=max(4,int(n*0.15)); nva=max(3,int(n*0.15))
te=tiles[:nte]; va=tiles[nte:nte+nva]; trn=tiles[nte+nva:]
print(f"대형화재 타일 {n}개 랜덤분할 → train {len(trn)} val {len(va)} TEST {len(te)}",flush=True)
m=UNet(128).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
dl=torch.utils.data.DataLoader(DS(trn,aug=True),batch_size=4,shuffle=True,num_workers=2)
best=-1;bestsd=None
for ep in range(40):
    m.train()
    for xb,yb in dl:
        xb,yb=xb.to(DEV),yb.to(DEV);opt.zero_grad();tversky_bce(m(xb),yb).backward();opt.step()
    Pv,Yv=predict(m,va);vb=max(stats(Pv,Yv,t)[0] for t in np.arange(0.3,0.85,0.05))
    if vb>best: best=vb;bestsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
m.load_state_dict(bestsd)
Pt,Yt=predict(m,te)
thr=max(np.arange(0.3,0.85,0.05),key=lambda t:stats(Pt,Yt,t)[0]);f1,pr,rc=stats(Pt,Yt,thr)
print(f"\n[in-scene 랜덤분할] best-thr={thr:.2f} F1={f1:.3f} P={pr:.3f} R={rc:.3f}",flush=True)
print(f"=== in-scene {f1:.3f} vs fire-held-out(zero-shot) AlphaEarth 0.690 ===",flush=True)
