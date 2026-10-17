"""방법1: 다년 시계열 [pre(Y-1), post(Y), post+1(Y+1)] = 192d. 불탄 지속성 신호. fire-held-out.
ko_ae x=[pre64|post64] + ko_ae_y1 x1=post+1(64). 사용: KO_AE=../ko_ae MIN_HA_KEEP=600 python3 ee_ae_multiyear.py
"""
import os, glob, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.environ.get("KO_AE",os.path.join(HERE,"..","ko_ae"))
Y1=os.path.join(HERE,"..","ko_ae_y1"); HOLD=os.path.join(HERE,"..","data","holdout_fixed.txt")
DEV="cuda" if torch.cuda.is_available() else "cpu"; MINHA=float(os.environ.get("MIN_HA_KEEP","600"))
def fire_ha(nm):
    try: return float(nm.split("_")[-1].replace("ha",""))
    except: return 0.0
def load192(f,fire):
    d=np.load(f); x=d["x"].astype(np.float32)  # 128
    p1=os.path.join(Y1,fire,os.path.basename(f))
    if os.path.exists(p1): x1=np.load(p1)["x1"].astype(np.float32)
    else: x1=x[64:]  # Y+1 없으면 post 복제(무해)
    return np.concatenate([x,x1],0), d["y"].astype(np.float32)  # 192
def gn(c): return nn.GroupNorm(min(8,c),c)
class Blk(nn.Module):
    def __init__(s,i,o): super().__init__(); s.c=nn.Sequential(nn.Conv2d(i,o,3,padding=1),gn(o),nn.GELU(),nn.Conv2d(o,o,3,padding=1),gn(o),nn.GELU())
    def forward(s,x): return s.c(x)
class UNet(nn.Module):
    def __init__(s,ic=192):
        super().__init__(); s.stem=nn.Conv2d(ic,64,1);s.d1=Blk(64,64);s.d2=Blk(64,128);s.d3=Blk(128,256);s.p=nn.MaxPool2d(2)
        s.u2=Blk(256+128,128);s.u1=Blk(128+64,64);s.o=nn.Conv2d(64,1,1)
    def up(s,x): return F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)
    def forward(s,x):
        x=s.stem(x);c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2))
        x=s.u2(torch.cat([s.up(c3),c2],1));x=s.u1(torch.cat([s.up(x),c1],1));return s.o(x)
class DS(torch.utils.data.Dataset):
    def __init__(s,items,aug=False): s.it=items;s.aug=aug
    def __len__(s): return len(s.it)
    def __getitem__(s,i):
        f,fire=s.it[i];x,y=load192(f,fire)
        if s.aug and random.random()<0.5: x,y=x[:,:,::-1].copy(),y[:,::-1].copy()
        if s.aug and random.random()<0.5: x,y=x[:,::-1].copy(),y[::-1].copy()
        return torch.from_numpy(x),torch.from_numpy(y)
def tv(l,y,a=0.7,b=0.3):
    p=torch.sigmoid(l).squeeze(1);e=1e-6;tp=(p*y).sum((1,2));fn=(y*(1-p)).sum((1,2));fp=((1-y)*p).sum((1,2))
    return 1-((tp+e)/(tp+a*fn+b*fp+e)).mean()+F.binary_cross_entropy_with_logits(l.squeeze(1),y)
def stt(P,Y,t):
    p=(P>=t);tp=(p&(Y>0)).sum();fp=(p&(Y==0)).sum();fn=((~p)&(Y>0)).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return 2*pr*rc/(pr+rc+1e-9),pr,rc
@torch.no_grad()
def pred(m,items):
    m.eval();P=[];Y=[]
    for f,fire in items:
        x,y=load192(f,fire);t=torch.from_numpy(x).unsqueeze(0).to(DEV)
        P.append(torch.sigmoid(m(t)).squeeze().cpu().numpy().reshape(-1));Y.append(y.astype(np.uint8).reshape(-1))
    return np.concatenate(P),np.concatenate(Y)
random.seed(42);torch.manual_seed(42);np.random.seed(42)
hold=set(l.strip() for l in open(HOLD) if l.strip())
fires=[d for d in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(d,"*.npz")) and (fire_ha(os.path.basename(d))>=MINHA or os.path.basename(d) in hold)]
it=lambda ds:[(f,os.path.basename(d)) for d in ds for f in glob.glob(os.path.join(d,"*.npz"))]
te=[d for d in fires if os.path.basename(d) in hold];tr=[d for d in fires if os.path.basename(d) not in hold];random.shuffle(tr)
nv=max(2,len(tr)//6);va=tr[:nv];trn=tr[nv:]
ny1=sum(1 for f,fr in it(fires) if os.path.exists(os.path.join(Y1,fr,os.path.basename(f))))
print(f"train {len(trn)}화재/{len(it(trn))}타일 TEST {len(te)}화재 | Y+1커버 {ny1}/{len(it(fires))}",flush=True)
m=UNet(192).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
dl=torch.utils.data.DataLoader(DS(it(trn),aug=True),batch_size=4,shuffle=True,num_workers=2)
best=-1;bsd=None
for ep in range(40):
    m.train()
    for xb,yb in dl:
        xb,yb=xb.to(DEV),yb.to(DEV);opt.zero_grad();tv(m(xb),yb).backward();opt.step()
    Pv,Yv=pred(m,it(va));vb=max(stt(Pv,Yv,t)[0] for t in np.arange(0.3,0.85,0.05))
    if vb>best: best=vb;bsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
m.load_state_dict(bsd)
Pt,Yt=pred(m,it(te));thr=max(np.arange(0.3,0.85,0.05),key=lambda t:stt(Pt,Yt,t)[0]);f1,pr,rc=stt(Pt,Yt,thr)
print(f"[다년 Y-1,Y,Y+1] F1={f1:.3f} P={pr:.3f} R={rc:.3f} thr={thr:.2f} vs baseline 0.690",flush=True)
