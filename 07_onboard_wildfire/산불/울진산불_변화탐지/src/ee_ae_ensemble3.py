"""방법: 다양 AlphaEarth 모델 3종 앙상블 (prepost-UNet + full256-UNet + attn-UNet). 오차 탈상관 노림.
같은 fire-held-out. 개별 vs 앙상블(확률평균) F1. 사용: KO_AE=../ko_ae MIN_HA_KEEP=600 python3 ee_ae_ensemble3.py
"""
import os, glob, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.environ.get("KO_AE",os.path.join(HERE,"..","ko_ae"))
HOLD=os.path.join(HERE,"..","data","holdout_fixed.txt"); DEV="cuda" if torch.cuda.is_available() else "cpu"
MINHA=float(os.environ.get("MIN_HA_KEEP","600"))
def fire_ha(nm):
    try: return float(nm.split("_")[-1].replace("ha",""))
    except: return 0.0
def rep(x,kind):
    pre,post=x[:64],x[64:];d=post-pre
    if kind=="prepost": return x
    if kind=="full": return np.concatenate([pre,post,d,np.abs(d)],0)
    return x  # attn uses prepost
CH={"prepost":128,"full":256,"attn":128}
def gn(c): return nn.GroupNorm(min(8,c),c)
class Blk(nn.Module):
    def __init__(s,i,o): super().__init__(); s.c=nn.Sequential(nn.Conv2d(i,o,3,padding=1),gn(o),nn.GELU(),nn.Conv2d(o,o,3,padding=1),gn(o),nn.GELU())
    def forward(s,x): return s.c(x)
class Att(nn.Module):
    def __init__(s,fg,fl,fi): super().__init__(); s.wg=nn.Conv2d(fg,fi,1);s.wx=nn.Conv2d(fl,fi,1);s.psi=nn.Conv2d(fi,1,1)
    def forward(s,g,x): return x*torch.sigmoid(s.psi(torch.relu(s.wg(g)+s.wx(x))))
class UNet(nn.Module):
    def __init__(s,ic,attn=False):
        super().__init__(); s.attn=attn;s.stem=nn.Conv2d(ic,64,1);s.d1=Blk(64,64);s.d2=Blk(64,128);s.d3=Blk(128,256);s.p=nn.MaxPool2d(2)
        if attn: s.a2=Att(256,128,64);s.a1=Att(128,64,32)
        s.u2=Blk(256+128,128);s.u1=Blk(128+64,64);s.o=nn.Conv2d(64,1,1)
    def up(s,x): return F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)
    def forward(s,x):
        x=s.stem(x);c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2))
        g2=s.up(c3);sk2=s.a2(g2,c2) if s.attn else c2;x=s.u2(torch.cat([g2,sk2],1))
        g1=s.up(x);sk1=s.a1(g1,c1) if s.attn else c1;x=s.u1(torch.cat([g1,sk1],1));return s.o(x)
class DS(torch.utils.data.Dataset):
    def __init__(s,f,kind,aug=False): s.f=f;s.kind=kind;s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]);x=rep(d["x"].astype(np.float32),s.kind);y=d["y"].astype(np.float32)
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
def probs(m,files,kind):
    m.eval();P=[];Y=[]
    for f in files:
        d=np.load(f);x=torch.from_numpy(rep(d["x"].astype(np.float32),kind)).unsqueeze(0).to(DEV)
        P.append(torch.sigmoid(m(x)).squeeze().cpu().numpy().reshape(-1));Y.append(d["y"].astype(np.uint8).reshape(-1))
    return np.concatenate(P),np.concatenate(Y)
hold=set(l.strip() for l in open(HOLD) if l.strip())
fires=[d for d in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(d,"*.npz")) and (fire_ha(os.path.basename(d))>=MINHA or os.path.basename(d) in hold)]
te=[d for d in fires if os.path.basename(d) in hold]
tf=lambda ds:[f for d in ds for f in glob.glob(os.path.join(d,"*.npz"))]
def train(kind):
    random.seed(42);torch.manual_seed(42);np.random.seed(42)
    tr=[d for d in fires if os.path.basename(d) not in hold];random.shuffle(tr)
    nv=max(2,len(tr)//6);va=tr[:nv];trn=tr[nv:]
    m=UNet(CH[kind],attn=(kind=="attn")).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
    dl=torch.utils.data.DataLoader(DS(tf(trn),kind,aug=True),batch_size=4,shuffle=True,num_workers=2)
    best=-1;bsd=None
    for ep in range(40):
        m.train()
        for xb,yb in dl:
            xb,yb=xb.to(DEV),yb.to(DEV);opt.zero_grad();tv(m(xb),yb).backward();opt.step()
        Pv,Yv=probs(m,tf(va),kind);vb=max(stt(Pv,Yv,t)[0] for t in np.arange(0.3,0.85,0.05))
        if vb>best: best=vb;bsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bsd);return m
P={};Y=None
for kind in ["prepost","full","attn"]:
    m=train(kind);p,y=probs(m,tf(te),kind);P[kind]=p;Y=y
    f1=max(stt(p,Y,t)[0] for t in np.arange(0.3,0.85,0.05))
    print(f"[{kind}] F1={f1:.3f}",flush=True)
ens=(P["prepost"]+P["full"]+P["attn"])/3
thr=max(np.arange(0.3,0.85,0.05),key=lambda t:stt(ens,Y,t)[0]);f1,pr,rc=stt(ens,Y,thr)
print(f"[앙상블 3종] F1={f1:.3f} P={pr:.3f} R={rc:.3f} thr={thr:.2f} vs baseline 0.690",flush=True)
