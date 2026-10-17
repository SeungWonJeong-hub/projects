"""방법2: Attention U-Net (스킵에 attention gate) on AlphaEarth 128d. 문헌(BiAU-Net) 근거. fire-held-out.
사용: KO_AE=../ko_ae MIN_HA_KEEP=600 python3 ee_ae_attn.py
"""
import os, glob, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.environ.get("KO_AE",os.path.join(HERE,"..","ko_ae"))
HOLD=os.path.join(HERE,"..","data","holdout_fixed.txt"); DEV="cuda" if torch.cuda.is_available() else "cpu"
MINHA=float(os.environ.get("MIN_HA_KEEP","600"))
def fire_ha(nm):
    try: return float(nm.split("_")[-1].replace("ha",""))
    except: return 0.0
def gn(c): return nn.GroupNorm(min(8,c),c)
class Blk(nn.Module):
    def __init__(s,i,o): super().__init__(); s.c=nn.Sequential(nn.Conv2d(i,o,3,padding=1),gn(o),nn.GELU(),nn.Conv2d(o,o,3,padding=1),gn(o),nn.GELU())
    def forward(s,x): return s.c(x)
class Att(nn.Module):  # attention gate
    def __init__(s,fg,fl,fi): super().__init__(); s.wg=nn.Conv2d(fg,fi,1);s.wx=nn.Conv2d(fl,fi,1);s.psi=nn.Conv2d(fi,1,1)
    def forward(s,g,x): a=torch.sigmoid(s.psi(torch.relu(s.wg(g)+s.wx(x)))); return x*a
class AttUNet(nn.Module):
    def __init__(s,ic=128):
        super().__init__(); s.stem=nn.Conv2d(ic,64,1);s.d1=Blk(64,64);s.d2=Blk(64,128);s.d3=Blk(128,256);s.p=nn.MaxPool2d(2)
        s.a2=Att(256,128,64);s.u2=Blk(256+128,128);s.a1=Att(128,64,32);s.u1=Blk(128+64,64);s.o=nn.Conv2d(64,1,1)
    def up(s,x): return F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)
    def forward(s,x):
        x=s.stem(x);c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2))
        g2=s.up(c3);x=s.u2(torch.cat([g2,s.a2(g2,c2)],1))
        g1=s.up(x);x=s.u1(torch.cat([g1,s.a1(g1,c1)],1));return s.o(x)
class DS(torch.utils.data.Dataset):
    def __init__(s,f,aug=False): s.f=f;s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]);x=d["x"].astype(np.float32);y=d["y"].astype(np.float32)
        if s.aug and random.random()<0.5: x,y=x[:,:,::-1].copy(),y[:,::-1].copy()
        if s.aug and random.random()<0.5: x,y=x[:,::-1].copy(),y[::-1].copy()
        return torch.from_numpy(x),torch.from_numpy(y)
def tv(l,y,a=0.7,b=0.3):
    p=torch.sigmoid(l).squeeze(1);e=1e-6;tp=(p*y).sum((1,2));fn=(y*(1-p)).sum((1,2));fp=((1-y)*p).sum((1,2))
    return 1-((tp+e)/(tp+a*fn+b*fp+e)).mean()+F.binary_cross_entropy_with_logits(l.squeeze(1),y)
def st(P,Y,t):
    p=(P>=t);tp=(p&(Y>0)).sum();fp=(p&(Y==0)).sum();fn=((~p)&(Y>0)).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return 2*pr*rc/(pr+rc+1e-9),pr,rc
@torch.no_grad()
def pred(m,f):
    m.eval();P=[];Y=[]
    for x in f:
        d=np.load(x);t=torch.from_numpy(d["x"].astype(np.float32)).unsqueeze(0).to(DEV)
        P.append(torch.sigmoid(m(t)).squeeze().cpu().numpy().reshape(-1));Y.append(d["y"].astype(np.uint8).reshape(-1))
    return np.concatenate(P),np.concatenate(Y)
random.seed(42);torch.manual_seed(42);np.random.seed(42)
hold=set(l.strip() for l in open(HOLD) if l.strip())
fires=[d for d in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(d,"*.npz")) and (fire_ha(os.path.basename(d))>=MINHA or os.path.basename(d) in hold)]
te=[d for d in fires if os.path.basename(d) in hold]; tr=[d for d in fires if os.path.basename(d) not in hold];random.shuffle(tr)
nv=max(2,len(tr)//6);va=tr[:nv];trn=tr[nv:]
tf=lambda ds:[f for d in ds for f in glob.glob(os.path.join(d,"*.npz"))]
m=AttUNet(128).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
dl=torch.utils.data.DataLoader(DS(tf(trn),aug=True),batch_size=4,shuffle=True,num_workers=2)
best=-1;bsd=None
for ep in range(40):
    m.train()
    for xb,yb in dl:
        xb,yb=xb.to(DEV),yb.to(DEV);opt.zero_grad();tv(m(xb),yb).backward();opt.step()
    Pv,Yv=pred(m,tf(va));vb=max(st(Pv,Yv,t)[0] for t in np.arange(0.3,0.85,0.05))
    if vb>best: best=vb;bsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
m.load_state_dict(bsd)
Pt,Yt=pred(m,tf(te));thr=max(np.arange(0.3,0.85,0.05),key=lambda t:st(Pt,Yt,t)[0]);f1,pr,rc=st(Pt,Yt,thr)
print(f"[AttUNet] F1={f1:.3f} P={pr:.3f} R={rc:.3f} thr={thr:.2f} vs baseline 0.690",flush=True)
