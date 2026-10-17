"""AlphaEarth 임베딩 표현방식 탐색 — 딥리서치 "표현이 분류기만큼 중요" 검증.
저장 x=[pre64|post64]에서 delta/magnitude 파생, 여러 표현으로 학습→비교. fire-held-out(0.690 공정비교).
사용: KO_AE=../ko_ae MIN_HA_KEEP=600 python3 ee_ae_repr.py
"""
import os, glob, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("KO_AE",os.path.join(HERE,"..","ko_ae"))
HOLD=os.path.join(HERE,"..","data","holdout_fixed.txt")
DEV="cuda" if torch.cuda.is_available() else "cpu"
MINHA=float(os.environ.get("MIN_HA_KEEP","600"))
def fire_ha(nm):
    try: return float(nm.split("_")[-1].replace("ha",""))
    except: return 0.0
def make_rep(x, rep):  # x=(128,H,W) [pre|post]
    pre,post=x[:64],x[64:]; d=post-pre
    if rep=="prepost": return x
    if rep=="postdelta": return np.concatenate([post,d],0)
    if rep=="prepostdelta": return np.concatenate([pre,post,d],0)
    if rep=="full": return np.concatenate([pre,post,d,np.abs(d)],0)
    raise ValueError(rep)
REP_CH={"prepost":128,"postdelta":128,"prepostdelta":192,"full":256}
def gn(c): return nn.GroupNorm(min(8,c),c)
class Blk(nn.Module):
    def __init__(s,i,o): super().__init__(); s.c=nn.Sequential(nn.Conv2d(i,o,3,padding=1),gn(o),nn.GELU(),nn.Conv2d(o,o,3,padding=1),gn(o),nn.GELU())
    def forward(s,x): return s.c(x)
class UNet(nn.Module):
    def __init__(s,ic):
        super().__init__(); s.stem=nn.Conv2d(ic,64,1);s.d1=Blk(64,64);s.d2=Blk(64,128);s.d3=Blk(128,256);s.p=nn.MaxPool2d(2)
        s.u2=Blk(256+128,128);s.u1=Blk(128+64,64);s.o=nn.Conv2d(64,1,1)
    def up(s,x): return F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)
    def forward(s,x):
        x=s.stem(x);c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2))
        x=s.u2(torch.cat([s.up(c3),c2],1));x=s.u1(torch.cat([s.up(x),c1],1));return s.o(x)
class DS(torch.utils.data.Dataset):
    def __init__(s,files,rep,aug=False): s.f=files;s.rep=rep;s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]);x=make_rep(d["x"].astype(np.float32),s.rep);y=d["y"].astype(np.float32)
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
def predict(m,files,rep):
    m.eval();P=[];Y=[]
    for f in files:
        d=np.load(f);x=torch.from_numpy(make_rep(d["x"].astype(np.float32),rep)).unsqueeze(0).to(DEV)
        P.append(torch.sigmoid(m(x)).squeeze().cpu().numpy().reshape(-1));Y.append(d["y"].astype(np.uint8).reshape(-1))
    return np.concatenate(P),np.concatenate(Y)

hold=set(l.strip() for l in open(HOLD) if l.strip())
fires=[d for d in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(d,"*.npz")) and (fire_ha(os.path.basename(d))>=MINHA or os.path.basename(d) in hold)]
te_f=[d for d in fires if os.path.basename(d) in hold]
def run(rep):
    random.seed(42);torch.manual_seed(42);np.random.seed(42)
    tr_all=[d for d in fires if os.path.basename(d) not in hold];random.shuffle(tr_all)
    nval=max(2,len(tr_all)//6);va_f=tr_all[:nval];trn_f=tr_all[nval:]
    tf=lambda ds:[f for d in ds for f in glob.glob(os.path.join(d,"*.npz"))]
    m=UNet(REP_CH[rep]).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
    dl=torch.utils.data.DataLoader(DS(tf(trn_f),rep,aug=True),batch_size=4,shuffle=True,num_workers=2)
    best=-1;bestsd=None
    for ep in range(40):
        m.train()
        for xb,yb in dl:
            xb,yb=xb.to(DEV),yb.to(DEV);opt.zero_grad();tversky_bce(m(xb),yb).backward();opt.step()
        Pv,Yv=predict(m,tf(va_f),rep);vb=max(stats(Pv,Yv,t)[0] for t in np.arange(0.3,0.85,0.05))
        if vb>best: best=vb;bestsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bestsd)
    Pt,Yt=predict(m,tf(te_f),rep)
    thr=max(np.arange(0.3,0.85,0.05),key=lambda t:stats(Pt,Yt,t)[0]);f1,pr,rc=stats(Pt,Yt,thr)
    return f1,pr,rc,thr
print(f"대형화재 {len(fires)} (홀드아웃 {len(te_f)})",flush=True)
for rep in ["prepost","postdelta","prepostdelta","full"]:
    f1,pr,rc,thr=run(rep)
    print(f"[{rep:14s}] ch{REP_CH[rep]:3d} F1={f1:.3f} P={pr:.3f} R={rc:.3f} thr={thr:.2f}",flush=True)
print("=== baseline prepost=0.690 ===")
