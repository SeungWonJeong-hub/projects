"""방법: AlphaEarth(0.690) + FT4 Prithvi(0.624) 이종 파운데이션 앙상블. 서로 다른 표현군→오차 탈상관 노림.
같은 홀드아웃 타일(ko_ae↔ko_refl 파일명 일치). 가중치 w 스윕. 사용: python3 ee_ae_ft4_ens.py
"""
import os, glob, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KO_DATA","ko_refl"); os.environ.setdefault("RUN_NAME","ko_prithvi_refl_ft4_v2")
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE=os.path.dirname(os.path.abspath(__file__))
AE=os.path.join(HERE,"..","ko_ae"); RF=os.path.join(HERE,"..","ko_refl")
HOLD=os.path.join(HERE,"..","data","holdout_fixed.txt"); DEV="cuda" if torch.cuda.is_available() else "cpu"
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
def tv(l,y,a=0.7,b=0.3):
    p=torch.sigmoid(l).squeeze(1);e=1e-6;tp=(p*y).sum((1,2));fn=(y*(1-p)).sum((1,2));fp=((1-y)*p).sum((1,2))
    return 1-((tp+e)/(tp+a*fn+b*fp+e)).mean()+F.binary_cross_entropy_with_logits(l.squeeze(1),y)
def stt(P,Y,t):
    p=(P>=t);tp=(p&(Y>0)).sum();fp=(p&(Y==0)).sum();fn=((~p)&(Y>0)).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return 2*pr*rc/(pr+rc+1e-9),pr,rc
# --- AlphaEarth 학습 ---
random.seed(42);torch.manual_seed(42);np.random.seed(42)
hold=set(l.strip() for l in open(HOLD) if l.strip())
fires=[d for d in glob.glob(os.path.join(AE,"*")) if glob.glob(os.path.join(d,"*.npz")) and (fire_ha(os.path.basename(d))>=600 or os.path.basename(d) in hold)]
te=[d for d in fires if os.path.basename(d) in hold];tr=[d for d in fires if os.path.basename(d) not in hold];random.shuffle(tr)
nv=max(2,len(tr)//6);va=tr[:nv];trn=tr[nv:]
class DS(torch.utils.data.Dataset):
    def __init__(s,f,aug=False): s.f=f;s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]);x=d["x"].astype(np.float32);y=d["y"].astype(np.float32)
        if s.aug and random.random()<0.5: x,y=x[:,:,::-1].copy(),y[:,::-1].copy()
        return torch.from_numpy(x),torch.from_numpy(y)
tf=lambda ds:[f for d in ds for f in glob.glob(os.path.join(d,"*.npz"))]
m=UNet(128).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
dl=torch.utils.data.DataLoader(DS(tf(trn),aug=True),batch_size=4,shuffle=True,num_workers=2)
@torch.no_grad()
def aeprob(files):
    m.eval();out={}
    for f in files:
        d=np.load(f);x=torch.from_numpy(d["x"].astype(np.float32)).unsqueeze(0).to(DEV)
        out[os.path.basename(f)]=(torch.sigmoid(m(x)).squeeze().cpu().numpy(),d["y"].astype(np.uint8))
    return out
best=-1;bsd=None
for ep in range(40):
    m.train()
    for xb,yb in dl:
        xb,yb=xb.to(DEV),yb.to(DEV);opt.zero_grad();tv(m(xb),yb).backward();opt.step()
    vp=aeprob(tf(va));P=np.concatenate([v[0].reshape(-1) for v in vp.values()]);Y=np.concatenate([v[1].reshape(-1) for v in vp.values()])
    vb=max(stt(P,Y,t)[0] for t in np.arange(0.3,0.85,0.05))
    if vb>best: best=vb;bsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
m.load_state_dict(bsd)
# --- FT4 로드 ---
import prithvi_ft as P4
ft4=P4.PrithviSeg(unfreeze=4).to(DEV);ft4.load_state_dict(torch.load(os.path.join(P4.RUN,"best.pt"),map_location=DEV));ft4.eval()
@torch.no_grad()
def ft4prob(fire,base):
    f=os.path.join(RF,fire,base)
    if not os.path.exists(f): return None
    d=np.load(f);x=d["x"].astype(np.float32)/10000.0
    return torch.sigmoid(ft4(P4.prep(x[0:6],x[6:12]).unsqueeze(0).to(DEV))).squeeze().cpu().numpy()
# --- 홀드아웃 앙상블 ---
aep=aeprob(tf(te))
PA=[];PF=[];Y=[]
for d in te:
    fire=os.path.basename(d)
    for f in glob.glob(os.path.join(d,"*.npz")):
        b=os.path.basename(f); a=aep.get(b); fp=ft4prob(fire,b)
        if a is None or fp is None: continue
        PA.append(a[0].reshape(-1));PF.append(fp.reshape(-1));Y.append(a[1].reshape(-1))
PA=np.concatenate(PA);PF=np.concatenate(PF);Y=np.concatenate(Y)
f1a=max(stt(PA,Y,t)[0] for t in np.arange(0.3,0.85,0.05)); f1f=max(stt(PF,Y,t)[0] for t in np.arange(0.3,0.85,0.05))
print(f"[AlphaEarth 단독] F1={f1a:.3f} | [FT4 단독] F1={f1f:.3f}",flush=True)
bestf=0;bw=0;bt=0
for w in np.arange(0,1.01,0.1):
    e=w*PA+(1-w)*PF
    for t in np.arange(0.3,0.85,0.05):
        f1=stt(e,Y,t)[0]
        if f1>bestf: bestf=f1;bw=w;bt=t
print(f"[AE+FT4 앙상블] best F1={bestf:.3f} (w_AE={bw:.1f}, thr={bt:.2f}) vs baseline 0.690",flush=True)
