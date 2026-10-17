"""앙상블 실험: FT4(Prithvi ViT) + from-scratch GN-UNet(CNN), 둘 다 ko_refl 12ch 동일타일.
구조 다름(ViT vs CNN)→오차 탈상관→확률평균으로 F1↑ 기대. 봄/여름 계절별 비교.
사용: KO_DATA=../ko_refl RUN_NAME=ko_prithvi_refl_ft4_v2 python ensemble_experiment.py
"""
import os, glob, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KO_DATA","ko_refl"); os.environ.setdefault("RUN_NAME","ko_prithvi_refl_ft4_v2")
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prithvi_ft as P

# ---- FT4 (Prithvi) ----
ft4=P.PrithviSeg(unfreeze=4).to(P.DEV); ft4.load_state_dict(torch.load(os.path.join(P.RUN,"best.pt"),map_location=P.DEV)); ft4.eval()

# ---- CNN (from-scratch GN-UNet on ko_refl 12ch) ----
def gn(c): return nn.GroupNorm(min(8,c),c)
class Blk(nn.Module):
    def __init__(s,i,o): super().__init__(); s.c=nn.Sequential(nn.Conv2d(i,o,3,padding=1),gn(o),nn.GELU(),nn.Conv2d(o,o,3,padding=1),gn(o),nn.GELU())
    def forward(s,x): return s.c(x)
class UNet(nn.Module):
    def __init__(s,ic=12):
        super().__init__(); s.d1=Blk(ic,32);s.d2=Blk(32,64);s.d3=Blk(64,128);s.d4=Blk(128,256);s.p=nn.MaxPool2d(2)
        s.u3=Blk(256+128,128);s.u2=Blk(128+64,64);s.u1=Blk(64+32,32);s.o=nn.Conv2d(32,1,1)
    def up(s,x): return F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        x=s.u3(torch.cat([s.up(c4),c3],1));x=s.u2(torch.cat([s.up(x),c2],1));x=s.u1(torch.cat([s.up(x),c1],1));return s.o(x)
class DS(torch.utils.data.Dataset):
    def __init__(s,files,aug=False): s.f=files;s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]);x=d["x"].astype(np.float32)/10000.0;y=d["y"].astype(np.float32)
        if s.aug and random.random()<0.5: x,y=x[:,:,::-1].copy(),y[:,::-1].copy()
        if s.aug and random.random()<0.5: x,y=x[:,::-1].copy(),y[::-1].copy()
        return torch.from_numpy(x),torch.from_numpy(y)
def tversky_bce(l,y,a=0.7,b=0.3):
    p=torch.sigmoid(l).squeeze(1);e=1e-6;tp=(p*y).sum((1,2));fn=(y*(1-p)).sum((1,2));fp=((1-y)*p).sum((1,2))
    return 1-((tp+e)/(tp+a*fn+b*fp+e)).mean()+F.binary_cross_entropy_with_logits(l.squeeze(1),y)

random.seed(42);torch.manual_seed(42);np.random.seed(42)
hold=set(l.strip() for l in open(P.HOLD) if l.strip())
fires=sorted([d for d in glob.glob(os.path.join(P.DATA,"*")) if glob.glob(os.path.join(d,"*.npz"))])
te_f=[d for d in fires if os.path.basename(d) in hold]
tr_all=[d for d in fires if os.path.basename(d) not in hold];random.shuffle(tr_all)
nval=max(3,len(tr_all)//6);va_f=tr_all[:nval];trn_f=tr_all[nval:]
tf=lambda ds:[f for d in ds for f in glob.glob(os.path.join(d,"*.npz"))]
cnn=UNet(12).to(P.DEV);opt=torch.optim.AdamW(cnn.parameters(),lr=3e-4,weight_decay=1e-4)
dl=torch.utils.data.DataLoader(DS(tf(trn_f),aug=True),batch_size=8,shuffle=True,num_workers=2)
print(f"CNN 학습: train {len(trn_f)}화재/{len(tf(trn_f))}타일 …",flush=True)
for ep in range(40):
    cnn.train()
    for xb,yb in dl:
        xb,yb=xb.to(P.DEV),yb.to(P.DEV);opt.zero_grad();tversky_bce(cnn(xb),yb).backward();opt.step()
cnn.eval()

def season(m): return "봄" if 3<=m<=5 else "여름" if 6<=m<=8 else "가을" if 9<=m<=11 else "겨울"
@torch.no_grad()
def probs(fire):
    out=[]
    for f in glob.glob(os.path.join(P.DATA,fire,"*.npz")):
        d=np.load(f);x=d["x"].astype(np.float32)/10000.0;y=d["y"].astype(np.uint8)
        pf=torch.sigmoid(ft4(P.prep(x[0:6],x[6:12]).unsqueeze(0).to(P.DEV))).squeeze().cpu().numpy()
        pc=torch.sigmoid(cnn(torch.from_numpy(x).unsqueeze(0).to(P.DEV))).squeeze().cpu().numpy()
        out.append((pf,pc,y))
    return out

# 홀드아웃 전체 수집
data={}
for fire in [os.path.basename(d) for d in te_f]:
    data[fire]=(season(int(fire.split("_")[1][4:6])),probs(fire))
def stats(P_,Y_,t):
    p=(P_>=t);tp=(p&(Y_>0)).sum();fp=(p&(Y_==0)).sum();fn=((~p)&(Y_>0)).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return 2*pr*rc/(pr+rc+1e-9),pr,rc
def eval_combo(w):  # w: 'ft4'|'cnn'|'ens'
    from collections import defaultdict
    THR=np.arange(0.3,0.85,0.05);grp=defaultdict(lambda:{t:[[],[]] for t in THR})
    for fire,(se,tiles) in data.items():
        for pf,pc,y in tiles:
            pr={'ft4':pf,'cnn':pc,'ens':(pf+pc)/2}[w]
            for g in("ALL",se):
                for t in THR: grp[g][t][0].append(pr.reshape(-1));grp[g][t][1].append(y.reshape(-1))
    out={}
    for g in("ALL","봄","여름"):
        best=max(THR,key=lambda t:stats(np.concatenate(grp[g][t][0]),np.concatenate(grp[g][t][1]),t)[0])
        f1,pr,rc=stats(np.concatenate(grp[g][best][0]),np.concatenate(grp[g][best][1]),best)
        out[g]=(round(float(f1),3),round(float(pr),3),round(float(rc),3),round(float(best),2))
    return out
R={w:eval_combo(w) for w in("ft4","cnn","ens")}
print(f"\n{'':6s} {'전체F1(P,R@thr)':26s} {'봄':22s} {'여름':22s}")
for w in("ft4","cnn","ens"):
    fmt=lambda g:f"{R[w][g][0]}({R[w][g][1]},{R[w][g][2]}@{R[w][g][3]})"
    print(f"{w:6s} {fmt('ALL'):26s} {fmt('봄'):22s} {fmt('여름'):22s}")
print(f"\n=== 전체 F1: ft4 {R['ft4']['ALL'][0]} · cnn {R['cnn']['ALL'][0]} → 앙상블 {R['ens']['ALL'][0]} (Δvs최고 {R['ens']['ALL'][0]-max(R['ft4']['ALL'][0],R['cnn']['ALL'][0]):+.3f})")
print(f"=== 봄 F1: 앙상블 {R['ens']['봄'][0]} (vs ft4 {R['ft4']['봄'][0]}) | 여름: 앙상블 {R['ens']['여름'][0]} (vs ft4 {R['ft4']['여름'][0]})")
import json;os.makedirs(os.path.join(P.RUN,"..","ensemble"),exist_ok=True);json.dump(R,open(os.path.join(P.RUN,"..","ensemble","results.json"),"w"),ensure_ascii=False,indent=1)
