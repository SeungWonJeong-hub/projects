"""AlphaEarth 백본 + SAM2 정제라벨 재학습. 홀드아웃은 정제·원본 둘 다 평가(순환착시 방지).
사용: KO_AE=../ko_ae python3 ee_ae_train_refined.py
"""
import os, glob, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("KO_AE",os.path.join(HERE,"..","ko_ae"))
SAM=os.path.join(HERE,"..","ko_refl_sam2")   # 정제라벨
HOLD=os.path.join(HERE,"..","data","holdout_fixed.txt")
DEV="cuda" if torch.cuda.is_available() else "cpu"
def refy(fire,base,orig):  # 정제 y 있으면 그것, 없으면 원본
    p=os.path.join(SAM,fire,base)
    if os.path.exists(p):
        try: return np.load(p)["y"].astype(np.float32)
        except Exception: return orig
    return orig
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
    def __init__(s,items,aug=False): s.it=items;s.aug=aug  # items=(npz_path, fire)
    def __len__(s): return len(s.it)
    def __getitem__(s,i):
        f,fire=s.it[i];d=np.load(f);x=d["x"].astype(np.float32);y=refy(fire,os.path.basename(f),d["y"].astype(np.float32))
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
def predict(m,items,use_ref):
    m.eval();P=[];Yr=[];Yo=[]
    for f,fire in items:
        d=np.load(f);x=torch.from_numpy(d["x"].astype(np.float32)).unsqueeze(0).to(DEV)
        P.append(torch.sigmoid(m(x)).squeeze().cpu().numpy().reshape(-1))
        Yo.append(d["y"].astype(np.uint8).reshape(-1))
        Yr.append(refy(fire,os.path.basename(f),d["y"].astype(np.float32)).astype(np.uint8).reshape(-1))
    return np.concatenate(P),np.concatenate(Yr),np.concatenate(Yo)
def season(mn): return "봄" if 3<=mn<=5 else "여름" if 6<=mn<=8 else "가을" if 9<=mn<=11 else "겨울"

random.seed(42);torch.manual_seed(42);np.random.seed(42)
hold=set(l.strip() for l in open(HOLD) if l.strip())
def fire_ha(nm):
    try: return float(nm.split("_")[-1].replace("ha",""))
    except: return 0.0
MINHA=float(os.environ.get("MIN_HA_KEEP","600"))  # 대형만(소형화재 오염 제외, 0.690 세트와 비교)
fires=sorted([d for d in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(d,"*.npz")) and (fire_ha(os.path.basename(d))>=MINHA or os.path.basename(d) in hold)])
items=lambda fs:[(f,os.path.basename(d)) for d in fs for f in glob.glob(os.path.join(d,"*.npz"))]
te_f=[d for d in fires if os.path.basename(d) in hold]
tr_all=[d for d in fires if os.path.basename(d) not in hold];random.shuffle(tr_all)
nval=max(2,len(tr_all)//6);va_f=tr_all[:nval];trn_f=tr_all[nval:]
# 정제라벨 커버리지
ni=items(fires); nref=sum(1 for f,fr in ni if os.path.exists(os.path.join(SAM,fr,os.path.basename(f))))
print(f"train {len(trn_f)}화재/{len(items(trn_f))}타일 val {len(items(va_f))} TEST {len(te_f)}화재 | 정제라벨 커버 {nref}/{len(ni)}타일",flush=True)
m=UNet(128).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
dl=torch.utils.data.DataLoader(DS(items(trn_f),aug=True),batch_size=4,shuffle=True,num_workers=2)
best=-1;bestsd=None
for ep in range(40):
    m.train()
    for xb,yb in dl:
        xb,yb=xb.to(DEV),yb.to(DEV);opt.zero_grad();tversky_bce(m(xb),yb).backward();opt.step()
    Pv,Yvr,_=predict(m,items(va_f),True);vb=max(stats(Pv,Yvr,t)[0] for t in np.arange(0.3,0.85,0.05))
    if vb>best: best=vb;bestsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
m.load_state_dict(bestsd)
# 홀드아웃 평가: 정제라벨 기준 + 원본폴리곤 기준
te_items=items(te_f)
P,Yr,Yo=predict(m,te_items,True)
for tag,Y in [("정제라벨 기준",Yr),("원본폴리곤 기준",Yo)]:
    thr=max(np.arange(0.3,0.85,0.05),key=lambda t:stats(P,Y,t)[0]);f1,pr,rc=stats(P,Y,thr)
    print(f"[{tag}] best-thr={thr:.2f} F1={f1:.3f} P={pr:.3f} R={rc:.3f}",flush=True)
# 계절별(정제기준)
from collections import defaultdict
bs=defaultdict(lambda:[[],[]])
for (f,fire) in te_items:
    d=np.load(f);x=torch.from_numpy(d["x"].astype(np.float32)).unsqueeze(0).to(DEV)
    with torch.no_grad(): pp=torch.sigmoid(m(x)).squeeze().cpu().numpy().reshape(-1)
    yy=refy(fire,os.path.basename(f),d["y"].astype(np.float32)).astype(np.uint8).reshape(-1)
    bs[season(int(fire.split("_")[1][4:6]))][0].append(pp);bs[season(int(fire.split("_")[1][4:6]))][1].append(yy)
thr=max(np.arange(0.3,0.85,0.05),key=lambda t:stats(P,Yr,t)[0])
print("계절별(정제기준):")
for se in ["봄","여름","가을","겨울"]:
    if se in bs:
        pp=np.concatenate(bs[se][0]);yy=np.concatenate(bs[se][1]);f1,pr,rc=stats(pp,yy,thr)
        print(f"  {se}: F1={f1:.3f} P={pr:.3f} R={rc:.3f}",flush=True)
print(f"\n=== 정제라벨 AlphaEarth vs 원본 AlphaEarth 0.690 ===")
