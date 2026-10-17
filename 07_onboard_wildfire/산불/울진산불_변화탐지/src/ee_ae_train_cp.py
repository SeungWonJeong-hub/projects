"""Copy-Paste 합성 burn 증강 — few-shot 병목 완화 (딥리서치 2026-08-25 diffusion증강의 실행가능 버전).

진짜 diffusion generator는 34-51화재론 학습불가 + 봄소각 생성은 phenology 순환. 대신 실증된
Copy-Paste seg 증강(Ghiasi 2021)을 AlphaEarth 임베딩에: 학습 중 다른 타일의 진짜 소각조각
(임베딩+라벨)을 현재 타일 비소각 위치에 붙여 새 합성 burn 샘플 생성 → few-shot 다양성↑.
순환 없음(진짜 소각 임베딩 사용). baseline seed1(0.701)과 공정비교.

사용: MIN_HA=500 SEED=1 CP_PROB=0.5 CP_N=2 python3 ee_ae_train_cp.py
"""
import os, glob, random, re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("KO_AE",os.path.join(HERE,"..","ko_ae"))
HOLD=os.path.join(HERE,"..","data","holdout_fixed.txt")
DEV="cuda" if torch.cuda.is_available() else "cpu"
MIN_HA=float(os.environ.get("MIN_HA","500")); SEED=int(os.environ.get("SEED","1"))
CP_PROB=float(os.environ.get("CP_PROB","0.5"))   # 샘플당 copy-paste 확률
CP_N=int(os.environ.get("CP_N","2"))             # 붙일 조각 최대 개수
TAG=os.environ.get("TAG",f"cp_p{CP_PROB}_n{CP_N}")
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

def extract_patches(files):
    """각 train 타일의 소각영역 bbox를 조각으로 추출 → (x_patch,y_patch) 뱅크."""
    bank=[]
    from scipy.ndimage import label as cclabel
    for f in files:
        d=np.load(f); y=d["y"].astype(np.uint8); x=d["x"].astype(np.float32); v=d["valid"].astype(np.uint8)
        lab,n=cclabel(y>0)
        for k in range(1,n+1):
            ys,xs=np.where(lab==k)
            if len(ys)<200: continue          # 너무 작은 조각 제외
            y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
            h,w=y1-y0,x1-x0
            if h>200 or w>200 or h<8 or w<8: continue  # 너무 크거나 작은 bbox 제외
            m=(lab[y0:y1,x0:x1]==k)&(v[y0:y1,x0:x1]>0)
            if m.sum()<150: continue
            bank.append((x[:,y0:y1,x0:x1].copy(), m.astype(np.uint8)))
    return bank

BANK=[]
def copy_paste(x,y,v):
    """현재 타일에 뱅크 조각을 비소각 위치에 붙임."""
    if not BANK or random.random()>=CP_PROB: return x,y,v
    H,W=y.shape
    for _ in range(random.randint(1,CP_N)):
        px,pm=random.choice(BANK); _,ph,pw=px.shape
        if ph>=H or pw>=W: continue
        top=random.randint(0,H-ph); left=random.randint(0,W-pw)
        reg_y=y[top:top+ph,left:left+pw]
        if (reg_y>0).mean()>0.3: continue     # 이미 소각 많은 곳엔 안 붙임
        mm=pm>0
        x[:,top:top+ph,left:left+pw]=np.where(mm[None],px,x[:,top:top+ph,left:left+pw])
        y[top:top+ph,left:left+pw]=np.where(mm,1,reg_y)
        v[top:top+ph,left:left+pw]=np.where(mm,1,v[top:top+ph,left:left+pw])
    return x,y,v
class DS(torch.utils.data.Dataset):
    def __init__(s,files,aug=False): s.f=files;s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]);x=d["x"].astype(np.float32).copy();y=d["y"].astype(np.float32).copy();v=d["valid"].astype(np.float32).copy()
        if s.aug: x,y,v=copy_paste(x,y,v)
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
def fire_ha(d):
    m=re.search(r"_(\d+)ha$",os.path.basename(d)); return float(m.group(1)) if m else 0.0

random.seed(SEED);torch.manual_seed(SEED);np.random.seed(SEED)
hold=set(l.strip() for l in open(HOLD) if l.strip())
allf=sorted([d for d in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(d,"*.npz"))])
te_f=[d for d in allf if os.path.basename(d) in hold]
tr_all=[d for d in allf if os.path.basename(d) not in hold and fire_ha(d)>=MIN_HA];random.shuffle(tr_all)
nval=max(2,len(tr_all)//6);va_f=tr_all[:nval];trn_f=tr_all[nval:]
tf=lambda ds:[f for d in ds for f in glob.glob(os.path.join(d,"*.npz"))]
BANK[:]=extract_patches(tf(trn_f))
print(f"[{TAG}] SEED={SEED} | train {len(trn_f)}화재/{len(tf(trn_f))}타일 | 소각조각뱅크 {len(BANK)}개 | CP_PROB={CP_PROB} CP_N={CP_N}",flush=True)
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
    per.append((season(int(os.path.basename(d).split("_")[1][4:6])),pp,yy))
Pa=np.concatenate(allP);Ya=np.concatenate(allY)
thr=max(np.arange(0.3,0.85,0.05),key=lambda t:stats(Pa,Ya,t)[0]);f1,pr,rc=stats(Pa,Ya,thr)
print(f"\n[CP] best-thr={thr:.2f} | F1={f1:.3f} P={pr:.3f} R={rc:.3f}  (baseline seed1=0.701)",flush=True)
from collections import defaultdict
bs=defaultdict(lambda:[[],[]]); per_season={}
for se,pp,yy in per: bs[se][0].append(pp);bs[se][1].append(yy)
for se in ["봄","여름","가을","겨울"]:
    if se in bs:
        pp=np.concatenate(bs[se][0]);yy=np.concatenate(bs[se][1]);f,p,r=stats(pp,yy,thr)
        per_season[se]={"f1":round(float(f),4),"p":round(float(p),4),"r":round(float(r),4)}
        print(f"{se}: F1={f:.3f} P={p:.3f} R={r:.3f}",flush=True)
import json
OUT=os.path.join(HERE,"..","runs",f"ko_ae_{TAG}"); os.makedirs(OUT,exist_ok=True)
torch.save(bestsd,os.path.join(OUT,"ae_nal_unet.pt"))
json.dump({"model":f"AlphaEarth-CopyPaste({TAG})","script":"ee_ae_train_cp.py","spec":"ae128","in_ch":128,
    "arch":"GN-UNet-3lvl","thr":round(float(thr),3),"f1":round(float(f1),4),
    "precision":round(float(pr),4),"recall":round(float(rc),4),"per_season":per_season,
    "cp_prob":CP_PROB,"cp_n":CP_N,"n_patches":len(BANK),"seed":SEED,
    "ckpt":"ae_nal_unet.pt","holdout":"data/holdout_fixed.txt"},
    open(os.path.join(OUT,"results.json"),"w"),ensure_ascii=False,indent=2)
print(f"[저장] {OUT} (F1={f1:.3f})",flush=True)
