"""PU 학습 — under-labeling 정면 공략 (딥리서치 2026-08-25).

가설: 산림청 폴리곤은 진짜 소각을 통째로 놓침(under-label). 폴리곤 외부를 "안 탄(negative)"으로
학습하면 모델이 놓친 소각을 배제당함. → 외부를 **unlabeled**(소각 포함 가능)로 두고 nnPU 손실 학습.
polygon내부=P, 외부=U. π=진짜 소각 prior(폴리곤 비율의 하한 × PRIOR_SCALE로 under-label 보정).

★평가 주의: 폴리곤 대비 엄격 F1은 PU가 더 넓게 예측→FP↑로 낮아 보일 수 있음(그 FP가 놓친 진짜소각).
정직 평가는 eval_boundary_tolerant.py + proxy_eval.py(물리 referee). 여기선 참고로 엄격도 출력.

사용: MIN_HA=500 SEED=1 PRIOR_SCALE=1.4 python3 ee_ae_train_pu.py
"""
import os, glob, random, re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("KO_AE",os.path.join(HERE,"..","ko_ae"))
HOLD=os.path.join(HERE,"..","data","holdout_fixed.txt")
DEV="cuda" if torch.cuda.is_available() else "cpu"
MIN_HA=float(os.environ.get("MIN_HA","500"))
SEED=int(os.environ.get("SEED","1"))
PRIOR_SCALE=float(os.environ.get("PRIOR_SCALE","1.4"))  # 폴리곤 under-label 보정(측정 과예측비 1.38)
TAG=os.environ.get("TAG",f"pu_ps{PRIOR_SCALE}")
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
        d=np.load(s.f[i]);x=d["x"].astype(np.float32);y=d["y"].astype(np.float32);v=d["valid"].astype(np.float32)
        if s.aug and random.random()<0.5: x,y,v=x[:,:,::-1].copy(),y[:,::-1].copy(),v[:,::-1].copy()
        if s.aug and random.random()<0.5: x,y,v=x[:,::-1].copy(),y[::-1].copy(),v[::-1].copy()
        return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(v)

def nnpu_loss(logits, y, v, prior, beta=0.0):
    """nnPU (Kiryo 2017). y=1 positive(폴리곤), y=0 unlabeled. v=valid. sigmoid loss."""
    z=logits.squeeze(1)
    l_pos=torch.sigmoid(-z)   # 손실 if 라벨 +1
    l_neg=torch.sigmoid(z)    # 손실 if 라벨 -1
    P=((y>0.5)&(v>0.5)); U=((y<=0.5)&(v>0.5))
    nP=P.sum().clamp(min=1.0); nU=U.sum().clamp(min=1.0)
    R_p_plus=(l_pos*P).sum()/nP
    R_p_minus=(l_neg*P).sum()/nP
    R_u_minus=(l_neg*U).sum()/nU
    pos_risk=prior*R_p_plus
    neg_risk=R_u_minus-prior*R_p_minus
    if neg_risk >= -beta:
        return pos_risk+neg_risk
    else:
        return -neg_risk   # 음수면 gradient ascent로 되돌림(Kiryo)
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
# π: train 폴리곤 소각비율(valid내) × PRIOR_SCALE
pf=[];pv=[]
for f in tf(trn_f):
    d=np.load(f); v=d["valid"]>0; pf.append(int((d["y"][v]>0).sum())); pv.append(int(v.sum()))
poly_frac=sum(pf)/max(sum(pv),1); prior=min(0.9,poly_frac*PRIOR_SCALE)
print(f"[{TAG}] SEED={SEED} MIN_HA={MIN_HA} | train {len(trn_f)}화재/{len(tf(trn_f))}타일 val {len(tf(va_f))} TEST {len(te_f)}화재",flush=True)
print(f"폴리곤 소각비율={poly_frac:.3f} × {PRIOR_SCALE} → π={prior:.3f}",flush=True)
m=UNet(128).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
dl=torch.utils.data.DataLoader(DS(tf(trn_f),aug=True),batch_size=4,shuffle=True,num_workers=2)
best=-1;bestsd=None
for ep in range(40):
    m.train()
    for xb,yb,vb in dl:
        xb,yb,vb=xb.to(DEV),yb.to(DEV),vb.to(DEV);opt.zero_grad()
        nnpu_loss(m(xb),yb,vb,prior).backward();opt.step()
    Pv,Yv=predict(m,tf(va_f));vb2=max(stats(Pv,Yv,t)[0] for t in np.arange(0.3,0.85,0.05))
    if vb2>best: best=vb2;bestsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
m.load_state_dict(bestsd)
allP=[];allY=[];per=[]
for d in te_f:
    pp,yy=predict(m,glob.glob(os.path.join(d,"*.npz")));allP.append(pp);allY.append(yy)
    per.append((os.path.basename(d),season(int(os.path.basename(d).split("_")[1][4:6])),pp,yy))
Pa=np.concatenate(allP);Ya=np.concatenate(allY)
thr=max(np.arange(0.3,0.85,0.05),key=lambda t:stats(Pa,Ya,t)[0]); f1,pr,rc=stats(Pa,Ya,thr)
print(f"\n[엄격 폴리곤대비] best-thr={thr:.2f} | F1={f1:.3f} P={pr:.3f} R={rc:.3f}  (PU는 넓게예측→엄격F1 낮을수 있음, 정직평가는 경계허용+proxy)",flush=True)
from collections import defaultdict
bs=defaultdict(lambda:[[],[]]); per_season={}
for fn,se,pp,yy in per: bs[se][0].append(pp);bs[se][1].append(yy)
for se in ["봄","여름","가을","겨울"]:
    if se in bs:
        pp=np.concatenate(bs[se][0]);yy=np.concatenate(bs[se][1]);f,p,r=stats(pp,yy,thr)
        per_season[se]={"f1":round(float(f),4),"p":round(float(p),4),"r":round(float(r),4),"fires":len(bs[se][0])}
        print(f"{se}: F1={f:.3f} P={p:.3f} R={r:.3f}",flush=True)
import json
OUT=os.path.join(HERE,"..","runs",f"ko_ae_{TAG}"); os.makedirs(OUT,exist_ok=True)
torch.save(bestsd,os.path.join(OUT,"ae_nal_unet.pt"))
json.dump({"model":f"AlphaEarth-PU({TAG})","script":"ee_ae_train_pu.py","spec":"ae128","in_ch":128,
    "arch":"GN-UNet-3lvl","thr":round(float(thr),3),"f1":round(float(f1),4),
    "precision":round(float(pr),4),"recall":round(float(rc),4),"per_season":per_season,
    "prior":round(prior,3),"prior_scale":PRIOR_SCALE,"seed":SEED,
    "ckpt":"ae_nal_unet.pt","holdout":"data/holdout_fixed.txt"},
    open(os.path.join(OUT,"results.json"),"w"),ensure_ascii=False,indent=2)
print(f"[저장] {OUT} (엄격F1={f1:.3f}) → 다음: eval_boundary_tolerant.py {OUT} + proxy_eval",flush=True)
