"""AlphaEarth 128ch + Noise-Aware(boundary-aware) loss — 라벨 벽 공략 실험(딥리서치 2026-08-24 1위 레버).

가설: 산림청 GT=경계 폴리곤(픽셀정밀 아님) → 경계 ±BAND px 는 라벨이 불확실.
그 구간의 loss 를 down-weight 하면 모델이 노이즈 경계에 과적합하지 않고 실제 분광신호로
불탄 범위를 학습 → 미학습 F1 상승 기대. ★v5(라벨 하드제거, negative)와 다름: 라벨 유지, loss만 완화.

baseline 재현: MIN_HA=500(대형 ~51화재, 0.690 세트) + BAND=0(가중 없음).
NAL 실험:    같은 데이터 + BAND=3(경계 ±3px down-weight W_LOW).
같은 고정홀드아웃(data/holdout_fixed.txt)·seed 로 공정비교.

사용:
  MIN_HA=500 BAND=0 python3 ee_ae_train_nal.py   # baseline
  MIN_HA=500 BAND=3 python3 ee_ae_train_nal.py   # noise-aware
"""
import os, glob, random, re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("KO_AE",os.path.join(HERE,"..","ko_ae"))
HOLD=os.path.join(HERE,"..","data","holdout_fixed.txt")
DEV="cuda" if torch.cuda.is_available() else "cpu"
MIN_HA=float(os.environ.get("MIN_HA","500"))      # 대형화재만(소형=경계노이즈 과예측, 0.664)
IN_CH=int(os.environ.get("IN_CH","128"))          # CF feature 등 추가채널용
BAND=int(os.environ.get("BAND","3"))              # 경계 ±BAND px down-weight (0=끔=baseline)
W_LOW=float(os.environ.get("W_LOW","0.15"))       # 불확실 경계구간 loss 가중
TAG=os.environ.get("TAG",f"nal_ha{int(MIN_HA)}_band{BAND}")

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

def boundary_weight(y):
    """경계 ±BAND px 를 W_LOW 로 down-weight 한 픽셀 가중맵. BAND=0 이면 전부 1."""
    w=np.ones_like(y,dtype=np.float32)
    if BAND<=0: return w
    yb=y>0.5
    if yb.all() or (~yb).all(): return w      # 단일클래스 타일은 경계 없음
    # 불탄까지 거리(밖) + 안탄까지 거리(안) → 경계로부터 최소거리
    d_out=distance_transform_edt(~yb); d_in=distance_transform_edt(yb)
    dist=np.where(yb,d_in,d_out)
    w[dist<=BAND]=W_LOW
    return w
class DS(torch.utils.data.Dataset):
    def __init__(s,files,aug=False): s.f=files;s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]);x=d["x"].astype(np.float32);y=d["y"].astype(np.float32)
        w=boundary_weight(y)
        if s.aug and random.random()<0.5: x,y,w=x[:,:,::-1].copy(),y[:,::-1].copy(),w[:,::-1].copy()
        if s.aug and random.random()<0.5: x,y,w=x[:,::-1].copy(),y[::-1].copy(),w[::-1].copy()
        return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(w)
def tversky_bce_w(l,y,w,a=0.7,b=0.3):
    """recall-Tversky(v3와 동일) + 경계가중 BCE(noise-aware)."""
    p=torch.sigmoid(l).squeeze(1);e=1e-6
    tp=(p*y).sum((1,2));fn=(y*(1-p)).sum((1,2));fp=((1-y)*p).sum((1,2))
    tv=1-((tp+e)/(tp+a*fn+b*fp+e)).mean()
    bce=F.binary_cross_entropy_with_logits(l.squeeze(1),y,reduction="none")
    bce=(bce*w).sum()/(w.sum()+e)             # 경계 down-weight
    return tv+bce
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

SEED=int(os.environ.get("SEED","42"))
random.seed(SEED);torch.manual_seed(SEED);np.random.seed(SEED)
hold=set(l.strip() for l in open(HOLD) if l.strip())
allfires=sorted([d for d in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(d,"*.npz"))])
# 홀드아웃은 크기무관 유지, train/val 만 MIN_HA 필터(대형=명확라벨)
te_f=[d for d in allfires if os.path.basename(d) in hold]
tr_all=[d for d in allfires if os.path.basename(d) not in hold and fire_ha(d)>=MIN_HA];random.shuffle(tr_all)
nval=max(2,len(tr_all)//6);va_f=tr_all[:nval];trn_f=tr_all[nval:]
tf=lambda ds:[f for d in ds for f in glob.glob(os.path.join(d,"*.npz"))]
print(f"[{TAG}] MIN_HA={MIN_HA} BAND={BAND} W_LOW={W_LOW}",flush=True)
print(f"train {len(trn_f)}화재/{len(tf(trn_f))}타일 val {len(tf(va_f))} TEST {len(te_f)}화재",flush=True)
m=UNet(IN_CH).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
dl=torch.utils.data.DataLoader(DS(tf(trn_f),aug=True),batch_size=4,shuffle=True,num_workers=2)
best=-1;bestsd=None
for ep in range(40):
    m.train()
    for xb,yb,wb in dl:
        xb,yb,wb=xb.to(DEV),yb.to(DEV),wb.to(DEV);opt.zero_grad();tversky_bce_w(m(xb),yb,wb).backward();opt.step()
    Pv,Yv=predict(m,tf(va_f));vb=max(stats(Pv,Yv,t)[0] for t in np.arange(0.3,0.85,0.05))
    if vb>best: best=vb;bestsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
m.load_state_dict(bestsd)
allP=[];allY=[];per=[]
for d in te_f:
    pp,yy=predict(m,glob.glob(os.path.join(d,"*.npz")));allP.append(pp);allY.append(yy)
    per.append((os.path.basename(d),season(int(os.path.basename(d).split("_")[1][4:6])),pp,yy))
Pa=np.concatenate(allP);Ya=np.concatenate(allY)
thr=max(np.arange(0.3,0.85,0.05),key=lambda t:stats(Pa,Ya,t)[0])
f1,pr,rc=stats(Pa,Ya,thr)
print(f"\n전체 홀드아웃 best-thr={thr:.2f} | F1={f1:.3f} P={pr:.3f} R={rc:.3f}",flush=True)
from collections import defaultdict
bs=defaultdict(lambda:[[],[]])
for fn,se,pp,yy in per: bs[se][0].append(pp);bs[se][1].append(yy)
per_season={}
for se in ["봄","여름","가을","겨울"]:
    if se in bs:
        pp=np.concatenate(bs[se][0]);yy=np.concatenate(bs[se][1]);f,p,r=stats(pp,yy,thr)
        per_season[se]={"f1":round(float(f),4),"p":round(float(p),4),"r":round(float(r),4),"fires":len(bs[se][0])}
        print(f"{se}: F1={f:.3f} P={p:.3f} R={r:.3f} (화재 {len(bs[se][0])})",flush=True)
print(f"\n=== NAL {TAG} 전체 {f1:.3f} vs AlphaEarth baseline 0.690 vs v3 0.616 ===")

import json
OUT=os.path.join(HERE,"..","runs",f"ko_ae_{TAG}"); os.makedirs(OUT,exist_ok=True)
torch.save(bestsd,os.path.join(OUT,"ae_nal_unet.pt"))
json.dump({"model":f"AlphaEarth-NAL({TAG})","script":"ee_ae_train_nal.py","spec":"ae128",
    "in_ch":IN_CH,"arch":"GN-UNet-3lvl","thr":round(float(thr),3),
    "f1":round(float(f1),4),"precision":round(float(pr),4),"recall":round(float(rc),4),
    "per_season":per_season,"min_ha":MIN_HA,"band":BAND,"w_low":W_LOW,
    "n_train_fires":len(trn_f),"n_test_fires":len(te_f),
    "ckpt":"ae_nal_unet.pt","holdout":"data/holdout_fixed.txt"},
    open(os.path.join(OUT,"results.json"),"w"),ensure_ascii=False,indent=2)
print(f"[저장] {OUT} (F1={f1:.3f})",flush=True)
