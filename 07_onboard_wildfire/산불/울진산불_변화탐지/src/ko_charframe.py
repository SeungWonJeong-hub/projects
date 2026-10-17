"""Stage 1: 한국 단독 char-프레이밍 격리실험 (일본/EU 사전학습 前 핵심변수 판정).

가설: 봄 실패는 "신호없음"이 아니라 "change(dNBR) 프레이밍 반전". 절대 char(post SWIR↑/NIR↓)는
봄 post에 존재 → post-기반 절대프레이밍이면 봄 F1이 오를 것. 안 오르면 사전학습 무의미.
세 입력구성을 같은 홀드아웃·같은 자로, 3시드 평균, 계절별 F1 비교:
  bitemporal : 12ch(pre+post)  = dNBR식 baseline
  postonly   : 6ch(post만)     = 순수 절대 char (반전 불가능)
  postchar   : 8ch(post6+NBR_post+NDVI_post) = 절대 char 인덱스 명시
밴드순서 [B02,B03,B04,B8A,B11,B12]: idx 2=Red 3=NIR(B8A) 5=SWIR2(B12).
사용: CONFIG=postchar python3 ko_charframe.py   (SEEDS=1,2,3 기본)
"""
import os, glob, re, json, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
DATA=os.path.join(PROJ,"ko_refl"); HOLD=os.path.join(PROJ,"data","holdout_fixed.txt")
DEV="cuda" if torch.cuda.is_available() else "cpu"
CONFIG=os.environ.get("CONFIG","bitemporal")
SEEDS=[int(s) for s in os.environ.get("SEEDS","1,2,3").split(",")]
MIN_HA=float(os.environ.get("MIN_HA","500"))
IN_CH={"bitemporal":12,"postonly":6,"postchar":8}[CONFIG]

def build_x(x12):  # x12: [12,H,W] float(/1e4). CONFIG별 입력 구성.
    post=x12[6:12]
    if CONFIG=="bitemporal": return x12
    if CONFIG=="postonly":   return post
    nir,red,sw2=post[3],post[2],post[5]
    nbr=(nir-sw2)/(nir+sw2+1e-6); ndvi=(nir-red)/(nir+red+1e-6)   # 절대 char 인덱스(post)
    return np.concatenate([post,nbr[None],ndvi[None]],0)
def gn(c): return nn.GroupNorm(min(8,c),c)
class Blk(nn.Module):
    def __init__(s,i,o): super().__init__(); s.c=nn.Sequential(nn.Conv2d(i,o,3,padding=1),gn(o),nn.GELU(),nn.Conv2d(o,o,3,padding=1),gn(o),nn.GELU())
    def forward(s,x): return s.c(x)
class UNet(nn.Module):
    def __init__(s,ic):
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
        d=np.load(s.f[i]);x=build_x(d["x"].astype(np.float32)/10000.0);y=d["y"].astype(np.float32)
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
        d=np.load(f);x=torch.from_numpy(build_x(d["x"].astype(np.float32)/10000.0)).unsqueeze(0).to(DEV)
        P.append(torch.sigmoid(m(x)).squeeze().cpu().numpy().reshape(-1));Y.append(d["y"].astype(np.uint8).reshape(-1))
    return np.concatenate(P),np.concatenate(Y)
def season(mn): return "봄" if 3<=mn<=5 else "여름" if 6<=mn<=8 else "가을" if 9<=mn<=11 else "겨울"
def fire_ha(d):
    m=re.search(r"_(\d+)ha$",os.path.basename(d)); return float(m.group(1)) if m else 0.0
tf=lambda ds:[f for d in ds for f in glob.glob(os.path.join(d,"*.npz"))]
hold=set(l.strip() for l in open(HOLD) if l.strip())
allf=sorted([d for d in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(d,"*.npz"))])
te_f=[d for d in allf if os.path.basename(d) in hold]

def run(seed):
    random.seed(seed);torch.manual_seed(seed);np.random.seed(seed)
    tr_all=[d for d in allf if os.path.basename(d) not in hold and fire_ha(d)>=MIN_HA];random.shuffle(tr_all)
    nval=max(2,len(tr_all)//6);va_f=tr_all[:nval];trn_f=tr_all[nval:]
    m=UNet(IN_CH).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
    dl=torch.utils.data.DataLoader(DS(tf(trn_f),aug=True),batch_size=4,shuffle=True,num_workers=2)
    best=-1;bestsd=None
    for ep in range(40):
        m.train()
        for xb,yb in dl:
            xb,yb=xb.to(DEV),yb.to(DEV);opt.zero_grad();tversky_bce(m(xb),yb).backward();opt.step()
        Pv,Yv=predict(m,tf(va_f));vb=max(stats(Pv,Yv,t)[0] for t in np.arange(0.3,0.85,0.05))
        if vb>best: best=vb;bestsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bestsd)
    allP=[];allY=[];per=defaultdict(lambda:[[],[]])
    for d in te_f:
        pp,yy=predict(m,glob.glob(os.path.join(d,"*.npz")));allP.append(pp);allY.append(yy)
        se=season(int(os.path.basename(d).split("_")[1][4:6]));per[se][0].append(pp);per[se][1].append(yy)
    Pa=np.concatenate(allP);Ya=np.concatenate(allY)
    thr=max(np.arange(0.3,0.85,0.05),key=lambda t:stats(Pa,Ya,t)[0]);f1,pr,rc=stats(Pa,Ya,thr)
    ps={se:float(stats(np.concatenate(per[se][0]),np.concatenate(per[se][1]),thr)[0]) for se in per}
    return float(f1),float(pr),float(rc),float(thr),ps

print(f"=== CONFIG={CONFIG} (in_ch={IN_CH}) | TEST {len(te_f)}화재 | seeds {SEEDS} ===",flush=True)
res=[run(s) for s in SEEDS]
f1s=[r[0] for r in res]; prs=[r[1] for r in res]; rcs=[r[2] for r in res]
allse=set(k for r in res for k in r[4])
print(f"[{CONFIG}] 전체 F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  P={np.mean(prs):.3f}  R={np.mean(rcs):.3f}",flush=True)
for se in ["봄","여름","가을","겨울"]:
    v=[r[4][se] for r in res if se in r[4]]
    if v: print(f"   {se}: F1={np.mean(v):.3f}±{np.std(v):.3f}  (n_seed={len(v)})",flush=True)
OUT=os.path.join(PROJ,"runs",f"charframe_{CONFIG}.json"); os.makedirs(os.path.dirname(OUT),exist_ok=True)
json.dump({"config":CONFIG,"in_ch":IN_CH,"f1_mean":float(np.mean(f1s)),"f1_std":float(np.std(f1s)),
    "p_mean":float(np.mean(prs)),"r_mean":float(np.mean(rcs)),"seeds":SEEDS,
    "per_season":{se:float(np.mean([r[4][se] for r in res if se in r[4]])) for se in allse},
    "per_seed_f1":f1s},open(OUT,"w"),ensure_ascii=False,indent=2)
print(f"[저장] {OUT}",flush=True)
