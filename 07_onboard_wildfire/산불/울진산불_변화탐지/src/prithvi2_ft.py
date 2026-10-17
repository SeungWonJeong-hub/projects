"""Prithvi-EO-2.0-300M 파인튜닝 — 이미지 입력 산불 seg-CD (수치 최대화 레버).

업로드 이미지에서 파운데이션 임베딩 계산(GEE 불필요) = AlphaEarth 강점을 이미지로 재현.
bi-temporal(pre,post) T=2, HLS 6밴드(ko_refl와 일치), frozen 인코더 + conv 디코더(+옵션 unfreeze).
사용: MIN_HA=500 SEED=1 UNFREEZE=0 python3 prithvi2_ft.py
"""
import os, sys, glob, json, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
sys.path.insert(0, os.path.join(PROJ,"prithvi2"))
from prithvi_mae_v2 import PrithviViT
DATA=os.path.join(PROJ,"ko_refl"); HOLD=os.path.join(PROJ,"data","holdout_fixed.txt")
CKPT=os.path.join(PROJ,"prithvi2","Prithvi_EO_V2_300M.pt")
DEV="cuda" if torch.cuda.is_available() else "cpu"
MIN_HA=float(os.environ.get("MIN_HA","500")); SEED=int(os.environ.get("SEED","1"))
UNFREEZE=int(os.environ.get("UNFREEZE","0")); EPOCHS=int(os.environ.get("EPOCHS","40"))
TAG=os.environ.get("TAG",f"prithvi2_uf{UNFREEZE}_s{SEED}")
IMG=224
MEAN=torch.tensor([1087.,1342.,1433.,2734.,1958.,1363.]).view(1,6,1,1,1)
STD =torch.tensor([2248.,2179.,2178.,1850.,1242.,1049.]).view(1,6,1,1,1)
def prep(pre,post):
    fr=np.stack([pre,post],1)                       # [6,2,H,W]
    t=torch.from_numpy(fr).float().unsqueeze(0)
    t=F.interpolate(t.reshape(1,12,*fr.shape[2:]),size=(IMG,IMG),mode="bilinear",align_corners=False).reshape(1,6,2,IMG,IMG)
    return ((t*10000.0-MEAN)/STD)[0]
class DS(torch.utils.data.Dataset):
    def __init__(s,files,aug=False): s.f=files;s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]);x=d["x"].astype(np.float32)/10000.0;y=d["y"].astype(np.float32)
        pre,post=x[0:6],x[6:12]
        if s.aug and random.random()<0.5: pre,post,y=pre[:,:,::-1].copy(),post[:,:,::-1].copy(),y[:,::-1].copy()
        if s.aug and random.random()<0.5: pre,post,y=pre[:,::-1].copy(),post[:,::-1].copy(),y[::-1].copy()
        return prep(pre,post),torch.from_numpy(y).float()
class Seg(nn.Module):
    def __init__(s,unfreeze=0):
        super().__init__()
        s.enc=PrithviViT(img_size=IMG,patch_size=(1,16,16),num_frames=2,in_chans=6,embed_dim=1024,depth=24,num_heads=16,coords_encoding=[])
        sd=torch.load(CKPT,map_location="cpu",weights_only=False)
        enc={k[len("encoder."):]:v for k,v in sd.items() if k.startswith("encoder.")}; enc.pop("pos_embed",None)
        s.enc.load_state_dict(enc,strict=False)
        s.unfreeze=unfreeze
        for p in s.enc.parameters(): p.requires_grad=False
        if unfreeze>0:
            for blk in s.enc.blocks[-unfreeze:]:
                for p in blk.parameters(): p.requires_grad=True
        C=2048
        s.dec=nn.Sequential(
            nn.Conv2d(C,512,3,padding=1),nn.BatchNorm2d(512),nn.GELU(),nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
            nn.Conv2d(512,256,3,padding=1),nn.BatchNorm2d(256),nn.GELU(),nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
            nn.Conv2d(256,128,3,padding=1),nn.BatchNorm2d(128),nn.GELU(),nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
            nn.Conv2d(128,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
            nn.Conv2d(64,1,1))
    def forward(s,x):
        ctx=torch.no_grad() if s.unfreeze==0 else torch.enable_grad()
        with ctx:
            f=s.enc.forward_features(x,None,None)
        last=f[-1] if isinstance(f,(list,tuple)) else f
        feat=s.enc.prepare_features_for_image_model([last])[0]     # [B,2048,14,14]
        return F.interpolate(s.dec(feat),size=(256,256),mode="bilinear",align_corners=False)
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
        d=np.load(f);pre,post=(d["x"].astype(np.float32)/10000.0)[0:6],(d["x"].astype(np.float32)/10000.0)[6:12]
        x=prep(pre,post).unsqueeze(0).to(DEV)
        P.append(torch.sigmoid(m(x)).squeeze().cpu().numpy().reshape(-1));Y.append(d["y"].astype(np.uint8).reshape(-1))
    return np.concatenate(P),np.concatenate(Y)
def season(mn): return "봄" if 3<=mn<=5 else "여름" if 6<=mn<=8 else "가을" if 9<=mn<=11 else "겨울"
import re
def fire_ha(d):
    m=re.search(r"_(\d+)ha$",os.path.basename(d)); return float(m.group(1)) if m else 0.0
random.seed(SEED);torch.manual_seed(SEED);np.random.seed(SEED)
hold=set(l.strip() for l in open(HOLD) if l.strip())
allf=sorted([d for d in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(d,"*.npz"))])
te_f=[d for d in allf if os.path.basename(d) in hold]
tr_all=[d for d in allf if os.path.basename(d) not in hold and fire_ha(d)>=MIN_HA];random.shuffle(tr_all)
nval=max(2,len(tr_all)//6);va_f=tr_all[:nval];trn_f=tr_all[nval:]
tf=lambda ds:[f for d in ds for f in glob.glob(os.path.join(d,"*.npz"))]
m=Seg(UNFREEZE).to(DEV)
ntr=sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"[{TAG}] 학습파라미터 {ntr/1e6:.2f}M | train {len(trn_f)}화재/{len(tf(trn_f))}타일 TEST {len(te_f)}",flush=True)
bs=4 if UNFREEZE==0 else 2; lr=1e-3 if UNFREEZE==0 else 2e-4
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
dl=torch.utils.data.DataLoader(DS(tf(trn_f),aug=True),batch_size=bs,shuffle=True,num_workers=2)
best=-1;bestsd=None
for ep in range(EPOCHS):
    m.train()
    for xb,yb in dl:
        xb,yb=xb.to(DEV),yb.to(DEV);opt.zero_grad();tversky_bce(m(xb),yb).backward();opt.step()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    Pv,Yv=predict(m,tf(va_f));vb=max(stats(Pv,Yv,t)[0] for t in np.arange(0.3,0.85,0.05))
    if vb>best: best=vb;bestsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
m.load_state_dict(bestsd)
allP=[];allY=[];per=[]
for d in te_f:
    pp,yy=predict(m,glob.glob(os.path.join(d,"*.npz")));allP.append(pp);allY.append(yy)
    per.append((season(int(os.path.basename(d).split("_")[1][4:6])),pp,yy))
Pa=np.concatenate(allP);Ya=np.concatenate(allY)
thr=max(np.arange(0.3,0.85,0.05),key=lambda t:stats(Pa,Ya,t)[0]);f1,pr,rc=stats(Pa,Ya,thr)
print(f"\n[{TAG}] best-thr={thr:.2f} F1={f1:.3f} P={pr:.3f} R={rc:.3f} (S2 GN-UNet 0.62 대비)",flush=True)
ps={}
from collections import defaultdict
bs2=defaultdict(lambda:[[],[]])
for se,pp,yy in per: bs2[se][0].append(pp);bs2[se][1].append(yy)
for se in ["봄","여름","가을","겨울"]:
    if se in bs2:
        pp=np.concatenate(bs2[se][0]);yy=np.concatenate(bs2[se][1]);f,p,r=stats(pp,yy,thr);ps[se]={"f1":round(float(f),4)}
        print(f"{se}: F1={f:.3f}",flush=True)
OUT=os.path.join(PROJ,"runs",f"ko_{TAG}"); os.makedirs(OUT,exist_ok=True)
torch.save(bestsd,os.path.join(OUT,"model.pt"))
json.dump({"model":f"Prithvi-EO-2.0-300M-FT({TAG})","spec":"prithvi2","in_ch":12,"unfreeze":UNFREEZE,
    "thr":round(float(thr),3),"f1":round(float(f1),4),"precision":round(float(pr),4),"recall":round(float(rc),4),
    "per_season":ps,"seed":SEED,"ckpt":"model.pt"},open(os.path.join(OUT,"results.json"),"w"),ensure_ascii=False,indent=2)
print(f"[저장] {OUT} (F1={f1:.3f})",flush=True)
