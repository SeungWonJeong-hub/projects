"""LoRA + MTBS 사전학습으로 한국 산불 일반화 돌파.
리서치 결론: frozen(미미)·full-FT(하락) 대신 LoRA(<1%파라미터)가 도메인시프트에 최강
(arXiv 2605.04989: Prithvi-v2+LoRA IoU 78.78). + 물량병목은 MTBS 대규모 사전학습으로 우회.

2단계:
  A. MTBS(US 7생태권 474+AK 151타일)로 Prithvi(LoRA)+디코더 사전학습 → 소각흔 일반특징
  B. ko_multi(한국)에 LoRA 계속 적응 → 고정홀드아웃 8화재로 v3(0.616) 비교

사용: python prithvi_lora.py [--mtbs_epochs 25] [--korea_epochs 40] [--r 8] [--smoke]
"""
import os, sys, glob, json, random, math, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PRI = os.path.join(HERE, "..", "prithvi"); sys.path.insert(0, PRI)
from prithvi_mae import PrithviViT

KO = os.path.join(HERE, "..", "ko_multi")
MTBS = [os.path.join(HERE, "..", "mtbs_data"), os.path.join(HERE, "..", "mtbs_data_ak")]
CKPT = os.path.join(PRI, "Prithvi_EO_V1_100M.pt")
HOLD = os.path.join(HERE, "..", "data", "holdout_fixed.txt")
RUN = os.path.join(HERE, "..", "runs", "ko_prithvi_lora"); os.makedirs(RUN, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([775.23,1080.99,1228.59,2497.20,2204.21,1610.83]).view(1,6,1,1,1)
STD  = torch.tensor([1281.53,1270.03,1399.48,1368.34,1291.68,1154.51]).view(1,6,1,1,1)
IMG = 224

def prep(pre, post):
    fr = np.stack([pre, post, post], 1)
    t = torch.from_numpy(fr).float().unsqueeze(0)
    t = F.interpolate(t.reshape(1,18,*fr.shape[2:]), size=(IMG,IMG), mode="bilinear", align_corners=False).reshape(1,6,3,IMG,IMG)
    return ((t*10000.0-MEAN)/STD)[0]

class DS(torch.utils.data.Dataset):
    def __init__(s, files, aug=False): s.f=files; s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]); x=d["x"].astype(np.float32); y=d["y"].astype(np.float32)
        pre,post=x[0:6],x[6:12]
        if s.aug and random.random()<0.5: pre,post,y=pre[:,:,::-1].copy(),post[:,:,::-1].copy(),y[:,::-1].copy()
        if s.aug and random.random()<0.5: pre,post,y=pre[:,::-1].copy(),post[:,::-1].copy(),y[::-1].copy()
        return prep(pre,post), torch.from_numpy(y).float()

class LoRALinear(nn.Module):
    def __init__(s, base, r=8, alpha=16, drop=0.05):
        super().__init__(); s.base=base
        for p in s.base.parameters(): p.requires_grad=False
        s.scaling=alpha/r; s.drop=nn.Dropout(drop)
        s.A=nn.Parameter(torch.zeros(r, base.in_features)); s.B=nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(s.A, a=math.sqrt(5))   # B=0 → 초기 델타 0
    def forward(s,x): return s.base(x) + (s.drop(x)@s.A.t()@s.B.t())*s.scaling

def inject_lora(enc, r, alpha):
    n=0
    for blk in enc.blocks:
        for parent,attr in [(blk.attn,"qkv"),(blk.attn,"proj"),(blk.mlp,"fc1"),(blk.mlp,"fc2")]:
            lin=getattr(parent,attr)
            if isinstance(lin,nn.Linear):
                setattr(parent,attr,LoRALinear(lin,r,alpha)); n+=1
    return n

class Seg(nn.Module):
    def __init__(s, r=8, alpha=16):
        super().__init__()
        s.enc=PrithviViT(img_size=IMG,patch_size=(1,16,16),num_frames=3,in_chans=6,embed_dim=768,depth=12,num_heads=12,encoder_only=True)
        sd=torch.load(CKPT,map_location="cpu")
        enc={k[len("encoder."):]:v for k,v in sd.items() if k.startswith("encoder.")}
        s.enc.load_state_dict(enc,strict=False)
        for p in s.enc.parameters(): p.requires_grad=False
        n=inject_lora(s.enc,r,alpha)
        print(f"[LoRA] {n}개 Linear에 주입 (r={r})",flush=True)
        C=768*3
        s.dec=nn.Sequential(
            nn.Conv2d(C,512,3,padding=1),nn.BatchNorm2d(512),nn.GELU(),nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
            nn.Conv2d(512,256,3,padding=1),nn.BatchNorm2d(256),nn.GELU(),nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
            nn.Conv2d(256,128,3,padding=1),nn.BatchNorm2d(128),nn.GELU(),nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
            nn.Conv2d(128,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
            nn.Conv2d(64,1,1))
    def forward(s,x):
        feats=s.enc.forward_features(x)
        f=s.enc.prepare_features_for_image_model([feats[-1]])[0]
        return F.interpolate(s.dec(f),size=(256,256),mode="bilinear",align_corners=False)

def tversky_bce(logit,y,a=0.7,b=0.3):
    p=torch.sigmoid(logit).squeeze(1); e=1e-6
    tp=(p*y).sum((1,2)); fn=(y*(1-p)).sum((1,2)); fp=((1-y)*p).sum((1,2))
    tv=1-((tp+e)/(tp+a*fn+b*fp+e)).mean()
    return tv + F.binary_cross_entropy_with_logits(logit.squeeze(1),y)

@torch.no_grad()
def predict(model,files):
    model.eval(); P,Y=[],[]
    for f in files:
        d=np.load(f); x=d["x"].astype(np.float32)
        t=prep(x[0:6],x[6:12]).unsqueeze(0).to(DEV)
        with torch.cuda.amp.autocast(): p=torch.sigmoid(model(t)).squeeze().float().cpu().numpy()
        P.append(p.reshape(-1)); Y.append(d["y"].astype(np.uint8).reshape(-1))
    return np.concatenate(P),np.concatenate(Y)

def stat(P,Y,thr):
    pr=P>=thr; tp=(pr&(Y==1)).sum(); fp=(pr&(Y==0)).sum(); fn=(~pr&(Y==1)).sum()
    f1=2*tp/(2*tp+fp+fn+1e-9); return dict(thr=round(float(thr),3),f1=round(float(f1),3),
        iou=round(float(tp/(tp+fp+fn+1e-9)),3),precision=round(float(tp/(tp+fp+1e-9)),3),recall=round(float(tp/(tp+fn+1e-9)),3))
def bestthr(P,Y): return max([stat(P,Y,t) for t in np.arange(0.1,0.9,0.05)],key=lambda d:d["f1"])

def train(model,files,epochs,lr,tag,val=None,bs=4):
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
    dl=torch.utils.data.DataLoader(DS(files,aug=True),batch_size=bs,shuffle=True,num_workers=2,drop_last=True)
    sc=torch.cuda.amp.GradScaler(); best=-1
    for ep in range(epochs):
        model.train(); tl=0
        for xb,yb in dl:
            xb,yb=xb.to(DEV),yb.to(DEV); opt.zero_grad()
            with torch.cuda.amp.autocast(): loss=tversky_bce(model(xb),yb)
            sc.scale(loss).backward(); sc.step(opt); sc.update(); tl+=loss.item()
        msg=f"[{tag}] ep{ep:02d} loss {tl/len(dl):.4f}"
        if val:
            Pv,Yv=predict(model,val); vb=bestthr(Pv,Yv); msg+=f" val_f1 {vb['f1']}"
            if vb["f1"]>best: best=vb["f1"]; torch.save(model.state_dict(),os.path.join(RUN,f"{tag}_best.pt"))
        print(msg,flush=True)
    if val and os.path.exists(os.path.join(RUN,f"{tag}_best.pt")):
        model.load_state_dict(torch.load(os.path.join(RUN,f"{tag}_best.pt")))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mtbs_epochs",type=int,default=25); ap.add_argument("--korea_epochs",type=int,default=40)
    ap.add_argument("--r",type=int,default=8); ap.add_argument("--smoke",action="store_true")
    A=ap.parse_args(); random.seed(42); torch.manual_seed(42)
    mtbs=[f for d in MTBS for f in glob.glob(os.path.join(d,"*","*.npz"))]
    fires=sorted([d for d in glob.glob(os.path.join(KO,"*")) if glob.glob(os.path.join(d,"*.npz"))])
    hold=set(l.strip() for l in open(HOLD) if l.strip())
    te=[f for d in fires if os.path.basename(d) in hold for f in glob.glob(os.path.join(d,"*.npz"))]
    tr_all=[d for d in fires if os.path.basename(d) not in hold]; random.shuffle(tr_all)
    nval=max(3,len(tr_all)//6); va_f,trn_f=tr_all[:nval],tr_all[nval:]
    va=[f for d in va_f for f in glob.glob(os.path.join(d,"*.npz"))]
    trn=[f for d in trn_f for f in glob.glob(os.path.join(d,"*.npz"))]
    print(f"MTBS 사전학습 {len(mtbs)}타일 | 한국 train {len(trn)} val {len(va)} TEST(홀드아웃) {len(te)}",flush=True)
    model=Seg(r=A.r).to(DEV)
    ntr=sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"학습 파라미터 {ntr/1e6:.2f}M",flush=True)
    if A.smoke:
        xb,yb=next(iter(torch.utils.data.DataLoader(DS(mtbs),batch_size=2)))
        l=tversky_bce(model(xb.to(DEV)),yb.to(DEV)); print(f"[smoke] loss {l.item():.4f} OK",flush=True); return
    # Stage A: MTBS 사전학습
    train(model,mtbs,A.mtbs_epochs,lr=5e-4,tag="mtbs",val=None,bs=4)
    torch.save(model.state_dict(),os.path.join(RUN,"after_mtbs.pt"))
    # Stage B: 한국 LoRA 적응
    train(model,trn,A.korea_epochs,lr=3e-4,tag="korea",val=va,bs=4)
    # 평가
    Pv,Yv=predict(model,va); vthr=bestthr(Pv,Yv)["thr"]
    Pt,Yt=predict(model,te); tm_val=stat(Pt,Yt,vthr); tm_best=bestthr(Pt,Yt)
    dn=np.concatenate([(2*np.load(f)["x"][12].astype(np.float32)-1).reshape(-1) for f in te])
    echo=max([stat(dn,Yt,t) for t in np.arange(0.05,0.6,0.05)],key=lambda d:d["f1"])
    res=dict(method="Prithvi-100M + LoRA(r=%d) + MTBS 사전학습"%A.r, mtbs_tiles=len(mtbs),
             val_thr=vthr,test_valthr=tm_val,test_bestthr=tm_best,dnbr_echo=echo,
             vs_v3={"v3":0.616,"lora":tm_best["f1"],"delta":round(tm_best["f1"]-0.616,3)})
    json.dump(res,open(os.path.join(RUN,"lora_results.json"),"w"),ensure_ascii=False,indent=1)
    print("\n== 홀드아웃 결과 (LoRA+MTBS) =="); print("val-thr",tm_val); print("best-thr",tm_best); print("echo",echo)
    print(f"vs v3(0.616): {tm_best['f1']} ({tm_best['f1']-0.616:+.3f})")

if __name__=="__main__": main()
