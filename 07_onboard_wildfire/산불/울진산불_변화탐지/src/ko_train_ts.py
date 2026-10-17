"""Stage2 — 시계열(다시점) 학습. 봄 개엽기 dNBR 반전을 '회복 궤적'으로 해소.
입력: ko_ts/*/*.npz  ts[6,6,H,W](t0=pre,1-5=post창) · days[6] · tvalid[6] · dnbr[H,W] · y[H,W].
v3 승리 레시피 계승: resnet34 U-Net + GroupNorm + recall-Tversky(a0.7/b0.3) + 고정 홀드아웃 + dNBR-echo 대비.

MODEL=ef      : 조기융합(6시점×6밴드=36ch + dNBR echo 1ch = 37ch)를 2D U-Net에 입력(기본, 소데이터 강건).
MODEL=convgru : 시점별 공유 CNN stem → ConvGRU 궤적집약 → 디코더(진짜 순차모델).
비교 대상은 native v3 홀드아웃 F1 0.616 / echo마진 +0.145.
"""
from __future__ import annotations
import os, glob, random, json
import numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp

HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("TS_DATA",os.path.join(HERE,"..","ko_ts"))
OUT=os.path.join(HERE,"..","runs","ko_ts"); os.makedirs(OUT,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS=int(os.environ.get("EPOCHS","60")); BATCH=int(os.environ.get("BATCH","6"))
MODEL=os.environ.get("MODEL","ef")
TEST_FRAC=0.22

def idx(a): return ((np.clip(a,-1,1)+1)/2).astype(np.float32)  # [-1,1]→[0,1]

def load_x(f):
    """ts[6,6,H,W]→ 모델 입력. ef=[37,H,W](36 조기융합+echo), convgru=[6,6,H,W]."""
    d=np.load(f); ts=d["ts"].astype(np.float32); dn=idx(d["dnbr"].astype(np.float32))
    if MODEL=="convgru": return ts  # [T,6,H,W]
    T,C,H,W=ts.shape
    return np.concatenate([ts.reshape(T*C,H,W), dn[None]],0)  # [37,H,W]

class DS(torch.utils.data.Dataset):
    def __init__(s,files,aug=False): s.f=files; s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]); x=load_x(s.f[i]); y=d["y"].astype(np.float32)
        if s.aug:  # 공간 증강만(시점축 불변). x는 마지막 2축이 항상 H,W.
            if random.random()<0.5: x=x[...,:,::-1].copy(); y=y[:,::-1].copy()
            if random.random()<0.5: x=x[...,::-1,:].copy(); y=y[::-1,:].copy()
            k=random.randint(0,3)
            if k: x=np.rot90(x,k,axes=(-2,-1)).copy(); y=np.rot90(y,k).copy()
        return torch.from_numpy(x), torch.from_numpy(y)

def _wbce(p,y): return nn.functional.binary_cross_entropy_with_logits(p,y)
class TverskyBCE(nn.Module):
    def __init__(s,a=0.7,b=0.3): super().__init__(); s.a=a; s.b=b
    def forward(s,p,y):
        y=y.unsqueeze(1); pb=torch.sigmoid(p)
        tp=(pb*y).sum((2,3)); fp=(pb*(1-y)).sum((2,3)); fn=((1-pb)*y).sum((2,3))
        tv=1-(tp+1)/(tp+s.a*fn+s.b*fp+1); return _wbce(p,y)+tv.mean()

def mets(P,Y,thr=None):
    def at(t):
        pr=P>=t; tp=(pr&(Y>0)).sum();fp=(pr&(Y==0)).sum();fn=((~pr)&(Y>0)).sum()
        pe=tp/(tp+fp+1e-9);re=tp/(tp+fn+1e-9); return 2*pe*re/(pe+re+1e-9),tp/(tp+fp+fn+1e-9),pe,re
    if thr is None: thr=float(max(np.linspace(0.1,0.9,33),key=lambda t:at(t)[0]))
    f1,iou,pe,re=at(thr); return dict(thr=round(thr,3),f1=round(float(f1),3),iou=round(float(iou),3),precision=round(float(pe),3),recall=round(float(re),3))

# ---- ConvGRU 시계열 모델 (경량, 소데이터용) ----
class ConvGRUCell(nn.Module):
    def __init__(s,ci,ch):
        super().__init__()
        s.z=nn.Conv2d(ci+ch,ch,3,padding=1); s.r=nn.Conv2d(ci+ch,ch,3,padding=1); s.h=nn.Conv2d(ci+ch,ch,3,padding=1)
    def forward(s,x,h):
        zr=torch.cat([x,h],1); z=torch.sigmoid(s.z(zr)); r=torch.sigmoid(s.r(zr))
        hh=torch.tanh(s.h(torch.cat([x,r*h],1))); return (1-z)*h+z*hh
class TSNet(nn.Module):
    """시점별 공유 stem(1/4 다운) → ConvGRU 궤적집약 → U-Net식 업샘플 디코더."""
    def __init__(s,in_ch=6,base=32):
        super().__init__()
        def blk(i,o,st): return nn.Sequential(nn.Conv2d(i,o,3,st,1),nn.GroupNorm(min(16,o),o),nn.ReLU(inplace=True),
                                              nn.Conv2d(o,o,3,1,1),nn.GroupNorm(min(16,o),o),nn.ReLU(inplace=True))
        s.s1=blk(in_ch,base,2); s.s2=blk(base,base*2,2)          # 1/2,1/4
        s.gru=ConvGRUCell(base*2,base*2)
        s.up1=nn.Sequential(nn.ConvTranspose2d(base*2,base,2,2),nn.GroupNorm(min(16,base),base),nn.ReLU(inplace=True))
        s.up2=nn.Sequential(nn.ConvTranspose2d(base,base,2,2),nn.GroupNorm(min(16,base),base),nn.ReLU(inplace=True))
        s.out=nn.Conv2d(base,1,1)
    def forward(s,x):  # x:[B,T,6,H,W]
        B,T,C,H,W=x.shape; h=None
        for t in range(T):
            f=s.s2(s.s1(x[:,t]))
            if h is None: h=torch.zeros_like(f)
            h=s.gru(f,h)
        return s.out(s.up2(s.up1(h)))

def build_model():
    if MODEL=="convgru":
        m=TSNet(in_ch=6,base=int(os.environ.get("BASE","32"))).to(DEV); print("[모델] ConvGRU 시계열(공유stem+궤적집약)",flush=True); return m
    IN=6*6+1  # 조기융합 37ch
    m=smp.Unet("resnet34",encoder_weights="imagenet",in_channels=IN,classes=1).to(DEV)
    def bn2gn(mod):
        for n,ch in mod.named_children():
            if isinstance(ch,nn.BatchNorm2d):
                c=ch.num_features; gn=nn.GroupNorm(min(32,c),c)
                with torch.no_grad(): gn.weight.copy_(ch.weight); gn.bias.copy_(ch.bias)
                setattr(mod,n,gn.to(DEV))
            else: bn2gn(ch)
    bn2gn(m); print(f"[모델] 조기융합 U-Net resnet34 {IN}ch + GroupNorm",flush=True); return m

@torch.no_grad()
def infer(m,files):
    m.eval(); P=[];Y=[]
    for i in range(0,len(files),BATCH):
        xb=np.stack([load_x(f) for f in files[i:i+BATCH]])
        yb=np.stack([np.load(f)["y"].astype(np.uint8) for f in files[i:i+BATCH]])
        with torch.autocast(device_type=DEV,dtype=torch.float16,enabled=(DEV=="cuda")):
            p=torch.sigmoid(m(torch.from_numpy(xb).to(DEV))).float()[:,0].cpu().numpy()
        P.append(p.reshape(-1)); Y.append(yb.reshape(-1))
    return np.concatenate(P),np.concatenate(Y)

def main():
    SEED=int(os.environ.get("SEED","42")); torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    fires=sorted([d for d in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(d,"*.npz"))])
    HF=os.path.join(HERE,"..","data","holdout_fixed.txt")
    fixed=set(l.strip() for l in open(HF) if l.strip()) if (os.path.exists(HF) and os.environ.get("FIXED_HOLDOUT","1")=="1") else None
    if fixed:
        te_fires=[d for d in fires if os.path.basename(d) in fixed]
        tr_fires=[d for d in fires if os.path.basename(d) not in fixed]
        miss=fixed-set(os.path.basename(d) for d in fires)
        print(f"[고정 홀드아웃] test {len(te_fires)}화재 (누락 {len(miss)}: {sorted(miss)})",flush=True)
    else:
        random.shuffle(fires); nte=max(3,int(len(fires)*TEST_FRAC)); te_fires=fires[:nte]; tr_fires=fires[nte:]
    tr=[f for d in tr_fires for f in glob.glob(os.path.join(d,"*.npz"))]
    te=[f for d in te_fires for f in glob.glob(os.path.join(d,"*.npz"))]
    random.shuffle(tr); nv=max(4,len(tr)//10); va=tr[:nv]; trn=tr[nv:]
    print(f"화재 {len(fires)}개 → train {len(tr_fires)}화재/{len(trn)}타일, val {len(va)}, TEST(미학습 {len(te_fires)}화재) {len(te)}타일",flush=True)
    print(f"홀드아웃 화재: {[os.path.basename(d) for d in te_fires]}",flush=True)
    # dNBR-echo baseline
    Pe,Ye=[],[]
    for f in te: d=np.load(f); Pe.append(idx(d["dnbr"].astype(np.float32)).reshape(-1)); Ye.append(d["y"].reshape(-1))
    echo=mets(np.concatenate(Pe),np.concatenate(Ye))
    print(f"[홀드아웃 dNBR-echo] {echo}",flush=True)
    tl=torch.utils.data.DataLoader(DS(trn,aug=True),batch_size=BATCH,shuffle=True,num_workers=2,drop_last=len(trn)>=BATCH)
    m=build_model()
    a=float(os.environ.get("TV_A","0.7")); b=float(os.environ.get("TV_B","0.3"))
    print(f"[손실] TverskyBCE a={a} b={b} (recall 지향)",flush=True); lf=TverskyBCE(a,b)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    sc=torch.cuda.amp.GradScaler(enabled=(DEV=="cuda")); best=-1; bs=None
    for ep in range(EPOCHS):
        m.train(); run=0
        for x,y in tl:
            x=x.to(DEV);y=y.to(DEV); opt.zero_grad()
            with torch.autocast(device_type=DEV,dtype=torch.float16,enabled=(DEV=="cuda")): loss=lf(m(x),y)
            sc.scale(loss).backward(); sc.step(opt); sc.update(); run+=loss.item()
        sch.step()
        Pv,Yv=infer(m,va); vf=mets(Pv,Yv)["f1"]
        if vf>best: best=vf; bs={k:v.cpu().clone() for k,v in m.state_dict().items()}
        if ep%5==0 or ep==EPOCHS-1: print(f"[ep {ep:02d}] loss={run/max(len(tl),1):.3f} val_f1={vf:.3f}",flush=True)
    m.load_state_dict(bs)
    Pv,Yv=infer(m,va); vm=mets(Pv,Yv)
    Pt,Yt=infer(m,te); tm=mets(Pt,Yt,thr=vm["thr"]); tmb=mets(Pt,Yt)
    margin=round(tmb["f1"]-echo["f1"],3)
    print(f"\n=== 한국 다산불 시계열 ({MODEL}) — fire-disjoint ===")
    print(f"VAL: {vm}")
    print(f"TEST(미학습 화재, val-thr): {tm}")
    print(f"TEST(미학습 화재, best-thr): {tmb}")
    print(f"[대비] 시계열 echo마진 {margin:+.3f} | native v3 F1 0.616/마진+0.145 | echo {echo['f1']} | 울진 0.772")
    torch.save(bs, os.path.join(OUT,f"ko_ts_{MODEL}.pt"))
    json.dump(dict(model=MODEL,fires=len(fires),test_fires=len(te_fires),val=vm,test_valthr=tm,test_bestthr=tmb,
                   dnbr_echo=echo,echo_margin=margin,holdout=[os.path.basename(d) for d in te_fires]),
              open(os.path.join(OUT,f"ko_ts_{MODEL}_results.json"),"w"),indent=2,ensure_ascii=False)
    print("저장:",OUT)

if __name__=="__main__": main()
