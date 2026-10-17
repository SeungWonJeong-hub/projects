"""한국 다산불 native 학습 — 단일 산불(울진 1건) 한계 해결.
ko_multi/*/ (화재별 서브디렉토리) → 화재-분리(fire-disjoint) train/test 스플릿.
라벨=실측 경계(raw 픽셀, dNBR 독립). smp.Unet resnet34 13ch, DiceBCE+AMP.
홀드아웃 화재(미학습)로 교차-산불 일반화 측정 = US LORO 0.885의 한국판.
"""
from __future__ import annotations
import os, glob, random, json
import numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp

HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("KO_DATA",os.path.join(HERE,"..","ko_multi"))

# 계절 조건화: 화재 발생일(DOY)을 sin/cos 상수 2채널로 추가(13→15ch).
# 봄 개엽기 dNBR 반전을 모델이 계절로 조건부 해석하게 함. 발생일=디렉토리명 kf_YYYYMMDD_.
import re, datetime as _dt
SEASON=os.environ.get("SEASON","0")=="1"
IN_CH=15 if SEASON else 13
def season_ch(path,H,W):
    m=re.search(r'kf_(\d{8})_',path); s8=m.group(1)
    doy=_dt.date(int(s8[:4]),int(s8[4:6]),int(s8[6:8])).timetuple().tm_yday
    a=2*np.pi*doy/365.25
    return np.concatenate([np.full((1,H,W),np.sin(a),np.float32),
                           np.full((1,H,W),np.cos(a),np.float32)],0)
# ④ dNBR 부호-불변 표현: 봄=음수·여름=양수 반전을 통일. ch12=idx(dnbr)=(dnbr+1)/2.
#   raw=그대로 | abs=|dnbr| | rel=dnbr-배경중앙값(→배경상대, 불탄곳이 봄·여름 모두 배경위)
DNBR_MODE=os.environ.get("DNBR_MODE","raw")
def apply_dnbr(x):
    if DNBR_MODE=="raw": return x
    x=x.copy(); dnbr=2*x[12]-1
    if DNBR_MODE=="abs": x[12]=np.abs(dnbr).astype(np.float32)
    elif DNBR_MODE=="rel":
        r=np.clip(dnbr-np.median(dnbr),-1,1); x[12]=((r+1)/2).astype(np.float32)
    return x
def read_x(f):
    x=apply_dnbr(np.load(f)["x"].astype(np.float32))
    if SEASON: x=np.concatenate([x,season_ch(f,x.shape[1],x.shape[2])],0)
    return x

# ⑤ 소프트 라벨: 경계 폴리곤의 '안 탄 섬' 노이즈를 삭제 대신 '손실 가중치 하향'으로 처리(v5 파괴 회피).
#   경계 안(y>0) 픽셀이 '안 탄 배경(y==0)'과 얼마나 다른가(robust z-score)로 확신도 산정.
#   배경과 비슷=섬 후보=저가중(W_MIN), 배경과 크게 다름=확실한 연소=1.0. 부호 무관(|편차|).
SOFT_LABEL=os.environ.get("SOFT_LABEL","0")=="1"
W_MIN=float(os.environ.get("W_MIN","0.3"))
def soft_weight(x,y):
    w=np.ones(y.shape,np.float32)
    if not SOFT_LABEL: return w
    dnbr=2*x[12]-1; bg=dnbr[y==0]
    if (y>0).sum()>0 and bg.size>50:
        med=np.median(bg); mad=np.median(np.abs(bg-med))*1.4826+1e-3
        ramp=np.clip((np.abs(dnbr-med)/mad-1.0)/2.0,0,1)   # 배경대비 1σ..3σ → 0..1
        w=np.where(y>0,(W_MIN+(1-W_MIN)*ramp),1.0).astype(np.float32)
    return w
OUT=os.path.join(HERE,"..","runs","ko_native"); os.makedirs(OUT,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS=int(os.environ.get("EPOCHS","40")); BATCH=int(os.environ.get("BATCH","6"))
TEST_FRAC=0.22  # 화재의 ~22%를 미학습 홀드아웃으로

class DS(torch.utils.data.Dataset):
    def __init__(s,files,aug=False): s.f=files; s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]); x=apply_dnbr(d["x"].astype(np.float32)); y=d["y"].astype(np.float32)
        w=soft_weight(x,y)  # 소프트 가중치(SOFT_LABEL 아니면 전부 1.0)
        if s.aug:
            if random.random()<0.5: x=x[:,:,::-1].copy(); y=y[:,::-1].copy(); w=w[:,::-1].copy()
            if random.random()<0.5: x=x[:,::-1].copy(); y=y[::-1].copy(); w=w[::-1].copy()
            k=random.randint(0,3)
            if k: x=np.rot90(x,k,axes=(1,2)).copy(); y=np.rot90(y,k).copy(); w=np.rot90(w,k).copy()
            if os.environ.get("AUG_STRONG","0")=="1":  # 밝기·게인 지터(x만, w·y 불변)
                if random.random()<0.5:
                    x=np.clip(x*np.float32(random.uniform(0.9,1.1))+np.float32(random.uniform(-0.03,0.03)),0,1)
        if SEASON: x=np.concatenate([x,season_ch(s.f[i],x.shape[1],x.shape[2])],0)  # 증강 뒤 추가(지터 오염 방지)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w.copy())

def _wbce(p,y,w):  # 픽셀 가중 BCE (w=None이면 균일)
    bce=nn.functional.binary_cross_entropy_with_logits(p,y,reduction='none')
    if w is None: return bce.mean()
    return (bce*w).sum()/(w.sum()+1e-6)

class DiceBCE(nn.Module):
    def __init__(s): super().__init__()
    def forward(s,p,y,w=None):
        y=y.unsqueeze(1); w=w.unsqueeze(1) if w is not None else None
        pb=torch.sigmoid(p); ww=1.0 if w is None else w
        inter=(pb*y*ww).sum((2,3)); dice=1-(2*inter+1)/((pb*ww).sum((2,3))+(y*ww).sum((2,3))+1)
        return _wbce(p,y,w)+dice.mean()

class TverskyBCE(nn.Module):
    """BCE + Tversky. TI=TP/(TP+aFN+bFP). b>a면 FP 강페널티, a>b면 recall 지향.
    소프트 라벨 w로 애매한 픽셀의 tp/fp/fn 기여를 하향(삭제 없이 노이즈 완화)."""
    def __init__(s,a=0.3,b=0.7): super().__init__(); s.a=a; s.b=b
    def forward(s,p,y,w=None):
        y=y.unsqueeze(1); w=w.unsqueeze(1) if w is not None else None
        pb=torch.sigmoid(p); ww=1.0 if w is None else w
        tp=(pb*y*ww).sum((2,3)); fp=(pb*(1-y)*ww).sum((2,3)); fn=((1-pb)*y*ww).sum((2,3))
        tv=1-(tp+1)/(tp+s.a*fn+s.b*fp+1); return _wbce(p,y,w)+tv.mean()

def make_loss():
    L=os.environ.get("LOSS","dicebce")
    if L=="tversky":
        a=float(os.environ.get("TV_A","0.3")); b=float(os.environ.get("TV_B","0.7"))
        lean="recall 지향(FN 페널티↑)" if a>b else ("precision 지향(FP 페널티↑)" if b>a else "균형")
        print(f"[손실] TverskyBCE a={a} b={b} ({lean})",flush=True); return TverskyBCE(a,b)
    print("[손실] DiceBCE",flush=True); return DiceBCE()

def mets(P,Y,thr=None):
    def at(t):
        pr=P>=t; tp=(pr&(Y>0)).sum();fp=(pr&(Y==0)).sum();fn=((~pr)&(Y>0)).sum()
        pe=tp/(tp+fp+1e-9);re=tp/(tp+fn+1e-9); return 2*pe*re/(pe+re+1e-9),tp/(tp+fp+fn+1e-9),pe,re
    if thr is None: thr=float(max(np.linspace(0.1,0.9,33),key=lambda t:at(t)[0]))
    f1,iou,pe,re=at(thr); return dict(thr=round(thr,3),f1=round(float(f1),3),iou=round(float(iou),3),precision=round(float(pe),3),recall=round(float(re),3))

MODEL=os.environ.get("MODEL","unet")  # unet(early-fusion smp) | snunet(샴 변화탐지)
def fwd(m,x):
    """모델 포워드 디스패처. snunet은 pre6(ch0-5)·post6(ch6-11) 분리 입력."""
    if MODEL=="snunet": return m(x[:,:6],x[:,6:12])
    return m(x)
@torch.no_grad()
def infer(m,files):
    m.eval(); P=[];Y=[]
    for i in range(0,len(files),BATCH):
        xb=np.stack([read_x(f) for f in files[i:i+BATCH]])
        yb=np.stack([np.load(f)["y"].astype(np.uint8) for f in files[i:i+BATCH]])
        with torch.autocast(device_type=DEV,dtype=torch.float16,enabled=(DEV=="cuda")):
            p=torch.sigmoid(fwd(m,torch.from_numpy(xb).to(DEV))).float()[:,0].cpu().numpy()
        P.append(p.reshape(-1)); Y.append(yb.reshape(-1))
    return np.concatenate(P),np.concatenate(Y)

def main():
    SEED=int(os.environ.get("SEED","42")); torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    fires=sorted([d for d in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(d,"*.npz"))])
    # 대형화재만 필터(소형 경계노이즈 제거). 디렉토리명 ..._XXXXha에서 면적 파싱.
    MIN_TRAIN_HA=float(os.environ.get("MIN_TRAIN_HA","0"))
    if MIN_TRAIN_HA>0:
        def _ha(d):
            m=re.search(r'_(\d+)ha',os.path.basename(d))
            return int(m.group(1)) if m else 0
        n0=len(fires); fires=[d for d in fires if _ha(d)>=MIN_TRAIN_HA]
        print(f"[대형필터] ≥{MIN_TRAIN_HA:.0f}ha → {len(fires)}/{n0}화재",flush=True)
    # A. 고정 홀드아웃(data/holdout_fixed.txt) — 모든 버전이 같은 test set으로 비교(평가 노이즈 제거).
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
    random.shuffle(tr); nv=max(6,len(tr)//10); va=tr[:nv]; trn=tr[nv:]
    print(f"화재 {len(fires)}개 → train {len(tr_fires)}화재/{len(trn)}타일, val {len(va)}, TEST(미학습 {len(te_fires)}화재) {len(te)}타일",flush=True)
    print(f"홀드아웃 화재: {[os.path.basename(d) for d in te_fires]}",flush=True)
    # dNBR-echo baseline on held-out
    Pe,Ye=[],[]
    for f in te: d=np.load(f); Pe.append(d["x"][12].reshape(-1)); Ye.append(d["y"].reshape(-1))
    echo=mets(np.concatenate(Pe),np.concatenate(Ye))
    print(f"[홀드아웃 dNBR-echo] {echo}",flush=True)
    tl=torch.utils.data.DataLoader(DS(trn,aug=True),batch_size=BATCH,shuffle=True,num_workers=2,drop_last=True)
    print(f"[입력] 모델={MODEL} {'(pre6|post6 샴)' if MODEL=='snunet' else str(IN_CH)+'채널'} (계절={'ON' if SEASON else 'OFF'})",flush=True)
    if MODEL=="snunet":
        from snunet import SNUNetCD; m=SNUNetCD(in_ch=6,out_ch=1,base=int(os.environ.get("SNUNET_BASE","32"))).to(DEV)
    else:
        m=smp.Unet("resnet34",encoder_weights="imagenet",in_channels=IN_CH,classes=1).to(DEV)
    if os.environ.get("NORM","bn")=="group":  # BN→GroupNorm: running통계 안씀→미학습화재 도메인시프트 강건(과소확신 완화)
        def bn2gn(mod):
            for n,ch in mod.named_children():
                if isinstance(ch,nn.BatchNorm2d):
                    c=ch.num_features; gn=nn.GroupNorm(min(32,c),c)
                    with torch.no_grad(): gn.weight.copy_(ch.weight); gn.bias.copy_(ch.bias)
                    setattr(mod,n,gn.to(DEV))
                else: bn2gn(ch)
        bn2gn(m); print("[정규화] BatchNorm→GroupNorm (도메인시프트 강건)",flush=True)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    lf=make_loss(); sc=torch.cuda.amp.GradScaler(enabled=(DEV=="cuda")); best=-1; bs=None
    for ep in range(EPOCHS):
        m.train(); run=0
        for x,y,w in tl:
            x=x.to(DEV);y=y.to(DEV); w=w.to(DEV) if SOFT_LABEL else None; opt.zero_grad()
            with torch.autocast(device_type=DEV,dtype=torch.float16,enabled=(DEV=="cuda")): loss=lf(fwd(m,x),y,w)
            sc.scale(loss).backward(); sc.step(opt); sc.update(); run+=loss.item()
        sch.step()
        Pv,Yv=infer(m,va); vf=mets(Pv,Yv)["f1"]
        if vf>best: best=vf; bs={k:v.cpu().clone() for k,v in m.state_dict().items()}
        if ep%5==0 or ep==EPOCHS-1: print(f"[ep {ep:02d}] loss={run/max(len(tl),1):.3f} val_f1={vf:.3f}",flush=True)
    m.load_state_dict(bs)
    Pv,Yv=infer(m,va); vm=mets(Pv,Yv)
    Pt,Yt=infer(m,te); tm=mets(Pt,Yt,thr=vm["thr"]); tmb=mets(Pt,Yt)
    print(f"\n=== 한국 다산불 native (fire-disjoint) ===")
    print(f"VAL: {vm}")
    print(f"TEST(미학습 화재, val-thr): {tm}")
    print(f"TEST(미학습 화재, best-thr): {tmb}")
    print(f"[대비] 홀드아웃 dNBR-echo {echo['f1']} | 단일산불 울진(동일씬 스플릿) 0.772 | US모델→한국 0.000(붕괴)")
    ENS_TAG=os.environ.get("ENS_TAG")
    if ENS_TAG:  # 앙상블용 홀드아웃 원시확률 저장(시드별)
        np.savez_compressed(os.path.join(OUT,f"ens_{ENS_TAG}.npz"),Pt=Pt,Yt=Yt)
        print(f"[앙상블] 확률저장 ens_{ENS_TAG}.npz",flush=True)
    torch.save(bs, os.path.join(OUT,"ko_native_unet.pt"))
    json.dump(dict(fires=len(fires),test_fires=len(te_fires),val=vm,test_valthr=tm,test_bestthr=tmb,dnbr_echo=echo,
                   holdout=[os.path.basename(d) for d in te_fires]),
              open(os.path.join(OUT,"ko_native_results.json"),"w"),indent=2,ensure_ascii=False)
    print("저장:",OUT)

if __name__=="__main__": main()
