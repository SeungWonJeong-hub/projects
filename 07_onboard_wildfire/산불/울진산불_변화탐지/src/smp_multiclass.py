"""다중클래스 semantic 변화탐지 — RGB+NIR(ET-02) → 6클래스 (WorldCover+산림청 라벨).

클래스: 0기타 1산림 2물 3도시 4나대지 5소각. 입력=8ch(B02·B03·B04·B8A×pre/post) bi-temporal.
smp UNet(ResNet34) classes=6, CE loss(소각 가중). 평가=클래스별 IoU + 소각 F1.
사용: SEED=1 python3 smp_multiclass.py
"""
import os, glob, re, json, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
DATA=os.path.join(PROJ,"ko_refl"); HOLD=os.path.join(PROJ,"data","holdout_fixed.txt")
DEV="cuda" if torch.cuda.is_available() else "cpu"
SEED=int(os.environ.get("SEED","1")); EPOCHS=int(os.environ.get("EPOCHS","40"))
DECODER=os.environ.get("DECODER","unet"); MIN_HA=float(os.environ.get("MIN_HA","500"))
NCLS=6; NAMES=["기타","산림","물","도시","나대지","소각"]
CW=torch.tensor([0.3,1.0,1.0,1.5,1.0,3.0]).to(DEV)   # 소각·도시 가중(희소)
_ARCH={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}

def rgbnir(x12):  # [12] → [8] B02,B03,B04,B8A ×pre,post
    pre,post=x12[0:6],x12[6:12]
    return np.concatenate([pre[[0,1,2,3]],post[[0,1,2,3]]],0)
class DS(torch.utils.data.Dataset):
    def __init__(s,files,aug=False): s.f=files;s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]);x=rgbnir(d["x"].astype(np.float32)/1e4);y=d["y_multi"].astype(np.int64)
        if s.aug:
            k=random.randint(0,3)
            if k: x,y=np.rot90(x,k,axes=(1,2)).copy(),np.rot90(y,k).copy()
            if random.random()<0.5: x,y=x[:,:,::-1].copy(),y[:,::-1].copy()
            if random.random()<0.5: x,y=x[:,::-1].copy(),y[::-1].copy()
        return torch.from_numpy(x),torch.from_numpy(y)
tf=lambda ds:[f for d in ds for f in glob.glob(os.path.join(d,"*.npz")) if "y_multi" in np.load(f)]
def fire_ha(d):
    m=re.search(r"_(\d+)ha$",os.path.basename(d)); return float(m.group(1)) if m else 9999.0
@torch.no_grad()
def evaluate(m,files):
    m.eval(); inter=np.zeros(NCLS); union=np.zeros(NCLS); tpb=fpb=fnb=0
    for f in files:
        d=np.load(f)
        if "y_multi" not in d: continue
        x=torch.from_numpy(rgbnir(d["x"].astype(np.float32)/1e4)).unsqueeze(0).to(DEV)
        pr=m(x).argmax(1)[0].cpu().numpy(); y=d["y_multi"]
        for c in range(NCLS):
            inter[c]+=((pr==c)&(y==c)).sum(); union[c]+=((pr==c)|(y==c)).sum()
        tpb+=((pr==5)&(y==5)).sum(); fpb+=((pr==5)&(y!=5)).sum(); fnb+=((pr!=5)&(y==5)).sum()
    iou=inter/(union+1e-9); f1b=2*tpb/(2*tpb+fpb+fnb+1e-9)
    return iou, f1b
random.seed(SEED);torch.manual_seed(SEED);np.random.seed(SEED)
hold=set(l.strip() for l in open(HOLD) if l.strip())
allf=sorted([d for d in glob.glob(os.path.join(DATA,"*")) if tf([d])])
lc=sorted([d for d in glob.glob(os.path.join(PROJ,"ko_landcover","*")) if tf([d])])
lc2=sorted([d for d in glob.glob(os.path.join(PROJ,"ko_landcover2","*")) if tf([d])])  # 아카이브 대량
random.Random(42).shuffle(lc2)
lc2_te=lc2[:max(6,len(lc2)//6)]; lc2_tr=lc2[len(lc2_te):]  # 다양성 일부 홀드아웃(도시/나대지 측정용)
te_d=[d for d in allf if os.path.basename(d) in hold]+lc2_te  # 화재(소각) + 다양성(토지피복)
tr_all=[d for d in allf if os.path.basename(d) not in hold and fire_ha(d)>=MIN_HA]+lc+lc2_tr
random.shuffle(tr_all)
nval=max(2,len(tr_all)//6);va_d=tr_all[:nval];trn_d=tr_all[nval:]
trn_f,va_f,te_f=tf(trn_d),tf(va_d),tf(te_d)
m=_ARCH[DECODER]("resnet34",encoder_weights="imagenet",in_channels=8,classes=NCLS).to(DEV)
print(f"[multiclass {DECODER}] train {len(trn_f)}타일 val {len(va_f)} TEST {len(te_f)} | 6클래스 RGB+NIR",flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
dl=torch.utils.data.DataLoader(DS(trn_f,aug=True),batch_size=8,shuffle=True,num_workers=2,drop_last=True)
best=-1;bestsd=None
for ep in range(EPOCHS):
    m.train()
    for xb,yb in dl:
        xb,yb=xb.to(DEV),yb.to(DEV);opt.zero_grad()
        F.cross_entropy(m(xb),yb,weight=CW).backward();opt.step()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    iou,f1b=evaluate(m,va_f); score=np.nanmean(iou[1:])  # 소각+토지 mIoU(배경제외)
    if score>best: best=score;bestsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
m.load_state_dict(bestsd)
iou,f1b=evaluate(m,te_f)
print(f"\n[결과] TEST 소각F1={f1b:.3f} | mIoU(1-5)={np.nanmean(iou[1:]):.3f}",flush=True)
for c in range(1,NCLS): print(f"   {NAMES[c]}: IoU={iou[c]:.3f}",flush=True)
OUT=os.path.join(PROJ,"runs",f"multiclass_{DECODER}_s{SEED}"); os.makedirs(OUT,exist_ok=True)
torch.save(bestsd,os.path.join(OUT,"model.pt"))
json.dump({"model":f"다중클래스 semantic ({DECODER})","spec":"rgbnir_multiclass","config":"rgbnir",
    "in_ch":8,"n_classes":NCLS,"class_names":NAMES,"decoder":DECODER,"encoder":"resnet34","arch":"smp-multiclass-rgbnir",
    "burn_f1":round(float(f1b),4),"miou":round(float(np.nanmean(iou[1:])),4),
    "per_class_iou":{NAMES[c]:round(float(iou[c]),4) for c in range(NCLS)},"seed":SEED},
    open(os.path.join(OUT,"results.json"),"w"),ensure_ascii=False,indent=2)
print(f"[저장] {OUT}",flush=True)
