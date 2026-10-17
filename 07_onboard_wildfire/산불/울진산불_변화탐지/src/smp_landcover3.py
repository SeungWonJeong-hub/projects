"""토지피복 3클래스 (0기타 1식생 2물) — RGB+NIR, WorldCover 라벨. 피해구역은 이진모델 별도.

아카이브 대량 타일(ko_landcover2)로 식생/물만 학습(단순→고IoU). 소각·도시·나대지 제외.
smp UNet(ResNet34) classes=3, CE. 평가=식생/물 IoU.
사용: SEED=1 python3 smp_landcover3.py
"""
import os, glob, json, random, numpy as np, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
DEV="cuda" if torch.cuda.is_available() else "cpu"
SEED=int(os.environ.get("SEED","1")); EPOCHS=int(os.environ.get("EPOCHS","40"))
NCLS=3; NAMES=["기타","식생","물"]; CW=torch.tensor([0.4,1.0,1.3]).to(DEV)

def rgbnir(x12):
    pre,post=x12[0:6],x12[6:12]; return np.concatenate([pre[[0,1,2,3]],post[[0,1,2,3]]],0)
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
from scipy.ndimage import label as cc_label
from segmentation_models_pytorch.losses import LovaszLoss
LOVASZ=LovaszLoss(mode="multiclass")   # 치트키#1: IoU 직접근사 손실
@torch.no_grad()
def get_probs(m,x,tta=False):
    if not tta: return m(x).softmax(1)  # 치트키#4: 8-view(회전4×좌우2) 평균
    acc=None
    for k in range(4):
        xr=torch.rot90(x,k,dims=(2,3))
        for fl in (False,True):
            xi=torch.flip(xr,dims=(3,)) if fl else xr
            p=m(xi).softmax(1)
            if fl: p=torch.flip(p,dims=(3,))
            p=torch.rot90(p,-k,dims=(2,3)); acc=p if acc is None else acc+p
    return acc/8
@torch.no_grad()
def evaluate(m,files,tta=False,wbias=1.0):   # 치트키#2: wbias=물 확률 스케일(argmax 전)
    m.eval(); inter=np.zeros(NCLS); union=np.zeros(NCLS); i_in=0; u_in=0
    for f in files:
        d=np.load(f)
        x=torch.from_numpy(rgbnir(d["x"].astype(np.float32)/1e4)).unsqueeze(0).to(DEV)
        p=get_probs(m,x,tta)[0].cpu().numpy(); p[2]*=wbias
        pr=p.argmax(0); y=d["y_multi"]
        for c in range(NCLS):
            inter[c]+=((pr==c)&(y==c)).sum(); union[c]+=((pr==c)|(y==c)).sum()
        # #2 내륙 물: 바다=이미지경계 접촉 water 컴포넌트 제외
        wy=(y==2); lab,_=cc_label(wy)
        bset=set(lab[0,:])|set(lab[-1,:])|set(lab[:,0])|set(lab[:,-1]); bset.discard(0)
        sea=np.isin(lab,list(bset)) if bset else np.zeros_like(wy)
        keep=~sea
        pw=(pr==2)&keep; yw=(y==2)&keep
        i_in+=(pw&yw).sum(); u_in+=(pw|yw).sum()
    return inter/(union+1e-9), i_in/(u_in+1e-9)
random.seed(SEED);torch.manual_seed(SEED);np.random.seed(SEED)
LC_DATA=os.environ.get("LC_DATA","ko_landcover2")   # ko_landcover2(WorldCover) | ko_landcover_egis(환경부) | "a+b" 결합
_dirs=LC_DATA.split("+")
allf=sorted(sum([glob.glob(os.path.join(PROJ,d,"*","*.npz")) for d in _dirs],[]))
random.shuffle(allf)
n=len(allf); te=allf[:n//6]; va=allf[n//6:n//5]; tr=allf[n//5:]
m=smp.Unet("resnet34",encoder_weights="imagenet",in_channels=8,classes=NCLS).to(DEV)
print(f"[landcover3] train {len(tr)} val {len(va)} TEST {len(te)} | 3클래스 식생/물",flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
dl=torch.utils.data.DataLoader(DS(tr,aug=True),batch_size=8,shuffle=True,num_workers=2,drop_last=True)
best=-1;bestsd=None
for ep in range(EPOCHS):
    m.train()
    for xb,yb in dl:
        xb,yb=xb.to(DEV),yb.to(DEV);opt.zero_grad()
        out=m(xb)
        LW=float(os.environ.get("LOVASZ_W","0.5"))     # LOVASZ_W=0 → CE-only(0.650 복구)
        (F.cross_entropy(out,yb,weight=CW)+LW*LOVASZ(out,yb)).backward();opt.step()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    iou,_=evaluate(m,va); sc=np.nanmean(iou[1:])
    if sc>best: best=sc;bestsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
m.load_state_dict(bestsd)
# 치트키#2: val서 물 bias 그리드서치 (mIoU 최대)
best_b=1.0; best_vm=-1
for b in [1.0,1.15,1.3,1.5,1.7,2.0]:
    iou,_=evaluate(m,va,tta=False,wbias=b); vm=np.nanmean(iou[1:])
    if vm>best_vm: best_vm=vm; best_b=b
print(f"\n[val] 최적 물bias={best_b} (val mIoU={best_vm:.3f})",flush=True)
# 3단계 분리측정: plain / +bias / +bias+TTA
stages=[("plain",False,1.0),("+bias",False,best_b),("+bias+TTA",True,best_b)]
res={}
for tag,tta,b in stages:
    iou,wi=evaluate(m,te,tta=tta,wbias=b)
    res[tag]={"miou":round(float(np.nanmean(iou[1:])),4),"veg":round(float(iou[1]),4),
              "water":round(float(iou[2]),4),"water_inland":round(float(wi),4),"etc":round(float(iou[0]),4)}
    print(f"[TEST {tag:9s}] mIoU={res[tag]['miou']:.3f}  식생={iou[1]:.3f}  물={iou[2]:.3f}  물내륙={wi:.3f}",flush=True)
OUT=os.path.join(PROJ,"runs",os.environ.get("RUN","landcover3_unet")); os.makedirs(OUT,exist_ok=True)
torch.save(bestsd,os.path.join(OUT,"model.pt"))
fin=res["+bias+TTA"]
json.dump({"model":"토지피복3(식생/물)+Lovász","spec":"landcover3","config":"rgbnir","in_ch":8,"n_classes":NCLS,
    "class_names":NAMES,"decoder":"unet","encoder":"resnet34","seed":SEED,
    "loss":"CE+0.5*Lovász","water_bias":best_b,"tta":True,
    "miou":fin["miou"],"per_class_iou":{"기타":fin["etc"],"식생":fin["veg"],"물":fin["water"]},
    "water_inland_iou":fin["water_inland"],"ablation":res},
    open(os.path.join(OUT,"results.json"),"w"),ensure_ascii=False,indent=2)
print(f"[저장] {OUT}",flush=True)
