"""탄탄한 Early Fusion U-Net (segmentation_models_pytorch) — baseline·사전학습·파인튜닝 통합.

자작 경량 GN-UNet → 표준 검증 smp U-Net(ResNet34/ImageNet, in_ch=12 concat pre|post).
계절-매칭 사전학습(일본+미동부 봄 leaf-off) → 한국 파인튜닝 워크플로우 지원.
입력 = pre|post 12ch concat = Early Fusion(복잡한 siamese 아님).

env:
  ENCODER=resnet34  INIT=imagenet|<ckpt.pt>  TRAIN=korea|japan|eastus(콤마조합)
  CONFIG=bitemporal|postonly|postchar  MIN_HA=500  SEED=1  EPOCHS=40  SAVE=<name>
평가는 항상 한국 고정 홀드아웃(계절별). 사용례:
  # baseline:  TRAIN=korea INIT=imagenet SAVE=ko_smp
  # pretrain:  TRAIN=japan,eastus INIT=imagenet SAVE=pre_jpeus
  # finetune:  TRAIN=korea INIT=../runs/pre_jpeus/model.pt SAVE=ft_jpeus
"""
import os, glob, re, json, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
HOLD=os.path.join(PROJ,"data","holdout_fixed.txt")
DEV="cuda" if torch.cuda.is_available() else "cpu"
# DS_KOREA env로 한국 데이터셋 디렉터리 교체(10m A/B: ko_refl_10m_b08). 기본=기존 20m.
DSDIR={"korea":os.environ.get("DS_KOREA","ko_refl"),"japan":"jp_refl","eastus":"eastus_refl",
       "japan_harm":"jp_refl_harm","eastus_harm":"eastus_refl_harm"}  # 라벨 하모나이즈(팽창+밀도필터)
ENCODER=os.environ.get("ENCODER","resnet34")
DECODER=os.environ.get("DECODER","unet")   # unet|unetpp|deeplabv3plus|fpn|manet
INIT=os.environ.get("INIT","imagenet")
TRAIN=os.environ.get("TRAIN","korea").split(",")
CONFIG=os.environ.get("CONFIG","bitemporal")
MIN_HA=float(os.environ.get("MIN_HA","500")); SEED=int(os.environ.get("SEED","1"))
EPOCHS=int(os.environ.get("EPOCHS","40")); SAVE=os.environ.get("SAVE","")
FREEZE=int(os.environ.get("FREEZE","0"))      # 파인튜닝: 초반 인코더 스테이지 동결 수(stem+layer1..FREEZE)
LR=float(os.environ.get("LR","3e-4"))          # 파인튜닝 땐 1e-4 권장
BIGNORE=int(os.environ.get("BIGNORE","0"))    # under-label 대응: 라벨 경계 ±N px loss 무시(0=off)
SCENE_HEAD=int(os.environ.get("SCENE_HEAD","0"))   # 씬 단위 화재유무 분류 헤드(aux). 화재없는 타일 통째 게이팅→오탐↓
SCENE_LW=float(os.environ.get("SCENE_LW","0.5"))   # 씬 분류 손실 가중(seg + λ·BCE_scene)
NEG_DS=os.environ.get("NEG_DS","")   # 무화재 음성 타일 디렉터리(예 ko_neg_s50). 비홀드아웃→학습, 홀드아웃→씬게이팅 평가
# post_rgbnir = 온보드 단일시기 시나리오(pre 영상 없이 사후 영상만). ET-02(8ch)와 공정 비교용.
IN_CH={"bitemporal":12,"postonly":6,"postchar":8,"biidx":17,"rgbnir":8,"post_rgbnir":4}[CONFIG]

def _nbr(b): return (b[3]-b[5])/(b[3]+b[5]+1e-6)   # (NIR B8A - SWIR2 B12)
def build_x(x12):
    pre,post=x12[0:6],x12[6:12]
    if CONFIG=="bitemporal": return x12
    if CONFIG=="rgbnir":     return np.concatenate([pre[[0,1,2,3]],post[[0,1,2,3]]],0)  # B02,B03,B04,NIR ×pre/post (SWIR제외=ET-02)
    if CONFIG=="post_rgbnir": return post[[0,1,2,3]]        # 사후만 4ch — pre 없는 온보드 운용
    if CONFIG=="postonly":   return post
    if CONFIG=="postchar":
        nir,red,sw2=post[3],post[2],post[5]
        nbr=(nir-sw2)/(nir+sw2+1e-6); ndvi=(nir-red)/(nir+red+1e-6)
        return np.concatenate([post,nbr[None],ndvi[None]],0)
    if CONFIG=="biidx":                             # 12 raw + 산불지수 5 = 17ch
        nbrp,nbro=_nbr(pre),_nbr(post); dnbr=nbrp-nbro
        rbr=dnbr/(nbrp+1.001)                        # Relative Burn Ratio
        dnbrdt=dnbr-np.median(dnbr)                  # detrended: 균일 green-up 제거(봄 강건)
        ndvio=(post[3]-post[2])/(post[3]+post[2]+1e-6)
        return np.concatenate([x12,dnbr[None],rbr[None],dnbrdt[None],nbro[None],ndvio[None]],0)
def ignore_weight(y):     # 라벨 경계 ±BIGNORE px = 정답불확실 → weight 0 (loss에서 제외)
    if BIGNORE<=0: return np.ones_like(y)
    from scipy.ndimage import binary_dilation,binary_erosion
    yb=y>0.5
    band=binary_dilation(yb,iterations=BIGNORE)&~binary_erosion(yb,iterations=BIGNORE)
    return (~band).astype(np.float32)
class DS(torch.utils.data.Dataset):
    def __init__(s,files,aug=False): s.f=files;s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]);x=build_x(d["x"].astype(np.float32)/10000.0);y=d["y"].astype(np.float32)
        if s.aug:
            k=random.randint(0,3)                                   # 위성=상하개념無 → 90/180/270 회전
            if k: x,y=np.rot90(x,k,axes=(1,2)).copy(),np.rot90(y,k).copy()
            if random.random()<0.5: x,y=x[:,:,::-1].copy(),y[:,::-1].copy()   # 좌우
            if random.random()<0.5: x,y=x[:,::-1].copy(),y[::-1].copy()       # 상하
        return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(ignore_weight(y))
def tversky_bce(l,y,w,a=float(os.environ.get("TA","0.7")),b=float(os.environ.get("TB","0.3"))):  # TA/TB로 α/β 조정

    p=torch.sigmoid(l).squeeze(1);e=1e-6
    tp=(w*p*y).sum((1,2));fn=(w*y*(1-p)).sum((1,2));fp=(w*(1-y)*p).sum((1,2))
    bcep=F.binary_cross_entropy_with_logits(l.squeeze(1),y,reduction="none")
    return 1-((tp+e)/(tp+a*fn+b*fp+e)).mean()+(w*bcep).sum()/(w.sum()+1e-6)
def dice_focal(l,y,w,gamma=2.0,alpha=0.25):   # 소수 burn픽셀 집중(불균형 대응 대안)
    lg=l.squeeze(1);p=torch.sigmoid(lg);e=1e-6
    tp=(w*p*y).sum((1,2));fp=(w*(1-y)*p).sum((1,2));fn=(w*y*(1-p)).sum((1,2))
    dice=1-((2*tp+e)/(2*tp+fp+fn+e)).mean()
    bce=F.binary_cross_entropy_with_logits(lg,y,reduction="none");pt=torch.exp(-bce)
    focal=(w*alpha*(1-pt)**gamma*bce).sum()/(w.sum()+1e-6)
    return dice+focal
LOSS={"tversky_bce":tversky_bce,"dice_focal":dice_focal}[os.environ.get("LOSS","tversky_bce")]
def stats(P,Y,t):
    p=(P>=t);tp=(p&(Y>0)).sum();fp=(p&(Y==0)).sum();fn=((~p)&(Y>0)).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return 2*pr*rc/(pr+rc+1e-9),pr,rc
@torch.no_grad()
def predict(m,files):
    m.eval();P=[];Y=[]
    for f in files:
        d=np.load(f);x=torch.from_numpy(build_x(d["x"].astype(np.float32)/10000.0)).unsqueeze(0).to(DEV)
        o=m(x);mask=o[0] if isinstance(o,tuple) else o   # 씬헤드시 (mask,scene_logit) → mask만
        P.append(torch.sigmoid(mask).squeeze().cpu().numpy().reshape(-1));Y.append(d["y"].astype(np.uint8).reshape(-1))
    return np.concatenate(P),np.concatenate(Y)
def season(mn): return "봄" if 3<=mn<=5 else "여름" if 6<=mn<=8 else "가을" if 9<=mn<=11 else "겨울"
def fire_ha(d):
    m=re.search(r"_(\d+)ha$",os.path.basename(d)); return float(m.group(1)) if m else 9999.0
tf=lambda ds:[f for d in ds for f in glob.glob(os.path.join(d,"*.npz"))]
def fires(key):
    return sorted([d for d in glob.glob(os.path.join(PROJ,DSDIR[key],"*")) if glob.glob(os.path.join(d,"*.npz"))])

random.seed(SEED);torch.manual_seed(SEED);np.random.seed(SEED)
hold=set(l.strip() for l in open(HOLD) if l.strip())
ko=fires("korea"); te_f=[d for d in ko if os.path.basename(d) in hold]      # 평가=한국 홀드아웃(고정)
# 학습 풀 구성
trn_pool=[]
for k in TRAIN:
    ff=fires(k)
    if k=="korea": ff=[d for d in ff if os.path.basename(d) not in hold and fire_ha(d)>=MIN_HA]
    trn_pool+=ff
if NEG_DS:   # 무화재 음성: 비홀드아웃 화재폴더의 음성타일을 학습풀에 추가(씬헤드가 화재/무화재 판별 학습)
    negdirs=[d for d in sorted(glob.glob(os.path.join(PROJ,NEG_DS,"*")))
             if os.path.basename(d) not in hold and glob.glob(os.path.join(d,"*.npz"))]
    NEG_MAX=int(os.environ.get("NEG_MAX","0"))    # 0=전부. 양성과 균형위해 캡(예 양성수). 클래스불균형→씬헤드 편향/ seg붕괴 방지
    if NEG_MAX>0 and len(negdirs)>NEG_MAX:
        random.shuffle(negdirs); negdirs=negdirs[:NEG_MAX]
    trn_pool+=negdirs
    print(f"[neg] 음성 화재폴더 {len(negdirs)}개(무화재 타일) 학습풀 추가 (양성 {len(trn_pool)-len(negdirs)}폴더)",flush=True)
random.shuffle(trn_pool)
nval=max(2,len(trn_pool)//6); va_d=trn_pool[:nval]; trn_d=trn_pool[nval:]
trn_f=tf(trn_d); va_f=tf(va_d)                 # 화재폴더 → npz파일 (fire레벨 split 유지=타일누수 방지)
KO_REPEAT=int(os.environ.get("KO_REPEAT","1"))  # 조인트학습: 한국 타일 오버샘플(해외 다수에 안 묻히게)
if KO_REPEAT>1:
    ko_tiles=[f for f in trn_f if os.sep+"ko_refl"+os.sep in f]
    trn_f=trn_f+ko_tiles*(KO_REPEAT-1)
    print(f"[oversample] 한국 타일 ×{KO_REPEAT} (해외와 균형)",flush=True)

_ARCH={"unet":smp.Unet,"unetpp":smp.UnetPlusPlus,"deeplabv3plus":smp.DeepLabV3Plus,
       "fpn":smp.FPN,"manet":smp.MAnet}[DECODER]
_aux=dict(classes=1,pooling="avg",dropout=0.2) if SCENE_HEAD else None
m=_ARCH(encoder_name=ENCODER,encoder_weights=("imagenet" if INIT=="imagenet" else None),
        in_channels=IN_CH,classes=1,aux_params=_aux).to(DEV)
if INIT!="imagenet" and os.path.exists(INIT):
    sd=torch.load(INIT,map_location="cpu"); m.load_state_dict(sd); print(f"[init] 사전학습 로드 {INIT}",flush=True)
frozen=0
if FREEZE>0:                                    # 초반 인코더 동결(저수준 특징 고정) → 디코더 위주 학습
    enc=m.encoder
    mods=[enc.conv1,enc.bn1]+[enc.layer1,enc.layer2,enc.layer3,enc.layer4][:FREEZE]
    for mod in mods:
        for p in mod.parameters(): p.requires_grad=False; frozen+=p.numel()
    print(f"[freeze] stem+layer1..{FREEZE} 동결 ({frozen/1e6:.1f}M 파라미터 고정)",flush=True)
ntr=sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"[smp {DECODER}/{ENCODER}|{CONFIG}|in{IN_CH}] TRAIN={TRAIN} {len(trn_f)}타일/val{len(va_f)} | TEST 한국 {len(te_f)}화재 | init={INIT if INIT=='imagenet' else 'ckpt'} | 학습param {ntr/1e6:.1f}M LR={LR}",flush=True)
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=LR,weight_decay=1e-4)
dl=torch.utils.data.DataLoader(DS(trn_f,aug=True),batch_size=8,shuffle=True,num_workers=2,drop_last=True)
best=-1;bestsd=None
for ep in range(EPOCHS):
    m.train()
    for xb,yb,wb in dl:
        xb,yb,wb=xb.to(DEV),yb.to(DEV),wb.to(DEV);opt.zero_grad()
        if SCENE_HEAD:
            mask,slog=m(xb)
            slab=(yb.flatten(1).amax(1)>0.5).float()                        # 타일에 burn 픽셀 존재?
            loss=LOSS(mask,yb,wb)+SCENE_LW*F.binary_cross_entropy_with_logits(slog.squeeze(1),slab)
        else:
            loss=LOSS(m(xb),yb,wb)
        loss.backward();opt.step()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    Pv,Yv=predict(m,va_f);vb=max(stats(Pv,Yv,t)[0] for t in np.arange(0.1,0.9,0.025))
    if vb>best: best=vb;bestsd={k:v.cpu().clone() for k,v in m.state_dict().items()}
m.load_state_dict(bestsd)
# 한국 홀드아웃 평가
allP=[];allY=[];per=defaultdict(lambda:[[],[]])
for d in te_f:
    pp,yy=predict(m,glob.glob(os.path.join(d,"*.npz")));allP.append(pp);allY.append(yy)
    se=season(int(os.path.basename(d).split("_")[1][4:6]));per[se][0].append(pp);per[se][1].append(yy)
Pa=np.concatenate(allP);Ya=np.concatenate(allY)
thr=max(np.arange(0.1,0.9,0.025),key=lambda t:stats(Pa,Ya,t)[0]);f1,pr,rc=stats(Pa,Ya,thr)
iou=f1/(2-f1)                                   # 이진 IoU=F1/(2-F1)
ps={se:round(float(stats(np.concatenate(per[se][0]),np.concatenate(per[se][1]),thr)[0]),3) for se in per}
print(f"\n[결과] TRAIN={'+'.join(TRAIN)} decoder={DECODER} loss={os.environ.get('LOSS','tversky_bce')} bignore={BIGNORE} freeze={FREEZE} init={'ckpt' if INIT!='imagenet' else 'imagenet'} | thr={thr:.2f} F1={f1:.3f} mIoU={iou:.3f} P={pr:.3f} R={rc:.3f}",flush=True)
for se in ["봄","여름","가을","겨울"]:
    if se in ps: print(f"   {se}: F1={ps[se]:.3f} IoU={ps[se]/(2-ps[se]):.3f}",flush=True)
scene_report=None
if SCENE_HEAD:
    # 씬 게이팅: scene_prob<τ 인 타일은 마스크를 통째로 0 → 화재없는 장면 오탐 제거
    # 평가셋 = 화재 홀드아웃(양성) + 홀드아웃 화재폴더의 음성타일(무화재). 음성이 있어야 게이팅 효과 측정 가능.
    m.eval();rows=[]   # (pixel_prob, y_flat, scene_prob, is_fire)
    def _collect(files):
        for f in files:
            dd=np.load(f);x=torch.from_numpy(build_x(dd["x"].astype(np.float32)/10000.0)).unsqueeze(0).to(DEV)
            mask,slog=m(x)
            pp=torch.sigmoid(mask).squeeze().cpu().numpy().reshape(-1)
            sp=float(torch.sigmoid(slog).reshape(-1)[0].cpu())
            yy=dd["y"].astype(np.uint8).reshape(-1);rows.append((pp,yy,sp,int(yy.sum()>0)))
    with torch.no_grad():
        for d in te_f: _collect(glob.glob(os.path.join(d,"*.npz")))
        if NEG_DS:
            neg_eval=[f for h in hold for f in glob.glob(os.path.join(PROJ,NEG_DS,h,"*.npz"))]
            _collect(neg_eval)
    npos=sum(r[3] for r in rows);nneg=len(rows)-npos
    sp_all=np.array([r[2] for r in rows]);isf=np.array([r[3] for r in rows])
    def gated(tau):
        Ps=[pp*(1.0 if sp>=tau else 0.0) for pp,_,sp,_ in rows];Ys=[yy for _,yy,_,_ in rows]
        return stats(np.concatenate(Ps),np.concatenate(Ys),thr)
    print(f"\n[씬헤드 게이팅] pixel thr={thr:.2f} · 평가 타일 {len(rows)}(화재 {npos}/무화재 {nneg})",flush=True)
    g0=gated(0.0);print(f"  τ=0.00(무게이팅) F1={g0[0]:.3f} P={g0[1]:.3f} R={g0[2]:.3f}",flush=True)
    scene_report={"npos":npos,"nneg":nneg,"ungated":{"f1":round(float(g0[0]),4),"p":round(float(g0[1]),4),"r":round(float(g0[2]),4)},"gated":{}}
    for tau in [0.3,0.5,0.7,0.9]:
        gg=gated(tau)
        negfilt=int(((sp_all<tau)&(isf==0)).sum());posdrop=int(((sp_all<tau)&(isf==1)).sum())
        print(f"  τ={tau:.2f} F1={gg[0]:.3f} P={gg[1]:.3f} R={gg[2]:.3f}  (무화재 {negfilt}/{nneg} 제거·화재 오제거 {posdrop}/{npos})",flush=True)
        scene_report["gated"][f"{tau:.2f}"]={"f1":round(float(gg[0]),4),"p":round(float(gg[1]),4),"r":round(float(gg[2]),4),"neg_filtered":negfilt,"pos_dropped":posdrop}
    if nneg>0 and npos>0:
        acc=float(((sp_all>=0.5).astype(int)==isf).mean());scene_report["scene_cls_acc"]=round(acc,4)
        print(f"  [씬분류] acc={acc:.3f} (화재 {npos} vs 무화재 {nneg})",flush=True)
if SAVE:
    OUT=os.path.join(PROJ,"runs",SAVE); os.makedirs(OUT,exist_ok=True)
    torch.save(bestsd,os.path.join(OUT,"model.pt"))
    json.dump({"encoder":ENCODER,"decoder":DECODER,"config":CONFIG,"in_ch":IN_CH,"train":TRAIN,"init":INIT,"bignore":BIGNORE,"freeze":FREEZE,
        "loss":os.environ.get("LOSS","tversky_bce"),"f1":round(float(f1),4),"miou":round(float(iou),4),
        "p":round(float(pr),4),"r":round(float(rc),4),"thr":round(float(thr),3),
        "per_season":ps,"seed":SEED,"n_train_tiles":len(trn_f),
        "scene_head":SCENE_HEAD,"scene_lw":SCENE_LW if SCENE_HEAD else None,"scene_report":scene_report},
        open(os.path.join(OUT,"results.json"),"w"),ensure_ascii=False,indent=2)
    print(f"[저장] {OUT} (F1={f1:.3f})",flush=True)
