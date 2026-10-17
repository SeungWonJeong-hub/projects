"""앙상블 평가 — 여러 학습모델의 확률맵을 평균내어 한국 홀드아웃 F1 측정.

각 run 디렉토리의 results.json(decoder/config/in_ch) 읽어 아키텍처 복원 → model.pt 로드 →
홀드아웃 타일별 sigmoid 확률 예측 → 모델간 평균 → best-thr F1·mIoU·계절별.
단일모델 천장(~0.616)을 앙상블로 정직하게 밀어올리는지 측정.

사용: python3 ensemble_eval.py ens_unet ens_fpn ens_deeplabv3plus [joint2_s1 ...]
"""
import os, sys, glob, json, re, numpy as np, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
HOLD=os.path.join(PROJ,"data","holdout_fixed.txt"); DEV="cuda" if torch.cuda.is_available() else "cpu"
RUNS=sys.argv[1:] or ["ens_unet","ens_fpn","ens_deeplabv3plus"]
_ARCH={"unet":smp.Unet,"unetpp":smp.UnetPlusPlus,"deeplabv3plus":smp.DeepLabV3Plus,"fpn":smp.FPN,"manet":smp.MAnet}
def _nbr(b): return (b[3]-b[5])/(b[3]+b[5]+1e-6)
def build_x(x12,cfg):
    pre,post=x12[0:6],x12[6:12]
    if cfg=="bitemporal": return x12
    if cfg=="rgbnir": return np.concatenate([pre[[0,1,2,3]],post[[0,1,2,3]]],0)  # ET-02(SWIR제외)
    if cfg=="postonly": return post
    if cfg=="postchar":
        nir,red,sw2=post[3],post[2],post[5]
        return np.concatenate([post,((nir-sw2)/(nir+sw2+1e-6))[None],((nir-red)/(nir+red+1e-6))[None]],0)
    if cfg=="biidx":
        nbrp,nbro=_nbr(pre),_nbr(post);dnbr=nbrp-nbro;rbr=dnbr/(nbrp+1.001);dnbrdt=dnbr-np.median(dnbr)
        ndvio=(post[3]-post[2])/(post[3]+post[2]+1e-6)
        return np.concatenate([x12,dnbr[None],rbr[None],dnbrdt[None],nbro[None],ndvio[None]],0)
def load_model(run):
    cfg=json.load(open(os.path.join(PROJ,"runs",run,"results.json")))
    m=_ARCH[cfg.get("decoder","unet")](encoder_name=cfg.get("encoder","resnet34"),encoder_weights=None,
        in_channels=cfg["in_ch"],classes=1).to(DEV)
    m.load_state_dict(torch.load(os.path.join(PROJ,"runs",run,"model.pt"),map_location=DEV)); m.eval()
    return m,cfg["config"]
def stats(P,Y,t):
    p=(P>=t);tp=(p&(Y>0)).sum();fp=(p&(Y==0)).sum();fn=((~p)&(Y>0)).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return 2*pr*rc/(pr+rc+1e-9),pr,rc
def season(mn): return "봄" if 3<=mn<=5 else "여름" if 6<=mn<=8 else "가을" if 9<=mn<=11 else "겨울"
hold=set(l.strip() for l in open(HOLD) if l.strip())
te=[d for d in sorted(glob.glob(os.path.join(PROJ,"ko_refl","*"))) if os.path.basename(d) in hold and glob.glob(os.path.join(d,"*.npz"))]
models=[load_model(r) for r in RUNS]
print(f"앙상블 {len(models)}모델: {RUNS}",flush=True)
allP=[];allY=[];per=defaultdict(lambda:[[],[]])
for d in te:
    for f in glob.glob(os.path.join(d,"*.npz")):
        z=np.load(f);x12=z["x"].astype(np.float32)/10000.0;y=z["y"].astype(np.uint8).reshape(-1)
        probs=[]
        for m,cfg in models:
            with torch.no_grad():
                xt=torch.from_numpy(build_x(x12,cfg)).unsqueeze(0).to(DEV)
                probs.append(torch.sigmoid(m(xt)).squeeze().cpu().numpy().reshape(-1))
        pmean=np.mean(probs,0)                       # 확률맵 평균 = 앙상블
        allP.append(pmean);allY.append(y)
        per[season(int(os.path.basename(d).split("_")[1][4:6]))][0].append(pmean)
        per[season(int(os.path.basename(d).split("_")[1][4:6]))][1].append(y)
Pa=np.concatenate(allP);Ya=np.concatenate(allY)
thr=max(np.arange(0.1,0.9,0.025),key=lambda t:stats(Pa,Ya,t)[0]);f1,pr,rc=stats(Pa,Ya,thr)
print(f"\n[앙상블] thr={thr:.2f} F1={f1:.3f} mIoU={f1/(2-f1):.3f} P={pr:.3f} R={rc:.3f}  (단일 native 0.616)",flush=True)
for se in ["봄","여름","가을","겨울"]:
    if se in per:
        pp=np.concatenate(per[se][0]);yy=np.concatenate(per[se][1]);print(f"   {se}: F1={stats(pp,yy,thr)[0]:.3f}",flush=True)
