"""#해결책: rehearsal 파인튜닝 — 파국망각 방지.
CIR-GT 봄타일(6화재) + 원래 산림청 ko_refl 타일(홀드아웃8 제외) 혼합 학습 → 단일 deeplab.
평가 양쪽: 봄 cross-fire(4 CIR 홀드아웃) + in-domain(산림청 8 홀드아웃).
목표: 봄 올리되(baseline단일0.097) 산림청(0.687) 유지. 둘다 좋으면 9모델 확장·배포.
사용: EPOCHS=15 LR=3e-5 python step4_rehearsal.py
시드 재현: SEED=1 (음성 홀드아웃은 NEG_SPLIT_SEED 로 따로 고정된다)
"""
import os, glob, random, json, numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
# 경로는 팀 규약 SYNTAX_INFERENCE_ROOT 사용(기본 ~/syntax_inference). WEIGHTS는 추가 오버라이드 허용.
_ROOT=os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference"))
CIRD=os.path.join(_ROOT, "Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires")
WEIGHTS=os.environ.get("WEIGHTS", os.path.join(_ROOT, "Steven/웹앱/weights"))
DEV="cuda" if torch.cuda.is_available() else "cpu"
# ★SEED 는 학습 난수(타일 순서·회전/뒤집기)만 흔든다. 기본 0 이라 기존 결과는 그대로 재현된다.
SEED=int(os.environ.get("SEED","0")); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# 음성 홀드아웃 분할은 **SEED 와 따로 고정**한다 — 시드 재현에서 학습 풀 구성까지 같이 바뀌면
# "학습 난수의 폭" 과 "어떤 음성이 뽑혔는가" 가 섞여 무엇이 흔든 건지 못 가른다.
NEG_SPLIT_SEED=int(os.environ.get("NEG_SPLIT_SEED","0"))
# ★해상도 파라미터화 — 기본값은 기존 20m 동작과 동일
GSD=float(os.environ.get("GSD","20"))
T=int(os.environ.get("T","256"))                       # 타일 픽셀(지상 시야 = T*GSD)
KO_DIR=os.environ.get("KO_DIR","ko_refl")              # 산림청 replay 데이터셋
# ★음성(소각 0) 타일 — 기본 off 라 기존 재현성은 그대로다(§13.4·§16.3 의 다음 실험).
#   왜: 학습 타일이 전부 화재 중심 크롭이라 모델이 **소각 0 인 타일을 본 적이 없다.**
#   그래서 "이 장면 어딘가는 탔다"를 전제하고 화재 없는 장면의 21~33% 를 소각으로 찍는다.
#   src/build_negatives.py 가 **같은 화재 위치·화재 이전 날짜**로 만든다(교란은 불의 유무뿐).
NEG_DIR=os.environ.get("NEG_DIR","")                   # 예: ko_neg_s50
NEG_FRAC=float(os.environ.get("NEG_FRAC","0.5"))       # 양성 타일 수 대비 음성 비율
NEG_HOLD_FRAC=float(os.environ.get("NEG_HOLD_FRAC","0.2"))  # 음성 중 홀드아웃 비율(오탐률 측정용)
RAW=os.environ.get("CIR_RAW","1")=="1"                # CIR-GT를 raw DN판으로 쓸지
_S="" if (GSD==20 and not RAW) else f"_{int(GSD)}m"+("_raw" if RAW else "")
# BANDS12=1 → SWIR 포함 12ch 학습. CIR-GT 도 12ch 판(gt_stac_*12.npz)이 있어야 한다.
BANDS12=os.environ.get("BANDS12","0")=="1"
CIR_NPZ=f"gt_stac{_S}{'12' if BANDS12 else ''}.npz"
# 재브래킷 A/B(2026-09-24): CIR_NPZ 를 env 로 갈아끼워 라벨판을 바꾼다(예 gt_stac_rb.npz).
# 파일별 폴백 — 오버라이드판이 없는 화재(다른 홀드아웃)는 표준 gt_stac.npz 로 되돌린다.
CIR_NPZ_OVERRIDE=os.environ.get("CIR_NPZ","")
_CIR_NPZ_DEFAULT=f"gt_stac{_S}{'12' if BANDS12 else ''}.npz"
CKPT=os.environ.get("CKPT",os.path.join(WEIGHTS,"et02_ens","et02_deeplabv3plus_s1.pt"))
EPOCHS=int(os.environ.get("EPOCHS","15")); LR=float(os.environ.get("LR","3e-5"))
# 학습 풀 — 기본 6화재는 전부 대형(138~22,191ha). 오프셋 버그(raw DN 미적용)로
# 0~23ha로 과소탐지되던 소형화재 10건이 GT 수정으로 살아났고, CIR_POOL=expanded 로 합류시킨다.
_CIR_BASE=["250325_경북영덕청송","250322_의성안평면","250325_영덕축산면","220304_울진북면","220528_울진근남면","230403_영주평은면"]
_CIR_SMALL=["220305_강릉옥계면","220411_군위의흥면","220531_밀양부북면","230308_합천용주면",
            "230402_금산복수면","230403_순천송광면","230403_함평대동면","230411_강릉난곡동"]  # 학습용 소형 8
# ★소형 전용 홀드아웃 — 두 arm(base/expanded) 모두 학습에서 제외해 공정 비교.
# 기존 CIR_HOLD 4화재가 283~1,714ha 중대형 편중이라, 소형 추가의 이득을 측정할 수 없다.
# 소형 홀드아웃. CIR-GT 소형은 총 10화재뿐이라 고정 2건으로는 판정이 안 된다(n=2).
# SMALL_HOLD env로 fold를 바꿔 교차검증한다 — 기본값은 기존 2건(하위호환).
SMALL_HOLD=[x for x in os.environ.get("SMALL_HOLD","250322_울주온양읍,250428_대구노곡동").split(",") if x]  # 70·84ha
_POOL=os.environ.get("CIR_POOL","base")                        # base | expanded | 콤마목록
if _POOL=="base":       CIR_TRAIN=_CIR_BASE
elif _POOL=="expanded": CIR_TRAIN=_CIR_BASE+_CIR_SMALL
else:                   CIR_TRAIN=[x for x in _POOL.split(",") if x]
CIR_HOLD=["250324_안동길안면","250321_산청시천면","230402_홍성서부면","220410_양구양구읍"]
FOREST_HOLD=[l.strip() for l in open(os.path.join(PROJ,"data","holdout_fixed.txt")) if l.strip()]
CIR_OS=int(os.environ.get("CIR_OS","3"))   # CIR 오버샘플(균형)
def rgbnir(x12): pre,post=x12[0:6],x12[6:12]; return np.concatenate([pre[[0,1,2,3]],post[[0,1,2,3]]],0)
def sel12(x12): return x12 if BANDS12 else rgbnir(x12)   # 산림청 타일의 밴드 선택 단일 소스
def cir_load(n):
    npz=CIR_NPZ_OVERRIDE if (CIR_NPZ_OVERRIDE and os.path.exists(os.path.join(CIRD,n,CIR_NPZ_OVERRIDE))) else _CIR_NPZ_DEFAULT
    d=np.load(os.path.join(CIRD,n,npz)); return d["x"].astype("float32"),d["gt"].astype(bool),d["valid"].astype(bool)
def cir_tiles(n,stride=None):
    if stride is None: stride=int(T*0.75)              # 원본 192/256 비율 유지
    x,gt,vd=cir_load(n); C,H,W=x.shape; out=[]
    for y in sorted(set(min(a,H-T) for a in range(0,max(H-T,0)+1,stride)) or [0]):
        for z in sorted(set(min(b,W-T) for b in range(0,max(W-T,0)+1,stride)) or [0]):
            xt=x[:,y:y+T,z:z+T]; yt=gt[y:y+T,z:z+T]; vt=vd[y:y+T,z:z+T]
            if xt.shape[1]==T and xt.shape[2]==T and vt.mean()>=0.5: out.append((xt.copy(),yt.astype("float32").copy()))
    return out
def forest_train_tiles():
    out=[]; dirs=[d for d in glob.glob(os.path.join(PROJ,KO_DIR,"*")) if os.path.isdir(d)]
    for d in dirs:
        if any(h.replace("kf_","") in os.path.basename(d) for h in FOREST_HOLD): continue  # 홀드아웃 제외
        for f in glob.glob(os.path.join(d,"*.npz")):
            z=np.load(f); x=np.clip(sel12(z["x"].astype(np.float32)/1e4),0,1); y=z["y"].astype("float32")
            v=z["valid"].astype(bool) if "valid" in z else np.ones(y.shape,bool)
            if v.mean()>=0.5: out.append((x.astype("float32"),y))
    return out
def neg_tiles():
    """음성 타일 → (학습용, 홀드아웃용). 화재 단위로 갈라 같은 장소가 양쪽에 안 걸치게 한다."""
    if not NEG_DIR: return [], []
    dirs=sorted(d for d in glob.glob(os.path.join(PROJ,NEG_DIR,"*")) if os.path.isdir(d))
    if not dirs: return [], []
    rnd=random.Random(NEG_SPLIT_SEED); rnd.shuffle(dirs)   # ★SEED 아님 — 시드 재현에서 홀드아웃 고정
    k=max(1,int(len(dirs)*NEG_HOLD_FRAC)); hold_d, tr_d = dirs[:k], dirs[k:]
    def load(ds):
        out=[]
        for d in ds:
            for f in glob.glob(os.path.join(d,"*.npz")):
                z=np.load(f)
                v=z["valid"].astype(bool) if "valid" in z else np.ones(z["y"].shape,bool)
                if v.mean()<0.5: continue
                x=np.clip(sel12(z["x"].astype(np.float32)/1e4),0,1).astype("float32")
                out.append((x, z["y"].astype("float32")))
        return out
    return load(tr_d), load(hold_d)


@torch.no_grad()
def eval_neg_fpr(m, tiles, thr=0.20):
    """음성 홀드아웃 오탐률 — **정답이 0** 이므로 재현율은 정의되지 않는다. FP/전체만 본다."""
    if not tiles: return float("nan")
    fp=tot=0
    for x,_ in tiles:
        p=infer(m,x,x.shape[1],x.shape[2])
        fp+=int((p>=thr).sum()); tot+=p.size
    return fp/max(tot,1)


_ARCHMAP={"deeplabv3plus":smp.DeepLabV3Plus,"unet":smp.Unet,"fpn":smp.FPN}
ARCH=os.environ.get("ARCH","deeplabv3plus")   # 기본=기존동작(deeplab). 인코더 A/B용 unet/fpn 허용
ENCODER=os.environ.get("ENCODER","resnet34")  # 기본=기존동작. resnet18 등 교체 검증용
def build():
    m=_ARCHMAP[ARCH](ENCODER,encoder_weights=None,in_channels=8,classes=1).to(DEV)
    m.load_state_dict(torch.load(CKPT,map_location=DEV)); return m
@torch.no_grad()
def infer(m,x,H,W):
    m.eval(); prob=np.zeros((H,W),"float32")
    for i in range(0,H,T):
        for j in range(0,W,T):
            p=x[:,i:i+T,j:j+T]; ph,pw=p.shape[1],p.shape[2]
            pad=np.zeros((8,T,T),"float32"); pad[:,:ph,:pw]=p
            prob[i:i+ph,j:j+pw]=torch.sigmoid(m(torch.from_numpy(pad[None]).to(DEV)))[0,0].cpu().numpy()[:ph,:pw]
    return prob
def bestf1(P,Y,region=None):
    b=0
    for t in np.arange(0.05,0.9,0.025):
        pr=P>=t
        if region is not None: pr=pr&region
        tp=(pr&(Y==1)).sum(); fp=(pr&(Y==0)).sum(); fn=((~pr)&(Y==1)).sum()
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); b=max(b,2*p*r/(p+r+1e-9))
    return b
def eval_cir(m):
    tot=[]
    for h in CIR_HOLD:
        x,gt,vd=cir_load(h); H,W=gt.shape; prob=infer(m,x,H,W)
        tot.append(bestf1(prob[vd],gt[vd].astype(int)))
    return float(np.mean(tot))
def eval_forest(m):
    tot=[]
    for h in FOREST_HOLD:
        dirs=[d for d in glob.glob(os.path.join(PROJ,KO_DIR,"*")) if h.replace("kf_","") in os.path.basename(d) and os.path.isdir(d)]
        if not dirs: continue
        P=[];Y=[]
        for f in glob.glob(os.path.join(dirs[0],"*.npz")):
            z=np.load(f); x=np.clip(sel12(z["x"].astype(np.float32)/1e4),0,1)
            pr=infer(m,x,x.shape[1],x.shape[2]); v=z["valid"].astype(bool) if "valid" in z else np.ones(z["y"].shape,bool)
            P.append(pr[v]); Y.append(z["y"][v].astype(int))
        tot.append(bestf1(np.concatenate(P),np.concatenate(Y)))
    return float(np.mean(tot))
def main():
    cir=[];
    for n in CIR_TRAIN: cir+=cir_tiles(n)
    forest=forest_train_tiles()
    neg_tr, neg_ho = neg_tiles()
    pos=forest+cir*CIR_OS
    if neg_tr and NEG_FRAC>0:
        want=int(len(pos)*NEG_FRAC)
        neg_use=(neg_tr*(want//max(len(neg_tr),1)+1))[:want]
    else:
        neg_use=[]
    tiles=pos+neg_use
    print(f"rehearsal 학습타일: 산림청 {len(forest)} + CIR {len(cir)}x{CIR_OS}"
          f" + 음성 {len(neg_use)}(풀 {len(neg_tr)}, 홀드아웃 {len(neg_ho)}) = {len(tiles)}"
          f" | LR={LR} EP={EPOCHS} SEED={SEED}",flush=True)
    m=build()
    print(f"[baseline] 봄cross-fire={eval_cir(m):.3f}  산림청in-domain={eval_forest(m):.3f}"
          f"  음성오탐률={eval_neg_fpr(m,neg_ho):.1%}",flush=True)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4); bce=nn.BCEWithLogitsLoss(); dice=DiceLoss(mode="binary"); BS=8
    for ep in range(EPOCHS):
        m.train(); random.shuffle(tiles); xs_,ys_=[],[]
        for xt,yt in tiles:
            k=random.randint(0,3); fl=random.random()<0.5
            xa=np.rot90(xt,k,axes=(1,2)); ya=np.rot90(yt,k)
            if fl: xa=xa[:,:,::-1]; ya=ya[:,::-1]
            xs_.append(xa.copy()); ys_.append(ya.copy())
        for i in range(0,len(xs_),BS):
            xb=torch.from_numpy(np.stack(xs_[i:i+BS])).to(DEV); yb=torch.from_numpy(np.stack(ys_[i:i+BS]))[:,None].to(DEV)
            if xb.shape[0]<2: continue
            opt.zero_grad(); out=m(xb); (bce(out,yb)+dice(out,yb)).backward(); opt.step()
        if (ep+1)%5==0: print(f"  ep{ep+1}: 봄={eval_cir(m):.3f} 산림청={eval_forest(m):.3f}"
                              f" 음성오탐률={eval_neg_fpr(m,neg_ho):.1%}",flush=True)
    c=eval_cir(m); f=eval_forest(m); nf=eval_neg_fpr(m,neg_ho)
    print(f"\n★ rehearsal 결과: 봄 cross-fire={c:.3f} (단일baseline 0.097) | 산림청 in-domain={f:.3f} (기존 0.687)"
          f" | 음성 홀드아웃 오탐률={nf:.1%}",flush=True)
    _out=os.path.join(PROJ,"runs",os.environ.get("OUT_DIR","cir_reh"))
    os.makedirs(_out,exist_ok=True)
    # ★파일명에 디코더 이름을 넣는다 — nifos_eval._load 가 파일명에서 디코더를 읽는다
    #   ("deeplab_reh.pt" 로 두면 채점 스크립트가 판별 실패로 죽는다).
    torch.save(m.state_dict(),os.path.join(_out,f"reh_{ARCH}_s1.pt"))
    json.dump(dict(cir=c,forest=f,neg_fpr=nf,arch=ARCH,encoder=ENCODER,neg_dir=NEG_DIR,neg_frac=NEG_FRAC,
                   n_neg_train=len(neg_use),n_neg_hold=len(neg_ho),epochs=EPOCHS,lr=LR,
                   seed=SEED,neg_split_seed=NEG_SPLIT_SEED),
              open(os.path.join(_out,"results_reh.json"),"w"),ensure_ascii=False,indent=1)
    print(f"저장: runs/{os.path.basename(_out)}/reh_{ARCH}_s1.pt + results_reh.json ({ARCH}/{ENCODER})",flush=True)
if __name__=="__main__": main()
