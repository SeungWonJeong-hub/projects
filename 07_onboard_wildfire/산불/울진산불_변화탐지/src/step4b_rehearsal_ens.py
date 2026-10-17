"""#해결책 배포판: 9모델 앙상블 rehearsal 파인튜닝(망각방지) → 앙상블 양쪽 재평가.
step4_rehearsal 함수 재사용. 산림청+CIR 혼합으로 9모델 각각 파인튜닝, 앙상블로 평가.
봄 cross-fire(4 CIR홀드아웃) + 산림청 in-domain(8홀드아웃) 둘다 확인. FT9 저장(배포용).
★경로 이식성: 이전 머신 절대경로 제거. 베이스 체크포인트는 BASE 글롭으로 지정한다.
env:
  BASE=<glob>   베이스 9모델. 기본 = 웹앱 weights/et02_ens/*.pt (20m 배포 베이스).
                10m: BASE="runs/et02_10mT512_*"  (각 디렉터리의 model.pt를 쓴다)
  OUT=<name>    runs/ 아래 출력 이름. 기본 cir_reh_ens
  GSD/T/KO_DIR/CIR_RAW  → step4_rehearsal 과 공유(거기 정의)
사용: 20m 재현  python step4b_rehearsal_ens.py
      10m       GSD=10 T=512 KO_DIR=ko_refl_10m_b08_t512 BASE="runs/et02_10mT512_*" \
                OUT=cir_reh_ens_10m python step4b_rehearsal_ens.py
"""
import os, glob, json, random, numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import step4_rehearsal as S
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
DEV=S.DEV; T=S.T                       # 경로·해상도 설정은 step4_rehearsal(S)에서 상속
OUT=os.path.join(PROJ,"runs",os.environ.get("OUT","cir_reh_ens")); os.makedirs(OUT,exist_ok=True)
EPOCHS=int(os.environ.get("EPOCHS","20")); LR=float(os.environ.get("LR","3e-5"))
ARCH={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}
BASE=os.environ.get("BASE",os.path.join(S.WEIGHTS,"et02_ens","*.pt"))
# CONFIG=rgbnir(기본, 8ch pre|post) | post_rgbnir(4ch post 만 — pre 업링크가 없는 온보드 운용)
#      | bitemporal(12ch, SWIR 포함 — 지상 전용. 큐브위성엔 SWIR 이 없다)
CONFIG=os.environ.get("CONFIG","rgbnir")
IN_CH={"rgbnir":8,"post_rgbnir":4,"bitemporal":12}[CONFIG]

def to_cfg(x):
    """입력 타일 → 설정 채널. smp_cd.build_x 와 같은 절단이어야 한다.

    bitemporal 은 12ch 를 그대로 쓴다. 8ch CIR 타일이 섞여 들어오면 **즉시 죽인다** —
    조용히 채널이 어긋나면 에러 없이 성능만 깎인다([[학습/서빙 계약]])."""
    if CONFIG=="bitemporal":
        if x.shape[0]!=12:
            raise ValueError(f"CONFIG=bitemporal 인데 타일이 {x.shape[0]}ch 다 — "
                             f"CIR-GT 12ch 판(BANDS12=1 로 재생성)이 필요하다")
        return x
    return x if CONFIG=="rgbnir" else np.ascontiguousarray(x[4:8])

def _resolve_base():
    """BASE 글롭 → [(decoder, 저장이름, ckpt경로)]. .pt 파일 또는 학습 디렉터리 둘 다 허용."""
    out=[]
    for e in sorted(glob.glob(BASE if os.path.isabs(BASE) else os.path.join(PROJ,BASE))):
        ck=e if e.endswith(".pt") else (glob.glob(os.path.join(e,"*.pt")) or [None])[0]
        if not ck: continue
        stem=os.path.basename(e)[:-3] if e.endswith(".pt") else os.path.basename(e)
        dec=next((d for d in ARCH if f"_{d}_" in stem or stem.endswith("_"+d)), None)
        assert dec, f"디코더 판별 실패: {stem}"
        # ★글롭이 채널 다른 베이스를 먹지 않았는지 본다. `runs/et02_20m_s50_*` 는 나중에 생긴
        #   `_bi12_`(12ch)·`_post_`(4ch) 까지 먹는다 — 실제로 2026-09-09 에 27개가 잡혔다.
        #   load_state_dict 가 결국 죽긴 하지만 학습을 한참 돌린 뒤라 사유가 안 보인다.
        rj=os.path.join(e,"results.json")
        if os.path.isdir(e) and os.path.exists(rj):
            ic=json.load(open(rj)).get("in_ch")
            if ic is not None and ic!=IN_CH:
                raise ValueError(f"BASE 글롭이 {IN_CH}ch 가 아닌 베이스를 먹었다: {stem} in_ch={ic} "
                                 f"(BASE={BASE}) — 글롭을 좁혀라")
        out.append((dec,stem,ck))
    assert out, f"베이스 체크포인트 없음: {BASE}"
    return out
ENS=_resolve_base()
random.seed(0); np.random.seed(0); torch.manual_seed(0)
@torch.no_grad()
def ens_infer(models,x,H,W):
    """가장자리 정렬 + 겹침 확률평균. 예전엔 부분타일을 0 으로 채워(`np.zeros((8,T,T))`)
    인공 경계를 만들었다 — 웹앱·nifos_eval 에서 2026-09-08 에 고친 결함이 여기 남아
    rehearsal 내부 평가(봄/산림청)를 깎고 있었다. 채널 수도 CONFIG 에서 온다."""
    for m in models: m.eval()
    ph,pw=max(T-H,0),max(T-W,0)
    xp=np.pad(x,((0,0),(0,ph),(0,pw)),mode="reflect") if (ph or pw) else x
    Hp,Wp=xp.shape[1],xp.shape[2]
    prob=np.zeros((Hp,Wp),"float32"); cnt=np.zeros((Hp,Wp),"float32")
    for i in sorted(set(list(range(0,Hp-T+1,T))+[Hp-T])):
        for j in sorted(set(list(range(0,Wp-T+1,T))+[Wp-T])):
            xt=torch.from_numpy(np.ascontiguousarray(xp[:,i:i+T,j:j+T])[None]).to(DEV)
            o=np.mean([torch.sigmoid(m(xt))[0,0].cpu().numpy() for m in models],0)
            prob[i:i+T,j:j+T]+=o; cnt[i:i+T,j:j+T]+=1
    return (prob/np.maximum(cnt,1))[:H,:W]
def eval_cir_ens(models):
    tot=[]
    for h in S.CIR_HOLD:
        x,gt,vd=S.cir_load(h); x=to_cfg(x); H,W=gt.shape; prob=ens_infer(models,x,H,W); tot.append(S.bestf1(prob[vd],gt[vd].astype(int)))
    return float(np.mean(tot))
def eval_forest_ens(models):
    tot=[]
    for h in S.FOREST_HOLD:
        dirs=[d for d in glob.glob(os.path.join(PROJ,S.KO_DIR,"*")) if h.replace("kf_","") in os.path.basename(d) and os.path.isdir(d)]
        if not dirs: continue
        P=[];Y=[]
        for f in glob.glob(os.path.join(dirs[0],"*.npz")):
            z=np.load(f); x=to_cfg(np.clip(S.sel12(z["x"].astype(np.float32)/1e4),0,1))
            pr=ens_infer(models,x,x.shape[1],x.shape[2]); v=z["valid"].astype(bool) if "valid" in z else np.ones(z["y"].shape,bool)
            P.append(pr[v]); Y.append(z["y"][v].astype(int))
        tot.append(S.bestf1(np.concatenate(P),np.concatenate(Y)))
    return float(np.mean(tot))
def finetune(dec,ckpt,tiles):
    m=ARCH[dec]("resnet34",encoder_weights=None,in_channels=IN_CH,classes=1).to(DEV)
    m.load_state_dict(torch.load(ckpt,map_location=DEV)); m.train()
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4); bce=nn.BCEWithLogitsLoss(); dice=DiceLoss(mode="binary"); BS=8
    for ep in range(EPOCHS):
        random.shuffle(tiles); xs_,ys_=[],[]
        for xt,yt in tiles:
            k=random.randint(0,3); fl=random.random()<0.5
            xa=np.rot90(xt,k,axes=(1,2)); ya=np.rot90(yt,k)
            if fl: xa=xa[:,:,::-1]; ya=ya[:,::-1]
            xs_.append(xa.copy()); ys_.append(ya.copy())
        for i in range(0,len(xs_),BS):
            xb=torch.from_numpy(np.stack(xs_[i:i+BS])).to(DEV); yb=torch.from_numpy(np.stack(ys_[i:i+BS]))[:,None].to(DEV)
            if xb.shape[0]<2: continue
            opt.zero_grad(); out=m(xb); (bce(out,yb)+dice(out,yb)).backward(); opt.step()
    m.eval(); return m
def main():
    cir=[]
    for n in S.CIR_TRAIN: cir+=S.cir_tiles(n)
    forest=S.forest_train_tiles(); tiles=[(to_cfg(a),b) for a,b in forest+cir*S.CIR_OS]
    print(f"rehearsal-ens 타일: 산림청{len(forest)}+CIR{len(cir)}x{S.CIR_OS}={len(tiles)} "
          f"| CONFIG={CONFIG}({IN_CH}ch) GSD={S.GSD:.0f}m T={T} KO={S.KO_DIR} CIR={S.CIR_NPZ} 모델{len(ENS)}개 LR={LR} EP={EPOCHS}",flush=True)
    base=[ARCH[d]("resnet34",encoder_weights=None,in_channels=IN_CH,classes=1).to(DEV).eval() for d,n,c in ENS]
    for (d,n,c),m in zip(ENS,base): m.load_state_dict(torch.load(c,map_location=DEV))
    print(f"[baseline 앙상블] 봄={eval_cir_ens(base):.3f} 산림청={eval_forest_ens(base):.3f}",flush=True)
    ft=[]
    for i,(d,n,c) in enumerate(ENS):
        m=finetune(d,c,tiles); torch.save(m.state_dict(),os.path.join(OUT,n+"_reh.pt")); ft.append(m)
        print(f"  [{i+1}/{len(ENS)}] {n} 완료",flush=True)
    c=eval_cir_ens(ft); f=eval_forest_ens(ft)
    # ⚠️ 이 두 자는 **하한(회귀 방지)** 확인용이다. 봄=CIR-GT 는 준순환(우리가 S2 dNBR 을 잘라
    #    만들었다), 산림청=경계 과대라벨이라 **순위 근거로 쓰면 안 된다.** 실제로 산림청 자는
    #    SWIR 과 post-only 를 각각 1위로 매겼고 둘 다 독립 실측(NIFoS 안동)에서 뒤집혔다
    #    — SESSION_2026-09-09 §2·§6e. 우열 판정은 nifos_eval.py(안동)로만 한다.
    print(f"\n★ rehearsal 앙상블: 봄 cross-fire={c:.3f} | 산림청 in-domain={f:.3f}"
          f"   ※ 하한 확인용 — 순위 근거 금지",flush=True)
    print(f"  배포게이트(하한): 봄>0.5 AND 산림청>=0.65 → {'통과' if c>0.5 and f>=0.65 else '미통과'}",flush=True)
if __name__=="__main__": main()
