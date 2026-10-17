"""②확장: ko_refl 봄화재(2017-21) dNBR 재라벨 → CIR봄 풀에 추가 → 확장 rehearsal 9모델.
학습풀 = CIR 6화재(dNBR) + ko_refl 봄 비홀드아웃(dNBR 재라벨) + ko_refl 산림청 replay(in-domain보존).
평가 = 봄 CIR 4홀드아웃(+TTA) + 산림청 8홀드아웃. 손라벨/STAC/API 불필요(npz 12밴드).
사용: EPOCHS=15 python step8_expanded.py
"""
import os, glob, re, random, numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import step4_rehearsal as S
import step4b_rehearsal_ens as E
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
WEIGHTS=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/웹앱/weights"); DEV=S.DEV
OUT=os.path.join(PROJ,"runs","cir_exp_ens"); os.makedirs(OUT,exist_ok=True)
EPOCHS=int(os.environ.get("EPOCHS","15")); LR=3e-5; T=256
random.seed(0); np.random.seed(0); torch.manual_seed(0)
FHOLD=[l.strip() for l in open(os.path.join(PROJ,"data","holdout_fixed.txt")) if l.strip()]
def month(b):
    m=re.search(r"kf_\d{4}(\d{2})",b); return int(m.group(1)) if m else 0
def dnbr_tiles_korefl_spring():
    """ko_refl 봄(3-5월) 비홀드아웃 화재 → npz 12밴드로 dNBR 라벨 재계산, rgbnir 8ch 타일."""
    out=[]
    for d in glob.glob(os.path.join(PROJ,"ko_refl","*")):
        if not os.path.isdir(d): continue
        b=os.path.basename(d)
        if month(b) not in (3,4,5): continue
        if any(h.replace("kf_","") in b for h in FHOLD): continue
        for f in glob.glob(os.path.join(d,"*.npz")):
            z=np.load(f); x=z["x"].astype(np.float32)  # 12ch reflectance(×1e4)
            pre,post=x[0:6],x[6:12]
            nbr=lambda a:(a[3]-a[5])/(a[3]+a[5]+1e-6); ndvi=lambda a:(a[3]-a[2])/(a[3]+a[2]+1e-6)
            dnbr=nbr(pre)-nbr(post); dndvi=ndvi(pre)-ndvi(post)
            v=z["valid"].astype(bool) if "valid" in z else np.ones(dnbr.shape,bool)
            burn=((dnbr>0.15)&(dndvi>0.08)&(ndvi(post)<0.45)&v).astype("float32")
            xt=np.clip(np.concatenate([pre[[0,1,2,3]],post[[0,1,2,3]]],0)/1e4,0,1).astype("float32")
            if v.mean()>=0.5: out.append((xt,burn))
    return out
def main():
    cir=[]
    for n in S.CIR_TRAIN: cir+=S.cir_tiles(n)
    kospring=dnbr_tiles_korefl_spring()
    forest=S.forest_train_tiles()   # 산림청 replay(전계절 in-domain 보존)
    tiles=forest + cir*S.CIR_OS + kospring
    print(f"확장풀: 산림청replay{len(forest)} + CIR{len(cir)}x{S.CIR_OS} + ko봄dNBR{len(kospring)} = {len(tiles)}",flush=True)
    # baseline(현 배포 rehearsal 앙상블)
    base=E.build_ens_from(os.path.join(WEIGHTS,"et02_ens"),"") if hasattr(E,"build_ens_from") else None
    ft=[]
    for i,(d,n) in enumerate(E.ENS):
        m=E.ARCH[d]("resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV)
        m.load_state_dict(torch.load(os.path.join(WEIGHTS,"et02_ens",n+".pt.bak_prespring"),map_location=DEV)); m.train()
        opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4); bce=nn.BCEWithLogitsLoss(); dice=DiceLoss(mode="binary"); BS=8
        for ep in range(EPOCHS):
            random.shuffle(tiles); xs_,ys_=[],[]
            for xt,yt in tiles:
                k=random.randint(0,3); fl=random.random()<0.5
                xa=np.rot90(xt,k,axes=(1,2)); ya=np.rot90(yt,k)
                if fl: xa=xa[:,:,::-1]; ya=ya[:,::-1]
                xs_.append(xa.copy()); ys_.append(ya.copy())
            for j in range(0,len(xs_),BS):
                xb=torch.from_numpy(np.stack(xs_[j:j+BS])).to(DEV); yb=torch.from_numpy(np.stack(ys_[j:j+BS]))[:,None].to(DEV)
                if xb.shape[0]<2: continue
                opt.zero_grad(); o=m(xb); (bce(o,yb)+dice(o,yb)).backward(); opt.step()
        m.eval(); torch.save(m.state_dict(),os.path.join(OUT,n+"_exp.pt")); ft.append(m)
        print(f"  [{i+1}/9] {n} 완료",flush=True)
    c=E.eval_cir_ens(ft); f=E.eval_forest_ens(ft)
    print(f"\n★ 확장 rehearsal 앙상블: 봄 cross-fire={c:.3f} (기존 0.764) | 산림청 in-domain={f:.3f} (기존 0.700)",flush=True)
    print(f"  게이트: 봄>0.5 AND 산림청>=0.65 → {'통과' if c>0.5 and f>=0.65 else '미통과'}",flush=True)
if __name__=="__main__": main()
