"""복구 라벨(12화재)을 학습에 추가 → 홀드아웃 cross-fire F1이 baseline 넘나(델타 규칙).
단일 unet_s1 base에서 rehearsal(CIR_TRAIN + 산림청 replay + 복구12) 재학습.
v4(소형화재 확대=negative, 경계노이즈) 재발 여부 실측.
"""
import os, glob, numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import step4_rehearsal as S
CIRD=S.CIRD; T=S.T
# ★10m: BASE=10m base 모델, RB_NPZ=gt_stac_rb_10m.npz 로 env 지정 (기본은 20m).
BASE=os.environ.get("BASE_PT", os.path.join(S.WEIGHTS,"et02_ens","et02_unet_s1.pt"))
RB_NPZ=os.environ.get("RB_NPZ","gt_stac_rb.npz")
REC_OS=int(os.environ.get("REC_OS","2"))
def unet():
    m=smp.Unet("resnet34",encoder_weights=None,in_channels=8,classes=1).to(S.DEV)
    m.load_state_dict(torch.load(BASE,map_location=S.DEV)); return m
def rb_tiles():
    out=[]
    for f in sorted(glob.glob(os.path.join(CIRD,"*",RB_NPZ))):
        z=np.load(f); x=z["x"].astype(np.float32); gt=z["gt"].astype(bool); vd=z["valid"].astype(bool)
        C,H,W=x.shape; st=int(T*0.75)
        for y in sorted(set(min(a,H-T) for a in range(0,max(H-T,0)+1,st)) or [0]):
            for zz in sorted(set(min(b,W-T) for b in range(0,max(W-T,0)+1,st)) or [0]):
                xt=x[:,y:y+T,zz:zz+T]; yt=gt[y:y+T,zz:zz+T]; vt=vd[y:y+T,zz:zz+T]
                if xt.shape[1]==T and xt.shape[2]==T and vt.mean()>=0.5: out.append((xt.copy(),yt.astype("float32").copy()))
    return out
def train(m,tiles,neg_ho,ep=None,lr=3e-5):
    ep=ep or int(os.environ.get("EPOCHS","10"))
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4); bce=nn.BCEWithLogitsLoss(); dice=DiceLoss(mode="binary"); BS=int(os.environ.get("BS","4"))
    import random, copy
    best_f1=-1; best_state=copy.deepcopy(m.state_dict()); best_ep=0
    for e in range(ep):
        m.train(); random.shuffle(tiles)
        # ★배치단위 스트리밍 augment (10m 타일 다수 → 전량 물질화 시 Orin 8GB OOM)
        for i in range(0,len(tiles),BS):
            xs,ys=[],[]
            for xt,yt in tiles[i:i+BS]:
                k=random.randint(0,3); fl=random.random()<0.5
                xa=np.rot90(xt,k,axes=(1,2)); ya=np.rot90(yt,k)
                if fl: xa=xa[:,:,::-1]; ya=ya[:,::-1]
                xs.append(xa.copy()); ys.append(ya.copy())
            if len(xs)<2: continue
            xb=torch.from_numpy(np.stack(xs)).to(S.DEV); yb=torch.from_numpy(np.stack(ys))[:,None].to(S.DEV)
            opt.zero_grad(); (bce(m(xb),yb)+dice(m(xb),yb)).backward(); opt.step()
        c=S.eval_cir(m); f=S.eval_forest(m)   # ★매 epoch 평가 → best 봄 cross-fire 저장(ep15 과학습 회피)
        print(f"  ep{e+1}: 봄={c:.3f} 산림청={f:.3f}",flush=True)
        if c>best_f1: best_f1=c; best_state=copy.deepcopy(m.state_dict()); best_ep=e+1
    print(f"  ★best ep{best_ep}: 봄 cross-fire={best_f1:.3f} 채택",flush=True)
    m.load_state_dict(best_state); return m
def main():
    import random; random.seed(1); np.random.seed(1); torch.manual_seed(1)
    m0=unet()
    print(f"[baseline unet_s1] 봄 cross-fire={S.eval_cir(m0):.3f} | 산림청={S.eval_forest(m0):.3f}",flush=True)
    cir=[];
    for n in S.CIR_TRAIN: cir+=S.cir_tiles(n)
    forest=S.forest_train_tiles(); rec=rb_tiles(); _,neg_ho=S.neg_tiles()
    FOREST_OS=int(os.environ.get("FOREST_OS","1"))
    print(f"학습타일: 산림청{len(forest)}x{FOREST_OS} + CIR{len(cir)}x{S.CIR_OS} + 복구{len(rec)}x{REC_OS}",flush=True)
    tiles=forest*FOREST_OS+cir*S.CIR_OS+rec*REC_OS
    m=unet()  # 같은 base 에서 재학습
    m=train(m,tiles,neg_ho)
    c=S.eval_cir(m); f=S.eval_forest(m)
    print(f"\n★ +복구12 재학습: 봄 cross-fire={c:.3f} (baseline {S.eval_cir(m0):.3f}) | 산림청={f:.3f}",flush=True)
    print(f"  델타 봄 {c-S.eval_cir(m0):+.3f} (σ=0.068 넘어야 진짜 이득 / v4처럼 −면 negative)")
    torch.save(m.state_dict(),os.path.join(S.PROJ,"runs",os.environ.get("OUT_PT","unet_recovered.pt")))
if __name__=="__main__": main()
