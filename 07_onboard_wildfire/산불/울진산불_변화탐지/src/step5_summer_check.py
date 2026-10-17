"""#③ 여름 홀드아웃 확대: 8개 여름화재 전부 학습제외(leave-all-summer-out)→1모델→8개 unseen 채점.
0.684(2화재)가 진짜인지 판정. rehearsal 방식(산림청 비여름 + CIR봄). 단일 deeplab.
비교: baseline(원본 deeplab, 여름 학습됨=in-sample참고) vs summer-out 모델(8개 unseen).
사용: EPOCHS=15 python step5_summer_check.py
"""
import os, glob, re, random, numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import step4_rehearsal as S
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
WEIGHTS=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/웹앱/weights"); DEV=S.DEV
EPOCHS=int(os.environ.get("EPOCHS","15")); LR=3e-5; T=256
random.seed(0); np.random.seed(0); torch.manual_seed(0)
SUMMER=["kf_20170603","kf_20170604","kf_20170623","kf_20180601","kf_20180801","kf_20190826","kf_20200608","kf_20200623"]
def is_summer(b): return any(s.replace("kf_","") in b for s in SUMMER)
def forest_tiles_no_summer():
    out=[]
    for d in glob.glob(os.path.join(PROJ,"ko_refl","*")):
        if not os.path.isdir(d): continue
        if is_summer(os.path.basename(d)): continue          # 여름 전부 제외
        for f in glob.glob(os.path.join(d,"*.npz")):
            z=np.load(f); x=np.clip(S.rgbnir(z["x"].astype(np.float32)/1e4),0,1); y=z["y"].astype("float32")
            v=z["valid"].astype(bool) if "valid" in z else np.ones(y.shape,bool)
            if v.mean()>=0.5: out.append((x.astype("float32"),y))
    return out
def eval_summer(m):
    res={}
    for s in SUMMER:
        dirs=[d for d in glob.glob(os.path.join(PROJ,"ko_refl","*")) if s.replace("kf_","") in os.path.basename(d) and os.path.isdir(d)]
        if not dirs: continue
        P=[];Y=[]
        for f in glob.glob(os.path.join(dirs[0],"*.npz")):
            z=np.load(f); x=np.clip(S.rgbnir(z["x"].astype(np.float32)/1e4),0,1)
            pr=S.infer(m,x,x.shape[1],x.shape[2]); v=z["valid"].astype(bool) if "valid" in z else np.ones(z["y"].shape,bool)
            P.append(pr[v]); Y.append(z["y"][v].astype(int))
        res[os.path.basename(dirs[0])]=S.bestf1(np.concatenate(P),np.concatenate(Y))
    return res
def main():
    cir=[]
    for n in S.CIR_TRAIN: cir+=S.cir_tiles(n)
    forest=forest_tiles_no_summer(); tiles=forest+cir*S.CIR_OS
    print(f"여름제외 학습타일: 비여름산림청{len(forest)}+CIR{len(cir)}x{S.CIR_OS}={len(tiles)}",flush=True)
    # baseline: 원본 deeplab(여름 학습됨) — 참고용 in-sample
    base=smp.DeepLabV3Plus("resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV).eval()
    base.load_state_dict(torch.load(os.path.join(WEIGHTS,"et02_ens","et02_deeplabv3plus_s1.pt.bak_prespring"),map_location=DEV))
    rb=eval_summer(base)
    print(f"\n[원본 deeplab, 여름 in-sample 참고] 평균={np.mean(list(rb.values())):.3f}")
    for k,v in rb.items(): print(f"  {k}: {v:.3f}",flush=True)
    # summer-out 모델
    m=smp.DeepLabV3Plus("resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV)
    m.load_state_dict(torch.load(os.path.join(WEIGHTS,"et02_ens","et02_deeplabv3plus_s1.pt.bak_prespring"),map_location=DEV))
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
    rs=eval_summer(m)
    print(f"\n★[summer-out 모델, 8개 전부 UNSEEN] 평균 best-F1={np.mean(list(rs.values())):.3f}")
    for k,v in rs.items(): print(f"  {k}: {v:.3f}",flush=True)
    print(f"\n판정: 여름 unseen 평균 {np.mean(list(rs.values())):.3f} — 0.684(2화재) 대비",flush=True)
if __name__=="__main__": main()
