"""RapidEye GT 사이트 점검 그림 — post 컬러 · dNBR · GT · 예측 4패널.

왜: 강릉에서 GT(361ha) 밖 378ha 가 dNBR 0.334 로 "탄 땅"으로 나왔다. GT 가 화재의 일부만
라벨한 것인지, 라벨 경계가 어떤 모양인지(장면 클립인지 등급판독 범위인지)는 수치로는 못 본다.

사용: SITE=gangneung GSD=20 SCENE_PICK=doy conda run -n ml python re_gt_figure.py
"""
import os, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
SITE=os.environ.get("SITE","gangneung"); GSD=float(os.environ.get("GSD","20"))
CACHE=os.path.join(PROJ,"nifos_cache",SITE+("_doy" if os.environ.get("SCENE_PICK")=="doy" else ""))
THR=float(os.environ.get("THR","0.2"))
gt=np.load(os.path.join(CACHE,f"gt_{int(GSD)}m.npy")).astype(bool)
xr=np.load(os.path.join(CACHE,f"scene_{int(GSD)}m.npz"))["x"]
H=min(gt.shape[0],xr.shape[1]); W=min(gt.shape[1],xr.shape[2]); gt=gt[:H,:W]; xr=xr[:,:H,:W]
x=xr.astype(np.float32)/1e4; pre,post=x[:6],x[6:]
def nbr(b): return (b[3]-b[5])/(b[3]+b[5]+1e-6)
dnbr=nbr(pre)-nbr(post)
sys.path.insert(0,HERE); os.environ.setdefault("NIR","B8A" if GSD==20 else "B08")
os.environ["SITE"]=SITE; os.environ["GSD"]=str(int(GSD)); os.environ.setdefault("TILE","256")
import nifos_eval as NE
runs=sorted(glob.glob(os.path.join(PROJ,os.environ.get("RUNS","runs/cir_reh_ens/*.pt"))))
prob,_=NE.infer(xr,runs); prob=prob[:H,:W]; pred=prob>=THR
rgb=np.stack([post[2],post[1],post[0]],-1); rgb=np.clip(rgb/0.30,0,1)
ov=np.zeros((H,W,3)); ov[...,0]=pred; ov[...,1]=gt      # 빨강=예측, 초록=GT, 노랑=일치
fig,ax=plt.subplots(1,4,figsize=(22,4.6),constrained_layout=True)
for a,(im,t,kw) in zip(ax,[(rgb,f"post RGB",{}),
                           (dnbr,"dNBR",dict(cmap="RdYlGn_r",vmin=-0.3,vmax=0.8)),
                           (gt,"GT (NIFoS RapidEye 5m)",dict(cmap="gray")),
                           (ov,f"빨강=예측(thr{THR}) 초록=GT 노랑=일치",{})]):
    a.imshow(im,**kw); a.set_title(t,fontsize=11); a.axis("off")
out=os.path.join(PROJ,"figs",f"re_gt_{SITE}_{int(GSD)}m.png")
os.makedirs(os.path.dirname(out),exist_ok=True)
fig.savefig(out,dpi=110); print("저장:",os.path.relpath(out,PROJ))
print(f"GT {gt.sum()*GSD*GSD/1e4:.0f}ha · 예측 {pred.sum()*GSD*GSD/1e4:.0f}ha · "
      f"dNBR>0.2 {(dnbr>0.2).sum()*GSD*GSD/1e4:.0f}ha")
