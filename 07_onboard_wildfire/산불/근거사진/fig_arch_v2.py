import os, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap
import segmentation_models_pytorch as smp
import step4_rehearsal as S
fp="/home/syntax_kmu/.fonts/NanumGothic.ttf"
fm.fontManager.addfont(fp); plt.rcParams["font.family"]=fm.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
OUT="/tmp/claude-1000/-home-syntax-kmu/96dc9619-541e-4fa1-8e2e-b6cafa39bb30/scratchpad"
DEV=S.DEV; REH=os.path.join(S.PROJ,"runs","cir_reh_ens")
MODELS=[("U-Net",smp.Unet,"et02_unet_s1"),("FPN",smp.FPN,"et02_fpn_s1"),("DeepLabV3+",smp.DeepLabV3Plus,"et02_deeplabv3plus_s1")]
def load(a,n): m=a("resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV); m.load_state_dict(torch.load(os.path.join(REH,n+"_reh.pt"),map_location=DEV)); m.eval(); return m
ms=[(lab,load(a,n)) for lab,a,n in MODELS]
def prf(P,Y,vd):
    best=(-1,0,0,0,None)
    for t in np.arange(0.05,0.9,0.025):
        pr=(P>=t)&vd; tp=(pr&Y).sum(); fp=(pr&~Y&vd).sum(); fn=(~pr&Y&vd).sum()
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); f=2*p*r/(p+r+1e-9)
        if f>best[0]: best=(f,p,r,t,pr.copy())
    return best
def st(a): lo,hi=np.nanpercentile(a,2),np.nanpercentile(a,98); return np.clip((a-lo)/(hi-lo+1e-9),0,1)
FIRES=["250324_안동길안면","250321_산청시천면","230402_홍성서부면","220410_양구양구읍"]
nice=["안동'25","산청'25","홍성'23","양구'22"]
F1=np.zeros((4,3))
fig=plt.figure(figsize=(16,20))
gs=fig.add_gridspec(5,4,height_ratios=[1,1,1,1,0.9],hspace=0.18,wspace=0.06)
orange=ListedColormap([(0,0,0,0),(1,0.5,0,0.55)])
for ri,h in enumerate(FIRES):
    x,gt,vd=S.cir_load(h); H,W=gt.shape; Y=gt&vd
    rgb=np.dstack([st(x[6]),st(x[5]),st(x[4])]); rgb[~vd]=0.15
    ax=fig.add_subplot(gs[ri,0]); ax.imshow(rgb); ax.contour(Y,[0.5],colors="red",linewidths=0.8)
    ax.set_ylabel(nice[ri],fontsize=14,fontweight="bold"); ax.set_xticks([]);ax.set_yticks([])
    if ri==0: ax.set_title("촬영영상 + 정답GT(빨강선)",fontsize=12)
    for ci,(lab,m) in enumerate(ms,1):
        P=S.infer(m,x,H,W); f,p,r,t,pr=prf(P,Y,vd); F1[ri,ci-1]=f
        ax=fig.add_subplot(gs[ri,ci]); ax.imshow(rgb); ax.imshow(pr,cmap=orange)
        ax.contour(Y,[0.5],colors="red",linewidths=0.6); ax.set_xticks([]);ax.set_yticks([])
        win = f==max(F1[ri]) if ci==3 else False
        head=(lab+"\n" if ri==0 else "")
        ax.set_title(head+f"R={r:.3f} P={p:.3f} F1={f:.3f}",fontsize=11,
                     color=("darkgreen" if lab=="U-Net" else "#333"),fontweight=("bold" if lab=="U-Net" else "normal"))
# 요약 바차트
axb=fig.add_subplot(gs[4,:]); w=0.25; xp=np.arange(4)
cols={"U-Net":"#2a7d2a","FPN":"#c07000","DeepLabV3+":"#888"}
for j,lab in enumerate(["U-Net","FPN","DeepLabV3+"]):
    axb.bar(xp+(j-1)*w,F1[:,j],w,label=lab,color=cols[lab])
axb.set_xticks(xp); axb.set_xticklabels(nice,fontsize=12)
axb.set_ylim(0.70,0.88); axb.set_ylabel("F1 (best-thr)",fontsize=12)
mean=F1.mean(0); worst=F1.min(0)
axb.set_title(f"화재별 F1  |  평균: U-Net {mean[0]:.3f} · FPN {mean[1]:.3f} · DeepLabV3+ {mean[2]:.3f}   "
              f"|  최악화재(하한): U-Net {worst[0]:.3f} · FPN {worst[1]:.3f} · DeepLabV3+ {worst[2]:.3f}",fontsize=12)
axb.legend(fontsize=11,loc="lower right"); axb.grid(axis="y",alpha=0.3)
fig.suptitle("모델별 소각영역 예측 비교 (주황=예측 · 빨강선=정답)  —  배포 3아키텍처, LORO 미학습 홀드아웃 4화재",
             fontsize=14,fontweight="bold",y=0.905)
plt.savefig(OUT+"/arch_compare_v2.png",dpi=105,bbox_inches="tight")
print("saved. per-fire F1 (U/F/D):")
for i in range(4): print(f"  {nice[i]}: {F1[i,0]:.3f} {F1[i,1]:.3f} {F1[i,2]:.3f}  승자={['U-Net','FPN','DeepLabV3+'][int(np.argmax(F1[i]))]}")
print(f"평균 {mean.round(3).tolist()} | 최악 {worst.round(3).tolist()} | 표준편차 {F1.std(0).round(3).tolist()}")
