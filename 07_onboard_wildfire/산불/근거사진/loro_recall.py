import os, numpy as np, torch
import segmentation_models_pytorch as smp
import step4_rehearsal as S
DEV=S.DEV; REH=os.path.join(S.PROJ,"runs","cir_reh_ens")
ARCH={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}
ENS=[("unet","et02_unet_s1"),("unet","et02_unet_s2"),("unet","et02_unet_s3"),
     ("fpn","et02_fpn_s1"),("fpn","et02_fpn_s2"),("fpn","et02_fpn_s3"),
     ("deeplabv3plus","et02_deeplabv3plus_s1"),("deeplabv3plus","et02_deeplabv3plus_s2"),("deeplabv3plus","et02_deeplabv3plus_s3")]
names=["unet_s1","unet_s2","unet_s3","fpn_s1","fpn_s2","fpn_s3","dlv3+_s1","dlv3+_s2","dlv3+_s3"]
def prf_at_bestf1(P,Y):
    best=(-1,0,0,0)
    for t in np.arange(0.05,0.9,0.025):
        pr=P>=t; tp=(pr&(Y==1)).sum(); fp=(pr&(Y==0)).sum(); fn=((~pr)&(Y==1)).sum()
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); f=2*p*r/(p+r+1e-9)
        if f>best[0]: best=(f,p,r,t)
    return best  # f1,p,r,thr
def rec_at(P,Y,t):
    pr=P>=t; tp=(pr&(Y==1)).sum(); fn=((~pr)&(Y==1)).sum()
    return tp/(tp+fn+1e-9)
ms=[]
for d,n in ENS:
    m=ARCH[d]("resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV)
    m.load_state_dict(torch.load(os.path.join(REH,n+"_reh.pt"),map_location=DEV)); m.eval(); ms.append(m)
REC=np.zeros((len(S.CIR_HOLD),9)); PRE=np.zeros_like(REC); RECF=np.zeros_like(REC)  # recall@bestF1, prec@bestF1, recall@fixed0.25
for fi,h in enumerate(S.CIR_HOLD):
    x,gt,vd=S.cir_load(h); H,W=gt.shape; Y=gt[vd].astype(int)
    for mi,m in enumerate(ms):
        P=S.infer(m,x,H,W)[vd]
        f,p,r,t=prf_at_bestf1(P,Y); REC[fi,mi]=r; PRE[fi,mi]=p; RECF[fi,mi]=rec_at(P,Y,0.25)
np.savez("/tmp/claude-1000/-home-syntax-kmu/96dc9619-541e-4fa1-8e2e-b6cafa39bb30/scratchpad/loro_recall.npz",REC=REC,PRE=PRE,RECF=RECF,fires=S.CIR_HOLD,names=names)
mean=REC.mean(0); worst=REC.min(0); order=np.argsort(-mean)
print("화재:",S.CIR_HOLD)
print(f"\n[recall @ 각 멤버 best-F1 임계값]  {'멤버':10s}{'평균R':>7s}{'최악화재R':>9s}{'평균P':>7s}")
for i in order:
    print(f"{'':0s}{names[i]:12s}{mean[i]:7.3f}{worst[i]:9.3f}{PRE.mean(0)[i]:8.3f}"+("   ★배포(unet_s1)" if names[i]=='unet_s1' else ""))
mf=RECF.mean(0); wf=RECF.min(0)
print(f"\n[recall @ 고정 thr 0.25]  unet_s1 평균={mf[0]:.3f} 최악={wf[0]:.3f} | 9멤버 평균R최고={names[int(np.argmax(mf))]}({mf.max():.3f}) 최악R최고={names[int(np.argmax(wf))]}({wf.max():.3f})")
print(f"\nunet_s1 recall 순위(평균, best-F1thr)= {list(order).index(0)+1}/9")
