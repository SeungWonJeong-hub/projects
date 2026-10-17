"""①TTA: 배포 rehearsal 앙상블(weights/et02_ens/*.pt)에 8-view TTA → 봄 4홀드아웃 이득 측정.
no-TTA vs TTA(회전4×반전2 평균) best-F1 비교.
"""
import os, glob, numpy as np, torch
import segmentation_models_pytorch as smp
import step4_rehearsal as S
WEIGHTS=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/웹앱/weights"); DEV=S.DEV
ENS=[("unet","et02_unet_s1"),("unet","et02_unet_s2"),("unet","et02_unet_s3"),
     ("fpn","et02_fpn_s1"),("fpn","et02_fpn_s2"),("fpn","et02_fpn_s3"),
     ("deeplabv3plus","et02_deeplabv3plus_s1"),("deeplabv3plus","et02_deeplabv3plus_s2"),("deeplabv3plus","et02_deeplabv3plus_s3")]
ARCH={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}
def load_ens():
    ms=[]
    for d,n in ENS:
        m=ARCH[d]("resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV).eval()
        m.load_state_dict(torch.load(os.path.join(WEIGHTS,"et02_ens",n+".pt"),map_location=DEV)); ms.append(m)
    return ms
@torch.no_grad()
def pred_tile(models,pad,tta):
    xt=torch.from_numpy(pad[None]).to(DEV)
    if not tta:
        return np.mean([torch.sigmoid(m(xt))[0,0].cpu().numpy() for m in models],0)
    acc=None; cnt=0
    for k in range(4):
        xr=torch.rot90(xt,k,dims=(2,3))
        for fl in (False,True):
            xi=torch.flip(xr,dims=(3,)) if fl else xr
            p=np.mean([torch.sigmoid(m(xi))[0,0].cpu().numpy() for m in models],0)
            if fl: p=p[:,::-1]
            p=np.rot90(p,-k)
            acc=p if acc is None else acc+p; cnt+=1
    return acc/cnt
def infer(models,x,H,W,tta):
    prob=np.zeros((H,W),"float32")
    for i in range(0,H,256):
        for j in range(0,W,256):
            p=x[:,i:i+256,j:j+256]; ph,pw=p.shape[1],p.shape[2]
            pad=np.zeros((8,256,256),"float32"); pad[:,:ph,:pw]=p
            o=pred_tile(models,pad,tta); prob[i:i+ph,j:j+pw]=o[:ph,:pw]
    return prob
def main():
    ms=load_ens()
    for tta in (False,True):
        tot=[]
        for h in S.CIR_HOLD:
            x,gt,vd=S.cir_load(h); H,W=gt.shape; prob=infer(ms,x,H,W,tta); tot.append(S.bestf1(prob[vd],gt[vd].astype(int)))
        print(f"{'TTA' if tta else 'no-TTA'}: 봄 4홀드아웃 best-F1={np.mean(tot):.3f}  개별={[round(t,3) for t in tot]}",flush=True)
if __name__=="__main__": main()
