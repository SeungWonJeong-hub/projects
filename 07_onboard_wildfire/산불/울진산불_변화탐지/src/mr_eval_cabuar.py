"""다지역 모델을 CaBuAr 실측(hand-label)로 재검증 — 순환 dNBR 우려 해소.
비교: ① 다지역 모델  ② dNBR 채널만 임계(echo 베이스라인)  ③ CaBuAr-native(F1 0.648, 기존)
실측 라벨에서 ①이 ②보다 높으면 = dNBR 재현이 아니라 진짜 산불 탐지.
"""
from __future__ import annotations
import os, glob
import numpy as np, torch
import segmentation_models_pytorch as smp

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
FILES = sorted(glob.glob(os.path.join(HERE,"..","data","cabuar","test","*.npz")))

def metrics(P, Y):
    def at(t):
        pr=P>=t; tp=(pr&(Y>0)).sum(); fp=(pr&(Y==0)).sum(); fn=((~pr)&(Y>0)).sum()
        pre=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
        return 2*pre*rec/(pre+rec+1e-9), tp/(tp+fp+fn+1e-9), pre, rec
    t=max(np.linspace(0.1,0.9,33), key=lambda t:at(t)[0])
    f1,iou,pre,rec=at(t)
    return dict(thr=round(float(t),3),f1=round(float(f1),3),iou=round(float(iou),3),
                precision=round(float(pre),3),recall=round(float(rec),3))

def main():
    imgs=[]; masks=[]
    for f in FILES:
        d=np.load(f); imgs.append(d["image"].astype(np.float32)); masks.append(d["mask"].astype(np.uint8))
    X=np.stack(imgs); Y=np.stack(masks)
    print(f"CaBuAr 실측 test: {len(FILES)} 타일, 불탄픽셀 {Y.mean()*100:.1f}%",flush=True)

    # ② dNBR echo 베이스라인 (13ch 중 마지막 채널 = dNBR)
    dnbr = X[:,12]  # [0,1], 0.5=변화없음
    print("② dNBR-only echo 베이스라인:", metrics(dnbr.reshape(-1), Y.reshape(-1)),flush=True)

    # ① 다지역 모델
    m=smp.Unet("resnet34",encoder_weights=None,in_channels=13,classes=1).to(DEV)
    m.load_state_dict(torch.load(os.path.join(HERE,"..","runs","mr_cd","mr_unet.pt"),map_location=DEV,weights_only=True)); m.eval()
    P=[]
    with torch.no_grad():
        for i in range(0,len(X),8):
            xb=torch.from_numpy(X[i:i+8]).to(DEV)
            with torch.cuda.amp.autocast():
                p=torch.sigmoid(m(xb)).float()[:,0].cpu().numpy()
            P.append(p)
    P=np.concatenate(P)
    print("① 다지역 모델 (실측 CaBuAr):", metrics(P.reshape(-1), Y.reshape(-1)),flush=True)
    print("③ [기존] CaBuAr-native 모델: F1 0.648 / IoU 0.479 (같은 실측 test)")

if __name__=="__main__": main()
