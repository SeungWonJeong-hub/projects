"""유럽(CEMS 실측) 먼 대륙 검증 — US 7생태권 학습 모델(mtbs_v2_unet)을 유럽 실측에 테스트.
비교: dNBR-echo(입력 dNBR채널) baseline.  '북미 밖에서도 되나'의 답.
"""
from __future__ import annotations
import os, glob
import numpy as np, torch
import segmentation_models_pytorch as smp

HERE=os.path.dirname(os.path.abspath(__file__)); DEV="cuda"
DATA=os.path.join(HERE,"..","cems_data_raw")
MODEL=os.path.join(HERE,"..","runs","mtbs_cd","mtbs_v2_unet.pt")

def metrics(P,Y):
    def at(t):
        pr=P>=t; tp=(pr&(Y>0)).sum();fp=(pr&(Y==0)).sum();fn=((~pr)&(Y>0)).sum()
        pe=tp/(tp+fp+1e-9);re=tp/(tp+fn+1e-9); return 2*pe*re/(pe+re+1e-9),tp/(tp+fp+fn+1e-9),pe,re
    t=max(np.linspace(0.1,0.9,33),key=lambda t:at(t)[0]); f1,iou,pe,re=at(t)
    return dict(thr=round(float(t),3),f1=round(float(f1),3),iou=round(float(iou),3),precision=round(float(pe),3),recall=round(float(re),3))

def main():
    files=sorted(glob.glob(os.path.join(DATA,"*","*.npz")))
    X=[];Y=[]
    for f in files:
        d=np.load(f); X.append(d["x"].astype(np.float32)); Y.append(d["y"].astype(np.uint8))
    X=np.stack(X); Y=np.stack(Y)
    print(f"유럽 CEMS 실측: {len(files)}타일, 불탄 {Y.mean()*100:.1f}%",flush=True)
    print("dNBR-echo:", metrics(X[:,12].reshape(-1), Y.reshape(-1)),flush=True)
    m=smp.Unet("resnet34",encoder_weights=None,in_channels=13,classes=1).to(DEV)
    m.load_state_dict(torch.load(MODEL,map_location=DEV,weights_only=True)); m.eval()
    P=[]
    with torch.no_grad():
        for i in range(0,len(X),8):
            xb=torch.from_numpy(X[i:i+8]).to(DEV)
            with torch.cuda.amp.autocast(): p=torch.sigmoid(m(xb)).float()[:,0].cpu().numpy()
            P.append(p)
    P=np.concatenate(P)
    print("US 7생태권 모델 → 유럽 실측:", metrics(P.reshape(-1),Y.reshape(-1)),flush=True)
    print("[비교] US내부 LORO 0.885 / US→CaBuAr(CA) 0.547 / dNBR-echo(CA) 0.386 / native 0.648")

if __name__=="__main__": main()
