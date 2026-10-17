"""우리 지표를 '이미지 전체'로도 환산 — 그들(accuracy/mean-IoU)과 같은 기준 비교용."""
import os, glob, numpy as np, torch
os.environ.setdefault("KO_DATA","ko_refl"); os.environ.setdefault("RUN_NAME","ko_prithvi_refl_ft4_v2")
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prithvi_ft as P
model=P.PrithviSeg(unfreeze=4).to(P.DEV)
model.load_state_dict(torch.load(os.path.join(P.RUN,"best.pt"),map_location=P.DEV))
hold=[l.strip() for l in open(P.HOLD) if l.strip()]
allP=[];allY=[]
for fire in hold:
    fs=glob.glob(os.path.join(P.DATA,fire,"*.npz"))
    if not fs: continue
    pp,yy=P.predict(model,fs); allP.append(pp);allY.append(yy)
Pa=np.concatenate(allP);Ya=np.concatenate(allY)
thr=P.best_thr(Pa,Ya)["thr"]; pred=(Pa>=thr)
tp=(pred&(Ya>0)).sum();tn=((~pred)&(Ya==0)).sum();fp=(pred&(Ya==0)).sum();fn=((~pred)&(Ya>0)).sum()
acc=(tp+tn)/len(Ya)
iou_fire=tp/(tp+fp+fn+1e-9)
iou_bg=tn/(tn+fp+fn+1e-9)
miou=(iou_fire+iou_bg)/2
f1=2*tp/(2*tp+fp+fn+1e-9)
print(f"홀드아웃 불탄비율(타일내): {Ya.mean()*100:.1f}%  thr={thr}")
print(f"[불탄클래스] F1={f1:.3f} IoU_fire={iou_fire:.3f}")
print(f"[이미지 전체] overall_accuracy={acc*100:.1f}%  mean_IoU={miou:.3f} (fire {iou_fire:.3f}+bg {iou_bg:.3f})")
