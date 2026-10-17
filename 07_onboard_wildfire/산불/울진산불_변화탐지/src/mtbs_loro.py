"""#1 leave-one-ecoregion-out (MTBS 실측, 일관 픽셀라벨) — 미학습 생태권 일반화 견고성.
서로 다른 생태권(사막 AZ·고산 CO·PNW OR·초원 MT)을 번갈아 홀드아웃 → 나머지로 학습 → 홀드아웃 실측 평가.
CaBuAr(0.547)에 더해 여러 미학습 지역에서 일반화가 유지되는지 확인.
"""
from __future__ import annotations
import os, glob, random, json
import numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp

HERE=os.path.dirname(os.path.abspath(__file__)); MTBS=os.path.join(HERE,"..","mtbs_data")
DEV="cuda"; EPOCHS=30; BATCH=6; THR_R=0.575
HOLDOUTS=["bush_az2020","cameronpeak_co2020","bootleg_or2021","richardspring_mt2021"]

class DS(torch.utils.data.Dataset):
    def __init__(s,f,aug=False): s.f=f; s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]); x=d["x"].astype(np.float32)
        y=((d["y"]>0)&(x[12]>THR_R)).astype(np.float32)     # 일관 픽셀라벨
        if s.aug:
            if random.random()<0.5: x=x[:,:,::-1].copy(); y=y[:,::-1].copy()
            if random.random()<0.5: x=x[:,::-1].copy(); y=y[::-1].copy()
            k=random.randint(0,3)
            if k: x=np.rot90(x,k,axes=(1,2)).copy(); y=np.rot90(y,k).copy()
        return torch.from_numpy(x), torch.from_numpy(y)

class DiceBCE(nn.Module):
    def __init__(s): super().__init__(); s.bce=nn.BCEWithLogitsLoss()
    def forward(s,p,y):
        y=y.unsqueeze(1); pb=torch.sigmoid(p); inter=(pb*y).sum((2,3))
        return s.bce(p,y)+(1-(2*inter+1)/(pb.sum((2,3))+y.sum((2,3))+1)).mean()

@torch.no_grad()
def ev(m,dl,thr=None):
    m.eval(); P=[];Y=[]
    for x,y in dl:
        p=torch.sigmoid(m(x.to(DEV))).float()[:,0].cpu().numpy(); P.append(p.reshape(-1)); Y.append(y.numpy().reshape(-1))
    P=np.concatenate(P);Y=np.concatenate(Y)
    def at(t):
        pr=P>=t; tp=(pr&(Y>0)).sum();fp=(pr&(Y==0)).sum();fn=((~pr)&(Y>0)).sum()
        pe=tp/(tp+fp+1e-9);re=tp/(tp+fn+1e-9); return 2*pe*re/(pe+re+1e-9),tp/(tp+fp+fn+1e-9),pe,re
    if thr is None: thr=float(max(np.linspace(0.1,0.9,25),key=lambda t:at(t)[0]))
    f1,iou,pe,re=at(thr); return dict(thr=round(thr,3),f1=round(float(f1),3),iou=round(float(iou),3),precision=round(float(pe),3),recall=round(float(re),3))

def train_eval(holdout):
    torch.manual_seed(42); random.seed(42)
    regions=[os.path.basename(d) for d in sorted(glob.glob(os.path.join(MTBS,"*")))]
    tr=[]; te=[]
    for r in regions:
        fs=glob.glob(os.path.join(MTBS,r,"*.npz"))
        (te if r==holdout else tr).extend(fs)
    random.shuffle(tr)
    tl=torch.utils.data.DataLoader(DS(tr,aug=True),batch_size=BATCH,shuffle=True,num_workers=2,drop_last=True)
    el=torch.utils.data.DataLoader(DS(te),batch_size=BATCH,num_workers=2)
    m=smp.Unet("resnet34",encoder_weights="imagenet",in_channels=13,classes=1).to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    lf=DiceBCE(); sc=torch.cuda.amp.GradScaler()
    for ep in range(EPOCHS):
        m.train()
        for x,y in tl:
            x=x.to(DEV);y=y.to(DEV); opt.zero_grad()
            with torch.cuda.amp.autocast(): loss=lf(m(x),y)
            sc.scale(loss).backward(); sc.step(opt); sc.update()
        sch.step()
    tm=ev(m,el)
    print(f"[홀드아웃 {holdout}] train {len(tr)}타일 / test {len(te)}타일 → {tm}",flush=True)
    return tm

def main():
    res={}
    for h in HOLDOUTS: res[h]=train_eval(h)
    print("\n=== #1 leave-one-ecoregion-out (MTBS 실측, 미학습 생태권) ===")
    for h,m in res.items(): print(f"  {h:22s} F1 {m['f1']:.3f} IoU {m['iou']:.3f} P {m['precision']:.2f} R {m['recall']:.2f}")
    f1s=[m['f1'] for m in res.values()]; print(f"  평균 F1 {np.mean(f1s):.3f} (범위 {min(f1s):.3f}~{max(f1s):.3f})")
    print(f"[참고] US→CaBuAr(CA) 0.547 / CaBuAr-native 0.648 / dNBR-echo 0.386")
    json.dump(res,open(os.path.join(HERE,"..","runs","mtbs_cd","loro_results.json"),"w"),indent=2)

if __name__=="__main__": main()
