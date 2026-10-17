"""다지역 산불 CD LORO 학습 — 일반화 측정.
mr_data 의 여러 산불(13채널 [pre6|post6|dNBR], 통일 정규화)로 학습,
울진(uljin2022)은 학습 제외 홀드아웃 → 교차지역 일반화 F1/IoU.
비교 기준: CaBuAr→울진 전이 실패(FIRMS 9.3%), 울진 in-domain IoU 0.63.
"""
from __future__ import annotations
import os, glob, random
import numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "mr_data")
DEV = "cuda"; TEST_REGION = "uljin2022"; EPOCHS = 40; BATCH = 6

def load_region(name):
    return sorted(glob.glob(os.path.join(DATA, name, "*.npz")))

class DS(torch.utils.data.Dataset):
    def __init__(self, files, aug=False): self.files=files; self.aug=aug
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d=np.load(self.files[i]); x=d["x"].astype(np.float32); y=d["y"].astype(np.float32)
        if self.aug:
            if random.random()<0.5: x=x[:,:,::-1].copy(); y=y[:,::-1].copy()
            if random.random()<0.5: x=x[:,::-1].copy(); y=y[::-1].copy()
            k=random.randint(0,3)
            if k: x=np.rot90(x,k,axes=(1,2)).copy(); y=np.rot90(y,k).copy()
        return torch.from_numpy(x), torch.from_numpy(y)

class DiceBCE(nn.Module):
    def __init__(self): super().__init__(); self.bce=nn.BCEWithLogitsLoss()
    def forward(self,p,y):
        y=y.unsqueeze(1); pb=torch.sigmoid(p)
        inter=(pb*y).sum((2,3)); dice=1-(2*inter+1)/(pb.sum((2,3))+y.sum((2,3))+1)
        return self.bce(p,y)+dice.mean()

@torch.no_grad()
def evaluate(model, dl, thr=None):
    model.eval(); P=[]; Y=[]
    for x,y in dl:
        p=torch.sigmoid(model(x.to(DEV))).float()[:,0].cpu().numpy(); P.append(p); Y.append(y.numpy())
    P=np.concatenate([p.reshape(-1) for p in P]); Y=np.concatenate([y.reshape(-1) for y in Y])
    def f1_at(t):
        pr=P>=t; tp=(pr&(Y>0)).sum(); fp=(pr&(Y==0)).sum(); fn=((~pr)&(Y>0)).sum()
        pre=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9); f1=2*pre*rec/(pre+rec+1e-9)
        iou=tp/(tp+fp+fn+1e-9); return f1,iou,pre,rec
    if thr is None:
        best=max(np.linspace(0.2,0.8,25), key=lambda t:f1_at(t)[0]); thr=float(best)
    f1,iou,pre,rec=f1_at(thr)
    return dict(thr=round(thr,3),f1=round(float(f1),3),iou=round(float(iou),3),
                precision=round(float(pre),3),recall=round(float(rec),3))

def main():
    torch.manual_seed(42); random.seed(42)
    regions=[os.path.basename(d) for d in sorted(glob.glob(os.path.join(DATA,"*"))) if os.path.isdir(d)]
    train_regions=[r for r in regions if r!=TEST_REGION]
    print("학습 지역:",train_regions,"| 홀드아웃:",TEST_REGION,flush=True)
    tr_files=[];
    for r in train_regions: tr_files+=load_region(r)
    te_files=load_region(TEST_REGION)
    random.shuffle(tr_files); nval=max(8,len(tr_files)//10); va_files=tr_files[:nval]; tr_files=tr_files[nval:]
    print(f"타일 — train {len(tr_files)} val {len(va_files)} test(울진) {len(te_files)}",flush=True)
    if not te_files or not tr_files: print("데이터 부족"); return
    tl=torch.utils.data.DataLoader(DS(tr_files,aug=True),batch_size=BATCH,shuffle=True,num_workers=2,drop_last=True)
    vl=torch.utils.data.DataLoader(DS(va_files),batch_size=BATCH,num_workers=2)
    el=torch.utils.data.DataLoader(DS(te_files),batch_size=BATCH,num_workers=2)
    model=smp.Unet("resnet34",encoder_weights="imagenet",in_channels=13,classes=1).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    lossf=DiceBCE(); scaler=torch.cuda.amp.GradScaler(); best_vf1=0; best_state=None
    for ep in range(EPOCHS):
        model.train(); run=0
        for x,y in tl:
            x=x.to(DEV); y=y.to(DEV)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss=lossf(model(x),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run+=loss.item()
        sched.step(); vm=evaluate(model,vl)
        if vm["f1"]>best_vf1: best_vf1=vm["f1"]; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep%5==0 or ep==EPOCHS-1: print(f"[ep {ep:02d}] loss={run/len(tl):.3f} val_f1={vm['f1']:.3f}",flush=True)
    model.load_state_dict(best_state)
    vm=evaluate(model,vl); tm=evaluate(model,el,thr=vm["thr"])
    print(f"\n=== 다지역→울진 홀드아웃 (교차지역 일반화) ===")
    print(f"VAL(학습지역): {vm}")
    print(f"TEST(울진, 미학습): {tm}")
    print(f"[비교] 울진 in-domain IoU 0.63 / CaBuAr→울진 전이 FIRMS 9.3%(사실상 실패)")
    od=os.path.join(HERE,"..","runs","mr_cd"); os.makedirs(od,exist_ok=True)
    torch.save(best_state, os.path.join(od,"mr_unet.pt"))
    import json; json.dump(dict(train_regions=train_regions,val=vm,test_uljin=tm),open(os.path.join(od,"mr_results.json"),"w"),indent=2)
    print("저장:", od)

if __name__=="__main__": main()
