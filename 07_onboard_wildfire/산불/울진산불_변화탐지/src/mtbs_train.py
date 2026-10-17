"""다지역 실측라벨 학습 — MTBS 7개 다생태권 → CaBuAr(캘리포니아, 홀드아웃 실측) 검증.
핵심 질문: 실측 라벨 다지역 학습이 dNBR-echo(0.386)를 넘고 native(0.648)에 근접/도달하는가?
= "다지역 실측라벨이면 dNBR 천장을 깬다"의 증명.
"""
from __future__ import annotations
import os, glob, random
import numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp

HERE = os.path.dirname(os.path.abspath(__file__))
MTBS = os.path.join(HERE, "..", "mtbs_data")
CABUAR_TEST = os.path.join(HERE, "..", "data", "cabuar", "test")
DEV="cuda"; EPOCHS=45; BATCH=6

class DS(torch.utils.data.Dataset):
    def __init__(self, files, kx, ky, aug=False): self.f=files; self.kx=kx; self.ky=ky; self.aug=aug
    def __len__(self): return len(self.f)
    def __getitem__(self,i):
        d=np.load(self.f[i]); x=d[self.kx].astype(np.float32); y=d[self.ky].astype(np.float32)
        if self.aug:
            if random.random()<0.5: x=x[:,:,::-1].copy(); y=y[:,::-1].copy()
            if random.random()<0.5: x=x[:,::-1].copy(); y=y[::-1].copy()
            k=random.randint(0,3)
            if k: x=np.rot90(x,k,axes=(1,2)).copy(); y=np.rot90(y,k).copy()
        return torch.from_numpy(x), torch.from_numpy(y)

class DiceBCE(nn.Module):
    def __init__(s): super().__init__(); s.bce=nn.BCEWithLogitsLoss()
    def forward(s,p,y):
        y=y.unsqueeze(1); pb=torch.sigmoid(p); inter=(pb*y).sum((2,3))
        dice=1-(2*inter+1)/(pb.sum((2,3))+y.sum((2,3))+1); return s.bce(p,y)+dice.mean()

@torch.no_grad()
def evaluate(model, dl, thr=None):
    model.eval(); P=[]; Y=[]
    for x,y in dl:
        p=torch.sigmoid(model(x.to(DEV))).float()[:,0].cpu().numpy(); P.append(p.reshape(-1)); Y.append(y.numpy().reshape(-1))
    P=np.concatenate(P); Y=np.concatenate(Y)
    def at(t):
        pr=P>=t; tp=(pr&(Y>0)).sum(); fp=(pr&(Y==0)).sum(); fn=((~pr)&(Y>0)).sum()
        pre=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
        return 2*pre*rec/(pre+rec+1e-9), tp/(tp+fp+fn+1e-9), pre, rec
    if thr is None: thr=float(max(np.linspace(0.1,0.9,33), key=lambda t:at(t)[0]))
    f1,iou,pre,rec=at(thr)
    return dict(thr=round(thr,3),f1=round(float(f1),3),iou=round(float(iou),3),precision=round(float(pre),3),recall=round(float(rec),3))

def main():
    torch.manual_seed(42); random.seed(42)
    tr=[]
    for d in sorted(glob.glob(os.path.join(MTBS,"*"))):
        tr+=glob.glob(os.path.join(d,"*.npz"))
    te=sorted(glob.glob(os.path.join(CABUAR_TEST,"*.npz")))
    random.shuffle(tr); nval=max(10,len(tr)//12); va=tr[:nval]; tr=tr[nval:]
    print(f"MTBS 실측 지역: {[os.path.basename(d) for d in sorted(glob.glob(os.path.join(MTBS,'*')))]}",flush=True)
    print(f"타일 — train {len(tr)} val {len(va)} test(CaBuAr 실측) {len(te)}",flush=True)
    if len(tr)<20 or not te: print("데이터 부족"); return
    tl=torch.utils.data.DataLoader(DS(tr,"x","y",aug=True),batch_size=BATCH,shuffle=True,num_workers=2,drop_last=True)
    vl=torch.utils.data.DataLoader(DS(va,"x","y"),batch_size=BATCH,num_workers=2)
    el=torch.utils.data.DataLoader(DS(te,"image","mask"),batch_size=BATCH,num_workers=2)
    model=smp.Unet("resnet34",encoder_weights="imagenet",in_channels=13,classes=1).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    lossf=DiceBCE(); scaler=torch.cuda.amp.GradScaler(); best=0; bstate=None
    for ep in range(EPOCHS):
        model.train(); run=0
        for x,y in tl:
            x=x.to(DEV); y=y.to(DEV); opt.zero_grad()
            with torch.cuda.amp.autocast(): loss=lossf(model(x),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run+=loss.item()
        sch.step(); vm=evaluate(model,vl)
        if vm["f1"]>best: best=vm["f1"]; bstate={k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep%5==0 or ep==EPOCHS-1: print(f"[ep {ep:02d}] loss={run/len(tl):.3f} val_f1={vm['f1']:.3f}",flush=True)
    model.load_state_dict(bstate); vm=evaluate(model,vl); tm=evaluate(model,el,thr=vm["thr"])
    print(f"\n=== 다지역 실측라벨 → CaBuAr(캘리포니아, 미학습 실측) ===")
    print(f"VAL(MTBS): {vm}")
    print(f"TEST(CaBuAr 실측): {tm}")
    print(f"[비교] dNBR-echo 0.386 / dNBR-유사라벨 다지역모델 0.373 / CaBuAr-native(in-domain) 0.648")
    od=os.path.join(HERE,"..","runs","mtbs_cd"); os.makedirs(od,exist_ok=True)
    torch.save(bstate, os.path.join(od,"mtbs_unet.pt"))
    import json; json.dump(dict(val=vm,test_cabuar=tm),open(os.path.join(od,"mtbs_results.json"),"w"),indent=2)
    print("저장:",od)

if __name__=="__main__": main()
