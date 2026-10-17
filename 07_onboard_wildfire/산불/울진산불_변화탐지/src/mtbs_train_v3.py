"""#1 라벨 정의 통일 — MTBS 경계전체 라벨을 '경계 ∩ 불탄픽셀'(픽셀단위)로 재정의해 CaBuAr 정의와 일치.
y_new = perimeter & (dNBR채널 > thr).  경계로 dNBR 오탐(화재밖)을 억제 + 픽셀단위로 CaBuAr와 정합.
검증: CaBuAr 실측 → 과예측(precision 0.156) 개선되는가.
"""
from __future__ import annotations
import os, glob, random
import numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp

HERE=os.path.dirname(os.path.abspath(__file__))
MTBS=os.path.join(HERE,"..","mtbs_data"); CABUAR=os.path.join(HERE,"..","data","cabuar","test")
DEV="cuda"; EPOCHS=45; BATCH=6; DNBR_THR=float(os.environ.get("DNBR_THR","0.575"))  # x12=(dnbr+1)/2, 0.575≈dnbr0.15

class DS(torch.utils.data.Dataset):
    def __init__(s,files,kx,ky,relabel=False,aug=False): s.f=files;s.kx=kx;s.ky=ky;s.rl=relabel;s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]); x=d[s.kx].astype(np.float32); y=d[s.ky].astype(np.float32)
        if s.rl: y=(y>0)&(x[12]>DNBR_THR); y=y.astype(np.float32)   # 경계 ∩ 불탄픽셀
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
        dice=1-(2*inter+1)/(pb.sum((2,3))+y.sum((2,3))+1); return s.bce(p,y)+dice.mean()

@torch.no_grad()
def ev(model,dl,thr=None):
    model.eval(); P=[];Y=[]
    for x,y in dl:
        p=torch.sigmoid(model(x.to(DEV))).float()[:,0].cpu().numpy(); P.append(p.reshape(-1)); Y.append(y.numpy().reshape(-1))
    P=np.concatenate(P);Y=np.concatenate(Y)
    def at(t):
        pr=P>=t; tp=(pr&(Y>0)).sum();fp=(pr&(Y==0)).sum();fn=((~pr)&(Y>0)).sum()
        pe=tp/(tp+fp+1e-9);re=tp/(tp+fn+1e-9); return 2*pe*re/(pe+re+1e-9),tp/(tp+fp+fn+1e-9),pe,re
    if thr is None: thr=float(max(np.linspace(0.1,0.9,33),key=lambda t:at(t)[0]))
    f1,iou,pe,re=at(thr); return dict(thr=round(thr,3),f1=round(float(f1),3),iou=round(float(iou),3),precision=round(float(pe),3),recall=round(float(re),3))

def main():
    torch.manual_seed(42); random.seed(42)
    tr=[]; [tr.extend(glob.glob(os.path.join(d,"*.npz"))) for d in sorted(glob.glob(os.path.join(MTBS,"*")))]
    cabuar_tr=sorted(glob.glob(os.path.join(HERE,"..","data","cabuar","train","*.npz")))
    te=sorted(glob.glob(os.path.join(CABUAR,"*.npz")))
    random.shuffle(tr); nv=max(10,len(tr)//12); va=tr[:nv]; tr=tr[nv:]
    # 재정의 후 불탄비율 확인
    s=np.load(tr[0]); yp=(s["y"]>0)&(s["x"][12]>DNBR_THR)
    print(f"[#3 하모나이제이션 CaBuAr+MTBS] MTBS train {len(tr)} + CaBuAr-train {len(cabuar_tr)} | val {len(va)} | test(CaBuAr) {len(te)} | MTBS재정의 불탄 {yp.mean()*100:.1f}%",flush=True)
    from torch.utils.data import ConcatDataset
    tl=torch.utils.data.DataLoader(ConcatDataset([DS(tr,"x","y",relabel=True,aug=True), DS(cabuar_tr,"image","mask",relabel=False,aug=True)]),batch_size=BATCH,shuffle=True,num_workers=2,drop_last=True)
    vl=torch.utils.data.DataLoader(DS(va,"x","y",relabel=True),batch_size=BATCH,num_workers=2)
    el=torch.utils.data.DataLoader(DS(te,"image","mask"),batch_size=BATCH,num_workers=2)
    m=smp.Unet("resnet34",encoder_weights="imagenet",in_channels=13,classes=1).to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    lf=DiceBCE(); sc=torch.cuda.amp.GradScaler(); best=0; bs=None
    for ep in range(EPOCHS):
        m.train(); run=0
        for x,y in tl:
            x=x.to(DEV);y=y.to(DEV); opt.zero_grad()
            with torch.cuda.amp.autocast(): loss=lf(m(x),y)
            sc.scale(loss).backward(); sc.step(opt); sc.update(); run+=loss.item()
        sch.step(); vm=ev(m,vl)
        if vm["f1"]>best: best=vm["f1"]; bs={k:v.cpu().clone() for k,v in m.state_dict().items()}
        if ep%5==0 or ep==EPOCHS-1: print(f"[ep {ep:02d}] loss={run/len(tl):.3f} val_f1={vm['f1']:.3f}",flush=True)
    m.load_state_dict(bs); vm=ev(m,vl); tm=ev(m,el,thr=vm["thr"])
    print(f"\n=== #3 하모나이제이션(CaBuAr실측+MTBS일관화) → CaBuAr-test ===")
    print(f"VAL(MTBS 재정의): {vm}")
    print(f"TEST(CaBuAr 실측): {tm}")
    print(f"[비교] #1 MTBS단독 0.547 / dNBR-echo 0.386 / CaBuAr-native 0.648")
    od=os.path.join(HERE,"..","runs","mtbs_cd"); os.makedirs(od,exist_ok=True)
    import json; json.dump(dict(val=vm,test_cabuar=tm,dnbr_thr=DNBR_THR),open(os.path.join(od,"mtbs_v3_results.json"),"w"),indent=2)
    print("저장:",od)

if __name__=="__main__": main()
