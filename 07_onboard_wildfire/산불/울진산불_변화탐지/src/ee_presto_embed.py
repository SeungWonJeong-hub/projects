"""ko_presto_ts (월별 S2 시계열) → Presto 임베딩 (128,256,256) 타일 → ko_presto/.

각 타일 256x256 픽셀을 시계열 (T,6) 로 만들어 Presto 인코더에 배치 통과 → 픽셀당 128d.
구름월(tvalid=0)은 mask=1 로 무시. y/valid 는 그대로 승계. 이후 ee_ae_train.py 식 GN-UNet 학습.

사용: python3 ee_presto_embed.py
"""
import os, glob, numpy as np, torch
from presto_loader import load_presto, OUR_S2_BANDS
from presto.dataops.utils import construct_batch_presto_input

HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
TS=os.path.join(PROJ,"ko_presto_ts"); OUT=os.path.join(PROJ,"ko_presto")
DEV="cuda" if torch.cuda.is_available() else "cpu"
BATCH=8192

def embed_tile(m, ts, months, tvalid, latlon):
    T,_,H,W=ts.shape
    # (T,6,H,W) → (H*W, T, 6)
    s2=np.transpose(ts,(2,3,0,1)).reshape(H*W,T,len(OUR_S2_BANDS))
    s2=torch.from_numpy(s2).float()
    x,mask,dw=construct_batch_presto_input(s2=s2, s2_bands=OUR_S2_BANDS)  # (N,T,17)
    # 구름월(tvalid=0) 전체밴드 mask=1(무시)
    miss=torch.from_numpy((tvalid==0)).bool()  # (T,)
    if miss.any(): mask[:,miss,:]=1
    N=x.shape[0]
    lat,lon=float(latlon[0]),float(latlon[1])
    latlons=torch.tensor([[lat,lon]]).float().repeat(N,1).to(DEV)
    month0=torch.tensor([int(months[0])]).long().repeat(N).to(DEV)
    embs=[]
    with torch.no_grad():
        for i in range(0,N,BATCH):
            xb=x[i:i+BATCH].to(DEV); mb=mask[i:i+BATCH].to(DEV); db=dw[i:i+BATCH].long().to(DEV)
            e=m.encoder(xb, dynamic_world=db, latlons=latlons[i:i+BATCH],
                        mask=mb, month=month0[i:i+BATCH], eval_task=True)
            embs.append(e.cpu().numpy())
    emb=np.concatenate(embs,0)                      # (N,128)
    return np.transpose(emb.reshape(H,W,-1),(2,0,1)).astype(np.float32)  # (128,H,W)

def main():
    m=load_presto(device=DEV)
    os.makedirs(OUT,exist_ok=True); n=0
    for f in sorted(glob.glob(os.path.join(TS,"*","*.npz"))):
        fire=os.path.basename(os.path.dirname(f)); tile=os.path.basename(f)
        od=os.path.join(OUT,fire); os.makedirs(od,exist_ok=True)
        op=os.path.join(od,tile)
        if os.path.exists(op): print(f"[skip] {fire}/{tile}"); continue
        d=np.load(f)
        emb=embed_tile(m, d["ts"], d["months"], d["tvalid"], d["latlon"])
        np.savez_compressed(op, x=emb, y=d["y"].astype(np.uint8), valid=d["valid"].astype(np.uint8))
        n+=1; print(f"[{n}] {fire}/{tile} → 임베딩 {emb.shape} (유효월 {int(d['tvalid'].sum())})",flush=True)
    print(f"완성 | ko_presto 타일 {len(glob.glob(os.path.join(OUT,'*','*.npz')))}",flush=True)

if __name__=="__main__": main()
