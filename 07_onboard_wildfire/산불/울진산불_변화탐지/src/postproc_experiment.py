"""precision 후처리 실험 — FT4 과예측(P0.48) 오탐 제거로 F1 올리기.
raw vs TTA(flip평균) vs TTA+형태학(opening+최소면적). 봄/여름 F1 비교. 후처리는 추론 후처리(재학습無)."""
import os, glob, numpy as np, torch
from scipy import ndimage as ndi
os.environ.setdefault("KO_DATA","ko_refl"); os.environ.setdefault("RUN_NAME","ko_prithvi_refl_ft4_v2")
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prithvi_ft as P

model=P.PrithviSeg(unfreeze=4).to(P.DEV)
model.load_state_dict(torch.load(os.path.join(P.RUN,"best.pt"),map_location=P.DEV))
model.eval()
hold=[l.strip() for l in open(P.HOLD) if l.strip()]
def season(m): return "봄" if 3<=m<=5 else "여름" if 6<=m<=8 else "가을" if 9<=m<=11 else "겨울"

@torch.no_grad()
def prob2d(x, tta):
    pre,post=x[0:6],x[6:12]
    def one(pr,po): return torch.sigmoid(model(P.prep(pr,po).unsqueeze(0).to(P.DEV))).squeeze().cpu().numpy()
    if not tta: return one(pre,post)
    ps=[one(pre,post),
        one(pre[:,:,::-1].copy(),post[:,:,::-1].copy())[:,::-1],
        one(pre[:,::-1].copy(),post[:,::-1].copy())[::-1,:]]
    return np.mean(ps,0)

# 타일별 prob(raw,tta), y, season 사전계산
tiles=[]
for fire in hold:
    se=season(int(fire.split("_")[1][4:6]))
    for f in glob.glob(os.path.join(P.DATA,fire,"*.npz")):
        d=np.load(f); x=d["x"].astype(np.float32)/10000.0; y=d["y"].astype(np.uint8)
        tiles.append((prob2d(x,False),prob2d(x,True),y,se))
print(f"타일 {len(tiles)}개 예측완료")

def morph(mask,min_px=25):
    m=ndi.binary_opening(mask,iterations=1)
    lab,n=ndi.label(m)
    if n>0:
        sz=ndi.sum(np.ones_like(lab),lab,range(1,n+1))
        rm=np.where(sz<min_px)[0]+1
        if len(rm): m[np.isin(lab,rm)]=0
    return m

def eval_variant(kind):
    # kind: raw / tta / tta_morph. thr 스윕 후 계절별 best-thr F1
    from collections import defaultdict
    THR=np.arange(0.3,0.85,0.05)
    # 각 thr에서 계절별 TP/FP/FN
    acc={t:defaultdict(lambda:[0,0,0]) for t in THR}
    for pr_raw,pr_tta,y,se in tiles:
        pr=pr_raw if kind=="raw" else pr_tta
        for t in THR:
            m=(pr>=t)
            if kind=="tta_morph": m=morph(m.astype(np.uint8)).astype(bool)
            tp=int((m&(y>0)).sum());fp=int((m&(y==0)).sum());fn=int((~m&(y>0)).sum())
            for grp in ("ALL",se):
                a=acc[t][grp]; a[0]+=tp;a[1]+=fp;a[2]+=fn
    def f1p(a):
        tp,fp,fn=a; pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return 2*pr*rc/(pr+rc+1e-9),pr,rc
    out={}
    for grp in ("ALL","봄","여름"):
        best=max(THR,key=lambda t:f1p(acc[t][grp])[0])
        f1,pr,rc=f1p(acc[best][grp]); out[grp]=(round(f1,3),round(pr,3),round(rc,3),round(float(best),2))
    return out

print(f"{'변형':10s} {'전체F1(P,R,thr)':28s} {'봄F1':22s} {'여름F1':22s}")
res={}
for kind in ["raw","tta","tta_morph"]:
    r=eval_variant(kind); res[kind]=r
    fmt=lambda g:f"{r[g][0]}({r[g][1]},{r[g][2]}@{r[g][3]})"
    print(f"{kind:10s} {fmt('ALL'):28s} {fmt('봄'):22s} {fmt('여름'):22s}")
print(f"\n=== 봄 F1: raw {res['raw']['봄'][0]} → tta {res['tta']['봄'][0]} → tta+morph {res['tta_morph']['봄'][0]} "
      f"(Δ {res['tta_morph']['봄'][0]-res['raw']['봄'][0]:+.3f})")
print(f"=== 전체 F1: raw {res['raw']['ALL'][0]} → tta+morph {res['tta_morph']['ALL'][0]} "
      f"(Δ {res['tta_morph']['ALL'][0]-res['raw']['ALL'][0]:+.3f}) | precision {res['raw']['ALL'][1]}→{res['tta_morph']['ALL'][1]}")
import json; os.makedirs(os.path.join(P.RUN,"..","postproc"),exist_ok=True)
json.dump(res,open(os.path.join(P.RUN,"..","postproc","results.json"),"w"),ensure_ascii=False,indent=1)
