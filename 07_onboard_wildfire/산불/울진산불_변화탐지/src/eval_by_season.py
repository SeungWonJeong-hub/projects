"""홀드아웃 화재를 계절별로 F1 분해 — '봄만 어렵고 비봄은 잘 되나?' 검증."""
import os, glob, numpy as np, torch
os.environ.setdefault("KO_DATA","ko_refl"); os.environ.setdefault("RUN_NAME","ko_prithvi_refl_ft4_v2")
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prithvi_ft as P

def season(m):
    return "봄" if 3<=m<=5 else "여름" if 6<=m<=8 else "가을" if 9<=m<=11 else "겨울"

model=P.PrithviSeg(unfreeze=4).to(P.DEV)
model.load_state_dict(torch.load(os.path.join(P.RUN,"best.pt"), map_location=P.DEV))
hold=[l.strip() for l in open(P.HOLD) if l.strip()]

# 전체 홀드아웃 best-thr(공정 단일임계)
allP=[];allY=[]
per=[]
for fire in hold:
    files=sorted(glob.glob(os.path.join(P.DATA,fire,"*.npz")))
    if not files: continue
    pp,yy=P.predict(model,files); allP.append(pp);allY.append(yy)
    m=int(fire.split("_")[1][4:6])
    per.append((fire,season(m),pp,yy))
Pa=np.concatenate(allP);Ya=np.concatenate(allY)
thr=P.best_thr(Pa,Ya)["thr"]
print(f"전체 홀드아웃 best-thr={thr} | 전체 F1={P.stats(Pa,Ya,thr)['f1']:.3f}\n")

from collections import defaultdict
byseas=defaultdict(lambda:[[],[]])
print(f"{'화재':28s} {'계절':4s} {'불탄%':6s} {'F1':6s} {'P':6s} {'R':6s}")
for fire,se,pp,yy in per:
    st=P.stats(pp,yy,thr)
    print(f"{fire:28s} {se:4s} {yy.mean()*100:5.1f}  {st['f1']:.3f}  {st['precision']:.3f}  {st['recall']:.3f}")
    byseas[se][0].append(pp);byseas[se][1].append(yy)
print("\n=== 계절별 통합 F1 (동일 임계) ===")
for se in ["봄","여름","가을","겨울"]:
    if se not in byseas: continue
    pp=np.concatenate(byseas[se][0]);yy=np.concatenate(byseas[se][1])
    st=P.stats(pp,yy,thr)
    print(f"{se}: F1={st['f1']:.3f} P={st['precision']:.3f} R={st['recall']:.3f} (화재 {len(byseas[se][0])})")
