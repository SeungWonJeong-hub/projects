"""SAM2로 ko_refl 전 타일 라벨 정제 → ko_refl_sam2/{fire}/{tile}.npz (refined y 저장).
post SWIR 위색합성 + 폴리곤 박스프롬프트 → 폴리곤 내부 제약. AlphaEarth 재학습에서 이 라벨 사용.
사용: KO_DATA=../ko_refl python3 sam2_refine_all.py
"""
import os, glob, numpy as np, torch
from transformers import Sam2Model, Sam2Processor
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("KO_DATA",os.path.join(HERE,"..","ko_refl"))
OUT=os.path.join(HERE,"..","ko_refl_sam2"); os.makedirs(OUT,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
proc=Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")
model=Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").to(DEV).eval()
def stretch(a,lo=2,hi=98):
    p1,p2=np.percentile(a,[lo,hi]); return np.clip((a-p1)/(p2-p1+1e-6),0,1)
def composite(post6):
    return (np.stack([stretch(post6[5]),stretch(post6[3]),stretch(post6[2])],-1)*255).astype(np.uint8)
def iou(a,b): return (a&b).sum()/((a|b).sum()+1e-9)
@torch.no_grad()
def refine(rgb,poly):
    ys,xs=np.where(poly>0)
    if len(xs)<10: return poly.copy()
    box=[[[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]]]
    inp=proc(images=rgb,input_boxes=box,return_tensors="pt").to(DEV)
    out=model(**inp); masks=proc.post_process_masks(out.pred_masks,inp["original_sizes"])[0].cpu().numpy()[0]>0.5
    best=max(masks,key=lambda m:iou(m,poly>0))
    return (best&(poly>0)).astype(np.uint8)
n=0; tot_p=0; tot_r=0
for fire_dir in sorted(glob.glob(os.path.join(DATA,"*"))):
    fire=os.path.basename(fire_dir); od=os.path.join(OUT,fire); os.makedirs(od,exist_ok=True)
    for f in glob.glob(os.path.join(fire_dir,"*.npz")):
        d=np.load(f); x=d["x"].astype(np.float32); y=d["y"].astype(np.uint8)
        if y.sum()<10:
            np.savez_compressed(os.path.join(od,os.path.basename(f)),y=y); n+=1; continue
        ref=refine(composite(x[6:12]),y)
        np.savez_compressed(os.path.join(od,os.path.basename(f)),y=ref.astype(np.uint8))
        tot_p+=int(y.sum()); tot_r+=int(ref.sum()); n+=1
print(f"정제 완료: {n}타일 | 폴리곤 {tot_p*0.04:.0f}ha → 정제 {tot_r*0.04:.0f}ha (제거 {(1-tot_r/max(tot_p,1))*100:.1f}%) → {OUT}")
