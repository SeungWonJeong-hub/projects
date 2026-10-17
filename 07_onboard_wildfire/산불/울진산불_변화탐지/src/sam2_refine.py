"""SAM2로 산림청 경계폴리곤 → 픽셀 정제. post SWIR 위색합성에 폴리곤 박스 프롬프트 →
실제 엣지 따라 마스크 → 폴리곤 내부로 제약(안탄 섬 제거). GT 화재서 정제 + 미리보기 PNG.
사용: KO_DATA=../ko_refl python3 sam2_refine.py
"""
import os, glob, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import Sam2Model, Sam2Processor
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("KO_DATA",os.path.join(HERE,"..","ko_refl"))
HOLD=os.path.join(HERE,"..","data","holdout_fixed.txt")
OUT=os.path.join(HERE,"..","sam2_refine"); os.makedirs(OUT,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
proc=Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")
model=Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").to(DEV).eval()

def stretch(a,lo=2,hi=98):
    p1,p2=np.percentile(a,[lo,hi]); return np.clip((a-p1)/(p2-p1+1e-6),0,1)
def composite(post6):  # [B02,B03,B04,B8A,B11,B12] → SWIR 위색 R=B12,G=B8A,B=B04
    r=stretch(post6[5]);g=stretch(post6[3]);b=stretch(post6[2])
    return (np.stack([r,g,b],-1)*255).astype(np.uint8)
def iou(a,b):
    i=(a&b).sum();u=(a|b).sum();return i/(u+1e-9)

@torch.no_grad()
def refine(rgb, poly):
    ys,xs=np.where(poly>0)
    if len(xs)<10: return poly.copy()
    box=[[[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]]]  # 폴리곤 bbox
    inp=proc(images=rgb,input_boxes=box,return_tensors="pt").to(DEV)
    out=model(**inp)
    masks=proc.post_process_masks(out.pred_masks,inp["original_sizes"])[0].cpu().numpy()  # (1,3,H,W)
    masks=masks[0]>0.5  # (3,H,W)
    best=max(masks,key=lambda m:iou(m,poly>0))       # 폴리곤과 가장 정합한 후보
    return (best & (poly>0)).astype(np.uint8)         # 폴리곤 내부로 제약

hold=[l.strip() for l in open(HOLD) if l.strip()]
print(f"{'화재':28s} {'폴리곤ha':9s} {'정제ha':8s} {'제거%':6s} {'IoU':5s}")
for fire in hold:
    for f in sorted(glob.glob(os.path.join(DATA,fire,"*.npz")))[:1]:  # 화재당 대표타일 1개 미리보기
        d=np.load(f); x=d["x"].astype(np.float32); y=d["y"].astype(np.uint8)
        rgb=composite(x[6:12]); ref=refine(rgb,y)
        ph=y.sum()*0.04; rh=ref.sum()*0.04; rem=(1-ref.sum()/max(y.sum(),1))*100
        print(f"{fire:28s} {ph:8.0f}  {rh:7.0f}  {rem:5.0f}  {iou(ref>0,y>0):.2f}")
        removed=((y>0)&(ref==0))  # SAM2가 없앤 픽셀
        # 제거픽셀 형광 오버레이: 초록(안탄) 위면 잘 없앤것, 빨강(불탄) 위면 잘못
        rm_rgb=rgb.copy(); rm_rgb[removed]=[0,255,255]  # 시안 형광
        fig,ax=plt.subplots(1,4,figsize=(16,4.2))
        ax[0].imshow(rgb);ax[0].set_title("1) post SWIR (red=burn, green=unburned)");ax[0].axis("off")
        ax[1].imshow(rgb);ax[1].imshow(y,alpha=0.4,cmap="autumn");ax[1].set_title(f"2) forest-service polygon {ph:.0f}ha");ax[1].axis("off")
        ax[2].imshow(rgb);ax[2].imshow(ref,alpha=0.4,cmap="autumn");ax[2].set_title(f"3) SAM2 refined {rh:.0f}ha");ax[2].axis("off")
        ax[3].imshow(rm_rgb);ax[3].set_title(f"4) REMOVED (cyan) -{rem:.0f}%: on green=good/on red=bad");ax[3].axis("off")
        plt.tight_layout();plt.savefig(os.path.join(OUT,f"{fire}.png"),dpi=85);plt.close()
print("미리보기 →",OUT)
