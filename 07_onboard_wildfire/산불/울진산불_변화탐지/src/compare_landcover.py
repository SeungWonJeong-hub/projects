"""WorldCover 모델 vs EGIS 모델 시각비교. 같은 타일에 두 모델 적용 + GT 나란히.

패널: RGB | WorldCover-예측 | EGIS-예측 | GT(해당 데이터 라벨)
사용: DATA=ko_landcover_egis N=6 python3 compare_landcover.py
출력: ~/Desktop/landcover_compare.png
"""
import os, glob, random, numpy as np, torch
import segmentation_models_pytorch as smp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
DEV="cuda" if torch.cuda.is_available() else "cpu"
DATA=os.environ.get("DATA","ko_landcover_egis"); N=int(os.environ.get("N","6"))
CMAP=ListedColormap(["#d9d9d9","#2ca02c","#1f77b4"])  # 기타 회색, 식생 초록, 물 파랑

def rgbnir(x12): pre,post=x12[0:6],x12[6:12]; return np.concatenate([pre[[0,1,2,3]],post[[0,1,2,3]]],0)
def load(run):
    m=smp.Unet("resnet34",encoder_weights=None,in_channels=8,classes=3).to(DEV)
    m.load_state_dict(torch.load(os.path.join(PROJ,"runs",run,"model.pt"),map_location=DEV)); m.eval(); return m
@torch.no_grad()
def pred(m,x8):
    t=torch.from_numpy(x8).unsqueeze(0).to(DEV); return m(t).softmax(1)[0].argmax(0).cpu().numpy()

def main():
    mwc=load("landcover3_unet"); meg=load("landcover3_egis")
    files=sorted(glob.glob(os.path.join(PROJ,DATA,"*","*.npz"))); random.seed(3); random.shuffle(files)
    files=files[:N]
    fig,ax=plt.subplots(N,4,figsize=(14,3.3*N))
    for i,f in enumerate(files):
        d=np.load(f); x=rgbnir(d["x"].astype(np.float32)/1e4); gt=d["y_multi"]
        rgb=np.clip(np.stack([x[2],x[1],x[0]],-1)*3.2,0,1)  # B04,B03,B02 post아닌 pre
        pw=pred(mwc,x); pe=pred(meg,x)
        for j,(img,t,cm) in enumerate([(rgb,"RGB",None),(pw,"WorldCover 모델",CMAP),
                                       (pe,"EGIS 모델",CMAP),(gt,f"GT ({DATA.split('_')[-1]})",CMAP)]):
            a=ax[i,j] if N>1 else ax[j]
            if cm: a.imshow(img,cmap=cm,vmin=0,vmax=2)
            else: a.imshow(img)
            if i==0: a.set_title(t,fontsize=12)
            a.axis("off")
    plt.tight_layout()
    out=os.path.join(os.environ.get("DESKTOP_DATA_ROOT", os.path.expanduser("~/Desktop")), "landcover_compare.png"); plt.savefig(out,dpi=90,bbox_inches="tight")
    print("저장:",out,flush=True)

if __name__=="__main__": main()
