"""새 GT 가 그 범위의 **전부**를 라벨했는가 — 예측·GT 불일치 구역의 소각 신호를 직접 잰다.

왜: 강릉(2019_63)에서 배포 앙상블이 GT 361ha 대비 1,316ha 를 예측했다(recall 0.995).
두 해석이 가능하다 ①모델 과대예측 ②GT 가 화재의 일부만 라벨. 둘은 정반대 처방이라
**모델 출력이 아닌 자료(S2 pre/post 반사율)로** 갈라야 한다.

세 구역의 dNBR·dNDVI 중앙값을 비교한다:
  GT소각 / 예측-GT(불일치) / 둘다아님(배경)
불일치 구역이 배경이 아니라 GT소각 쪽 신호를 보이면 **GT 가 불완전한 것**이다.

사용: SITE=gangneung GSD=20 conda run -n ml python re_gt_completeness.py [--runs "runs/cir_reh_ens/*.pt"]
"""
import os, sys, glob, json
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
SITE=os.environ.get("SITE","gangneung"); GSD=float(os.environ.get("GSD","20"))
CACHE=os.path.join(PROJ,"nifos_cache",SITE+("_doy" if os.environ.get("SCENE_PICK")=="doy" else ""))
THR=float(os.environ.get("THR","0.2"))

gt=np.load(os.path.join(CACHE,f"gt_{int(GSD)}m.npy")).astype(bool)
x=np.load(os.path.join(CACHE,f"scene_{int(GSD)}m.npz"))["x"].astype(np.float32)/1e4
H=min(gt.shape[0],x.shape[1]); W=min(gt.shape[1],x.shape[2])
gt=gt[:H,:W]; x=x[:,:H,:W]
# ORDER = [B02,B03,B04,NIR,B11,B12] x (pre,post)
pre,post=x[:6],x[6:]
def nbr(b): return (b[3]-b[5])/(b[3]+b[5]+1e-6)
def ndvi(b): return (b[3]-b[2])/(b[3]+b[2]+1e-6)
dnbr=nbr(pre)-nbr(post); dndvi=ndvi(pre)-ndvi(post)

sys.path.insert(0,HERE)
os.environ.setdefault("NIR","B8A" if GSD==20 else "B08")
runs_g=os.environ.get("RUNS","runs/cir_reh_ens/*.pt")
runs=sorted(glob.glob(os.path.join(PROJ,runs_g)))
os.environ["SITE"]=SITE; os.environ["GSD"]=str(int(GSD))
os.environ["TILE"]=os.environ.get("TILE","256")
import nifos_eval as NE
prob,_=NE.infer(np.load(os.path.join(CACHE,f"scene_{int(GSD)}m.npz"))["x"][:,:H,:W],runs)
prob=prob[:H,:W]; pred=prob>=THR

px_ha=GSD*GSD/1e4
if os.environ.get("EVAL_MASK")=="hull":
    import cv2
    ys,xs=np.nonzero(gt); hull=cv2.convexHull(np.stack([xs,ys],1).astype(np.int32))
    m=np.zeros(gt.shape,np.uint8); cv2.fillConvexPoly(m,hull,1)
    k=int(round(float(os.environ.get("HULL_ERODE_M","200"))/GSD))
    M=cv2.erode(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*k+1,2*k+1))).astype(bool)
    gt=gt&M; pred=pred&M; dnbr=np.where(M,dnbr,np.nan)
    print(f"[마스크] hull 안에서만: {M.sum()*px_ha:.0f}ha")
    zones_extra={"(마스크밖)":~M}
zones={"GT소각":gt, "예측-GT(불일치)":pred&~gt, "둘다아님(배경)":~pred&~gt, "GT-예측(미탐)":gt&~pred}
print(f"[{SITE}] GSD={GSD:.0f}m thr={THR}  격자 {H}x{W} = {H*W*px_ha:.0f}ha")
print(f"{'구역':18} {'면적ha':>8} {'비율':>6}  {'dNBR중앙':>9} {'dNDVI중앙':>10} {'post NDVI':>10}  {'확률중앙':>8}")
for k,m in zones.items():
    n=int(m.sum())
    if n==0: print(f"{k:18} {0:8.0f}"); continue
    print(f"{k:18} {n*px_ha:8.0f} {n/(H*W)*100:5.1f}%  {np.median(dnbr[m]):9.3f} {np.median(dndvi[m]):10.3f}"
          f" {np.median(ndvi(post)[m]):10.3f}  {np.median(prob[m]):8.3f}")
