"""경계허용(boundary-tolerant) F1 — 부정확 폴리곤 라벨에 대한 정직한 평가.

산림청 현장조사 폴리곤은 경계가 ±20-40m 부정확(20m 픽셀서 1-2px). 엄격 픽셀 F1은
모델을 그 경계 흐림만큼 부당하게 깎는다. 경계허용 F1(도로·건물·의료 세그서 표준):
  precision = |pred ∩ dilate(label,τ)| / |pred|
  recall    = |label ∩ dilate(pred,τ)| / |label|
τ는 **미터**로 지정한다(px 아님). 20-40m가 방어가능 구간(현장폴리곤 통상오차), 60m+는 관대.
★해상도 종속 버그 수정: 이전 판은 τ를 px로 고정해 `tau*20`으로 미터를 표기했다. 그러면 10m
  데이터에서 같은 τ=2px가 40m가 아니라 20m가 되어 고해상 모델만 2배 엄격하게 채점된다.
  이제 GSD env로 미터→px를 환산하므로 20m/10m A/B가 공정하다.

★한계: 경계 노이즈만 보정. 폴리곤이 통째로 놓친 소각(under-labeling)은 별개. 여전히 폴리곤 기준.

사용: python3 eval_boundary_tolerant.py [runs/<model>]        # 기본 baseline_ha500
      GSD=10 DATA=../ko_refl_10m_b08 python3 eval_boundary_tolerant.py runs/<model>
"""
import os, sys, glob, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import binary_dilation
PROJ=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
AE=os.environ.get("DATA",os.path.join(PROJ,"ko_ae"))
GSD=float(os.environ.get("GSD","20"))          # 데이터 격자 해상도(m) — 미터→px 환산에 사용
TAUS_M=[float(t) for t in os.environ.get("TAUS_M","0,20,40,60").split(",")]  # 허용오차(미터)
RUN=sys.argv[1] if len(sys.argv)>1 else os.path.join(PROJ,"runs","ko_ae_baseline_ha500")
CKPT=os.path.join(RUN,"ae_nal_unet.pt") if os.path.exists(os.path.join(RUN,"ae_nal_unet.pt")) else os.path.join(RUN,"ae_unet.pt")
THR=json.load(open(os.path.join(RUN,"results.json")))["thr"] if os.path.exists(os.path.join(RUN,"results.json")) else 0.4
DEV="cuda" if torch.cuda.is_available() else "cpu"
def gn(c): return nn.GroupNorm(min(8,c),c)
class Blk(nn.Module):
    def __init__(s,i,o): super().__init__(); s.c=nn.Sequential(nn.Conv2d(i,o,3,padding=1),gn(o),nn.GELU(),nn.Conv2d(o,o,3,padding=1),gn(o),nn.GELU())
    def forward(s,x): return s.c(x)
class UNet(nn.Module):
    def __init__(s,ic=128):
        super().__init__(); s.stem=nn.Conv2d(ic,64,1); s.d1=Blk(64,64);s.d2=Blk(64,128);s.d3=Blk(128,256);s.p=nn.MaxPool2d(2)
        s.u2=Blk(256+128,128);s.u1=Blk(128+64,64);s.o=nn.Conv2d(64,1,1)
    def up(s,x): return F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)
    def forward(s,x):
        x=s.stem(x);c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2))
        x=s.u2(torch.cat([s.up(c3),c2],1));x=s.u1(torch.cat([s.up(x),c1],1));return s.o(x)
_INCH=json.load(open(os.path.join(RUN,"results.json"))).get("in_ch",128) if os.path.exists(os.path.join(RUN,"results.json")) else 128
m=UNet(_INCH).to(DEV);m.load_state_dict(torch.load(CKPT,map_location=DEV));m.eval()
hold=[l.strip() for l in open(os.path.join(PROJ,"data","holdout_fixed.txt")) if l.strip()]
tiles=[]
for fire in hold:
    for f in sorted(glob.glob(os.path.join(AE,fire,"*.npz"))):
        d=np.load(f)
        with torch.no_grad(): prob=torch.sigmoid(m(torch.from_numpy(d["x"].astype(np.float32)).unsqueeze(0).to(DEV))).squeeze().cpu().numpy()
        tiles.append(((prob>=THR),d["y"].astype(bool),d["valid"].astype(bool)))
def tolF1(tau_m):
    """tau_m 미터 허용. px 반경 = round(tau_m/GSD) → 해상도가 달라도 물리 허용치는 동일."""
    tau=int(round(tau_m/GSD))
    TPp=TPr=Pp=Pr=0
    for pred,y,v in tiles:
        p=pred&v; g=y&v
        gd=binary_dilation(g,iterations=tau) if tau>0 else g
        pd=binary_dilation(p,iterations=tau) if tau>0 else p
        TPp+=int((p&gd).sum()); Pp+=int(p.sum()); TPr+=int((g&pd).sum()); Pr+=int(g.sum())
    pr=TPp/(Pp+1e-9); rc=TPr/(Pr+1e-9); return 2*pr*rc/(pr+rc+1e-9),pr,rc
print(f"모델 {os.path.basename(RUN)} thr={THR} · GSD={GSD:.0f}m · 경계허용 F1(미터 고정):\n")
out={"gsd_m":GSD}
for tau_m in TAUS_M:
    f1,pr,rc=tolF1(tau_m); px=int(round(tau_m/GSD))
    out[f"tau{int(tau_m)}m"]={"f1":round(f1,4),"p":round(pr,3),"r":round(rc,3),"px":px}
    tag="← 엄격" if tau_m==0 else ("← 방어가능(현장폴리곤 통상오차)" if tau_m<=40 else "← 관대")
    print(f"  τ={tau_m:>3.0f}m ({px}px)  F1={f1:.4f}  P={pr:.3f}  R={rc:.3f}   {tag}")
json.dump(out,open(os.path.join(PROJ,"runs",f"boundary_tolerant_{int(GSD)}m.json"),"w"),ensure_ascii=False,indent=2)
print("\n※ τ는 미터 고정 → 20m/10m 모델을 같은 물리 허용치로 비교할 수 있다.")
