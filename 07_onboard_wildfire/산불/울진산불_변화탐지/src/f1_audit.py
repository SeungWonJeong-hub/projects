"""F1 감사: 파이프라인에서 측정 F1을 새는(낮추는) 지점 탐지.
새 레버 아님 — 기존 baseline(AlphaEarth 0.677) 그대로, 어디서 F1이 손실되는지 분해.

점검:
 1) 화재별 F1 (전역임계 vs 화재별-최적임계) → 단일 임계가 얼마나 손해보나(적응임계 여지)
 2) 화재별 P/R/면적비 → 과예측/과소·이상치 화재
 3) valid 마스크가 불탄라벨을 얼마나 깎나(라벨 손실)
 4) 이상치: F1 급락 화재(파이프라인 결함 징후, 옛 '바다타일 F1=0'류)
 5) 폴리곤 경계 vs 코어 F1 분해 → 손실이 경계에 몰리나(라벨노이즈) 코어에도 있나(진짜 실패)
"""
import os, glob, json, numpy as np, torch
from scipy.ndimage import distance_transform_edt
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
AE=os.path.join(PROJ,"ko_ae"); HOLD=os.path.join(PROJ,"data","holdout_fixed.txt")
CKPT=os.path.join(PROJ,"runs","ko_ae_baseline_ha500","ae_nal_unet.pt")
CKJS=os.path.join(PROJ,"runs","ko_ae_baseline_ha500","results.json")
DEV="cuda" if torch.cuda.is_available() else "cpu"
import torch.nn as nn, torch.nn.functional as F
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
        x=s.stem(x); c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2))
        x=s.u2(torch.cat([s.up(c3),c2],1));x=s.u1(torch.cat([s.up(x),c1],1));return s.o(x)
def prf(pred,g,valid):
    p=(pred>0)&(valid>0); gg=(g>0)&(valid>0)
    tp=int((p&gg).sum());fp=int((p&~gg).sum());fn=int((~p&gg).sum())
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return 2*pr*rc/(pr+rc+1e-9),pr,rc
THRS=np.arange(0.2,0.9,0.05)
gthr=json.load(open(CKJS))["f1"] and json.load(open(CKJS))["thr"]
m=UNet(128).to(DEV); m.load_state_dict(torch.load(CKPT,map_location=DEV)); m.eval()
hold=[l.strip() for l in open(HOLD) if l.strip()]

# 전역 최적임계 재확인(전 홀드아웃 합쳐서)
allp=[];ally=[];allv=[]
tiles=[]
for fire in hold:
    for f in sorted(glob.glob(os.path.join(AE,fire,"*.npz"))):
        d=np.load(f)
        with torch.no_grad():
            prob=torch.sigmoid(m(torch.from_numpy(d["x"].astype(np.float32)).unsqueeze(0).to(DEV))).squeeze().cpu().numpy()
        tiles.append((fire,os.path.basename(f),prob,d["y"].astype(np.uint8),d["valid"].astype(np.uint8)))
        allp.append(prob.ravel());ally.append(d["y"].ravel());allv.append(d["valid"].ravel())
P=np.concatenate(allp);Y=np.concatenate(ally);V=np.concatenate(allv)
gthr=max(THRS,key=lambda t:prf(P>=t,Y,V)[0]); gF1=prf(P>=gthr,Y,V)[0]
print(f"전역 best-thr={gthr:.2f} 전체 F1={gF1:.3f}\n")

print(f"{'화재':<22}{'계절':<5}{'F1@전역':>8}{'F1@화재최적':>11}{'손실':>7}{'P':>6}{'R':>6}{'예측/폴리곤':>10}{'valid':>7}{'경계F1':>7}{'코어F1':>7}")
rows=[]
for fire,tile,prob,y,valid in tiles:
    mo=int(fire.split("_")[1][4:6]); se="봄" if 3<=mo<=5 else "여름" if 6<=mo<=8 else "가을" if 9<=mo<=11 else "겨울"
    fg=prf(prob>=gthr,y,valid)[0]
    bt=max(THRS,key=lambda t:prf(prob>=t,y,valid)[0]); fb,pb,rb=prf(prob>=bt,y,valid)
    ratio=(prob>=gthr).sum()/max((y>0).sum(),1)
    vf=valid.mean()
    # 경계 vs 코어 분해: 폴리곤 경계 ±3px = 경계, 나머지 = 코어
    yb=y>0.5; band=np.zeros_like(y,bool)
    if yb.any() and (~yb).any():
        dist=np.where(yb,distance_transform_edt(yb),distance_transform_edt(~yb)); band=dist<=3
    core_v=valid.copy(); core_v[band]=0   # 코어만
    edge_v=valid.copy(); edge_v[~band]=0  # 경계만
    ef=prf(prob>=gthr,y,edge_v)[0]; cf=prf(prob>=gthr,y,core_v)[0]
    print(f"{fire:<22}{se:<5}{fg:>8.3f}{fb:>11.3f}{fb-fg:>7.3f}{pb:>6.2f}{rb:>6.2f}{ratio:>10.2f}{vf:>7.2f}{ef:>7.3f}{cf:>7.3f}")
    rows.append(dict(fire=fire,season=se,f1_global=round(fg,3),f1_best=round(fb,3),
                     gain=round(fb-fg,3),ratio=round(float(ratio),2),valid=round(float(vf),2),
                     edge_f1=round(ef,3),core_f1=round(cf,3)))
# 집계
def A(k): v=[r[k] for r in rows]; return round(sum(v)/len(v),3)
print(f"\n[평균] F1@전역={A('f1_global')}  F1@화재최적={A('f1_best')}  적응임계 이득={A('gain')}")
print(f"[평균] 경계F1={A('edge_f1')}  코어F1={A('core_f1')}  → 손실이 경계에 몰리면 라벨노이즈, 코어도 낮으면 진짜 미탐")
print(f"[평균] 예측/폴리곤 면적비={A('ratio')} (>1=모델이 폴리곤보다 넓게 예측)")
json.dump(rows,open(os.path.join(PROJ,"runs","f1_audit.json"),"w"),ensure_ascii=False,indent=2)
