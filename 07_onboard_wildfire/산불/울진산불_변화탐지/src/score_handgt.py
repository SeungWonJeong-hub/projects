"""손라벨 픽셀 GT 대비 정직 채점 (export_label_package.py 로 만든 labelset/ 사용).

라벨 완성(hand_gt.tif 수정) 후 실행하면 3가지를 분리 측정:
  ① polygon vs hand   — 산림청 폴리곤이 실제로 얼마나 노이즈였나(라벨 벽 크기)
  ② baseline vs hand  — 모델의 진짜 정확도(폴리곤 아닌 진짜 소각 대비)
  ③ baseline vs polygon — 지금까지 보고한 값(0.677)이 착시였는지
  + NAL vs hand — 라벨 벽 레버가 진짜 GT선 이득인지

hand_gt 가 polygon 과 동일한(=아직 수정 안 한) 타일은 자동 제외.
사용: python3 score_handgt.py
"""
import os, glob, json, numpy as np, rasterio
import torch, torch.nn as nn, torch.nn.functional as F

HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
AE=os.path.join(PROJ,"ko_ae"); LS=os.path.join(PROJ,"labelset")
MODELS={"baseline":"runs/ko_ae_baseline_ha500","NAL":"runs/ko_ae_nal_band3"}

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

def rd(p):
    with rasterio.open(p) as d: return d.read(1)
def f1(pred,gt,valid):
    p=(pred>0)&(valid>0); g=(gt>0)&(valid>0)
    tp=int((p&g).sum());fp=int((p&~g).sum());fn=int((~p&g).sum())
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);iou=tp/(tp+fp+fn+1e-9)
    return 2*pr*rc/(pr+rc+1e-9),pr,rc,iou

# 모델별 예측 로드
loaded={}
for name,rel in MODELS.items():
    ck=os.path.join(PROJ,rel,"ae_nal_unet.pt"); js=os.path.join(PROJ,rel,"results.json")
    if not os.path.exists(ck): print(f"[skip {name}] ckpt 없음 {ck}"); continue
    m=UNet(128); m.load_state_dict(torch.load(ck,map_location="cpu")); m.eval()
    thr=json.load(open(js))["thr"] if os.path.exists(js) else 0.4
    loaded[name]=(m,thr)
if not loaded: raise SystemExit("모델 없음")

dirs=sorted(glob.glob(os.path.join(LS,"*","hand_gt.tif")))
rows=[]; edited=0
for hp in dirs:
    d=os.path.dirname(hp); name=os.path.basename(d)
    hand=rd(hp); poly=rd(os.path.join(d,"polygon_gt.tif")); valid=rd(os.path.join(d,"valid.tif"))
    if np.array_equal(hand,poly):
        continue   # 아직 수정 안 함 → 제외
    edited+=1
    # ae x 로 모델 예측
    fire,tile=name.split("__"); ae=np.load(os.path.join(AE,fire,tile+".npz"))
    x=torch.from_numpy(ae["x"].astype(np.float32)).unsqueeze(0)
    row={"tile":name}
    fp,pp,pr,pi=f1(poly,hand,valid); row["poly_vs_hand_F1"]=round(fp,3)
    for mn,(m,thr) in loaded.items():
        with torch.no_grad(): pred=(torch.sigmoid(m(x)).squeeze().numpy()>=thr)
        ff,_,_,_=f1(pred,hand,valid); row[f"{mn}_vs_hand_F1"]=round(ff,3)
        if mn=="baseline":
            fpp,_,_,_=f1(pred,poly,valid); row["baseline_vs_poly_F1"]=round(fpp,3)
    rows.append(row)

if edited==0:
    print("아직 수정된 hand_gt 가 없습니다. QGIS서 labelset/*/hand_gt.tif 를 실제 소각픽셀로 고친 뒤 다시 실행하세요.")
    raise SystemExit(0)

print(f"\n=== 손라벨 {edited}타일 정직 채점 ===")
hdr=["tile","poly_vs_hand_F1","baseline_vs_hand_F1","NAL_vs_hand_F1","baseline_vs_poly_F1"]
print(" | ".join(h for h in hdr if any(h in r for r in rows)))
for r in rows: print(" | ".join(str(r.get(h,"-")) for h in hdr if any(h in rr for rr in rows)))
# 집계
def avg(k):
    v=[r[k] for r in rows if k in r]; return round(sum(v)/len(v),3) if v else None
print(f"\n[평균] 폴리곤 노이즈(poly_vs_hand)={avg('poly_vs_hand_F1')} | "
      f"baseline 진짜정확도(vs_hand)={avg('baseline_vs_hand_F1')} | "
      f"NAL(vs_hand)={avg('NAL_vs_hand_F1')} | baseline(vs_poly, 기존보고)={avg('baseline_vs_poly_F1')}")
print("\n해석: baseline_vs_hand > baseline_vs_poly 이면 → 모델은 폴리곤보다 진짜 소각을 잘 맞힌 것(폴리곤이 천장을 눌렀던 것).")
print("      NAL_vs_hand > baseline_vs_hand 이면 → 라벨 벽 레버(NAL)가 진짜 GT선 실제 이득(폴리곤 평가가 가렸던 것).")
json.dump(rows,open(os.path.join(LS,"handgt_scores.json"),"w"),ensure_ascii=False,indent=2)
