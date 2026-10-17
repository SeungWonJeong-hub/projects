"""소량 픽셀 GT 수동제작용 QGIS 라벨링 패키지 생성 (딥리서치 2026-08-24 finding#5).

홀드아웃 11타일 각각에 대해 labelset/<fire>__<tile>/ 에:
  post_swir.tif   SWIR 위색(B12/B8A/B04) — 소각=빨강/주황 (라벨링 배경)
  pre_swir.tif    화재 전 위색 (비교용)
  polygon_gt.tif  산림청 경계폴리곤 라벨 (참조)
  model_pred.tif  baseline 모델 예측 (폴리곤과 불일치=수정 후보)
  valid.tif       유효 픽셀
  hand_gt.tif     ★편집대상 — polygon 으로 초기화됨. QGIS서 이걸 수정해 진짜 소각픽셀로.
  preview.png     pre|post|polygon|pred 4분할 미리보기

모든 GeoTIFF 동일 격자(256²·20m 더미 transform)라 QGIS서 정확히 겹침.
사용: python3 export_label_package.py
"""
import os, glob, json, numpy as np, rasterio
from rasterio.transform import Affine
import torch, torch.nn as nn, torch.nn.functional as F

HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
AE=os.path.join(PROJ,"ko_ae"); RF=os.path.join(PROJ,"ko_refl")
HOLD=os.path.join(PROJ,"data","holdout_fixed.txt")
OUT=os.path.join(PROJ,"labelset")
CKPT=os.path.join(PROJ,"runs","ko_ae_baseline_ha500","ae_nal_unet.pt")
CKPT_JSON=os.path.join(PROJ,"runs","ko_ae_baseline_ha500","results.json")
TRANSFORM=Affine(20,0,0,0,-20,0)   # 20m 픽셀, 원점 0,0 (더미 — 모든 레이어 공유해 겹침)

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

CRS="EPSG:32652"   # UTM52N — 좌표는 더미지만 CRS 부여로 QGIS 폴리곤 디지타이징 정상 동작
def wtif(path,arr,count,dtype):
    with rasterio.open(path,"w",driver="GTiff",height=256,width=256,count=count,
                       dtype=dtype,transform=TRANSFORM,crs=CRS) as d:
        if count==1: d.write(arr.astype(dtype),1)
        else:
            for i in range(count): d.write(arr[...,i].astype(dtype),i+1)

def falsecolor(refl12,off):
    """SWIR 위색: R=B12(idx off+5) G=B8A(off+3) B=B04(off+2). 2-98% 스트레치 → uint8."""
    r,g,b=refl12[off+5],refl12[off+3],refl12[off+2]
    img=np.stack([r,g,b],-1)
    lo,hi=np.percentile(img,2),np.percentile(img,98)
    return np.clip((img-lo)/(hi-lo+1e-6),0,1)*255

# 모델 로드(예측용)
thr=json.load(open(CKPT_JSON))["thr"] if os.path.exists(CKPT_JSON) else 0.4
m=UNet(128); m.load_state_dict(torch.load(CKPT,map_location="cpu")); m.eval()

hold=[l.strip() for l in open(HOLD) if l.strip()]
os.makedirs(OUT,exist_ok=True); n=0
manifest=[]
for fire in hold:
    for ae_f in sorted(glob.glob(os.path.join(AE,fire,"*.npz"))):
        tile=os.path.basename(ae_f); rf_f=os.path.join(RF,fire,tile)
        if not os.path.exists(rf_f):
            print(f"[skip] {fire}/{tile} refl 없음"); continue
        ae=np.load(ae_f); refl=np.load(rf_f)["x"].astype(np.float32)/10000.0
        y=ae["y"].astype(np.uint8); valid=ae["valid"].astype(np.uint8)
        with torch.no_grad():
            prob=torch.sigmoid(m(torch.from_numpy(ae["x"].astype(np.float32)).unsqueeze(0))).squeeze().numpy()
        pred=(prob>=thr).astype(np.uint8)
        d=os.path.join(OUT,f"{fire}__{tile.replace('.npz','')}"); os.makedirs(d,exist_ok=True)
        wtif(os.path.join(d,"pre_swir.tif"),falsecolor(refl,0),3,"uint8")
        wtif(os.path.join(d,"post_swir.tif"),falsecolor(refl,6),3,"uint8")
        wtif(os.path.join(d,"polygon_gt.tif"),y,1,"uint8")
        wtif(os.path.join(d,"model_pred.tif"),pred,1,"uint8")
        wtif(os.path.join(d,"valid.tif"),valid,1,"uint8")
        wtif(os.path.join(d,"hand_gt.tif"),y.copy(),1,"uint8")   # polygon 으로 초기화 — 편집대상
        # 미리보기 PNG (pre|post|polygon|pred)
        try:
            from PIL import Image
            pre=falsecolor(refl,0).astype(np.uint8); post=falsecolor(refl,6).astype(np.uint8)
            def m3(a,c):
                o=post.copy(); o[a>0]=c; return o
            quad=np.concatenate([np.concatenate([pre,post],1),
                                 np.concatenate([m3(y,[255,255,0]),m3(pred,[255,0,0])],1)],0)
            Image.fromarray(quad).save(os.path.join(d,"preview.png"))
        except Exception as ex: print("preview 실패",ex)
        mn=int(fire.split("_")[1][4:6]); se="봄" if 3<=mn<=5 else "여름" if 6<=mn<=8 else "가을" if 9<=mn<=11 else "겨울"
        manifest.append({"dir":os.path.basename(d),"fire":fire,"tile":tile,"season":se,
                         "polygon_burn_px":int(y.sum()),"model_pred_px":int(pred.sum())})
        n+=1; print(f"[{n}] {fire}/{tile} [{se}] polygon={int(y.sum())}px pred={int(pred.sum())}px")
json.dump(manifest,open(os.path.join(OUT,"manifest.json"),"w"),ensure_ascii=False,indent=2)
print(f"\n완료: {n}타일 → {OUT}/  (thr={thr})")
print("QGIS: post_swir(배경)+hand_gt(편집) 열고, 붉은 소각영역에 맞춰 hand_gt 수정 → score_handgt.py")
