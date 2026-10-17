"""RapidEye GT(1m 등급) → 각 사이트 평가격자(20m) 등급 래스터. 심각도 전이 검증용.

왜 20m 격자로 굽는가: `sev_train.py` 가 안동에서 쓰는 형식(`andong_severity_20m.tif`)과 같아야
학습(안동)→시험(새 사이트)이 같은 자로 이뤄진다. 셀 대표값은 최빈등급(mode).

사용: SITE=gangneung SCENE_PICK=doy conda run -n geo python re_gt_sev20m.py
"""
import os, sys
if not os.environ.get("PROJ_DATA"):
    os.environ["PROJ_DATA"]=os.path.join(sys.prefix,"share","proj")
    os.environ.setdefault("GDAL_DATA",os.path.join(sys.prefix,"share","gdal"))
import numpy as np
from osgeo import gdal; gdal.UseExceptions()

HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
SITE=os.environ["SITE"]; GSD=20.0
SUF="_doy" if os.environ.get("SCENE_PICK")=="doy" else ""
CACHE=os.path.join(PROJ,"nifos_cache",SITE+SUF)
sys.path.insert(0,HERE)
os.environ.setdefault("GSD","20")
import nifos_eval as NE
src=NE.SITES[SITE][0]
ute=np.load(os.path.join(CACHE,"ute_20m.npy"))
out=os.path.join(CACHE,"severity_20m.tif")
gdal.Warp(out,src,xRes=GSD,yRes=GSD,outputBounds=list(ute),resampleAlg="mode",
          format="GTiff",creationOptions=["COMPRESS=DEFLATE"])
# mode 는 nodata(0)를 무시하므로 라벨 화소가 **하나라도** 닿은 셀에 등급이 붙어 면적이 부푼다
# (강릉 벡터 361ha → mode 498ha). 소각 여부는 `gt_20m.npy`(셀 내 소각면적비 ≥0.5)가 진실이므로
# 그 마스크로 깎아 안동 파일(0.3m 2,020ha → 20m 2,010ha, 부풀지 않음)과 같은 규칙으로 맞춘다.
d=gdal.Open(out,gdal.GA_Update); a=d.ReadAsArray()
gt=np.load(os.path.join(CACHE,"gt_20m.npy"))
h=min(a.shape[0],gt.shape[0]); w=min(a.shape[1],gt.shape[1])
a=a[:h,:w].copy(); a[gt[:h,:w]==0]=0
d.GetRasterBand(1).WriteArray(np.pad(a,((0,d.RasterYSize-h),(0,d.RasterXSize-w))))
d.FlushCache(); d=None
px=GSD*GSD/1e4
print(f"[{SITE}] {a.shape}  심 {(a==1).sum()*px:7.1f} / 중 {(a==2).sum()*px:7.1f} / "
      f"경 {(a==3).sum()*px:7.1f} ha  (등급화소 {int(np.isin(a,[1,2,3]).sum()):,}) → {os.path.relpath(out,PROJ)}")
