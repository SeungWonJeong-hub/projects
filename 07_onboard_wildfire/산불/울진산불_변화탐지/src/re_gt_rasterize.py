"""RapidEye 5m NIFoS GT(폴리곤) → 피해강도 래스터(1m, 값 1/2/3=심/중/경).

왜 1m 인가: 폴리곤은 어느 격자에도 정렬돼 있지 않은 자유형 벡터다(격자정렬 검사 5m/10m/20m
모두 <1%). 원 자료가 5m RapidEye 이므로 5m 로 굽는 것도 되지만, 1m 로 구우면 폴리곤 형상을
그대로 보존해 10m·20m 어느 격자로 면적비율 리샘플해도 라벨이 병목이 되지 않는다.
(`nifos_eval.gt_grid()` 가 GT 원본 화소 > 평가 GSD 면 경고한다 — 울진 20m 사고 방지)

사용: conda run -n geo python re_gt_rasterize.py
산출: nifos_burn_gt/re/<code>_severity_1m.tif
"""
import os, sys, glob, json
if not os.environ.get("PROJ_DATA"):
    os.environ["PROJ_DATA"]=os.path.join(sys.prefix,"share","proj")
    os.environ.setdefault("GDAL_DATA",os.path.join(sys.prefix,"share","gdal"))
from osgeo import gdal, ogr, osr
gdal.UseExceptions(); ogr.UseExceptions()

HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
NB=os.path.join(PROJ,"..","nifos_burn_gt")
SRC=os.path.expanduser(os.environ.get("RE_GT_DIR",os.path.join(NB,"ext")))
OUT=os.path.join(NB,"re"); os.makedirs(OUT,exist_ok=True)
RES=float(os.environ.get("RES","1"))
PAD=float(os.environ.get("PAD","200"))     # 폴리곤 바깥 여유(비소각 배경) — 채점 격자에 배경이 필요

for shp in sorted(glob.glob(f"{SRC}/re_labels/*.shp")):
    code=os.path.basename(shp)[:-4]
    dst=os.path.join(OUT,f"{code}_severity_{int(RES)}m.tif")
    ds=ogr.Open(shp); ly=ds.GetLayer()
    x0,x1,y0,y1=ly.GetExtent()
    x0,y0=x0-PAD,y0-PAD; x1,y1=x1+PAD,y1+PAD
    w,h=int(round((x1-x0)/RES)),int(round((y1-y0)/RES))
    if os.path.exists(dst):
        print(f"{code:10} 이미 있음 ({w}x{h})"); continue
    drv=gdal.GetDriverByName("GTiff")
    t=drv.Create(dst,w,h,1,gdal.GDT_Byte,options=["COMPRESS=DEFLATE","TILED=YES"])
    t.SetGeoTransform((x0,RES,0,y1,0,-RES))
    t.SetProjection(ly.GetSpatialRef().ExportToWkt())
    t.GetRasterBand(1).Fill(0); t.GetRasterBand(1).SetNoDataValue(0)
    # 심(1)→중(2)→경(3) 순으로 구워 겹침이 있으면 더 심한 등급이 남게 한다
    for cls in (3,2,1):
        ly.SetAttributeFilter(f"Dam_class={cls}")
        if ly.GetFeatureCount()==0: continue
        gdal.RasterizeLayer(t,[1],ly,burn_values=[cls],options=["ALL_TOUCHED=FALSE"])
    ly.SetAttributeFilter(None)
    t.FlushCache()
    a=t.GetRasterBand(1).ReadAsArray()
    ha={c:int((a==c).sum())*RES*RES/1e4 for c in (1,2,3)}
    t=None
    print(f"{code:10} {w}x{h} @{int(RES)}m  심 {ha[1]:7.1f} / 중 {ha[2]:7.1f} / 경 {ha[3]:7.1f} ha "
          f"= {sum(ha.values()):7.1f}ha  → {os.path.relpath(dst,PROJ)}")
