"""손으로 그린 폴리곤 → hand_gt.tif 변환 (폴리곤 라벨링 방식용).

QGIS서 각 타일 폴더의 post_swir.tif 위에 소각영역을 폴리곤으로 그려
그 폴더에 hand_label.geojson (또는 .gpkg/.shp) 로 저장하면, 이 스크립트가
타일 격자에 정확히 래스터화해 hand_gt.tif 를 덮어쓴다(같은 transform/CRS 사용).

사용: python3 rasterize_handlabel.py           # 모든 타일 폴더 스캔
      python3 rasterize_handlabel.py <타일폴더> # 특정 폴더만
"""
import os, sys, glob, json, numpy as np, rasterio
from rasterio.features import rasterize
from osgeo import ogr

HERE=os.path.dirname(os.path.abspath(__file__))
LS=os.path.join(HERE,"..","labelset")

def find_vector(d):
    for ext in ("geojson","gpkg","shp"):
        g=glob.glob(os.path.join(d,f"hand_label.{ext}"))
        if g: return g[0]
    return None

def process(d):
    vec=find_vector(d)
    if not vec:
        return None
    with rasterio.open(os.path.join(d,"post_swir.tif")) as ref:
        transform=ref.transform; H,W=ref.height,ref.width
    ds=ogr.Open(vec); lyr=ds.GetLayer()
    shapes=[]
    for feat in lyr:
        g=feat.GetGeometryRef()
        if g is not None: shapes.append((json.loads(g.ExportToJson()),1))
    if not shapes:
        print(f"[{os.path.basename(d)}] 폴리곤 0개 — skip"); return None
    mask=rasterize(shapes,out_shape=(H,W),transform=transform,fill=0,
                   default_value=1,dtype="uint8")
    out=os.path.join(d,"hand_gt.tif")
    with rasterio.open(out,"w",driver="GTiff",height=H,width=W,count=1,
                       dtype="uint8",transform=transform) as dst:
        dst.write(mask,1)
    print(f"[{os.path.basename(d)}] {vec} → hand_gt.tif ({int(mask.sum())}px 소각)")
    return int(mask.sum())

if __name__=="__main__":
    if len(sys.argv)>1:
        process(sys.argv[1])
    else:
        dirs=[os.path.dirname(p) for p in glob.glob(os.path.join(LS,"*","post_swir.tif"))]
        done=[d for d in sorted(dirs) if process(d) is not None]
        print(f"\n{len(done)}타일 변환됨. 이제 python3 score_handgt.py")
