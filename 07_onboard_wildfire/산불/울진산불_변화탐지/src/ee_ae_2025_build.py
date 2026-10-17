"""2025 초대형 산불(의성·산청) AlphaEarth 임베딩 추출 — pre=2024/post=2025.
라벨 없음(2025 GT 부재) → x=(128,256,256) [pre64|post64]+valid만 저장. 라벨은 추후 world model 생성.
FIRMS 2025 열점으로 실제 화재영역에 집중(tight bbox). → ko_ae_2025/.
사용: python ee_ae_2025_build.py
"""
from __future__ import annotations
import os, math, json
import numpy as np
from osgeo import osr
import ee
ee.Initialize()
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("KO_OUT",os.path.join(HERE,"..","ko_ae_2025"))
FIRMS=os.path.join(HERE,"..","..","산불데이터","firms_korea_bbox_2025.geojson")
UTM=32652; T=256; STRIDE=256
COL=ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
BANDS=[f"A{i:02d}" for i in range(64)]
# 대략 bbox[W,S,E,N] (FIRMS로 tight화됨)
FIRES=[dict(name="kf2025_uiseong_48000ha", rough=[128.46,36.23,128.95,36.55]),
       dict(name="kf2025_sancheong_hadong", rough=[127.68,35.10,128.00,35.38])]

def firms_extent(rough):
    d=json.load(open(FIRMS,encoding="utf-8-sig")); feats=d["features"] if isinstance(d,dict) else d
    lo,la,hi,na=rough; xs=[];ys=[]
    for f in feats:
        g=f["geometry"]
        if g["type"]!="Point": continue
        x,y=g["coordinates"][:2]; dtc=str(f["properties"].get("acq_date","")).replace("/","-")
        if not("2025-03-01"<=dtc<="2025-04-15"): continue
        if lo<=x<=hi and la<=y<=na: xs.append(x);ys.append(y)
    if not xs: return None
    return [min(xs),min(ys),max(xs),max(ys)],len(xs)

def ae_tile(yr,ox,oy):
    img=COL.filterDate(f"{yr}-01-01",f"{yr}-12-31").mosaic().select(BANDS)
    grid={"dimensions":{"width":T,"height":T},
          "affineTransform":{"scaleX":20,"shearX":0,"translateX":ox,"shearY":0,"scaleY":-20,"translateY":oy},
          "crsCode":f"EPSG:{UTM}"}
    arr=ee.data.computePixels({"expression":img,"fileFormat":"NUMPY_NDARRAY","grid":grid})
    return np.stack([arr[b] for b in BANDS],0).astype(np.float32)

def build_one(fire):
    name=fire["name"]; ext=firms_extent(fire["rough"])
    if not ext: print(f"[{name}] FIRMS없음",flush=True); return 0
    (lo,la,hi,na),nfp=ext
    s=osr.SpatialReference(); s.ImportFromEPSG(4326); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    d=osr.SpatialReference(); d.ImportFromEPSG(UTM); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(s,d); xs=[];ys=[]
    for x,y in [(lo,la),(hi,la),(hi,na),(lo,na)]:
        px,py,_=ct.TransformPoint(x,y); xs.append(px);ys.append(py)
    b=2000  # 2km 버퍼
    ute=[math.floor(min(xs))-b,math.floor(min(ys))-b,math.ceil(max(xs))+b,math.ceil(max(ys))+b]
    W=int((ute[2]-ute[0])//20); H=int((ute[3]-ute[1])//20)
    od=os.path.join(OUT,name); os.makedirs(od,exist_ok=True); n=0
    yy_s=sorted(set(min(yy,H-T) for yy in range(0,max(H-T,0)+1,STRIDE)))
    xx_s=sorted(set(min(xx,W-T) for xx in range(0,max(W-T,0)+1,STRIDE)))
    print(f"[{name}] FIRMS {nfp}점 | 그리드 {W}x{H} = {len(yy_s)}x{len(xx_s)}타일 추출...",flush=True)
    for yy2 in yy_s:
        for xx2 in xx_s:
            ox=ute[0]+xx2*20; oy=ute[3]-yy2*20
            try:
                pre=ae_tile(2024,ox,oy); post=ae_tile(2025,ox,oy)
            except Exception as ex:
                print(f"  타일({yy2},{xx2}) 실패 {str(ex)[:70]}",flush=True); continue
            valid=(np.abs(pre).sum(0)>0)&(np.abs(post).sum(0)>0)&np.isfinite(pre).all(0)&np.isfinite(post).all(0)
            if valid.mean()<0.4: continue
            np.savez_compressed(os.path.join(od,f"{name}_{yy2}_{xx2}.npz"),
                x=np.concatenate([pre,post],0), valid=valid.astype(np.uint8),
                ute=np.array([ox,oy]))  # y는 추후 생성
            n+=1
    print(f"[{name}] 타일 {n} 저장(라벨無, 추후생성)",flush=True); return n

def main():
    os.makedirs(OUT,exist_ok=True); tot=0
    for f in FIRES:
        try: tot+=build_one(f)
        except Exception as ex: print(f"[{f['name']}] 실패 {str(ex)[:100]}",flush=True)
    print(f"완성 | 총 타일 {tot} → {OUT}",flush=True)

if __name__=="__main__": main()
