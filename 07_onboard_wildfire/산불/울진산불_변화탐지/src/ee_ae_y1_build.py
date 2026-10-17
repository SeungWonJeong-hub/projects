"""다년 시계열용 — 화재년+1(Y+1) AlphaEarth 임베딩 추출 (ko_ae와 동일 그리드/파일명).
불탄자국 지속성으로 진짜불탄 vs 일시변화 구분. → ko_ae_y1/{fire}/{tile}.npz (x1=(64,H,W)).
사용: python3 ee_ae_y1_build.py [N_FIRES]
"""
from __future__ import annotations
import os, sys, math, datetime as dt, glob
import numpy as np
from osgeo import ogr, osr
import ee; ee.Initialize()
HERE=os.path.dirname(os.path.abspath(__file__))
SHP=os.path.join(HERE,"..","..","산불데이터","korea","mtfire","MTfire.shp")
OUT=os.path.join(HERE,"..","ko_ae_y1")
AE=os.path.join(HERE,"..","ko_ae")   # 어떤 화재 있는지 매칭
UTM=32652; T=256; STRIDE=256
N=int(sys.argv[1]) if len(sys.argv)>1 else 60
MIN_HA=float(os.environ.get("MIN_HA","600")); MAX_HA=float(os.environ.get("MAX_HA","8000"))
YEARS=set("2018,2019,2020,2021".split(","))
CROP_M=2650.0
COL=ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"); BANDS=[f"A{i:02d}" for i in range(64)]
def tile(yr,ox,oy):
    img=COL.filterDate(f"{yr}-01-01",f"{yr}-12-31").mosaic().select(BANDS)
    g={"dimensions":{"width":T,"height":T},"affineTransform":{"scaleX":20,"shearX":0,"translateX":ox,"shearY":0,"scaleY":-20,"translateY":oy},"crsCode":f"EPSG:{UTM}"}
    a=ee.data.computePixels({"expression":img,"fileFormat":"NUMPY_NDARRAY","grid":g})
    return np.stack([a[b] for b in BANDS],0).astype(np.float32)
def build_one(name,gu,fdate):
    Y=fdate.year
    if not glob.glob(os.path.join(AE,name,"*.npz")): return 0  # ko_ae에 있는 화재만
    e=gu.GetEnvelope();cx=(e[0]+e[1])/2;cy=(e[2]+e[3])/2
    hx=max((e[1]-e[0])/2+500,CROP_M);hy=max((e[3]-e[2])/2+500,CROP_M)
    ute=[math.floor(cx-hx),math.floor(cy-hy),math.ceil(cx+hx),math.ceil(cy+hy)]
    W=int((ute[2]-ute[0])//20);H=int((ute[3]-ute[1])//20)
    if H<T or W<T: return 0
    od=os.path.join(OUT,name);os.makedirs(od,exist_ok=True);n=0
    ys=sorted(set(min(yy,H-T) for yy in range(0,H-T+1,STRIDE)));xs=sorted(set(min(xx,W-T) for xx in range(0,W-T+1,STRIDE)))
    for yy2 in ys:
        for xx2 in xs:
            fn=os.path.join(od,f"{name}_{yy2}_{xx2}.npz")
            if os.path.exists(fn): n+=1; continue
            ox=ute[0]+xx2*20;oy=ute[3]-yy2*20
            try: x1=tile(Y+1,ox,oy)
            except Exception as ex: print(f"[{name}]({yy2},{xx2}) 실패 {str(ex)[:60]}",flush=True); continue
            np.savez_compressed(fn,x1=x1); n+=1
    print(f"[{name}] Y+1={Y+1} 타일 {n}",flush=True); return n
def main():
    os.makedirs(OUT,exist_ok=True); ds=ogr.Open(SHP); ly=ds.GetLayer(); cand=[]
    for f in ly:
        if f.GetField("OCCU_YEAR") not in YEARS: continue
        g=f.GetGeometryRef()
        if not g: continue
        s=osr.SpatialReference();s.ImportFromEPSG(5186);s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        d=osr.SpatialReference();d.ImportFromEPSG(UTM);d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        gu=g.Clone();gu.Transform(osr.CoordinateTransformation(s,d));ha=gu.GetArea()/1e4
        if not(MIN_HA<=ha<=MAX_HA): continue
        od=f.GetField("OCCU_DATE")
        try: fdate=dt.date(int(od[:4]),int(od[4:6]),int(od[6:8]))
        except: continue
        cand.append((ha,fdate,gu.Clone()))
    cand.sort(key=lambda c:c[0],reverse=True); tot=0
    for ha,fdate,gu in cand[:N]:
        name=f"kf_{fdate.isoformat().replace('-','')}_{int(ha)}ha"
        try: tot+=build_one(name,gu,fdate)
        except Exception as ex: print(f"[{name}] 실패 {str(ex)[:80]}",flush=True)
    print(f"완성 Y+1 총 {tot}타일 → {OUT}",flush=True)
if __name__=="__main__": main()
