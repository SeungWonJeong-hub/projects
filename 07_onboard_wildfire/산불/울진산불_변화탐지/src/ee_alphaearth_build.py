"""AlphaEarth(Google Satellite Embedding, 64밴드·10m·연단위) bi-temporal CD 빌드.
우리 화재·크롭·라벨·홀드아웃 그대로 재사용. pre=화재전년 / post=화재년 연임베딩.
x=(128,256,256) float32 [pre64|post64], y=산림청라벨. → ko_ae/. computePixels로 타일직접추출(20m 정렬).
사용: python ee_alphaearth_build.py [N_FIRES]
"""
from __future__ import annotations
import os, sys, math, datetime as dt, glob
import numpy as np
from osgeo import gdal, ogr, osr
import ee
ee.Initialize()
gdal.UseExceptions()
HERE=os.path.dirname(os.path.abspath(__file__))
SHP=os.path.join(HERE,"..","..","산불데이터","korea","mtfire","MTfire.shp")
OUT=os.environ.get("KO_OUT",os.path.join(HERE,"..","ko_ae"))
UTM=32652
N_FIRES=int(sys.argv[1]) if len(sys.argv)>1 else 50
MIN_HA=float(os.environ.get("MIN_HA","600")); MAX_HA=float(os.environ.get("MAX_HA","8000"))
YEARS=set(os.environ.get("YEARS","2018,2019,2020,2021").split(","))  # pre=전년 필요, AE는 2017+
CROP_M=float(os.environ.get("CROP_M","2650")); STRIDE=256; T=256
COL=ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
BANDS=[f"A{i:02d}" for i in range(64)]

def ae_year(yr, ute, ox, oy):
    """해당 연도 임베딩을 [ox,oy] 좌상단 256x256 타일(20m,UTM)로 추출 → (64,256,256)."""
    img=COL.filterDate(f"{yr}-01-01",f"{yr}-12-31").mosaic().select(BANDS)
    grid={"dimensions":{"width":T,"height":T},
          "affineTransform":{"scaleX":20,"shearX":0,"translateX":ox,"shearY":0,"scaleY":-20,"translateY":oy},
          "crsCode":f"EPSG:{UTM}"}
    arr=ee.data.computePixels({"expression":img,"fileFormat":"NUMPY_NDARRAY","grid":grid})
    return np.stack([arr[b] for b in BANDS],0).astype(np.float32)  # (64,256,256)

def build_one(name, gu, fdate):
    Y=fdate.year
    if Y-1<2017: print(f"[{name}] pre({Y-1}) 없음",flush=True); return 0
    e=gu.GetEnvelope(); cx=(e[0]+e[1])/2; cy=(e[2]+e[3])/2
    hx=max((e[1]-e[0])/2+500,CROP_M); hy=max((e[3]-e[2])/2+500,CROP_M)
    ute=[math.floor(cx-hx),math.floor(cy-hy),math.ceil(cx+hx),math.ceil(cy+hy)]
    W=int((ute[2]-ute[0])//20); H=int((ute[3]-ute[1])//20)
    if H<T or W<T: print(f"[{name}] 크롭<{T}",flush=True); return 0
    # 라벨 래스터화 (전체 화재 그리드)
    srs=osr.SpatialReference(); srs.ImportFromEPSG(UTM)
    mem=ogr.GetDriverByName("Memory").CreateDataSource("m"); lyr=mem.CreateLayer("f",srs,ogr.wkbPolygon)
    ft=ogr.Feature(lyr.GetLayerDefn()); ft.SetGeometry(gu); lyr.CreateFeature(ft)
    rast=gdal.GetDriverByName("MEM").Create("",W,H,1,gdal.GDT_Byte); rast.SetGeoTransform((ute[0],20,0,ute[3],0,-20)); rast.SetProjection(srs.ExportToWkt())
    gdal.RasterizeLayer(rast,[1],lyr,burn_values=[1]); mask=rast.ReadAsArray().astype(np.uint8)
    od=os.path.join(OUT,name); os.makedirs(od,exist_ok=True); n=0
    ys=sorted(set(min(yy,H-T) for yy in range(0,H-T+1,STRIDE))); xs=sorted(set(min(xx,W-T) for xx in range(0,W-T+1,STRIDE)))
    for yy2 in ys:
        for xx2 in xs:
            ox=ute[0]+xx2*20; oy=ute[3]-yy2*20
            try:
                pre=ae_year(Y-1,ute,ox,oy); post=ae_year(Y,ute,ox,oy)
            except Exception as ex:
                print(f"[{name}] 타일({yy2},{xx2}) 추출실패 {str(ex)[:80]}",flush=True); continue
            yt=mask[yy2:yy2+T,xx2:xx2+T]
            valid=np.isfinite(pre).all(0)&np.isfinite(post).all(0)&(np.abs(pre).sum(0)>0)
            if valid.mean()<0.4: continue
            x=np.concatenate([pre,post],0)  # 128ch
            np.savez_compressed(os.path.join(od,f"{name}_{yy2}_{xx2}.npz"),
                x=x, y=(yt&valid).astype(np.uint8), valid=valid.astype(np.uint8)); n+=1
    if n==0:
        try: os.rmdir(od)
        except OSError: pass
    print(f"[{name}] {fdate} pre{Y-1}/post{Y} | 타일 {n} | 불탄 {mask.mean()*100:.1f}%",flush=True); return n

def main():
    os.makedirs(OUT,exist_ok=True); ds=ogr.Open(SHP); ly=ds.GetLayer(); cand=[]
    for f in ly:
        if f.GetField("OCCU_YEAR") not in YEARS: continue
        g=f.GetGeometryRef()
        if not g: continue
        s=osr.SpatialReference(); s.ImportFromEPSG(5186); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        d=osr.SpatialReference(); d.ImportFromEPSG(UTM); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        gu=g.Clone(); gu.Transform(osr.CoordinateTransformation(s,d)); ha=gu.GetArea()/1e4
        if not(MIN_HA<=ha<=MAX_HA): continue
        od=f.GetField("OCCU_DATE")
        try: fdate=dt.date(int(od[:4]),int(od[4:6]),int(od[6:8]))
        except: continue
        cand.append((ha,fdate,gu.Clone()))
    cand.sort(key=lambda c:c[0],reverse=True)
    print(f"후보 {len(cand)}, 상위 {N_FIRES} 빌드(AlphaEarth 128ch)",flush=True); tot=0
    for i,(ha,fdate,gu) in enumerate(cand[:N_FIRES]):
        name=f"kf_{fdate.isoformat().replace('-','')}_{int(ha)}ha"
        if glob.glob(os.path.join(OUT,name,"*.npz")): print(f"[{name}] skip",flush=True); continue
        try: tot+=build_one(name,gu,fdate)
        except Exception as ex: print(f"[{name}] 실패 {str(ex)[:100]}",flush=True)
        if (i+1)%5==0: print(f"  진행 {i+1}/{min(N_FIRES,len(cand))} 누적타일 {tot}",flush=True)
    print(f"완성 | 총 타일 {len(glob.glob(os.path.join(OUT,'*','*.npz')))}",flush=True)

if __name__=="__main__": main()
