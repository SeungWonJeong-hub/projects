"""Presto 멀티템포럴 레버용 월별 S2 시계열 빌드 (AlphaEarth 격자·라벨·홀드아웃 재사용).

AlphaEarth(연단위)의 한계 대체 가설: 봄 개엽 pre→post intra-year 신호를 월별 시계열로 포착.
각 타일에 대해 화재년 T_MONTHS개월 S2 월별합성(우리 6밴드, SCL 구름마스킹) → (T,6,256,256)
+ 유효월 플래그 + y(산림청). → ko_presto_ts/. 이후 ee_presto_embed.py 가 Presto로 (128,256,256) 임베딩.

사용: MIN_HA=500 N_FIRES=3 SMOKE=1 python ee_presto_build.py   # 스모크(소수)
      MIN_HA=500 N_FIRES=60 python ee_presto_build.py           # 전체
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
OUT=os.environ.get("KO_OUT",os.path.join(HERE,"..","ko_presto_ts"))
HOLD=os.path.join(HERE,"..","data","holdout_fixed.txt")
UTM=32652
N_FIRES=int(os.environ.get("N_FIRES", sys.argv[1] if len(sys.argv)>1 else 60))
MIN_HA=float(os.environ.get("MIN_HA","500")); MAX_HA=float(os.environ.get("MAX_HA","8000"))
YEARS=set(os.environ.get("YEARS","2018,2019,2020,2021").split(","))
CROP_M=float(os.environ.get("CROP_M","2650")); STRIDE=256; T=256
T_MONTHS=int(os.environ.get("T_MONTHS","12"))
SMOKE=os.environ.get("SMOKE","0")=="1"        # 스모크: 홀드아웃 우선 소수만
S2BANDS=["B2","B3","B4","B8A","B11","B12"]     # ko_refl 순서와 동일
COL=ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

def s2_month(yr, mo, ox, oy):
    """해당 연·월 S2 월별합성(구름마스킹 median)을 타일 격자로 추출 → (6,256,256), 반사율 0..1. 없으면 None."""
    start=ee.Date.fromYMD(yr,mo,1); end=start.advance(1,"month")
    ic=COL.filterDate(start,end)
    def mask(img):
        scl=img.select("SCL")
        keep=scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)).And(scl.neq(0)).And(scl.neq(1))
        return img.updateMask(keep)
    n=ic.size().getInfo()
    if n==0: return None
    img=ic.map(mask).median().select(S2BANDS)
    grid={"dimensions":{"width":T,"height":T},
          "affineTransform":{"scaleX":20,"shearX":0,"translateX":ox,"shearY":0,"scaleY":-20,"translateY":oy},
          "crsCode":f"EPSG:{UTM}"}
    try:
        arr=ee.data.computePixels({"expression":img,"fileFormat":"NUMPY_NDARRAY","grid":grid})
    except Exception:
        return None
    return np.stack([arr[b].astype(np.float32) for b in S2BANDS],0)/10000.0  # (6,256,256)

def build_one(name, gu, fdate):
    Y=fdate.year
    e=gu.GetEnvelope(); cx=(e[0]+e[1])/2; cy=(e[2]+e[3])/2
    hx=max((e[1]-e[0])/2+500,CROP_M); hy=max((e[3]-e[2])/2+500,CROP_M)
    ute=[math.floor(cx-hx),math.floor(cy-hy),math.ceil(cx+hx),math.ceil(cy+hy)]
    W=int((ute[2]-ute[0])//20); H=int((ute[3]-ute[1])//20)
    if H<T or W<T: print(f"[{name}] 크롭<{T}",flush=True); return 0
    srs=osr.SpatialReference(); srs.ImportFromEPSG(UTM)
    mem=ogr.GetDriverByName("Memory").CreateDataSource("m"); lyr=mem.CreateLayer("f",srs,ogr.wkbPolygon)
    ft=ogr.Feature(lyr.GetLayerDefn()); ft.SetGeometry(gu); lyr.CreateFeature(ft)
    rast=gdal.GetDriverByName("MEM").Create("",W,H,1,gdal.GDT_Byte); rast.SetGeoTransform((ute[0],20,0,ute[3],0,-20)); rast.SetProjection(srs.ExportToWkt())
    gdal.RasterizeLayer(rast,[1],lyr,burn_values=[1]); mask=rast.ReadAsArray().astype(np.uint8)
    cx_ll=(ute[0]+ute[2])/2; cy_ll=(ute[1]+ute[3])/2  # 대략 latlon 은 아래서 UTM→WGS
    od=os.path.join(OUT,name); os.makedirs(od,exist_ok=True); n=0
    ys=sorted(set(min(yy,H-T) for yy in range(0,H-T+1,STRIDE))); xs=sorted(set(min(xx,W-T) for xx in range(0,W-T+1,STRIDE)))
    for yy2 in ys:
        for xx2 in xs:
            ox=ute[0]+xx2*20; oy=ute[3]-yy2*20
            ts=[]; months=[]; tvalid=[]
            for mo in range(1,T_MONTHS+1):
                a=s2_month(Y,mo,ox,oy)
                if a is None:
                    ts.append(np.zeros((6,T,T),np.float32)); tvalid.append(0)
                else:
                    ts.append(a); tvalid.append(1)
                months.append(mo)
            ts=np.stack(ts,0)  # (T_MONTHS,6,256,256)
            if sum(tvalid)<4: print(f"[{name}] 타일({yy2},{xx2}) 유효월<4",flush=True); continue
            yt=mask[yy2:yy2+T,xx2:xx2+T]
            # 픽셀 유효 = 유효월들에서 반사율>0 인 곳
            vm=(ts.sum(1)>0)  # (T_MONTHS,256,256) 월별 픽셀유효
            valid=(vm.sum(0)>=2)  # 최소 2개월 관측
            if valid.mean()<0.4:
                print(f"[{name}] 타일({yy2},{xx2}) 픽셀유효<0.4",flush=True); continue
            # UTM 타일중심 → WGS84 latlon
            tcx=ox+T*10; tcy=oy-T*10
            s=osr.SpatialReference(); s.ImportFromEPSG(UTM); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            d=osr.SpatialReference(); d.ImportFromEPSG(4326); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            lon,lat,_=osr.CoordinateTransformation(s,d).TransformPoint(tcx,tcy)
            np.savez_compressed(os.path.join(od,f"{name}_{yy2}_{xx2}.npz"),
                ts=ts.astype(np.float32), months=np.array(months,np.int16),
                tvalid=np.array(tvalid,np.uint8), y=(yt&valid).astype(np.uint8),
                valid=valid.astype(np.uint8), latlon=np.array([lat,lon],np.float32)); n+=1
            print(f"[{name}] 타일({yy2},{xx2}) 저장 유효월{sum(tvalid)}/{T_MONTHS}",flush=True)
    if n==0:
        try: os.rmdir(od)
        except OSError: pass
    print(f"[{name}] {fdate} | 타일 {n}",flush=True); return n

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
    hold=set(l.strip() for l in open(HOLD) if l.strip())
    def nm(ha,fd): return f"kf_{fd.isoformat().replace('-','')}_{int(ha)}ha"
    if SMOKE:  # 홀드아웃(봄) 1 + train 2 우선
        ho=[c for c in cand if nm(c[0],c[1]) in hold and 3<=c[1].month<=5]
        tr=[c for c in cand if nm(c[0],c[1]) not in hold]
        cand=ho[:1]+tr[:max(0,N_FIRES-1)]
    else:
        cand=cand[:N_FIRES]
    print(f"빌드 대상 {len(cand)}화재 (T_MONTHS={T_MONTHS}, SMOKE={SMOKE})",flush=True); tot=0
    for i,(ha,fdate,gu) in enumerate(cand):
        name=nm(ha,fdate)
        if glob.glob(os.path.join(OUT,name,"*.npz")): print(f"[{name}] skip",flush=True); continue
        try: tot+=build_one(name,gu,fdate)
        except Exception as ex: print(f"[{name}] 실패 {str(ex)[:100]}",flush=True)
    print(f"완성 | 총 타일 {len(glob.glob(os.path.join(OUT,'*','*.npz')))}",flush=True)

if __name__=="__main__": main()
