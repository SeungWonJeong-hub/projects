"""환경부 토지피복 중분류(L2, 22종) 도엽 224개 → 20m 3클래스 모자이크.

리맵(대분류 첫자리): 식생=2xx농업/3xx산림/4xx초지 →1, 물=5xx습지/7xx수역 →2, 기타=1xx시가화/6xx나지 →0.
출력: data/landcover_egis/rast/cls_*.tif (EPSG:5186, 20m, 0/1/2, 255=nodata) + egis_cls.vrt
사용: python3 build_egis_mosaic.py
"""
import os, glob, subprocess, numpy as np, rasterio
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
BASE=os.path.join(PROJ,"data","landcover_egis")
SHP=os.path.join(BASE,"extracted","shp"); RAST=os.path.join(BASE,"rast"); os.makedirs(RAST,exist_ok=True)

def remap(code):
    d=code//100
    cls=np.full(code.shape,255,np.uint8)
    cls[np.isin(d,[2,3,4])]=1   # 식생
    cls[np.isin(d,[5,7])]=2     # 물
    cls[np.isin(d,[1,6])]=0     # 기타
    return cls

def main():
    shps=sorted(glob.glob(os.path.join(SHP,"lv2_*.shp")))
    print(f"도엽 {len(shps)}개 라스터화+리맵",flush=True)
    cls_tifs=[]; ok=0
    for i,shp in enumerate(shps):
        name=os.path.splitext(os.path.basename(shp))[0]     # lv2_357163
        layer=name; idx=name.split("_")[1]
        code_tif=os.path.join(RAST,f"code_{idx}.tif"); cls_tif=os.path.join(RAST,f"cls_{idx}.tif")
        if not os.path.exists(cls_tif):
            r=subprocess.run(["gdal_rasterize","-q","-sql",f"SELECT CAST(L2_CODE AS integer) AS c FROM {layer}",
                "-a","c","-tr","20","20","-tap","-a_nodata","255","-ot","UInt16",shp,code_tif],
                capture_output=True,text=True)
            if r.returncode!=0 or not os.path.exists(code_tif):
                print(f"  [{idx}] rasterize 실패 {r.stderr[:60]}",flush=True); continue
            with rasterio.open(code_tif) as s:
                code=s.read(1); prof=s.profile
            cls=remap(code)
            prof.update(dtype="uint8",nodata=255,count=1)
            with rasterio.open(cls_tif,"w",**prof) as d: d.write(cls,1)
            os.remove(code_tif)
        cls_tifs.append(cls_tif); ok+=1
        if (i+1)%40==0: print(f"  {i+1}/{len(shps)}",flush=True)
    # VRT 모자이크
    vrt=os.path.join(BASE,"egis_cls.vrt")
    subprocess.run(["gdalbuildvrt","-q","-srcnodata","255","-vrtnodata","255",vrt]+cls_tifs,check=True)
    print(f"완료 {ok}도엽 → {vrt}",flush=True)
    # 전체 클래스 분포
    cnt=np.zeros(3,dtype=np.int64)
    for t in cls_tifs:
        with rasterio.open(t) as s: a=s.read(1)
        for c in range(3): cnt[c]+=int((a==c).sum())
    tot=cnt.sum()
    print("클래스분포:",{["기타","식생","물"][c]:f"{cnt[c]/tot*100:.1f}%" for c in range(3)},flush=True)

if __name__=="__main__": main()
