"""다양성 land cover 타일 (비-화재) — 물/도시/농경을 모델이 배우게. GEE.

화재타일은 내륙산림 편중(물0.3%·도시0.5%)이라 해안/도시/농경 타일 추가 필요.
한국 다양 지점서 S2 RGB+NIR 2시점(pre/post, 변화없음) + WorldCover → ko_landcover/.
x=(12,256,256) int16 (화재타일과 동일포맷), y_multi=6클래스(소각=0), valid.
사용: python3 ko_build_landcover.py
"""
import os, io, urllib.request, numpy as np, ee
ee.Initialize()
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
OUT=os.path.join(PROJ,"ko_landcover"); os.makedirs(OUT,exist_ok=True)
BANDS=["B2","B3","B4","B8A","B11","B12"]
WC=ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
REMAP={10:1,20:1,95:1, 80:2,90:2, 50:3, 30:4,40:4,60:4,100:4, 70:0}
# 다양 지점 (해안·도시·농경·혼합)
PTS=[("부산해안",129.05,35.10),("인천해안",126.55,37.45),("목포해안",126.35,34.79),
 ("강릉해안",128.90,37.75),("포항해안",129.36,36.05),("여수해안",127.75,34.74),
 ("서울도심",126.99,37.55),("대구도심",128.60,35.87),("대전도심",127.40,36.35),
 ("광주도심",126.85,35.16),("수원도심",127.03,37.28),("울산공단",129.36,35.52),
 ("호남평야",126.85,35.82),("김제농경",126.88,35.80),("나주농경",126.72,35.03),
 ("당진농경",126.63,36.89),("예산농경",126.85,36.68),("논산농경",127.10,36.20),
 ("안동혼합",128.73,36.57),("춘천혼합",127.73,37.88),("원주혼합",127.92,37.34),
 ("청주혼합",127.49,36.64),("진주혼합",128.11,35.19),("제주해안",126.53,33.51)]
SCL_BAD=[0,1,3,8,9,10,11]

def clear(geom,d0,d1):
    col=ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(d0,d1)\
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",25)).sort("CLOUDY_PIXEL_PERCENTAGE")
    im=ee.Image(col.first()); good=im.select("SCL").remap(SCL_BAD,[0]*len(SCL_BAD),1)
    return im.select(BANDS).updateMask(good)

def npy(img,region,bands):
    url=img.getDownloadURL({"region":region,"scale":20,"crs":"EPSG:4326","format":"NPY","bands":bands})
    a=np.load(io.BytesIO(urllib.request.urlopen(url,timeout=120).read()))
    return a

def pad(a,T=256):
    a=a[:T,:T]
    if a.shape[0]<T or a.shape[1]<T:
        p=[(0,T-a.shape[0]),(0,T-a.shape[1])]+[(0,0)]*(a.ndim-2); a=np.pad(a,p)
    return a
ok=0
for name,lon,lat in PTS:
    if os.path.exists(os.path.join(OUT,f"lc_{name}",f"lc_{name}_0_0.npz")):  # 이어하기
        ok+=1; continue
    pt=ee.Geometry.Point([lon,lat]); region=pt.buffer(2560).bounds()
    try:
        pre=clear(region,"2020-01-01","2020-12-31"); post=clear(region,"2021-01-01","2021-12-31")
        pa=pad(np.stack([npy(pre,region,BANDS)[b] for b in BANDS],-1).astype(np.float32))
        pb=pad(np.stack([npy(post,region,BANDS)[b] for b in BANDS],-1).astype(np.float32))
        wc=pad(npy(WC.rename("Map"),region,["Map"])["Map"].astype(np.int16))
    except Exception as e: print(f"[{name}] 실패 {str(e)[:40]}",flush=True); continue
    ym=np.zeros(wc.shape,np.uint8)
    for k,v in REMAP.items(): ym[wc==k]=v
    valid=(pa[...,0]>0)&(pb[...,0]>0); ym[~valid]=0
    x=np.concatenate([pa,pb],-1).transpose(2,0,1).astype(np.int16)  # (12,256,256)
    d=os.path.join(OUT,f"lc_{name}"); os.makedirs(d,exist_ok=True)
    np.savez_compressed(os.path.join(d,f"lc_{name}_0_0.npz"),x=x,y=np.zeros(ym.shape,np.uint8),
        y_multi=ym,valid=valid.astype(np.uint8)); ok+=1
    print(f"[{name}] OK 물{(ym==2).mean()*100:.0f}% 도시{(ym==3).mean()*100:.0f}% 산림{(ym==1).mean()*100:.0f}% 나대지{(ym==4).mean()*100:.0f}%",flush=True)
print(f"완료 {ok}/{len(PTS)} 다양성 타일",flush=True)
