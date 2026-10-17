"""S2 아카이브(~/Desktop/sentinel2_data, RGB+NIR)로 대량 다양성 land-cover 타일 생성.

전국 MGRS 타일에서 저구름 2시점(pre/post) 크롭 + WorldCover(GEE) 정렬라벨.
다중클래스 모델(RGB+NIR)이 산림/물/도시/나대지를 제대로 배우게. 소각=0(비화재).
x=(12,256,256) int16 [B02,B03,B04,B08,0,0]×pre/post (SWIR=0, rgbnir가 앞4밴드 사용),
y_multi=6클래스(0기타1산림2물3도시4나대지), valid.
사용: N_TILES=50 WIN=3 python3 build_landcover_archive.py
"""
import os, glob, json, io, urllib.request, random, numpy as np, ee, rasterio
from rasterio.windows import Window
ee.Initialize()
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
ARC=os.path.join(os.environ.get("DESKTOP_DATA_ROOT", os.path.expanduser("~/Desktop")), "sentinel2_data")
OUT=os.path.join(PROJ,"ko_landcover2"); os.makedirs(OUT,exist_ok=True)
N_TILES=int(os.environ.get("N_TILES","50")); WIN=int(os.environ.get("WIN","3")); T=256
WC=ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
GSW=ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")   # 0-100 수체 발생빈도(정확)
# 3클래스 이미지 GEE서 합성: 1식생(WC tree/shrub/grass/crop) 2물(GSW>20 OR WC바다) 0기타. 물 우선.
_veg=WC.eq(10).Or(WC.eq(20)).Or(WC.eq(30)).Or(WC.eq(40)).Or(WC.eq(95))
_water=GSW.gt(20).unmask(0).Or(WC.eq(80)).Or(WC.eq(90))
CLS=ee.Image(0).where(_veg,1).where(_water,2).rename("cls").toUint8()
random.seed(0)

def scenes_by_tile():
    d={}
    for s in glob.glob(os.path.join(ARC,"S2*_L2A")):
        b=os.path.basename(s); parts=b.split("_")
        if len(parts)<4: continue
        tile=parts[1]; date=parts[2]
        try: cloud=json.load(open(os.path.join(s,"metadata.json"))).get("cloud",99)
        except: cloud=99
        if cloud>15 or not os.path.exists(os.path.join(s,"B08.tif")): continue
        d.setdefault(tile,[]).append((date,s))
    return {t:sorted(v) for t,v in d.items() if len(v)>=2}

def fetch_wc(bounds,crs):
    x0,y0,x1,y1=bounds
    region=ee.Geometry.Rectangle([x0,y0,x1,y1],proj=str(crs),geodesic=False)
    url=CLS.getDownloadURL({"region":region,"scale":20,"crs":str(crs),"format":"NPY","bands":["cls"]})
    a=np.load(io.BytesIO(urllib.request.urlopen(url,timeout=120).read()))
    return a["cls"].astype(np.uint8)   # 이미 0/1/2 remap됨

def read_win(scene,off_r,off_c):
    bands=[]
    for bn in ["B02","B03","B04","B08"]:
        with rasterio.open(os.path.join(scene,f"{bn}.tif")) as s:
            w=Window(off_c,off_r,T,T); arr=s.read(1,window=w); tr=s.window_transform(w); crs=s.crs
        bands.append(arr)
    return np.stack(bands), tr, crs   # (4,T,T)

def main():
    bt=scenes_by_tile()
    tiles=random.sample(list(bt.keys()),min(N_TILES,len(bt)))
    print(f"타일 {len(tiles)}개 선택 (저구름 2시점+ 있는 것)",flush=True)
    ok=0
    for ti,tile in enumerate(tiles):
        scs=bt[tile]; pre_s=scs[0][1]; post_s=scs[-1][1]  # 최초·최후 날짜
        # 타일 크기
        with rasterio.open(os.path.join(pre_s,"B02.tif")) as s: H,W=s.shape
        for w in range(WIN):
            r=random.randint(0,H-T); c=random.randint(0,W-T)
            try:
                pre,tr,crs=read_win(pre_s,r,c); post,_,_=read_win(post_s,r,c)
                if (pre[0]>0).mean()<0.6 or (post[0]>0).mean()<0.6: continue  # 결측많으면 skip
                x0=tr.c; y1=tr.f; x1=x0+T*20; y0=y1-T*20
                wc=fetch_wc([x0,y0,x1,y1],crs)[:T,:T]
                if wc.shape!=(T,T):
                    p=np.zeros((T,T),np.uint8);p[:wc.shape[0],:wc.shape[1]]=wc;wc=p
            except Exception as e: print(f"[{tile} w{w}] 실패 {str(e)[:35]}",flush=True); continue
            valid=(pre[0]>0)&(post[0]>0); wc[~valid]=0
            zero=np.zeros((2,T,T))
            x=np.concatenate([pre,zero,post,zero],0).astype(np.int16)  # 12ch, SWIR=0
            d=os.path.join(OUT,f"lc2_{tile}_{ok}"); os.makedirs(d,exist_ok=True)
            np.savez_compressed(os.path.join(d,f"lc2_{tile}_{ok}_0_0.npz"),
                x=x,y=np.zeros((T,T),np.uint8),y_multi=wc,valid=valid.astype(np.uint8)); ok+=1
        if (ti+1)%10==0: print(f"  {ti+1}/{len(tiles)} 누적타일 {ok}",flush=True)
    print(f"완료 {ok}타일 → {OUT}",flush=True)
    cnt=np.zeros(3)
    for f in glob.glob(os.path.join(OUT,"*","*.npz")):
        ym=np.load(f)["y_multi"]
        for c in range(3): cnt[c]+=(ym==c).sum()
    tot=cnt.sum()
    print("클래스분포:",{["기타","식생","물"][c]:f"{cnt[c]/tot*100:.1f}%" for c in range(3)})

if __name__=="__main__": main()
