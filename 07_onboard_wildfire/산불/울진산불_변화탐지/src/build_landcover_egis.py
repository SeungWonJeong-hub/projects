"""EGIS 3클래스 모자이크(egis_cls.vrt)를 라벨로 land-cover 타일 생성.

build_landcover_archive.py와 동일 포맷(x 12ch RGB+NIR pre/post, y_multi 0/1/2, valid)이나
라벨소스만 WorldCover(GEE) → 환경부 EGIS(항공정사 22종→3클래스). EGIS 커버 지역 S2 크롭만.
출력: ko_landcover_egis/  사용: N_TILES=400 WIN=6 python3 build_landcover_egis.py
"""
import os, glob, json, random, numpy as np, rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling, transform_bounds
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
ARC=os.path.join(os.environ.get("DESKTOP_DATA_ROOT", os.path.expanduser("~/Desktop")), "sentinel2_data")
VRT=os.path.join(PROJ,"data","landcover_egis","egis_cls.vrt")
OUT=os.path.join(PROJ,"ko_landcover_egis"); os.makedirs(OUT,exist_ok=True)
N_TILES=int(os.environ.get("N_TILES","400")); WIN=int(os.environ.get("WIN","6")); T=256
MINCOV=float(os.environ.get("MINCOV","0.85"))   # 윈도우 내 EGIS 유효라벨 최소비율
random.seed(0)
EG=rasterio.open(VRT); EG_CRS=EG.crs; EG_BOUNDS=EG.bounds

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

def overlaps_egis(scene):
    with rasterio.open(os.path.join(scene,"B02.tif")) as s:
        b=s.bounds; crs=s.crs
    x0,y0,x1,y1=transform_bounds(crs,EG_CRS,*b)
    return not (x1<EG_BOUNDS.left or x0>EG_BOUNDS.right or y1<EG_BOUNDS.bottom or y0>EG_BOUNDS.top)

def read_win(scene,off_r,off_c):
    bands=[]
    for bn in ["B02","B03","B04","B08"]:
        with rasterio.open(os.path.join(scene,f"{bn}.tif")) as s:
            w=Window(off_c,off_r,T,T); arr=s.read(1,window=w); tr=s.window_transform(w); crs=s.crs
        bands.append(arr)
    return np.stack(bands), tr, crs

def egis_label(tr,crs):
    """윈도우 그리드(T×T, crs, tr)로 EGIS 모자이크 재투영 샘플."""
    dst=np.full((T,T),255,np.uint8)
    reproject(source=rasterio.band(EG,1),destination=dst,
        src_transform=EG.transform,src_crs=EG_CRS,
        dst_transform=tr,dst_crs=crs,resampling=Resampling.nearest,
        src_nodata=255,dst_nodata=255)
    return dst

def main():
    bt=scenes_by_tile()
    # EGIS 커버 타일만
    cov={}
    for t,scs in bt.items():
        if overlaps_egis(scs[0][1]): cov[t]=scs
    print(f"전체 {len(bt)}타일 중 EGIS 커버 {len(cov)}타일",flush=True)
    tiles=list(cov.keys()); random.shuffle(tiles)
    ok=0
    for ti,tile in enumerate(tiles):
        scs=cov[tile]; pre_s=scs[0][1]; post_s=scs[-1][1]
        with rasterio.open(os.path.join(pre_s,"B02.tif")) as s: H,W=s.shape
        tries=0
        made=0
        while made<WIN and tries<WIN*8:
            tries+=1
            r=random.randint(0,H-T); c=random.randint(0,W-T)
            try:
                pre,tr,crs=read_win(pre_s,r,c); post,_,_=read_win(post_s,r,c)
                if (pre[0]>0).mean()<0.6 or (post[0]>0).mean()<0.6: continue
                lab=egis_label(tr,crs)
                if (lab!=255).mean()<MINCOV: continue     # EGIS 커버 부족→skip
            except Exception as e:
                print(f"[{tile}] 실패 {str(e)[:35]}",flush=True); continue
            valid=(pre[0]>0)&(post[0]>0)&(lab!=255)
            lab=lab.copy(); lab[~valid]=0
            zero=np.zeros((2,T,T))
            x=np.concatenate([pre,zero,post,zero],0).astype(np.int16)
            d=os.path.join(OUT,f"eg_{tile}_{ok}"); os.makedirs(d,exist_ok=True)
            np.savez_compressed(os.path.join(d,f"eg_{tile}_{ok}_0_0.npz"),
                x=x,y=np.zeros((T,T),np.uint8),y_multi=lab,valid=valid.astype(np.uint8)); ok+=1; made+=1
            if ok>=N_TILES: break
        if ok>=N_TILES: break
        if (ti+1)%10==0: print(f"  {ti+1}/{len(tiles)} 누적 {ok}",flush=True)
    print(f"완료 {ok}타일 → {OUT}",flush=True)
    cnt=np.zeros(3)
    for f in glob.glob(os.path.join(OUT,"*","*.npz")):
        ym=np.load(f)["y_multi"]
        for c in range(3): cnt[c]+=(ym==c).sum()
    tot=cnt.sum()
    print("클래스분포:",{["기타","식생","물"][c]:f"{cnt[c]/tot*100:.1f}%" for c in range(3)},flush=True)

if __name__=="__main__": main()
