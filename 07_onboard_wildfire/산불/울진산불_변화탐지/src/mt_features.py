"""다중시기 실험 3단계 — B200 에서 실행. 젯슨에서 잘라 온 클립으로 타일별 다중시기 채널을 만든다.

입력: ko_mt/clips/<화재>/<장면id>.tif (B02,B03,B04,B08 DN, ute 격자 20 m), ko_mt/clips/<화재>/clips.json, ko_refl/<화재>/*.npz (타일 위치는 파일 이름 _yy_xx)
출력: ko_mt/tiles/<화재>/<타일파일이름>.npz  키 mt (9,256,256) float32
채널 (모두 NDVI 기반, 화재 전 pre = 화재 120~8일 전, 화재 후 post = 8~130일 후)
  0 pre_med   화재 전 NDVI 중앙값 (구름 튀는 값에 강건한 기준 상태)
  1 pre_p90   화재 전 NDVI 90 백분위 (개엽 전 최대 녹색도)
  2 post_med  화재 후 NDVI 중앙값
  3 post_p20  화재 후 NDVI 20 백분위 (소각 뒤 지속되는 낮은 값)
  4 d_med     pre_med - post_med
  5 d_late    pre_p90 - post_p20 (지속성 낙차)
  6 n_pre/20  유효 장면 수
  7 n_post/20
  8 flag      1 = 자료 있음 (없는 화재는 전부 0 으로 채우고 flag 0)
구름과 눈: SCL 이 없어서 청색 DN 2000 초과, 0 값(자료 없음) 을 무효로 두고 백분위로 버틴다.
반사율 offset: 젯슨 보관 자료는 earth-search 가 이미 offset 을 보정한 DN 이라 빼지 않는다 (korea_s2_foundation/calibrate.py 실측, 2026-09-09). 이번 실험은 2020년 장면만 써서 영향 없음.
사용: env -u LD_LIBRARY_PATH PROJ_DATA=~/miniforge3/envs/ml/share/proj python src/mt_features.py [화재 ...]
"""
import os, sys, glob, json, re, datetime as dt, numpy as np
import rasterio
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
CLIPS=os.path.join(PROJ,"ko_mt","clips"); OUT=os.path.join(PROJ,"ko_mt","tiles"); T=256
BLUE_CLOUD=2000; NCH=9

def load_stack(fire, role):
    meta=json.load(open(os.path.join(CLIPS,fire,"clips.json")))
    arrs=[]
    for sid,m in meta.items():
        if m.get("role")!=role or m.get("status")!="완료": continue
        p=os.path.join(CLIPS,fire,sid+".tif")
        if not os.path.exists(p): continue
        with rasterio.open(p) as ds: a=ds.read().astype(np.float32)          # (4,H,W) B02,B03,B04,B08
        off=0.0   # 보관 자료는 offset 보정 완료 DN
        valid=(a>0).all(0)&(a[0]-off<BLUE_CLOUD)
        if valid.mean()<0.02: continue                                         # 모서리만 걸친 장면
        a=np.clip(a-off,0,None); ndvi=(a[3]-a[2])/(a[3]+a[2]+1e-6); ndvi[~valid]=np.nan
        arrs.append(ndvi)
    return np.stack(arrs,0) if arrs else None

def features(pre, post):
    H,W=(pre if pre is not None else post).shape[1:]
    f=np.zeros((NCH,H,W),np.float32)
    with np.errstate(all="ignore"):
        if pre is not None:
            f[0]=np.nanmedian(pre,0); f[1]=np.nanpercentile(pre,90,0); f[6]=np.isfinite(pre).sum(0)/20.0
        if post is not None:
            f[2]=np.nanmedian(post,0); f[3]=np.nanpercentile(post,20,0); f[7]=np.isfinite(post).sum(0)/20.0
        f[4]=f[0]-f[2]; f[5]=f[1]-f[3]
    f=np.nan_to_num(f,nan=0.0); f[8]=1.0
    return f

def main():
    fires=sys.argv[1:] or sorted(os.path.basename(d) for d in glob.glob(os.path.join(CLIPS,"kf_*")) if os.path.exists(os.path.join(d,"clips.json")))
    for fire in fires:
        pre=load_stack(fire,"pre"); post=load_stack(fire,"post")
        if pre is None and post is None: print(f"[{fire}] 유효 클립 없음"); continue
        f=features(pre,post); H,W=f.shape[1:]
        tiles=sorted(glob.glob(os.path.join(PROJ,"ko_refl",fire,"*.npz"))); od=os.path.join(OUT,fire); os.makedirs(od,exist_ok=True); n=0
        for tp in tiles:
            m=re.search(r"_(\d+)_(\d+)\.npz$",os.path.basename(tp)); yy,xx=int(m.group(1)),int(m.group(2))
            if yy+T>H or xx+T>W: print(f"  격자 불일치 {os.path.basename(tp)} 격자 {H}x{W}"); continue
            np.savez_compressed(os.path.join(od,os.path.basename(tp)),mt=f[:,yy:yy+T,xx:xx+T]); n+=1
        npre=0 if pre is None else pre.shape[0]; npost=0 if post is None else post.shape[0]
        print(f"[{fire}] pre {npre} post {npost} 격자 {H}x{W} 타일 {n}/{len(tiles)} | d_med 중앙 {np.median(f[4]):.3f} 최대 {f[4].max():.3f}",flush=True)
if __name__=="__main__": main()
