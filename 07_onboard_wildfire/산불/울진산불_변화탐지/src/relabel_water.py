"""물 라벨 정밀화 (#3 NDWI + #4 모폴로지) — ko_landcover2 로컬 relabel, 재fetch 없음.

정적 WC/JRC 물 경계 대신 실제 S2 NDWI로 물을 이미지-정합 → 해안선 혼합픽셀 라벨 정밀↑.
모폴로지 opening/closing으로 노이즈 정리. 식생/기타는 WorldCover 유지.
사용: python3 relabel_water.py
"""
import glob,os,numpy as np
from scipy.ndimage import binary_opening,binary_closing,generate_binary_structure
PROJ=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
st=generate_binary_structure(2,2)
n=0
for f in glob.glob(os.path.join(PROJ,"ko_landcover2","*","*.npz")):
    z=dict(np.load(f)); x=z["x"].astype(np.float32)/1e4; ym=z["y_multi"].copy()
    valid=z["valid"].astype(bool) if "valid" in z else (x[6]>0)
    grn,nir,red=x[6+1],x[6+3],x[6+2]                 # post: B03,B8A/B08,B04
    ndwi=(grn-nir)/(grn+nir+1e-6); ndvi=(nir-red)/(nir+red+1e-6)
    w=(ndwi>0.10)&valid                              # #3 NDWI 이미지-정합 물
    w=binary_opening(w,st,iterations=1)              # #4 opening: 흩뿌린 오탐 제거
    w=binary_closing(w,st,iterations=1)              # closing: 구멍 메움
    # 물 재구성: 새 물=NDWI, 옛 물인데 NDWI 아니면 식생/기타로 재분류
    ym[(ym==2)&~w]=np.where(ndvi[(ym==2)&~w]>0.35,1,0)
    ym[w]=2
    ym[~valid]=0
    z["y_multi"]=ym.astype(np.uint8)
    np.savez_compressed(f,**z); n+=1
print(f"물 라벨 정밀화(NDWI+모폴로지) 완료: {n}타일")
cnt=np.zeros(3)
for f in glob.glob(os.path.join(PROJ,"ko_landcover2","*","*.npz")):
    ym=np.load(f)["y_multi"]
    for c in range(3): cnt[c]+=(ym==c).sum()
print("클래스분포:",{["기타","식생","물"][c]:f"{cnt[c]/cnt.sum()*100:.1f}%" for c in range(3)})
