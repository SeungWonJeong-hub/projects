"""ko_refl 타일에 ESA WorldCover(실측 토지피복) 정렬 fetch → 다중클래스 라벨.

기존 ko_refl x(12ch pre/post 반사율)·y(산림청 소각)를 재사용(S2 재취득 없음).
각 타일의 UTM 윈도우를 ko_build_refl 크롭로직으로 재계산 → GEE WorldCover 20m fetch →
remap → 산림청 소각(y) 덮어씀 → y_multi(6클래스) 를 npz에 추가 저장.

클래스: 0=기타 1=산림 2=물 3=도시 4=나대지/농경 5=소각
사용: python3 build_multiclass_labels.py
"""
import os, glob, math, io, urllib.request, re, numpy as np, ee
from osgeo import ogr, osr
ee.Initialize()
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
DATA=os.path.join(PROJ,"ko_refl")
SHP=os.path.join(HERE,"..","..","산불데이터","korea","mtfire","MTfire.shp")
UTM=32652; CROP_M=2650.0; T=256
WC=ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
# WorldCover → 우리 클래스
REMAP={10:1,20:1,95:1, 80:2,90:2, 50:3, 30:4,40:4,60:4,100:4, 70:0}

def fire_ute(gu):
    e=gu.GetEnvelope(); cx=(e[0]+e[1])/2; cy=(e[2]+e[3])/2
    hx=max((e[1]-e[0])/2+500,CROP_M); hy=max((e[3]-e[2])/2+500,CROP_M)
    return [math.floor(cx-hx),math.floor(cy-hy),math.ceil(cx+hx),math.ceil(cy+hy)]

def fetch_wc(x0,y0,x1,y1):
    """UTM 윈도우 WorldCover 20m → HxW uint8 (remap 후)."""
    region=ee.Geometry.Rectangle([x0,y0,x1,y1],proj=f"EPSG:{UTM}",geodesic=False)
    url=WC.getDownloadURL({"region":region,"scale":20,"crs":f"EPSG:{UTM}","format":"NPY","bands":["Map"]})
    a=np.load(io.BytesIO(urllib.request.urlopen(url,timeout=120).read()))
    raw=a["Map"].astype(np.int16)
    out=np.zeros(raw.shape,np.uint8)
    for k,v in REMAP.items(): out[raw==k]=v
    return out

def load_fires():
    ds=ogr.Open(SHP); ly=ds.GetLayer(); d={}
    s=osr.SpatialReference();s.ImportFromEPSG(5186);s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    du=osr.SpatialReference();du.ImportFromEPSG(UTM);du.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(s,du)
    for f in ly:
        od=f.GetField("OCCU_DATE"); g=f.GetGeometryRef()
        if not od or not g: continue
        gu=g.Clone(); gu.Transform(ct); ha=int(gu.GetArea()/1e4)
        d[f"{od}_{ha}"]=gu.Clone()
    return d

def main():
    fires=load_fires()
    folders=sorted([x for x in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(x,"*.npz"))])
    ok=0; miss=0
    for fol in folders:
        b=os.path.basename(fol); m=re.match(r"kf_(\d+)_(\d+)ha",b)
        if not m: continue
        key=f"{m.group(1)}_{m.group(2)}"; gu=fires.get(key)
        if gu is None:   # ha 근사 매칭
            for k,g in fires.items():
                if k.startswith(m.group(1)+"_") and abs(int(k.split("_")[1])-int(m.group(2)))<=2: gu=g;break
        if gu is None: miss+=1; print(f"[{b}] MTfire 매칭실패",flush=True); continue
        ute=fire_ute(gu)
        for npz in glob.glob(os.path.join(fol,"*.npz")):
            z=dict(np.load(npz))
            mm=re.search(r"_(\d+)_(\d+)\.npz$",npz); yy2,xx2=int(mm.group(1)),int(mm.group(2))
            x0=ute[0]+xx2*20; y1=ute[3]-yy2*20; x1=x0+T*20; y0=y1-T*20
            try: wc=fetch_wc(x0,y0,x1,y1)
            except Exception as e: print(f"[{b}] WC실패 {str(e)[:40]}",flush=True); continue
            H,W=z["y"].shape; wc=wc[:H,:W]
            if wc.shape!=(H,W):
                pad=np.zeros((H,W),np.uint8); pad[:wc.shape[0],:wc.shape[1]]=wc; wc=pad
            ymulti=wc.copy(); ymulti[z["y"]>0]=5          # 소각 덮어씀
            if "valid" in z: ymulti[z["valid"]==0]=0
            z["y_multi"]=ymulti
            np.savez_compressed(npz,**z); ok+=1
        print(f"[{b}] OK",flush=True)
    print(f"완료: {ok}타일 y_multi 추가, {miss}화재 매칭실패",flush=True)
    # 클래스 분포
    cnt=np.zeros(6)
    for npz in glob.glob(os.path.join(DATA,"*","*.npz")):
        z=np.load(npz)
        if "y_multi" in z:
            for c in range(6): cnt[c]+=(z["y_multi"]==c).sum()
    tot=cnt.sum()
    print("클래스분포:",{["기타","산림","물","도시","나대지","소각"][c]:f"{cnt[c]/tot*100:.1f}%" for c in range(6)})

if __name__=="__main__": main()
