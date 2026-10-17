"""다지역 산불 CD 데이터 빌드 (일반화용).
STAC(earth-search)로 각 산불의 pre/post 저구름 S2 취득 -> 20m 6밴드+SCL ->
13채널 [pre6|post6|dNBR] + dNBR Otsu 유사라벨 -> 256 타일 npz.
build_dataset.py 규약(밴드순·stretch·refine_label)과 동일 정규화.
사용: mr_build.py <name>   (FIRES 에 정의)
"""
from __future__ import annotations
import os, sys, math
import numpy as np, cv2
from osgeo import gdal
from pystac_client import Client
gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "mr_data")
# name: (bbox W,S,E,N, pre 기간, post 기간, UTM EPSG)
FIRES = {
    "greece2023":    ([26.05,40.85,26.45,41.15], "2023-07-01/2023-08-10", "2023-09-01/2023-10-15", "EPSG:32635"),
    "australia2019": ([149.9,-36.6,150.3,-36.2], "2019-11-01/2019-12-10", "2020-01-20/2020-03-10", "EPSG:32756"),
    "chile2024":     ([-71.55,-33.15,-71.30,-32.95],"2024-01-01/2024-01-31","2024-02-10/2024-03-20","EPSG:32719"),
    "turkey2021":    ([31.30,36.70,31.60,36.95], "2021-07-01/2021-07-26", "2021-08-15/2021-09-30", "EPSG:32636"),
    "california2021":([-121.45,39.80,-121.10,40.10],"2021-06-01/2021-06-30","2021-10-01/2021-11-05","EPSG:32610"),
    "uljin2022":     ([129.30,36.90,129.55,37.15], "2021-04-01/2021-05-10", "2022-04-01/2022-05-10", "EPSG:32652"),  # 홀드아웃 테스트
}
BAND_ASSET = {"B02":"blue","B03":"green","B04":"red","B08":"nir","B11":"swir16","B12":"swir22","SCL":"scl"}

def pick(bbox, dt):
    cat = Client.open("https://earth-search.aws.element84.com/v1")
    it = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=dt,
              query={"eo:cloud_cover":{"lt":25}}, sortby=[{"field":"properties.eo:cloud_cover","direction":"asc"}], max_items=1).items())
    return it[0] if it else None

def warp(href, bbox, utm, out):
    gdal.Warp(out, f"/vsicurl/{href}", dstSRS=utm, xRes=20, yRes=20,
              outputBounds=bbox, outputBoundsSRS="EPSG:4326", resampleAlg="average", format="GTiff")
    return gdal.Open(out).ReadAsArray().astype(np.float32)

def load(item, bbox, utm, tag):
    b={}
    for k,a in BAND_ASSET.items():
        href = item.assets[a].href if a in item.assets else item.assets[k].href
        b[k]=warp(href, bbox, utm, f"/tmp/_mr_{tag}_{k}.tif")
    return b

def stretch(ch, valid, lo=2, hi=98):
    v=ch[valid]; a,b=np.percentile(v,[lo,hi]) if v.size else (0,1)
    return np.clip((ch-a)/max(b-a,1e-6),0,1).astype(np.float32)
def ratio(x,y): return (x-y)/(x+y+1e-6)
def idx(x): return ((np.clip(x,-1,1)+1)/2).astype(np.float32)

def stack6(bd, valid):
    return np.stack([stretch(bd["B12"],valid), stretch(bd["B08"],valid), stretch(bd["B02"],valid),
                     idx(ratio(bd["B08"],bd["B12"])), idx(ratio(bd["B08"],bd["B04"])), idx(ratio(bd["B03"],bd["B08"]))],0)

def refine(dnbr, valid, ndvi_pre):
    veg=valid&(ndvi_pre>0.2); vals=np.clip(dnbr[veg],0,1)
    if vals.size<50: return (valid&(dnbr>=0.27)).astype(np.uint8)
    u8=(vals*255).astype(np.uint8); otsu,_=cv2.threshold(u8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thr=float(np.clip(otsu/255.,0.15,0.35)); raw=(valid&(dnbr>=thr)).astype(np.uint8)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    raw=cv2.morphologyEx(raw,cv2.MORPH_OPEN,k); raw=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,k)
    n,lab,st,_=cv2.connectedComponentsWithStats(raw,8); out=np.zeros_like(raw)
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA]>=12: out[lab==i]=1     # ~0.5ha @20m
    return out

def main():
    name=sys.argv[1]; bbox,predt,postdt,utm=FIRES[name]
    pi=pick(bbox,predt); qi=pick(bbox,postdt)
    if not pi or not qi: print(f"[{name}] 장면 없음 pre={bool(pi)} post={bool(qi)}"); return
    print(f"[{name}] pre {pi.id}({pi.properties['eo:cloud_cover']:.0f}%) post {qi.id}({qi.properties['eo:cloud_cover']:.0f}%)",flush=True)
    pb=load(pi,bbox,utm,"pre"); qb=load(qi,bbox,utm,"post")
    H=min(pb["B02"].shape[0],qb["B02"].shape[0]); W=min(pb["B02"].shape[1],qb["B02"].shape[1])
    for d in (pb,qb):
        for k in d: d[k]=d[k][:H,:W]
    valid=(pb["B02"]>0)&(pb["B08"]>0)&(pb["B12"]>0)&(qb["B02"]>0)&(qb["B08"]>0)&(qb["B12"]>0)
    valid&= ~np.isin(qb["SCL"],[0,1,3,8,9,10,11]) & ~np.isin(pb["SCL"],[0,1,3,8,9,10,11])
    pre6=stack6(pb,valid); post6=stack6(qb,valid)
    dnbr=np.clip(ratio(pb["B08"],pb["B12"])-ratio(qb["B08"],qb["B12"]),-1,1)
    ndvi_pre=ratio(pb["B08"],pb["B04"])
    label=refine(dnbr,valid,ndvi_pre)
    x=np.concatenate([pre6,post6,idx(dnbr)[None]],0).astype(np.float32)   # (13,H,W)
    od=os.path.join(OUT,name); os.makedirs(od,exist_ok=True)
    T=256; n=0
    for y in range(0,H-T+1,T):
        for xx in range(0,W-T+1,T):
            v=valid[y:y+T,xx:xx+T]
            if v.mean()<0.5: continue
            np.savez_compressed(os.path.join(od,f"{name}_{y}_{xx}.npz"),
                                x=x[:,y:y+T,xx:xx+T], y=label[y:y+T,xx:xx+T].astype(np.uint8)); n+=1
    print(f"[{name}] 타일 {n} | 불탄픽셀비율 {label[valid].mean()*100:.1f}% | 저장 {od}",flush=True)

if __name__=="__main__": main()
