"""CEMS(유럽 실측 DEL) → 먼 대륙 실측 테스트 타일.
그들의 S2L2A(post, 12밴드 [B01..B12]) + DEL(실측 소실마스크) + 제 STAC pre → 13채널[pre6|post6|dNBR].
라벨 = DEL ∩ (dNBR>thr)  (학습(MTBS)과 동일 정의로 하모나이즈).  20m UTM 그리드.
사용: cems_build.py  (BURN_MIN 이상 AOI 자동)
"""
from __future__ import annotations
import os, glob, json, math, datetime as dt
import numpy as np, cv2, rasterio
from osgeo import gdal
from pystac_client import Client
gdal.UseExceptions()

HERE=os.path.dirname(os.path.abspath(__file__))
TEST=os.path.join(HERE,"..","cems","test_ex","test")
OUT=os.path.join(HERE,"..","cems_data_raw")
DNBR_THR=0.15; BURN_MIN=12
CAT=Client.open("https://earth-search.aws.element84.com/v1")
# 12밴드 순서 [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12] → 0-based
iB02,iB03,iB04,iB08,iB11,iB12=1,2,3,7,10,11

def utm_epsg(lo,la): return (32600 if la>=0 else 32700)+int((lo+180)//6)+1
def stretch(ch,v,lo=2,hi=98):
    x=ch[v]; a,b=np.percentile(x,[lo,hi]) if x.size else (0,1); return np.clip((ch-a)/max(b-a,1e-6),0,1).astype(np.float32)
def ratio(x,y): return (x-y)/(x+y+1e-6)
def idx(x): return ((np.clip(x,-1,1)+1)/2).astype(np.float32)
def stack6(b12,valid):
    B02,B03,B04,B08,B11,B12=b12[iB02],b12[iB03],b12[iB04],b12[iB08],b12[iB11],b12[iB12]
    return np.stack([stretch(B12,valid),stretch(B08,valid),stretch(B02,valid),
                     idx(ratio(B08,B12)),idx(ratio(B08,B04)),idx(ratio(B03,B08))],0), B08,B12

def warp_to(src, ute, epsg, out, band=None):
    opt=dict(dstSRS=f"EPSG:{epsg}",xRes=20,yRes=20,outputBounds=ute,resampleAlg="near" if band=="mask" else "average",format="GTiff")
    gdal.Warp(out, src, **opt); return gdal.Open(out).ReadAsArray().astype(np.float32)

def stac_pre(bbox_ll, firedate, ute, epsg):
    pre_r=f"{firedate-dt.timedelta(days=110)}/{firedate-dt.timedelta(days=25)}"
    its=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=pre_r,
             query={"eo:cloud_cover":{"lt":25}},sortby=[{"field":"properties.eo:cloud_cover","direction":"asc"}],max_items=1).items())
    if not its: return None
    it=its[0]; d=it.datetime.date()
    sib=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=f"{d-dt.timedelta(days=2)}/{d+dt.timedelta(days=2)}",query={"eo:cloud_cover":{"lt":35}},max_items=10).items()) or [it]
    b={}
    for k,a in [("B02","blue"),("B03","green"),("B04","red"),("B08","nir"),("B11","swir16"),("B12","swir22"),("SCL","scl")]:
        hrefs=[(x.assets[a].href if a in x.assets else x.assets[k].href) for x in sib]
        gdal.Warp(f"/tmp/_cp_{k}.tif",[f"/vsicurl/{h}" for h in hrefs],dstSRS=f"EPSG:{epsg}",xRes=20,yRes=20,outputBounds=ute,resampleAlg="average",format="GTiff",srcNodata=0)
        b[k]=gdal.Open(f"/tmp/_cp_{k}.tif").ReadAsArray().astype(np.float32)
    return b

def main():
    os.makedirs(OUT,exist_ok=True); done=0
    for aoi in sorted(glob.glob(os.path.join(TEST,"EMSR*","AOI*","*_01"))):
        name=os.path.basename(aoi)
        dl=glob.glob(os.path.join(aoi,"*_DEL.tif")); s2=glob.glob(os.path.join(aoi,"*_S2L2A.tif")); js=glob.glob(os.path.join(aoi,"*_S2L2A.json"))
        if not(dl and s2 and js): continue
        with rasterio.open(dl[0]) as r:
            if (r.read(1)>0).mean()*100 < BURN_MIN: continue
            c=r.bounds
        lo,la=(c.left+c.right)/2,(c.bottom+c.top)/2; epsg=utm_epsg(lo,la)
        firedate=dt.date.fromisoformat(json.load(open(js[0]))["payload"]["input"]["data"][0]["dataFilter"]["timeRange"]["from"][:10])
        # UTM 20m 그리드(그들 bbox)
        src=rasterio.open(s2[0]); import pyproj
        t=pyproj.Transformer.from_crs("EPSG:4326",f"EPSG:{epsg}",always_xy=True)
        x1,y1=t.transform(c.left,c.bottom); x2,y2=t.transform(c.right,c.top)
        ute=[min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)]
        try:
            post12=warp_to(s2[0],ute,epsg,"/tmp/_cpost.tif")
            mask=(warp_to(dl[0],ute,epsg,"/tmp/_cdel.tif",band="mask")>0).astype(np.uint8)
            pre=stac_pre([c.left-0.02,c.bottom-0.02,c.right+0.02,c.top+0.02],firedate,ute,epsg)
            if pre is None: print(f"[{name}] pre 없음",flush=True); continue
            H=min(post12.shape[1],pre["B02"].shape[0],mask.shape[0]); W=min(post12.shape[2],pre["B02"].shape[1],mask.shape[1])
            post12=post12[:,:H,:W]; mask=mask[:H,:W]
            for k in pre: pre[k]=pre[k][:H,:W]
            valid=(post12[iB02]>0)&(post12[iB08]>0)&(post12[iB12]>0)&(pre["B02"]>0)&(pre["B08"]>0)&(pre["B12"]>0)
            valid&=~np.isin(pre["SCL"],[0,1,3,8,9,10,11])
            pre6=stack6(np.stack([pre["B02"],pre["B03"],pre["B04"],pre["B08"],pre["B11"],pre["B12"]])[[0,0,1,2,0,0,3,0,0,0,4,5]] if False else None, valid) if False else None
            # pre6 는 12밴드 포맷이 아니라 6밴드 직접 구성
            def s6(B02,B03,B04,B08,B11,B12,v):
                return np.stack([stretch(B12,v),stretch(B08,v),stretch(B02,v),idx(ratio(B08,B12)),idx(ratio(B08,B04)),idx(ratio(B03,B08))],0)
            pre6=s6(pre["B02"],pre["B03"],pre["B04"],pre["B08"],pre["B11"],pre["B12"],valid)
            post6,post_B08,post_B12=stack6(post12,valid)
            dnbr=np.clip(ratio(pre["B08"],pre["B12"])-ratio(post_B08,post_B12),-1,1)
            x=np.concatenate([pre6,post6,idx(dnbr)[None]],0).astype(np.float32)
            y=(mask>0).astype(np.uint8)   # 순수 DEL 실측(dNBR 독립) — 정직한 먼대륙 테스트
            od=os.path.join(OUT,name); os.makedirs(od,exist_ok=True); T=256; n=0
            for yy in range(0,H-T+1,T):
                for xx in range(0,W-T+1,T):
                    v=valid[yy:yy+T,xx:xx+T]
                    if v.mean()<0.5: continue
                    np.savez_compressed(os.path.join(od,f"{name}_{yy}_{xx}.npz"),x=x[:,yy:yy+T,xx:xx+T],y=(y[yy:yy+T,xx:xx+T]&v).astype(np.uint8)); n+=1
            print(f"[{name}] pre {firedate}이전 | 타일 {n} | 하모나이즈 불탄 {y[valid].mean()*100:.1f}%",flush=True); done+=1
        except Exception as ex:
            print(f"[{name}] 실패 {ex}",flush=True)
    print(f"\n완성 화재 {done}개 | 총 타일 {len(glob.glob(os.path.join(OUT,'*','*.npz')))}",flush=True)

if __name__=="__main__": main()
