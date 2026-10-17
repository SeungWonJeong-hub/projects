"""안동 CIR-GT의 F1 천장 측정 = 오라클 상한.
같은 S2 post 반사율서 소각지수(dNBR/dNDVI/RBR/NBRpost) 뽑아 임계값을 CIR-GT에
맞춰 최적화 → "완벽한 S2 소각탐지기 vs CIR-GT" 최대 F1. 낮으면 CIR-GT/소스불일치가
벽, 높으면 모델(Prithvi 0.48)이 한계. 오라클(테스트서 임계fit)=낙관 상한임에 유의.
"""
import os, datetime as dt, socket, numpy as np
from osgeo import gdal, osr
gdal.UseExceptions(); socket.setdefaulttimeout(90)
for k,v in {"GDAL_HTTP_TIMEOUT":"40","GDAL_HTTP_CONNECTTIMEOUT":"15","GDAL_HTTP_MAX_RETRY":"3",
            "GDAL_HTTP_RETRY_DELAY":"3","GDAL_DISABLE_READDIR_ON_OPEN":"EMPTY_DIR"}.items():
    gdal.SetConfigOption(k,v)
from pystac_client import Client
CAT=Client.open("https://earth-search.aws.element84.com/v1")
FOLD=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/산불/산불데이터/국토정보플랫폼 데이터셋/200424_안동풍천면")
UTM=32652
BANDS=[("B02","blue"),("B03","green"),("B04","red"),("B8A","nir08"),("B11","swir16"),("B12","swir22")]
ORDER=["B02","B03","B04","B8A","B11","B12"]
def rfc(a,b): return f"{a}T00:00:00Z/{b}T23:59:59Z"

def stac_refl(bbox_ll,dtr,ute):
    its=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=dtr,
        query={"eo:cloud_cover":{"lt":30}},sortby=[{"field":"properties.eo:cloud_cover","direction":"asc"}],max_items=1).items())
    if not its: return None
    d=its[0].datetime.date()
    sib=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,
        datetime=rfc(d-dt.timedelta(days=2),d+dt.timedelta(days=2)),
        query={"eo:cloud_cover":{"lt":40}},max_items=10).items()) or its
    off=1000.0 if d>=dt.date(2022,1,25) else 0.0
    b={}
    for k,a in BANDS+[("SCL","scl")]:
        hrefs=[(x.assets[a].href if a in x.assets else x.assets.get(k).href) for x in sib if (a in x.assets or k in x.assets)]
        if not hrefs: return None
        gdal.Warp(f"/tmp/_anc_{k}.tif",[f"/vsicurl/{h}" for h in hrefs],dstSRS=f"EPSG:{UTM}",xRes=20,yRes=20,
                  outputBounds=ute,resampleAlg="average",format="GTiff",srcNodata=0)
        arr=gdal.Open(f"/tmp/_anc_{k}.tif").ReadAsArray().astype(np.float32)
        if k!="SCL": arr=np.clip(arr-off,0,None)
        b[k]=arr
    b["_date"]=str(d); return b

def main():
    X0,Y0,X1,Y1=1093825.712093,1835110.082559,1106075.371674,1841528.080140
    s=osr.SpatialReference(); s.ImportFromEPSG(5179); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    u=osr.SpatialReference(); u.ImportFromEPSG(UTM); u.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(s,u)
    c=[ct.TransformPoint(x,y) for x in (X0,X1) for y in (Y0,Y1)]
    xs=[p[0] for p in c]; ys=[p[1] for p in c]
    ute=[int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))]
    ute=[ute[0]-ute[0]%20,ute[1]-ute[1]%20,ute[2]-ute[2]%20,ute[3]-ute[3]%20]
    bbox_ll=[128.55,36.49,128.70,36.59]
    fdate=dt.date(2020,4,24); pc=fdate.replace(year=2019)
    print("취득 pre(2019)/post(2020)...",flush=True)
    pre=stac_refl(bbox_ll,rfc(pc-dt.timedelta(days=30),pc+dt.timedelta(days=45)),ute)
    post=stac_refl(bbox_ll,rfc(fdate+dt.timedelta(days=15),fdate+dt.timedelta(days=100)),ute)
    assert pre and post,"S2 취득 실패"
    H=min(pre['B02'].shape[0],post['B02'].shape[0]); W=min(pre['B02'].shape[1],post['B02'].shape[1])
    for dd in (pre,post):
        for kk in list(dd):
            if kk!="_date": dd[kk]=dd[kk][:H,:W]
    valid=(pre['B02']>0)&(pre['B8A']>0)&(pre['B12']>0)&(post['B02']>0)&(post['B8A']>0)&(post['B12']>0)
    valid&=~np.isin(pre['SCL'],[0,1,3,8,9,10,11])&~np.isin(post['SCL'],[0,1,3,8,9,10,11])
    print(f"격자 {H}x{W} valid {100*valid.mean():.0f}% pre={pre['_date']} post={post['_date']}",flush=True)

    def nbr(b): return (b['B8A']-b['B12'])/(b['B8A']+b['B12']+1e-6)
    def ndvi(b): return (b['B04']*0+ (b['B8A']-b['B04'])/(b['B8A']+b['B04']+1e-6))
    dnbr=nbr(pre)-nbr(post)
    dndvi=ndvi(pre)-ndvi(post)
    rbr=dnbr/(nbr(pre)+1.001)
    nbrpost=-nbr(post)                    # post 저NBR = 소각
    ndvipost=-ndvi(post)
    sigs={"dNBR":dnbr,"dNDVI":dndvi,"RBR":rbr,"NBR_post(-)":nbrpost,"NDVI_post(-)":ndvipost}

    # CIR-GT warp
    gtmask=os.path.join(FOLD,"burn_mask_200424_안동풍천면.tif")
    gdal.Warp("/tmp/_gtc_utm.tif",gtmask,dstSRS=f"EPSG:{UTM}",xRes=20,yRes=20,outputBounds=ute,resampleAlg="max",format="GTiff")
    gt=gdal.Open("/tmp/_gtc_utm.tif").ReadAsArray()[:H,:W].astype(bool)
    px=20*20/1e4
    print(f"CIR-GT 소각 {int((gt&valid).sum())}px ({(gt&valid).sum()*px:.0f}ha)\n",flush=True)

    def f1_of(pred,y,m):
        pred=pred&m; y=y&m
        tp=(pred&y).sum(); fp=(pred&~y).sum(); fn=(~pred&y).sum()
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); return 2*p*r/(p+r+1e-9),p,r
    print("===== CIR-GT F1 천장 (오라클: 임계값 CIR-GT에 최적화) =====",flush=True)
    best_overall=None
    for name,sig in sigs.items():
        v=sig[valid]; lo,hi=np.percentile(v,1),np.percentile(v,99)
        best=None
        for thr in np.linspace(lo,hi,60):
            f1,p,r=f1_of(sig>=thr,gt,valid)
            if best is None or f1>best[0]: best=(f1,thr,p,r)
        f1,thr,p,r=best
        print(f"  {name:13s} 천장 F1={f1:.3f}  P={p:.3f} R={r:.3f} (thr={thr:.3f})",flush=True)
        if best_overall is None or f1>best_overall[0]: best_overall=(f1,name)
    print(f"\n★ CIR-GT F1 천장 = {best_overall[0]:.3f} ({best_overall[1]})",flush=True)
    print(f"  (참고: Prithvi FT4 실측 0.480, 산림청 폴리곤 0.000)",flush=True)

if __name__=="__main__": main()
