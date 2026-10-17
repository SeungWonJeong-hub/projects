"""안동 풍천 2020-04-24 화재: Prithvi FT4 모델 추론 → CIR-GT와 F1.
학습과 동일 STAC/AWS 반사율(pre=2019, post=2020 화재후) 취득 → 모델 예측 →
CIR-GT(EPSG5179 10m)를 모델 격자(UTM52 20m)로 warp → 임계별 F1(best-thr 보고).
CIR-GT는 자동초벌 라벨임(정직: 양쪽 다 완벽 GT 아님)."""
import os, sys, datetime as dt, socket, glob
import numpy as np
from osgeo import gdal, ogr, osr
gdal.UseExceptions()
socket.setdefaulttimeout(90)
for k,v in {"GDAL_HTTP_TIMEOUT":"40","GDAL_HTTP_CONNECTTIMEOUT":"15","GDAL_HTTP_MAX_RETRY":"3",
            "GDAL_HTTP_RETRY_DELAY":"3","GDAL_DISABLE_READDIR_ON_OPEN":"EMPTY_DIR"}.items():
    gdal.SetConfigOption(k,v)
from pystac_client import Client
CAT=Client.open("https://earth-search.aws.element84.com/v1")
HERE=os.path.dirname(os.path.abspath(__file__))
FOLD=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/산불/산불데이터/국토정보플랫폼 데이터셋/200424_안동풍천면")
UTM=32652
BANDS=[("B02","blue"),("B03","green"),("B04","red"),("B8A","nir08"),("B11","swir16"),("B12","swir22")]
ORDER=["B02","B03","B04","B8A","B11","B12"]

def rfc(a,b): return f"{a}T00:00:00Z/{b}T23:59:59Z"

def stac_refl(bbox_ll, dtr, ute):
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
        gdal.Warp(f"/tmp/_an_{k}.tif",[f"/vsicurl/{h}" for h in hrefs],dstSRS=f"EPSG:{UTM}",xRes=20,yRes=20,
                  outputBounds=ute,resampleAlg="average",format="GTiff",srcNodata=0)
        arr=gdal.Open(f"/tmp/_an_{k}.tif").ReadAsArray().astype(np.float32)
        if k!="SCL": arr=np.clip(arr-off,0,None)
        b[k]=arr
    b["_date"]=str(d)
    return b

def main():
    # CIR extent 5179 → UTM52 bounds
    X0,Y0,X1,Y1=1093825.712093,1835110.082559,1106075.371674,1841528.080140
    s=osr.SpatialReference(); s.ImportFromEPSG(5179); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    u=osr.SpatialReference(); u.ImportFromEPSG(UTM); u.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(s,u)
    c=[ct.TransformPoint(x,y) for x in (X0,X1) for y in (Y0,Y1)]
    xs=[p[0] for p in c]; ys=[p[1] for p in c]
    ute=[int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))]
    # 20m 격자 정렬
    ute=[ute[0]-ute[0]%20, ute[1]-ute[1]%20, ute[2]-ute[2]%20, ute[3]-ute[3]%20]
    bbox_ll=[128.55,36.49,128.70,36.59]
    print("UTM52 bounds",ute,flush=True)

    fdate=dt.date(2020,4,24)
    pc=fdate.replace(year=2019)
    print("취득: pre(2019) ...",flush=True)
    pre=stac_refl(bbox_ll,rfc(pc-dt.timedelta(days=30),pc+dt.timedelta(days=45)),ute)
    print("취득: post(2020 화재후) ...",flush=True)
    post=stac_refl(bbox_ll,rfc(fdate+dt.timedelta(days=15),fdate+dt.timedelta(days=100)),ute)
    assert pre and post, "S2 취득 실패"
    print(f"  pre={pre['_date']}  post={post['_date']}",flush=True)
    H=min(pre['B02'].shape[0],post['B02'].shape[0]); W=min(pre['B02'].shape[1],post['B02'].shape[1])
    for dd in (pre,post):
        for kk in list(dd):
            if kk!="_date": dd[kk]=dd[kk][:H,:W]
    valid=(pre['B02']>0)&(pre['B8A']>0)&(pre['B12']>0)&(post['B02']>0)&(post['B8A']>0)&(post['B12']>0)
    valid&=~np.isin(pre['SCL'],[0,1,3,8,9,10,11])&~np.isin(post['SCL'],[0,1,3,8,9,10,11])
    x=np.stack([pre[k] for k in ORDER]+[post[k] for k in ORDER],0).astype(np.float32)  # (12,H,W) 반사율
    print(f"  격자 {H}x{W}, valid {100*valid.mean():.0f}%",flush=True)

    # 모델 로드
    os.environ.setdefault("KO_DATA","ko_refl"); os.environ["RUN_NAME"]="ko_prithvi_refl_ft4_v2"
    sys.path.insert(0,HERE)
    import torch, prithvi_ft as P
    model=P.PrithviSeg(unfreeze=4).to(P.DEV)
    model.load_state_dict(torch.load(os.path.join(P.RUN,"best.pt"),map_location=P.DEV)); model.eval()

    # 256 타일 슬라이딩 예측 → 스티치(평균)
    T=256
    prob=np.zeros((H,W),np.float32); cnt=np.zeros((H,W),np.float32)
    ys=sorted(set(min(yy,max(H-T,0)) for yy in range(0,max(H-T,0)+1,200))) or [0]
    xs2=sorted(set(min(xx,max(W-T,0)) for xx in range(0,max(W-T,0)+1,200))) or [0]
    xr=x/10000.0
    with torch.no_grad():
        for yy in ys:
            for xx in xs2:
                tile=xr[:, yy:yy+T, xx:xx+T]
                ph,pw=tile.shape[1],tile.shape[2]
                if ph<T or pw<T:
                    tile=np.pad(tile,((0,0),(0,T-ph),(0,T-pw)),mode="reflect")
                t=P.prep(tile[0:6],tile[6:12]).unsqueeze(0).to(P.DEV)
                pp=torch.sigmoid(model(t)).squeeze().cpu().numpy()[:ph,:pw]
                prob[yy:yy+ph,xx:xx+pw]+=pp; cnt[yy:yy+ph,xx:xx+pw]+=1
    prob/=np.maximum(cnt,1)

    # prob 저장(UTM52 20m)
    gt=(ute[0],20,0,ute[3],0,-20)
    drv=gdal.GetDriverByName("GTiff"); srs=osr.SpatialReference(); srs.ImportFromEPSG(UTM)
    pds=drv.Create(os.path.join(FOLD,"andong_model_prob.tif"),W,H,1,gdal.GDT_Float32)
    pds.SetGeoTransform(gt); pds.SetProjection(srs.ExportToWkt()); pds.GetRasterBand(1).WriteArray(prob); pds=None

    # CIR-GT(5179 10m) → UTM52 20m 격자로 warp
    gtmask=os.path.join(FOLD,"burn_mask_200424_안동풍천면.tif")
    gwarp=gdal.Warp("/tmp/_gt_utm.tif",gtmask,dstSRS=f"EPSG:{UTM}",xRes=20,yRes=20,
                    outputBounds=ute,resampleAlg="max",format="GTiff")
    gt_arr=gdal.Open("/tmp/_gt_utm.tif").ReadAsArray()[:H,:W].astype(bool)

    # F1 (valid 픽셀만)
    def prf(pred,y,m):
        pred=pred&m; y=y&m
        tp=(pred&y).sum(); fp=(pred&~y).sum(); fn=(~pred&y).sum()
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); f1=2*p*r/(p+r+1e-9)
        return f1,p,r,tp,fp,fn
    best=None
    for thr in np.arange(0.1,0.9,0.05):
        f1,p,r,tp,fp,fn=prf(prob>=thr,gt_arr,valid)
        if best is None or f1>best[0]: best=(f1,thr,p,r,tp,fp,fn)
    f1,thr,p,r,tp,fp,fn=best
    px_area=20*20/1e4
    print("\n===== 안동 풍천 2020-04-24 : Prithvi FT4 vs CIR-GT =====",flush=True)
    print(f"pre={pre['_date']} post={post['_date']}  valid {100*valid.mean():.0f}%",flush=True)
    print(f"CIR-GT 소각 {int((gt_arr&valid).sum())}px({(gt_arr&valid).sum()*px_area:.0f}ha)  "
          f"모델예측(thr={thr:.2f}) {int(((prob>=thr)&valid).sum())}px",flush=True)
    print(f"★ F1={f1:.3f}  P={p:.3f}  R={r:.3f}  (best-thr={thr:.2f})",flush=True)

    # 오버레이 저장
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def norm(a):
        lo,hi=np.percentile(a[a>0],2),np.percentile(a[a>0],98); return np.clip((a-lo)/(hi-lo+1e-6),0,1)
    rgb=np.dstack([norm(post['B12']),norm(post['B8A']),norm(post['B04'])])
    fig,ax=plt.subplots(1,3,figsize=(21,6))
    ax[0].imshow(rgb); ax[0].set_title(f"post S2 (SWIR-NIR-R) {post['_date']}")
    ax[1].imshow(rgb); ax[1].imshow(np.ma.masked_where(~gt_arr,gt_arr.astype(float)),cmap="autumn",alpha=0.4); ax[1].set_title("CIR-GT (draft)")
    ax[2].imshow(rgb); ax[2].imshow(np.ma.masked_where(~(prob>=thr),(prob>=thr).astype(float)),cmap="cool",alpha=0.4)
    ax[2].set_title(f"Prithvi FT4 pred  F1={f1:.3f}")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig("andong_model_vs_gt.png",dpi=90,bbox_inches="tight")
    print("saved viz",flush=True)

if __name__=="__main__":
    main()
