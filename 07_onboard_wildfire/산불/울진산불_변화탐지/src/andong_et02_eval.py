"""안동 풍천 2020-04-24: ET-02 RGB+NIR 9모델 앙상블 → CIR-GT F1.
배포 웹앱과 동일 구성(rgbnir 8ch, unet/fpn/deeplabv3+ ×3시드, sigmoid평균, thr0.2).
같은 STAC/AWS pre(2019)/post(2020) 반사율. CIR-GT(5179 10m)→UTM52 20m warp.
보고: 배포 thr0.2 F1 + best-thr F1. (CIR-GT 천장 0.887, Prithvi 0.48 대비)
"""
import os, datetime as dt, socket, numpy as np, torch
from osgeo import gdal, osr
import segmentation_models_pytorch as smp
gdal.UseExceptions(); socket.setdefaulttimeout(90)
for k,v in {"GDAL_HTTP_TIMEOUT":"40","GDAL_HTTP_CONNECTTIMEOUT":"15","GDAL_HTTP_MAX_RETRY":"3",
            "GDAL_HTTP_RETRY_DELAY":"3","GDAL_DISABLE_READDIR_ON_OPEN":"EMPTY_DIR"}.items():
    gdal.SetConfigOption(k,v)
from pystac_client import Client
CAT=Client.open("https://earth-search.aws.element84.com/v1")
FOLD=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/산불/산불데이터/국토정보플랫폼 데이터셋/200424_안동풍천면")
WEIGHTS=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/웹앱/weights")
DEV="cuda" if torch.cuda.is_available() else "cpu"
UTM=32652
BANDS=[("B02","blue"),("B03","green"),("B04","red"),("B8A","nir08"),("B11","swir16"),("B12","swir22")]
ENS=[("unet","et02_unet_s1"),("unet","et02_unet_s2"),("unet","et02_unet_s3"),
     ("fpn","et02_fpn_s1"),("fpn","et02_fpn_s2"),("fpn","et02_fpn_s3"),
     ("deeplabv3plus","et02_deeplabv3plus_s1"),("deeplabv3plus","et02_deeplabv3plus_s2"),("deeplabv3plus","et02_deeplabv3plus_s3")]
ARCH={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}
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
        gdal.Warp(f"/tmp/_ane_{k}.tif",[f"/vsicurl/{h}" for h in hrefs],dstSRS=f"EPSG:{UTM}",xRes=20,yRes=20,
                  outputBounds=ute,resampleAlg="average",format="GTiff",srcNodata=0)
        arr=gdal.Open(f"/tmp/_ane_{k}.tif").ReadAsArray().astype(np.float32)
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
    valid=(pre['B02']>0)&(pre['B8A']>0)&(post['B02']>0)&(post['B8A']>0)
    valid&=~np.isin(pre['SCL'],[0,1,3,8,9,10,11])&~np.isin(post['SCL'],[0,1,3,8,9,10,11])
    print(f"격자 {H}x{W} valid {100*valid.mean():.0f}% pre={pre['_date']} post={post['_date']}",flush=True)

    # rgbnir 8ch (반사율 0-1, 웹앱 _rgbnir과 동일)
    def four(b): return [b["B02"],b["B03"],b["B04"],b["B8A"]]
    x=np.stack([np.clip(c/1e4,0,1) for c in four(pre)+four(post)]).astype("float32")  # (8,H,W)

    # 앙상블 로드
    print("앙상블 9모델 로드...",flush=True)
    mods=[]
    for dec,name in ENS:
        m=ARCH[dec](encoder_name="resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV).eval()
        m.load_state_dict(torch.load(os.path.join(WEIGHTS,"et02_ens",name+".pt"),map_location=DEV))
        mods.append(m)

    # 256 타일 추론, sigmoid 평균
    T=256; prob=np.zeros((H,W),"float32")
    for i in range(0,H,T):
        for j in range(0,W,T):
            p=x[:,i:i+T,j:j+T]; ph,pw=p.shape[1],p.shape[2]
            pad=np.zeros((8,T,T),"float32"); pad[:,:ph,:pw]=p
            xt=torch.from_numpy(pad[None]).to(DEV)
            with torch.no_grad():
                o=np.mean([torch.sigmoid(m(xt))[0,0].cpu().numpy() for m in mods],0)
            prob[i:i+ph,j:j+pw]=o[:ph,:pw]

    # CIR-GT warp
    gtmask=os.path.join(FOLD,"burn_mask_200424_안동풍천면.tif")
    gdal.Warp("/tmp/_gte_utm.tif",gtmask,dstSRS=f"EPSG:{UTM}",xRes=20,yRes=20,outputBounds=ute,resampleAlg="max",format="GTiff")
    gt=gdal.Open("/tmp/_gte_utm.tif").ReadAsArray()[:H,:W].astype(bool)
    px=20*20/1e4
    def prf(pred,y,m):
        pred=pred&m; y=y&m
        tp=(pred&y).sum(); fp=(pred&~y).sum(); fn=(~pred&y).sum()
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); return 2*p*r/(p+r+1e-9),p,r
    print(f"\nCIR-GT 소각 {int((gt&valid).sum())}px ({(gt&valid).sum()*px:.0f}ha)",flush=True)
    print("===== 안동 : ET-02 RGB+NIR 9모델 앙상블 vs CIR-GT =====",flush=True)
    f1d,pd,rd=prf(prob>=0.2,gt,valid)
    print(f"  배포 thr=0.20 : F1={f1d:.3f} P={pd:.3f} R={rd:.3f}",flush=True)
    best=None
    for thr in np.arange(0.05,0.95,0.05):
        f1,p,r=prf(prob>=thr,gt,valid)
        if best is None or f1>best[0]: best=(f1,thr,p,r)
    print(f"  best-thr={best[1]:.2f} : F1={best[0]:.3f} P={best[2]:.3f} R={best[3]:.3f}",flush=True)
    print(f"\n  대조: CIR-GT 천장 0.887 · Prithvi FT4 0.480 · 산림청 0.000",flush=True)

if __name__=="__main__": main()
