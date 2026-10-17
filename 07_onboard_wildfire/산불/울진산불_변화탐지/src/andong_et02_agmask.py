"""안동 ET-02 앙상블 봄 과예측을 농경지(N3A_D) 마스킹으로 잡기 → CIR-GT F1 비교.
앙상블 prob 1회 계산 후 캐시(/tmp/_et02_and.npz). 농경지 벡터를 격자에 rasterize,
pred & ~ag 로 오탐 제거. 마스크 없음 vs 농경지마스킹(원본/침식) F1 비교 + 재현율 손실.
"""
import os, datetime as dt, socket, numpy as np, torch
from osgeo import gdal, osr
from scipy.ndimage import binary_erosion, binary_dilation
import segmentation_models_pytorch as smp
gdal.UseExceptions(); socket.setdefaulttimeout(90)
for k,v in {"GDAL_HTTP_TIMEOUT":"40","GDAL_HTTP_CONNECTTIMEOUT":"15","GDAL_HTTP_MAX_RETRY":"3",
            "GDAL_HTTP_RETRY_DELAY":"3","GDAL_DISABLE_READDIR_ON_OPEN":"EMPTY_DIR"}.items():
    gdal.SetConfigOption(k,v)
FOLD=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/산불/산불데이터/국토정보플랫폼 데이터셋/200424_안동풍천면")
AG=os.path.join(FOLD,"(농경지레이어)N3A_D0010000.shp")
WEIGHTS=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/웹앱/weights")
DEV="cuda" if torch.cuda.is_available() else "cpu"; UTM=32652
CACHE="/tmp/_et02_and.npz"
BANDS=[("B02","blue"),("B03","green"),("B04","red"),("B8A","nir08"),("B11","swir16"),("B12","swir22")]
ENS=[("unet","et02_unet_s1"),("unet","et02_unet_s2"),("unet","et02_unet_s3"),
     ("fpn","et02_fpn_s1"),("fpn","et02_fpn_s2"),("fpn","et02_fpn_s3"),
     ("deeplabv3plus","et02_deeplabv3plus_s1"),("deeplabv3plus","et02_deeplabv3plus_s2"),("deeplabv3plus","et02_deeplabv3plus_s3")]
ARCH={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}
def rfc(a,b): return f"{a}T00:00:00Z/{b}T23:59:59Z"

def get_ute():
    X0,Y0,X1,Y1=1093825.712093,1835110.082559,1106075.371674,1841528.080140
    s=osr.SpatialReference(); s.ImportFromEPSG(5179); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    u=osr.SpatialReference(); u.ImportFromEPSG(UTM); u.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(s,u); c=[ct.TransformPoint(x,y) for x in (X0,X1) for y in (Y0,Y1)]
    xs=[p[0] for p in c]; ys=[p[1] for p in c]
    ute=[int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))]
    return [ute[0]-ute[0]%20,ute[1]-ute[1]%20,ute[2]-ute[2]%20,ute[3]-ute[3]%20]

def compute_prob(ute):
    from pystac_client import Client
    CAT=Client.open("https://earth-search.aws.element84.com/v1")
    def stac_refl(bbox_ll,dtr):
        its=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=dtr,query={"eo:cloud_cover":{"lt":30}},
            sortby=[{"field":"properties.eo:cloud_cover","direction":"asc"}],max_items=1).items())
        if not its: return None
        d=its[0].datetime.date()
        sib=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox_ll,datetime=rfc(d-dt.timedelta(days=2),d+dt.timedelta(days=2)),
            query={"eo:cloud_cover":{"lt":40}},max_items=10).items()) or its
        off=1000.0 if d>=dt.date(2022,1,25) else 0.0; b={}
        for k,a in BANDS+[("SCL","scl")]:
            hrefs=[(x.assets[a].href if a in x.assets else x.assets.get(k).href) for x in sib if (a in x.assets or k in x.assets)]
            if not hrefs: return None
            gdal.Warp(f"/tmp/_am_{k}.tif",[f"/vsicurl/{h}" for h in hrefs],dstSRS=f"EPSG:{UTM}",xRes=20,yRes=20,
                      outputBounds=ute,resampleAlg="average",format="GTiff",srcNodata=0)
            arr=gdal.Open(f"/tmp/_am_{k}.tif").ReadAsArray().astype(np.float32)
            if k!="SCL": arr=np.clip(arr-off,0,None)
            b[k]=arr
        b["_date"]=str(d); return b
    bbox_ll=[128.55,36.49,128.70,36.59]; fdate=dt.date(2020,4,24); pc=fdate.replace(year=2019)
    print("취득 pre/post...",flush=True)
    pre=stac_refl(bbox_ll,rfc(pc-dt.timedelta(days=30),pc+dt.timedelta(days=45)))
    post=stac_refl(bbox_ll,rfc(fdate+dt.timedelta(days=15),fdate+dt.timedelta(days=100)))
    assert pre and post
    H=min(pre['B02'].shape[0],post['B02'].shape[0]); W=min(pre['B02'].shape[1],post['B02'].shape[1])
    for dd in (pre,post):
        for kk in list(dd):
            if kk!="_date": dd[kk]=dd[kk][:H,:W]
    valid=(pre['B02']>0)&(pre['B8A']>0)&(post['B02']>0)&(post['B8A']>0)
    valid&=~np.isin(pre['SCL'],[0,1,3,8,9,10,11])&~np.isin(post['SCL'],[0,1,3,8,9,10,11])
    def four(b): return [b["B02"],b["B03"],b["B04"],b["B8A"]]
    x=np.stack([np.clip(c/1e4,0,1) for c in four(pre)+four(post)]).astype("float32")
    print("앙상블 9모델 추론...",flush=True)
    mods=[]
    for dec,name in ENS:
        m=ARCH[dec](encoder_name="resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV).eval()
        m.load_state_dict(torch.load(os.path.join(WEIGHTS,"et02_ens",name+".pt"),map_location=DEV)); mods.append(m)
    T=256; prob=np.zeros((H,W),"float32")
    for i in range(0,H,T):
        for j in range(0,W,T):
            p=x[:,i:i+T,j:j+T]; ph,pw=p.shape[1],p.shape[2]
            pad=np.zeros((8,T,T),"float32"); pad[:,:ph,:pw]=p
            xt=torch.from_numpy(pad[None]).to(DEV)
            with torch.no_grad(): o=np.mean([torch.sigmoid(m(xt))[0,0].cpu().numpy() for m in mods],0)
            prob[i:i+ph,j:j+pw]=o[:ph,:pw]
    # CIR-GT
    gdal.Warp("/tmp/_gtm.tif",os.path.join(FOLD,"burn_mask_200424_안동풍천면.tif"),dstSRS=f"EPSG:{UTM}",
              xRes=20,yRes=20,outputBounds=ute,resampleAlg="max",format="GTiff")
    gt=gdal.Open("/tmp/_gtm.tif").ReadAsArray()[:H,:W].astype(bool)
    np.savez_compressed(CACHE,prob=prob,valid=valid,gt=gt,H=H,W=W)
    return prob,valid,gt,H,W

def rasterize_ag(ute,H,W):
    # GDAL 3.4 Rasterize는 벡터 재투영 안함 → ogr2ogr로 UTM52 재투영 먼저
    utm_shp="/tmp/_ag_utm.shp"
    gdal.VectorTranslate(utm_shp,AG,dstSRS=f"EPSG:{UTM}",srcSRS="EPSG:5179",reproject=True)
    tif="/tmp/_ag.tif"
    gdal.Rasterize(tif,utm_shp,format="GTiff",outputSRS=f"EPSG:{UTM}",outputBounds=ute,
                   xRes=20,yRes=20,burnValues=[1],allTouched=True,outputType=gdal.GDT_Byte,noData=0)
    return gdal.Open(tif).ReadAsArray()[:H,:W].astype(bool)

def prf(pred,y,m):
    pred=pred&m; y=y&m; tp=(pred&y).sum(); fp=(pred&~y).sum(); fn=(~pred&y).sum()
    p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); return 2*p*r/(p+r+1e-9),p,r
def bestf1(prob,gt,valid,keep):
    b=None
    for thr in np.arange(0.05,0.95,0.05):
        f1,p,r=prf((prob>=thr)&keep,gt,valid)
        if b is None or f1>b[0]: b=(f1,thr,p,r)
    return b

def main():
    ute=get_ute()
    if os.path.exists(CACHE):
        d=np.load(CACHE); prob,valid,gt=d["prob"],d["valid"].astype(bool),d["gt"].astype(bool); H,W=int(d["H"]),int(d["W"])
        print("prob 캐시 로드",flush=True)
    else:
        prob,valid,gt,H,W=compute_prob(ute)
    ag=rasterize_ag(ute,H,W)
    ones=np.ones((H,W),bool)
    px=20*20/1e4
    print(f"\n격자 {H}x{W} | CIR-GT 소각 {int((gt&valid).sum())}px({(gt&valid).sum()*px:.0f}ha) | 농경지 {100*(ag&valid).mean():.0f}% of AOI",flush=True)
    print(f"CIR-GT 소각 중 농경지에 걸친 비율 = {100*(gt&ag&valid).sum()/max((gt&valid).sum(),1):.1f}% (마스킹 재현율 손실 상한)",flush=True)
    ag_er=binary_erosion(ag,iterations=1)   # 경계 1px 침식(소각 경계 살리기)
    variants=[("마스크없음",ones),("농경지마스킹",~ag),("농경지(1px침식)마스킹",~ag_er)]
    print("\n===== 배포 thr=0.20 =====",flush=True)
    for nm,keep in variants:
        f1,p,r=prf((prob>=0.2)&keep,gt,valid); print(f"  {nm:20s} F1={f1:.3f} P={p:.3f} R={r:.3f}",flush=True)
    print("===== best-thr =====",flush=True)
    for nm,keep in variants:
        f1,thr,p,r=bestf1(prob,gt,valid,keep); print(f"  {nm:20s} F1={f1:.3f} P={p:.3f} R={r:.3f} (thr={thr:.2f})",flush=True)
    print(f"\n  대조: CIR-GT 천장 0.887 · 마스크없음 best 0.555 · Prithvi 0.480",flush=True)

if __name__=="__main__": main()
