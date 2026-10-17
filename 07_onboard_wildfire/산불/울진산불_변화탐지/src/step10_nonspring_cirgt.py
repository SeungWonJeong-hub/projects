"""비봄 홀드아웃 4화재 진짜 CIR-GT F1 — STAC 타이트 브래킷(pre 화재직전·post 화재+6~28일).
ngii_targets.csv 좌표·날짜 사용. raw DN(NO_OFFSET, 배포모델·2018-21 스케일 일치). 봄 0.828과 비교.
사용: python step10_nonspring_cirgt.py
"""
import os, csv, datetime as dt, numpy as np, torch
os.environ["NO_OFFSET"]="1"   # raw DN (배포모델 스케일 일치)
from osgeo import gdal, osr
from scipy import ndimage as ndi
import segmentation_models_pytorch as smp
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
import build_cir_gt_stac as B
WEIGHTS=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/웹앱/weights"); DEV="cuda" if torch.cuda.is_available() else "cpu"
FIRES=["kf_20190826_3047ha","kf_20200608_2433ha","kf_20181004_2666ha","kf_20210220_1504ha"]
SEAS={"kf_20190826_3047ha":"여름","kf_20200608_2433ha":"여름","kf_20181004_2666ha":"가을","kf_20210220_1504ha":"겨울"}
ENS=[("unet","et02_unet_s1"),("unet","et02_unet_s2"),("unet","et02_unet_s3"),
     ("fpn","et02_fpn_s1"),("fpn","et02_fpn_s2"),("fpn","et02_fpn_s3"),
     ("deeplabv3plus","et02_deeplabv3plus_s1"),("deeplabv3plus","et02_deeplabv3plus_s2"),("deeplabv3plus","et02_deeplabv3plus_s3")]
ARCH={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}
def rows():
    r={}
    for d in csv.DictReader(open(os.path.join(PROJ,"data","ngii_targets.csv"))):
        r[d["fire_id"]]=d
    return r
def ute_from_ll(bbox):
    lon0,lat0,lon1,lat1=bbox; utm=32652 if (lon0+lon1)/2>=126 else 32651
    ll=osr.SpatialReference(); ll.ImportFromEPSG(4326); ll.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    u=osr.SpatialReference(); u.ImportFromEPSG(utm); u.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(ll,u); pts=[ct.TransformPoint(x,y) for x in (lon0,lon1) for y in (lat0,lat1)]
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ute=[int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))]; return [ute[0]-ute[0]%20,ute[1]-ute[1]%20,ute[2]-ute[2]%20,ute[3]-ute[3]%20],utm
def load_ens():
    ms=[]
    for d,n in ENS:
        m=ARCH[d]("resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV).eval()
        m.load_state_dict(torch.load(os.path.join(WEIGHTS,"et02_ens",n+".pt"),map_location=DEV)); ms.append(m)
    return ms
def bestf1(P,Y):
    b=0
    for t in np.arange(0.05,0.9,0.025):
        pr=P>=t; tp=(pr&(Y==1)).sum(); fp=(pr&(Y==0)).sum(); fn=((~pr)&(Y==1)).sum()
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); b=max(b,2*p*r/(p+r+1e-9))
    return b
@torch.no_grad()
def ens_prob(ms,x8,H,W):
    prob=np.zeros((H,W),"float32")
    for i in range(0,H,256):
        for j in range(0,W,256):
            p=x8[:,i:i+256,j:j+256]; ph,pw=p.shape[1],p.shape[2]
            pad=np.zeros((8,256,256),"float32"); pad[:,:ph,:pw]=p
            prob[i:i+ph,j:j+pw]=np.mean([torch.sigmoid(m(torch.from_numpy(pad[None]).to(DEV)))[0,0].cpu().numpy() for m in ms],0)[:ph,:pw]
    return prob
def main():
    R=rows(); ms=load_ens(); D=dt.timedelta; per={}
    print("비봄 홀드아웃 4화재 진짜 CIR-GT F1 (STAC 타이트 브래킷):",flush=True)
    for fid in FIRES:
        d=R[fid]; fd=dt.date.fromisoformat(d["occu_date"])
        bbox=(float(d["lon_min"]),float(d["lat_min"]),float(d["lon_max"]),float(d["lat_max"]))
        ute,utm=ute_from_ll(bbox)
        pre=B.stac_refl(bbox,B.rfc(fd-D(25),fd-D(2)),ute,utm); post=B.stac_refl(bbox,B.rfc(fd+D(6),fd+D(28)),ute,utm)
        if not pre or not post: print(f"  {fid} 취득실패"); continue
        H=min(pre['B02'].shape[0],post['B02'].shape[0]); W=min(pre['B02'].shape[1],post['B02'].shape[1])
        for dd in (pre,post):
            for k in ["B02","B03","B04","B8A","B11","B12"]: dd[k]=dd[k][:H,:W]
        nbr=lambda b:(b['B8A']-b['B12'])/(b['B8A']+b['B12']+1e-6); ndvi=lambda b:(b['B8A']-b['B04'])/(b['B8A']+b['B04']+1e-6)
        valid=pre['_valid'][:H,:W]&post['_valid'][:H,:W]&~np.isin(pre['SCL'][:H,:W],[0,1,3,8,9,10,11])&~np.isin(post['SCL'][:H,:W],[0,1,3,8,9,10,11])
        dnbr=nbr(pre)-nbr(post); dndvi=ndvi(pre)-ndvi(post)
        gt=(dnbr>0.15)&(dndvi>0.08)&(ndvi(post)<0.45)&valid
        gt=ndi.binary_opening(gt,iterations=1); gt=ndi.binary_closing(gt,iterations=2); gt=ndi.binary_fill_holes(gt)
        four=lambda b:[b['B02'],b['B03'],b['B04'],b['B8A']]
        x8=np.clip(np.stack(four(pre)+four(post))/1e4,0,1).astype("float32")
        prob=ens_prob(ms,x8,H,W)
        f=bestf1(prob[valid],gt[valid].astype(int)); ha=gt.sum()*400/1e4
        per.setdefault(SEAS[fid],[]).append(f)
        print(f"  {fid:22s} {SEAS[fid]} pre={pre['_date']} post={post['_date']} GT소각{ha:.0f}ha valid{100*valid.mean():.0f}% → F1={f:.3f}",flush=True)
    print("\n=== 계절별 (진짜 CIR-GT, 봄 미학습 0.828과 동일자尺) ===",flush=True)
    allf=[]
    for s in ["여름","가을","겨울"]:
        if s in per: print(f"  {s}: {np.mean(per[s]):.3f} (n{len(per[s])})",flush=True); allf+=per[s]
    if allf: print(f"\n★ 비봄 홀드아웃 평균 CIR-GT F1 = {np.mean(allf):.3f} (n{len(allf)}) vs 봄 미학습 0.828",flush=True)
if __name__=="__main__": main()
