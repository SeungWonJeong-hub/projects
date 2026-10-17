"""안동 CIR-GT 봄 파인튜닝 개념검증 (공간 홀드아웃).
안동 씬을 좌/우 분할 → 한쪽 CIR-GT로 ET-02 앙상블 파인튜닝 → 반대쪽(미학습)서 평가.
순환 방지. 단 같은 화재/날짜라 cross-fire 일반화 증명 아님(더 많은 CIR-GT 씬 필요).
캐시 /tmp/_ftand.npz (x 8ch, gt, valid). 배포 앙상블 baseline vs 파인튜닝 후 비교.
"""
import os, datetime as dt, socket, random, numpy as np, torch, torch.nn as nn
from osgeo import gdal, osr
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
gdal.UseExceptions(); socket.setdefaulttimeout(90)
for k,v in {"GDAL_HTTP_TIMEOUT":"40","GDAL_HTTP_CONNECTTIMEOUT":"15","GDAL_HTTP_MAX_RETRY":"3",
            "GDAL_HTTP_RETRY_DELAY":"3","GDAL_DISABLE_READDIR_ON_OPEN":"EMPTY_DIR"}.items():
    gdal.SetConfigOption(k,v)
FOLD=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/산불/산불데이터/국토정보플랫폼 데이터셋/200424_안동풍천면")
WEIGHTS=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/웹앱/weights")
DEV="cuda" if torch.cuda.is_available() else "cpu"; UTM=32652; CACHE="/tmp/_ftand.npz"
SEED=0; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
BANDS=[("B02","blue"),("B03","green"),("B04","red"),("B8A","nir08"),("B11","swir16"),("B12","swir22")]
ENS=[("unet","et02_unet_s1"),("unet","et02_unet_s2"),("unet","et02_unet_s3"),
     ("fpn","et02_fpn_s1"),("fpn","et02_fpn_s2"),("fpn","et02_fpn_s3"),
     ("deeplabv3plus","et02_deeplabv3plus_s1"),("deeplabv3plus","et02_deeplabv3plus_s2"),("deeplabv3plus","et02_deeplabv3plus_s3")]
ARCH={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}
def rfc(a,b): return f"{a}T00:00:00Z/{b}T23:59:59Z"

def fetch():
    if os.path.exists(CACHE):
        d=np.load(CACHE); print("캐시 로드",flush=True); return d["x"],d["gt"].astype(bool),d["valid"].astype(bool)
    from pystac_client import Client
    CAT=Client.open("https://earth-search.aws.element84.com/v1")
    def stac(bbox,dtr,ute):
        its=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox,datetime=dtr,query={"eo:cloud_cover":{"lt":30}},
            sortby=[{"field":"properties.eo:cloud_cover","direction":"asc"}],max_items=1).items())
        if not its: return None
        dd=its[0].datetime.date()
        sib=list(CAT.search(collections=["sentinel-2-l2a"],bbox=bbox,datetime=rfc(dd-dt.timedelta(days=2),dd+dt.timedelta(days=2)),
            query={"eo:cloud_cover":{"lt":40}},max_items=10).items()) or its
        off=1000.0 if dd>=dt.date(2022,1,25) else 0.0; b={}
        for k,a in BANDS+[("SCL","scl")]:
            hrefs=[(x.assets[a].href if a in x.assets else x.assets.get(k).href) for x in sib if (a in x.assets or k in x.assets)]
            gdal.Warp(f"/tmp/_ft_{k}.tif",[f"/vsicurl/{h}" for h in hrefs],dstSRS=f"EPSG:{UTM}",xRes=20,yRes=20,
                      outputBounds=ute,resampleAlg="average",format="GTiff",srcNodata=0)
            arr=gdal.Open(f"/tmp/_ft_{k}.tif").ReadAsArray().astype(np.float32)
            if k!="SCL": arr=np.clip(arr-off,0,None)
            b[k]=arr
        return b
    X0,Y0,X1,Y1=1093825.712093,1835110.082559,1106075.371674,1841528.080140
    s=osr.SpatialReference(); s.ImportFromEPSG(5179); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    u=osr.SpatialReference(); u.ImportFromEPSG(UTM); u.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(s,u); c=[ct.TransformPoint(x,y) for x in (X0,X1) for y in (Y0,Y1)]
    xs=[p[0] for p in c]; ys=[p[1] for p in c]; ute=[int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))]
    ute=[ute[0]-ute[0]%20,ute[1]-ute[1]%20,ute[2]-ute[2]%20,ute[3]-ute[3]%20]
    bbox=[128.55,36.49,128.70,36.59]; fdate=dt.date(2020,4,24); pc=fdate.replace(year=2019)
    print("취득...",flush=True)
    pre=stac(bbox,rfc(pc-dt.timedelta(days=30),pc+dt.timedelta(days=45)),ute)
    post=stac(bbox,rfc(fdate+dt.timedelta(days=15),fdate+dt.timedelta(days=100)),ute)
    H=min(pre['B02'].shape[0],post['B02'].shape[0]); W=min(pre['B02'].shape[1],post['B02'].shape[1])
    for dd2 in (pre,post):
        for kk in list(dd2): dd2[kk]=dd2[kk][:H,:W]
    valid=(pre['B02']>0)&(pre['B8A']>0)&(post['B02']>0)&(post['B8A']>0)
    valid&=~np.isin(pre['SCL'],[0,1,3,8,9,10,11])&~np.isin(post['SCL'],[0,1,3,8,9,10,11])
    four=lambda b:[b["B02"],b["B03"],b["B04"],b["B8A"]]
    x=np.stack([np.clip(c/1e4,0,1) for c in four(pre)+four(post)]).astype("float32")
    gdal.Warp("/tmp/_ftgt.tif",os.path.join(FOLD,"burn_mask_200424_안동풍천면.tif"),dstSRS=f"EPSG:{UTM}",
              xRes=20,yRes=20,outputBounds=ute,resampleAlg="max",format="GTiff")
    gt=gdal.Open("/tmp/_ftgt.tif").ReadAsArray()[:H,:W].astype(bool)
    np.savez_compressed(CACHE,x=x,gt=gt,valid=valid)
    return x,gt,valid

def prf(pred,y,m):
    pred=pred&m; y=y&m; tp=(pred&y).sum(); fp=(pred&~y).sum(); fn=(~pred&y).sum()
    p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); return 2*p*r/(p+r+1e-9),p,r
def bestf1(prob,gt,valid,region):
    b=None
    for thr in np.arange(0.05,0.95,0.05):
        f1,p,r=prf((prob>=thr)&region,gt,valid)
        if b is None or f1>b[0]: b=(f1,thr,p,r)
    return b

def ens_prob(mods,x,H,W):
    T=256; prob=np.zeros((H,W),"float32")
    for i in range(0,H,T):
        for j in range(0,W,T):
            p=x[:,i:i+T,j:j+T]; ph,pw=p.shape[1],p.shape[2]
            pad=np.zeros((8,T,T),"float32"); pad[:,:ph,:pw]=p
            xt=torch.from_numpy(pad[None]).to(DEV)
            with torch.no_grad(): o=np.mean([torch.sigmoid(m(xt))[0,0].cpu().numpy() for m in mods],0)
            prob[i:i+ph,j:j+pw]=o[:ph,:pw]
    return prob

def crop_tiles(x,gt,valid,cols,T=128,stride=48):
    """cols=(c0,c1) 훈련영역서 128타일 추출."""
    c0,c1=cols; H=x.shape[1]; tiles=[]
    ys=sorted(set(min(yy,H-T) for yy in range(0,max(H-T,0)+1,stride))) or [0]
    xs=sorted(set(min(xx,c1-T) for xx in range(c0,max(c1-T,c0)+1,stride))) or [c0]
    for yy in ys:
        for xx in xs:
            xt=x[:,yy:yy+T,xx:xx+T]; yt=gt[yy:yy+T,xx:xx+T]; vt=valid[yy:yy+T,xx:xx+T]
            if xt.shape[1]<T or xt.shape[2]<T: continue
            if vt.mean()<0.4: continue
            tiles.append((xt.copy(),yt.copy(),vt.copy()))
    return tiles

def finetune_one(dec,name,tiles,epochs=25):
    m=ARCH[dec](encoder_name="resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV)
    m.load_state_dict(torch.load(os.path.join(WEIGHTS,"et02_ens",name+".pt"),map_location=DEV)); m.train()
    opt=torch.optim.AdamW(m.parameters(),lr=5e-5,weight_decay=1e-4)
    bce=nn.BCEWithLogitsLoss(); dice=DiceLoss(mode="binary"); BS=8
    for ep in range(epochs):
        random.shuffle(tiles)
        xs_,ys_=[],[]
        for xt,yt,vt in tiles:
            k=random.randint(0,3); fl=random.random()<0.5
            xa=np.rot90(xt,k,axes=(1,2)); ya=np.rot90(yt,k)
            if fl: xa=xa[:,:,::-1]; ya=ya[:,::-1]
            xs_.append(xa.copy()); ys_.append(ya.astype("float32").copy())
        for i in range(0,len(xs_),BS):
            xb=torch.from_numpy(np.stack(xs_[i:i+BS])).to(DEV)
            yb=torch.from_numpy(np.stack(ys_[i:i+BS]))[:,None].to(DEV)
            if xb.shape[0]<2: continue      # BN batch>=2
            opt.zero_grad(); out=m(xb); loss=bce(out,yb)+dice(out,yb); loss.backward(); opt.step()
    m.eval(); return m

def main():
    x,gt,valid=fetch(); C,H,W=x.shape
    # 분할축 선택: 좌우 소각 균형
    mid=W//2
    lb=(gt&valid)[:, :mid].sum(); rb=(gt&valid)[:, mid:].sum()
    print(f"격자 {H}x{W} | 소각 좌{int(lb)} 우{int(rb)}px",flush=True)
    # 훈련=소각 많은쪽, 테스트=반대(미학습). 둘다 소각 있어야.
    if lb>=rb: train_cols=(0,mid); test_region=np.zeros((H,W),bool); test_region[:,mid:]=True; tr_lbl="좌"
    else: train_cols=(mid,W); test_region=np.zeros((H,W),bool); test_region[:,:mid]=True; tr_lbl="우"
    print(f"훈련={tr_lbl}반, 테스트=반대반(미학습)",flush=True)
    tiles=crop_tiles(x,gt,valid,train_cols)
    print(f"훈련타일 {len(tiles)}개",flush=True)

    # baseline 앙상블 (파인튜닝 전)
    base=[]
    for dec,name in ENS:
        m=ARCH[dec](encoder_name="resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV).eval()
        m.load_state_dict(torch.load(os.path.join(WEIGHTS,"et02_ens",name+".pt"),map_location=DEV)); base.append(m)
    pb=ens_prob(base,x,H,W)
    # 파인튜닝 앙상블
    print("파인튜닝 9모델...",flush=True)
    ft=[finetune_one(dec,name,tiles) for dec,name in ENS]
    pf=ens_prob(ft,x,H,W)

    print(f"\n===== 테스트(미학습 반) : CIR-GT F1 =====",flush=True)
    tb=int((gt&valid&test_region).sum())
    print(f"테스트영역 소각 {tb}px",flush=True)
    for tag,prob in [("baseline(파인튜닝前)",pb),("파인튜닝後",pf)]:
        f1d,pd,rd=prf((prob>=0.2)&test_region,gt,valid)
        f1b,thr,pbb,rb2=bestf1(prob,gt,valid,test_region)
        print(f"  {tag:16s} thr0.20 F1={f1d:.3f}(P{pd:.2f}/R{rd:.2f}) | best F1={f1b:.3f}(P{pbb:.2f}/R{rb2:.2f},thr{thr:.2f})",flush=True)
    print(f"\n  대조: CIR-GT 천장 0.887 · 전역앙상블(안동전체) 0.555",flush=True)

if __name__=="__main__": main()
