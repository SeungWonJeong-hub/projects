"""추론 전용: 국토정보플랫폼 CIR-GT 파인튜닝 前/後 앙상블을 안동(2025) 10m 에서 비교.
학습 없음 — 이미 학습된 가중치를 그대로 불러 P/R 만 잰다.
  前(파인튜닝 전) = weights/et02_ens/*.bak_prespring  (봄 rehearsal 이전 base 9모델)
  後(파인튜닝 후) = weights/et02_ens_10m/*_reh.pt      (10m 배포 rehearsed 9모델)
평가 GT = 국토정보플랫폼 CIR-GT (gt_stac_10m_raw.npz), 홀드아웃(미학습) 화재.
"""
import os, numpy as np, torch
import segmentation_models_pytorch as smp
ROOT=os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference"))
FIRES=os.path.join(ROOT,"Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires")
W=os.path.join(ROOT,"Steven/웹앱/weights")
DEV="cuda" if torch.cuda.is_available() else "cpu"
NPZ="gt_stac_10m_raw.npz"
HOLD=os.environ.get("HOLD","250324_안동길안면,250321_산청시천면,230402_홍성서부면,220410_양구양구읍").split(",")
ARCH={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}
NAMES=[("unet","unet_s1"),("unet","unet_s2"),("unet","unet_s3"),
       ("fpn","fpn_s1"),("fpn","fpn_s2"),("fpn","fpn_s3"),
       ("deeplabv3plus","deeplabv3plus_s1"),("deeplabv3plus","deeplabv3plus_s2"),("deeplabv3plus","deeplabv3plus_s3")]
BEFORE=[(a,os.path.join(W,"et02_ens",f"et02_{n}.pt.bak_prespring")) for a,n in NAMES]
AFTER =[(a,os.path.join(W,"et02_ens_10m",f"et02_10mT512_s50_{n}_reh.pt")) for a,n in NAMES]

def load(name):
    d=np.load(os.path.join(FIRES,name,NPZ)); return d["x"].astype("float32"),d["gt"].astype(bool),d["valid"].astype(bool)
def build(arch,path):
    m=ARCH[arch](encoder_name="resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV)
    m.load_state_dict(torch.load(path,map_location=DEV)); m.eval(); return m
@torch.no_grad()
def infer1(m,x,H,Wd):
    prob=np.zeros((H,Wd),"float32")
    for i in range(0,H,256):
        for j in range(0,Wd,256):
            p=x[:,i:i+256,j:j+256]; ph,pw=p.shape[1],p.shape[2]
            pad=np.zeros((8,256,256),"float32"); pad[:,:ph,:pw]=p
            o=torch.sigmoid(m(torch.from_numpy(pad[None]).to(DEV)))[0,0].cpu().numpy()
            prob[i:i+ph,j:j+pw]=o[:ph,:pw]
    return prob
def ens_prob(models,x,H,Wd):
    acc=np.zeros((H,Wd),"float32")
    for m in models: acc+=infer1(m,x,H,Wd)
    return acc/len(models)
def prf(pred,y,mm):
    pred=pred&mm; y=y&mm; tp=(pred&y).sum(); fp=(pred&~y).sum(); fn=(~pred&y).sum()
    p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); return 2*p*r/(p+r+1e-9),p,r
def bestf1(prob,gt,vd):
    b=None
    for t in np.arange(0.05,0.95,0.05):
        f,p,r=prf(prob>=t,gt,vd)
        if b is None or f>b[0]: b=(f,t,p,r)
    return b

def run(tag,ckpts):
    print(f"\n=== {tag} ===",flush=True)
    models=[build(a,p) for a,p in ckpts]
    rows={}
    for h in HOLD:
        x,gt,vd=load(h); C,H,Wd=x.shape
        prob=ens_prob(models,x,H,Wd)
        f,t,p,r=bestf1(prob,gt,vd); f2,p2,r2=prf(prob>=0.2,gt,vd)
        rows[h]=(f,p,r,t,f2,p2,r2)
        print(f"  {h:20s} bestF1={f:.3f}(P{p:.2f}/R{r:.2f},t{t:.2f})  thr0.2 F1={f2:.3f}(P{p2:.2f}/R{r2:.2f})",flush=True)
    for m in models: del m
    if DEV=="cuda": torch.cuda.empty_cache()
    return rows

def main():
    b=run("파인튜닝 前 (base .bak_prespring, 9모델 앙상블)",BEFORE)
    a=run("파인튜닝 後 (10m 배포 _reh, 9모델 앙상블)",AFTER)
    print("\n=== 安東(2025) 10m 파인튜닝 前→後 (bestF1 임계값 기준) ===",flush=True)
    for h in HOLD:
        fb,pb,rb=b[h][:3]; fa,pa,ra=a[h][:3]
        print(f"  {h:20s} F1 {fb:.3f}->{fa:.3f}({fa-fb:+.3f})  P {pb:.2f}->{pa:.2f}  R {rb:.2f}->{ra:.2f}",flush=True)

if __name__=="__main__": main()
