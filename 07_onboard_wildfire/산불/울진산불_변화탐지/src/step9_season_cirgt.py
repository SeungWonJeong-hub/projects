"""비봄(여름/가을/겨울) CIR-GT(dNBR) F1 측정 — 봄 0.828과 같은 자尺 비교.
ko_refl 비봄 화재의 12밴드 npz로 dNBR GT 직접계산 → 배포 앙상블 채점.
홀드아웃(미학습) vs in-sample(학습) 구분 보고. 손라벨/STAC/API 불필요.
사용: python step9_season_cirgt.py
"""
import os, re, glob, numpy as np, torch
import segmentation_models_pytorch as smp
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
WEIGHTS=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/웹앱/weights"); DEV="cuda" if torch.cuda.is_available() else "cpu"
FHOLD=[l.strip() for l in open(os.path.join(PROJ,"data","holdout_fixed.txt")) if l.strip()]
ENS=[("unet","et02_unet_s1"),("unet","et02_unet_s2"),("unet","et02_unet_s3"),
     ("fpn","et02_fpn_s1"),("fpn","et02_fpn_s2"),("fpn","et02_fpn_s3"),
     ("deeplabv3plus","et02_deeplabv3plus_s1"),("deeplabv3plus","et02_deeplabv3plus_s2"),("deeplabv3plus","et02_deeplabv3plus_s3")]
ARCH={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}
SEAS={12:"겨울",1:"겨울",2:"겨울",3:"봄",4:"봄",5:"봄",6:"여름",7:"여름",8:"여름",9:"가을",10:"가을",11:"가을"}
def month(b):
    m=re.search(r"kf_\d{4}(\d{2})",b); return int(m.group(1)) if m else 0
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
def fire_f1(ms,fdir):
    P=[];Y=[]
    for f in glob.glob(os.path.join(fdir,"*.npz")):
        z=np.load(f); x=z["x"].astype(np.float32)  # 12ch reflectance
        pre,post=x[0:6],x[6:12]
        nbr=lambda a:(a[3]-a[5])/(a[3]+a[5]+1e-6); ndvi=lambda a:(a[3]-a[2])/(a[3]+a[2]+1e-6)
        dnbr=nbr(pre)-nbr(post); dndvi=ndvi(pre)-ndvi(post)
        v=z["valid"].astype(bool) if "valid" in z else np.ones(dnbr.shape,bool)
        gt=((dnbr>0.15)&(dndvi>0.08)&(ndvi(post)<0.45)&v).astype(np.uint8)   # dNBR CIR-GT (봄과 동일 임계)
        xin=np.clip(np.concatenate([pre[[0,1,2,3]],post[[0,1,2,3]]],0)/1e4,0,1).astype("float32")
        xt=torch.from_numpy(xin[None]).to(DEV)
        prob=np.mean([torch.sigmoid(m(xt))[0,0].cpu().numpy() for m in ms],0)
        P.append(prob[v]); Y.append(gt[v])
    if not P: return None
    return bestf1(np.concatenate(P),np.concatenate(Y))
def main():
    ms=load_ens()
    dirs=[d for d in glob.glob(os.path.join(PROJ,"ko_refl","*")) if os.path.isdir(d)]
    res={}  # (계절, 홀드아웃?) -> [f1]
    print("화재별 비봄 dNBR-GT F1:",flush=True)
    for d in sorted(dirs):
        b=os.path.basename(d); mo=month(b); s=SEAS.get(mo)
        if s=="봄" or s is None: continue
        hold=any(h.replace("kf_","") in b for h in FHOLD)
        f=fire_f1(ms,d)
        if f is None: continue
        res.setdefault((s,hold),[]).append(f)
        print(f"  {b:26s} {s} {'홀드아웃' if hold else 'in-sample':9s} F1={f:.3f}",flush=True)
    print("\n=== 계절별 평균 (비봄 CIR-GT, 봄 미학습 0.828과 비교) ===",flush=True)
    for s in ["여름","가을","겨울"]:
        h=res.get((s,True),[]); t=res.get((s,False),[])
        hs=f"홀드아웃(미학습) {np.mean(h):.3f}(n{len(h)})" if h else "홀드아웃 없음"
        ts=f"in-sample {np.mean(t):.3f}(n{len(t)})" if t else ""
        print(f"  {s}: {hs}  |  {ts}",flush=True)
    allh=[f for (s,hold),fs in res.items() if hold for f in fs]
    if allh: print(f"\n★ 비봄 홀드아웃(미학습) 전체 평균 F1 = {np.mean(allh):.3f} (n{len(allh)}) vs 봄 미학습 0.828",flush=True)
if __name__=="__main__": main()
