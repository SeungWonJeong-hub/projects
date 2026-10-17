"""#3 in-domain 망각 점검: 봄 파인튜닝(CIR-GT)이 기존 여름/가을 산림청-라벨 성능을 해치나.
ko_refl 홀드아웃 8화재(rgbnir 8ch, 산림청 y)에 pre-FT(*.bak_prespring) vs post-FT(*.pt)
앙상블 채점. 화재별+계절별 best-thr F1 비교. 배포 전처리(/1e4 clip 0-1)로 공정비교.
사용: python step3_indomain.py
"""
import os, glob, re, datetime as dt, numpy as np, torch
import segmentation_models_pytorch as smp
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
WEIGHTS=os.path.join(os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference")), "Steven/웹앱/weights"); DEV="cuda" if torch.cuda.is_available() else "cpu"
HOLD=[l.strip() for l in open(os.path.join(PROJ,"data","holdout_fixed.txt")) if l.strip()]
ENS=[("unet","et02_unet_s1"),("unet","et02_unet_s2"),("unet","et02_unet_s3"),
     ("fpn","et02_fpn_s1"),("fpn","et02_fpn_s2"),("fpn","et02_fpn_s3"),
     ("deeplabv3plus","et02_deeplabv3plus_s1"),("deeplabv3plus","et02_deeplabv3plus_s2"),("deeplabv3plus","et02_deeplabv3plus_s3")]
ARCH={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}
def rgbnir(x12): pre,post=x12[0:6],x12[6:12]; return np.concatenate([pre[[0,1,2,3]],post[[0,1,2,3]]],0)
def season(name):
    m=re.search(r"_(\d{4})(\d{2})(\d{2})_",name) or re.search(r"kf_(\d{4})(\d{2})(\d{2})",name)
    mo=int(name[7:9]) if False else int(re.search(r"kf_\d{4}(\d{2})",name).group(1))
    return {12:"겨울",1:"겨울",2:"겨울",3:"봄",4:"봄",5:"봄",6:"여름",7:"여름",8:"여름",9:"가을",10:"가을",11:"가을"}[mo]
def build(suffix):
    ms=[]
    for dec,name in ENS:
        m=ARCH[dec](encoder_name="resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV).eval()
        m.load_state_dict(torch.load(os.path.join(WEIGHTS,"et02_ens",name+".pt"+suffix),map_location=DEV)); ms.append(m)
    return ms
@torch.no_grad()
def predict_fire(models,fdir):
    P=[];Y=[]
    for f in glob.glob(os.path.join(fdir,"*.npz")):
        d=np.load(f); x=np.clip(rgbnir(d["x"].astype(np.float32)/1e4),0,1)
        xt=torch.from_numpy(x).unsqueeze(0).to(DEV)
        p=np.mean([torch.sigmoid(m(xt))[0,0].cpu().numpy() for m in models],0)
        v=d["valid"].astype(bool) if "valid" in d else np.ones(p.shape,bool)
        P.append(p[v]); Y.append(d["y"][v].astype(np.uint8))
    return np.concatenate(P),np.concatenate(Y)
def bestf1(P,Y):
    b=0
    for t in np.arange(0.1,0.9,0.025):
        pr=P>=t; tp=(pr&(Y==1)).sum(); fp=(pr&(Y==0)).sum(); fn=((~pr)&(Y==1)).sum()
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); f=2*p*r/(p+r+1e-9); b=max(b,f)
    return b
def evalset(models):
    per={}; allP=[]; allY=[]; seas={}
    for fire in HOLD:
        dirs=glob.glob(os.path.join(PROJ,"ko_refl",fire+"*"))+glob.glob(os.path.join(PROJ,"ko_refl","*"+fire.replace('kf_','')+"*"))
        dirs=[d for d in set(dirs) if os.path.isdir(d)]
        if not dirs: per[fire]=None; continue
        P,Y=predict_fire(models,dirs[0]); allP.append(P); allY.append(Y)
        per[fire]=bestf1(P,Y); seas.setdefault(season(fire),[[],[]]); seas[season(fire)][0].append(P); seas[season(fire)][1].append(Y)
    Pa=np.concatenate(allP); Ya=np.concatenate(allY); overall=bestf1(Pa,Ya)
    sf={s:bestf1(np.concatenate(v[0]),np.concatenate(v[1])) for s,v in seas.items()}
    return overall,per,sf
def main():
    print("=== #3 in-domain 망각 점검 (산림청 라벨 8홀드아웃) ===",flush=True)
    for tag,suf in [("파인튜닝 前(bak_prespring)",".bak_prespring"),("파인튜닝 後(배포)","")]:
        ov,per,sf=evalset(build(suf))
        print(f"\n[{tag}] 전체 best-F1={ov:.3f}",flush=True)
        print("  계절:", {s:round(f,3) for s,f in sf.items()},flush=True)
        print("  화재별:", {k:(round(v,3) if v else None) for k,v in per.items()},flush=True)

if __name__=="__main__": main()
