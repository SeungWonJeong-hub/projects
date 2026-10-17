"""배포된 새 unet_s1(FOREST_OS=4)로 성능표 P/R 재측정 @배포임계 0.25.
CIR 홀드아웃(국토정보플랫폼 CIR-GT) + 산림청 홀드아웃. wildfire_cd.json 20m metrics 갱신.
"""
import os, glob, json, numpy as np, torch
import segmentation_models_pytorch as smp
import step4_rehearsal as S
WEB=os.path.join(S._ROOT,"Steven","웹앱")
DEPLOYED=os.path.join(WEB,"weights","et02_ens","et02_unet_s1.pt")
THR=0.25
def load():
    m=smp.Unet("resnet34",encoder_weights=None,in_channels=8,classes=1).to(S.DEV)
    m.load_state_dict(torch.load(DEPLOYED,map_location=S.DEV)); m.eval(); return m
def pr(P,Y):
    pr_=P>=THR; tp=(pr_&(Y>0)).sum(); fp=(pr_&(Y==0)).sum(); fn=((~pr_)&(Y>0)).sum()
    return tp/(tp+fp+1e-9), tp/(tp+fn+1e-9)
def main():
    m=load()
    # CIR 홀드아웃(국토 CIR-GT)
    P=[];Y=[]
    for h in S.CIR_HOLD:
        x,gt,vd=S.cir_load(h); prob=S.infer(m,x,*gt.shape); P.append(prob[vd]); Y.append(gt[vd].astype(int))
    cp,cr=pr(np.concatenate(P),np.concatenate(Y))
    # 산림청 홀드아웃
    P=[];Y=[]
    for h in S.FOREST_HOLD:
        dirs=[d for d in glob.glob(os.path.join(S.PROJ,S.KO_DIR,"*")) if h.replace("kf_","") in os.path.basename(d) and os.path.isdir(d)]
        if not dirs: continue
        for f in glob.glob(os.path.join(dirs[0],"*.npz")):
            z=np.load(f); x=np.clip(S.sel12(z["x"].astype(np.float32)/1e4),0,1)
            prob=S.infer(m,x,x.shape[1],x.shape[2]); v=z["valid"].astype(bool) if "valid" in z else np.ones(z["y"].shape,bool)
            P.append(prob[v]); Y.append(z["y"][v].astype(int))
    fp_,fr=pr(np.concatenate(P),np.concatenate(Y))
    print(f"새 모델 @thr {THR}:")
    print(f"  국토 CIR-GT:  P {cp:.3f} · R {cr:.3f}")
    print(f"  산림청 폴리곤: P {fp_:.3f} · R {fr:.3f}")
    # wildfire_cd.json 20m metrics 갱신
    p=os.path.join(WEB,"weights","wildfire_cd.json"); d=json.load(open(p))
    M=d["variants"]["20m"].setdefault("metrics",{})
    M.update(cir_p=round(float(cp),3),cir_r=round(float(cr),3),forest_p=round(float(fp_),3),forest_r=round(float(fr),3),
             remeasured_2026_09_20="FOREST_OS=4 재브래킷 배포본 @thr0.25")
    json.dump(d,open(p,"w"),ensure_ascii=False,indent=1)
    print("wildfire_cd.json 20m metrics 갱신 완료")
if __name__=="__main__": main()
