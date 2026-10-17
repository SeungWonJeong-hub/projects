"""LORO 게이트 Tier1: 이미 학습된 9모델(cir_reh_ens)로 재학습 없이
단일(멤버) vs 앙상블 이득이 멤버 간 편차를 넘는지 판정.
평가 = CIR_HOLD 4화재(9모델이 학습때 못 본 홀드아웃). 추론만.
사용: python loro_gate_tier1.py
"""
import os, numpy as np, torch
import segmentation_models_pytorch as smp
import step4_rehearsal as S

DEV=S.DEV
REH=os.path.join(S.PROJ,"runs","cir_reh_ens")
ARCH={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}
ENS=[("unet","et02_unet_s1"),("unet","et02_unet_s2"),("unet","et02_unet_s3"),
     ("fpn","et02_fpn_s1"),("fpn","et02_fpn_s2"),("fpn","et02_fpn_s3"),
     ("deeplabv3plus","et02_deeplabv3plus_s1"),("deeplabv3plus","et02_deeplabv3plus_s2"),("deeplabv3plus","et02_deeplabv3plus_s3")]

def load_models():
    ms=[]
    for d,n in ENS:
        m=ARCH[d]("resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV)
        m.load_state_dict(torch.load(os.path.join(REH,n+"_reh.pt"),map_location=DEV)); m.eval()
        ms.append((n,m))
    return ms

def main():
    ms=load_models()
    print(f"로드 9모델: {[n for n,_ in ms]}",flush=True)
    print(f"홀드아웃 화재: {S.CIR_HOLD}\n",flush=True)
    member_by_fire=[]   # per-fire list of 9 member F1
    ens_by_fire=[]      # per-fire ensemble F1
    for h in S.CIR_HOLD:
        x,gt,vd=S.cir_load(h); H,W=gt.shape
        Y=gt[vd].astype(int)
        probs=[S.infer(m,x,H,W) for _,m in ms]           # 9 prob maps
        memf1=[S.bestf1(p[vd],Y) for p in probs]         # 9 member F1
        ens=np.mean(probs,0)                             # 앙상블 = 확률 평균
        ensf1=S.bestf1(ens[vd],Y)
        member_by_fire.append(memf1); ens_by_fire.append(ensf1)
        mm=np.mean(memf1); ms_=np.std(memf1)
        print(f"[{h:20s}] 멤버F1 평균={mm:.3f}±{ms_:.3f} (min{min(memf1):.3f}/max{max(memf1):.3f}) | 앙상블={ensf1:.3f} | 이득 vs평균멤버={ensf1-mm:+.3f} vs최고멤버={ensf1-max(memf1):+.3f}",flush=True)

    member_by_fire=np.array(member_by_fire)   # (4,9)
    ens_by_fire=np.array(ens_by_fire)         # (4,)
    mean_member_per_fire=member_by_fire.mean(1)   # (4,)
    best_member_per_fire=member_by_fire.max(1)    # (4,)

    print("\n===== 집계 (4-홀드아웃) =====",flush=True)
    print(f"멤버 F1: 화재평균의 평균={mean_member_per_fire.mean():.3f}, 화재 간 표준편차(fold σ)={mean_member_per_fire.std():.3f}",flush=True)
    print(f"앙상블 F1: 평균={ens_by_fire.mean():.3f}, 화재 간 표준편차={ens_by_fire.std():.3f}",flush=True)

    gain_vs_mean=ens_by_fire-mean_member_per_fire   # 페어드 이득 (4,)
    gain_vs_best=ens_by_fire-best_member_per_fire
    print(f"\n페어드 이득(앙상블 - 평균멤버) per fire: {np.round(gain_vs_mean,3).tolist()}",flush=True)
    print(f"  평균 이득={gain_vs_mean.mean():+.4f}  이득 표준편차={gain_vs_mean.std():.4f}",flush=True)
    print(f"페어드 이득(앙상블 - 최고멤버) per fire: {np.round(gain_vs_best,3).tolist()}",flush=True)
    print(f"  평균 이득={gain_vs_best.mean():+.4f}",flush=True)

    # 게이트: 평균 이득이 fold σ(화재 간 변동)을 넘는가?
    fold_sigma=mean_member_per_fire.std()
    print(f"\n===== 게이트 판정 =====",flush=True)
    print(f"앙상블 평균 이득(vs 평균멤버) = {gain_vs_mean.mean():+.4f}",flush=True)
    print(f"fold σ (화재 간 F1 변동)      = {fold_sigma:.4f}",flush=True)
    verdict = "앙상블 정당화 O (이득 > fold σ)" if gain_vs_mean.mean()>fold_sigma else "앙상블 정당화 X (이득 ≤ fold σ = 노이즈 수준)"
    print(f"판정: {verdict}",flush=True)
    print(f"\n※ n=4 홀드아웃, 통계검정력 약함 → 애매하면 Tier2(10-fold 재학습) 필요",flush=True)

    # 정직검증용 원자료 저장
    import json
    out={"fires":S.CIR_HOLD,"member_f1":member_by_fire.tolist(),"ens_f1":ens_by_fire.tolist(),
         "gain_vs_mean":gain_vs_mean.tolist(),"gain_vs_best":gain_vs_best.tolist(),
         "mean_gain_vs_mean":float(gain_vs_mean.mean()),"fold_sigma":float(fold_sigma)}
    with open(os.path.join(S.PROJ,"runs","loro_gate_tier1.json"),"w") as f: json.dump(out,f,ensure_ascii=False,indent=2)
    print(f"저장: runs/loro_gate_tier1.json",flush=True)

if __name__=="__main__": main()
