"""Cross-domain baseline: 온보드 단일 unet_s1(8ch RGBNIR pre/post)의
Korea in-domain(CIR 홀드아웃) vs US cross-biome(MTBS drop-in) 붕괴폭 실측.
같은 자(尺): 양쪽 다 step4_rehearsal.infer + bestf1 사용.
학습 없음(순수 추론). MTBS 라벨=sev {2,3,4}=burned, valid=sev{1,2,3,4}.
결과 runs/xdomain_baseline.json.
"""
from __future__ import annotations
import os, glob, json, numpy as np, torch
import segmentation_models_pytorch as smp
import step4_rehearsal as S

HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
CKPT=os.path.join(PROJ,"runs","cir_reh_ens","et02_unet_s1_reh.pt")
MTBS=os.path.join(PROJ,"..","mtbs_pretrain","patches")
DEV=S.DEV
CLOUD_MAX=float(os.environ.get("CLOUD_MAX","30"))
MIN_BURN=int(os.environ.get("MIN_BURN","300"))    # valid 내 최소 소각픽셀(F1 계산 가능성)
BURN=set(int(c) for c in os.environ.get("BURN","2,3,4").split(","))
VALIDC=set(int(c) for c in os.environ.get("VALIDC","1,2,3,4").split(","))
# MTBS X 밴드 [B02,B03,B04,B8A,B11,B12]x pre/post → 8ch RGBNIR pre/post
SEL=[0,1,2,3, 6,7,8,9]

def load_unet():
    m=smp.Unet("resnet34",encoder_weights=None,in_channels=8,classes=1).to(DEV)
    m.load_state_dict(torch.load(CKPT,map_location=DEV)); m.eval(); return m

def mtbs_iter():
    for f in sorted(glob.glob(os.path.join(MTBS,"*.npz"))):
        try: d=np.load(f, allow_pickle=True)
        except Exception: continue
        if "X" not in d or "sev" not in d: continue
        try: cl=float(d["cloud_pct"])
        except Exception: cl=0.0
        if cl>CLOUD_MAX: continue
        X=d["X"].astype(np.float32); sev=d["sev"]
        x=X[SEL]                                    # (8,H,W)
        gt=np.isin(sev,list(BURN))
        valid=np.isin(sev,list(VALIDC))
        if int((gt&valid).sum())<MIN_BURN: continue
        st=os.path.basename(f)[:2]
        yield os.path.basename(f), st, x, gt, valid

def main():
    m=load_unet()
    print(f"모델: {os.path.basename(CKPT)} (8ch RGBNIR pre/post)\n",flush=True)

    # --- Korea in-domain (CIR 홀드아웃, 같은 자) ---
    ko={}
    for h in S.CIR_HOLD:
        try:
            x,gt,vd=S.cir_load(h)
        except Exception as e:
            print(f"[KO skip {h}] {e}",flush=True); continue
        H,W=gt.shape; p=S.infer(m,x,H,W)
        f1=S.bestf1(p[vd],gt[vd].astype(int))
        ko[h]=round(float(f1),4); print(f"[KO {h:20s}] best-F1 {f1:.4f}",flush=True)
    ko_vals=np.array(list(ko.values()))

    # --- US cross-biome (MTBS drop-in) ---
    us_by_patch={}; us_by_state={}
    n=0
    for name,st,x,gt,valid in mtbs_iter():
        H,W=gt.shape; p=S.infer(m,x,H,W)
        f1=S.bestf1(p[valid],gt[valid].astype(int))
        us_by_patch[name]=round(float(f1),4)
        us_by_state.setdefault(st,[]).append(float(f1)); n+=1
        if n%50==0: print(f"  ...MTBS {n} patches",flush=True)
    us_vals=np.array(list(us_by_patch.values()))
    state_mean={s:round(float(np.mean(v)),4) for s,v in sorted(us_by_state.items()) if len(v)>=3}
    state_n={s:len(v) for s,v in us_by_state.items()}

    out={
        "model":"et02_unet_s1_reh (8ch RGBNIR, onboard single)",
        "korea_indomain":{"per_fire":ko,"mean":round(float(ko_vals.mean()),4),
                          "min":round(float(ko_vals.min()),4),"sigma":round(float(ko_vals.std()),4),"n":len(ko)},
        "us_crossbiome":{"mean":round(float(us_vals.mean()),4),"min":round(float(us_vals.min()),4),
                         "max":round(float(us_vals.max()),4),"sigma":round(float(us_vals.std()),4),"n":len(us_vals),
                         "by_state_mean":state_mean,"by_state_n":state_n},
        "collapse_delta":round(float(us_vals.mean()-ko_vals.mean()),4),
        "params":{"CLOUD_MAX":CLOUD_MAX,"MIN_BURN":MIN_BURN,"BURN":sorted(BURN),"VALIDC":sorted(VALIDC)},
    }
    os.makedirs(os.path.join(PROJ,"runs"),exist_ok=True)
    json.dump(out,open(os.path.join(PROJ,"runs","xdomain_baseline.json"),"w"),ensure_ascii=False,indent=2)
    print("\n=== Cross-domain baseline (온보드 단일 unet_s1) ===",flush=True)
    print(f"Korea in-domain : mean {out['korea_indomain']['mean']:.3f} (min {out['korea_indomain']['min']:.3f}, σ {out['korea_indomain']['sigma']:.3f}, n{out['korea_indomain']['n']})",flush=True)
    print(f"US cross-biome  : mean {out['us_crossbiome']['mean']:.3f} (min {out['us_crossbiome']['min']:.3f}, max {out['us_crossbiome']['max']:.3f}, σ {out['us_crossbiome']['sigma']:.3f}, n{out['us_crossbiome']['n']})",flush=True)
    print(f"붕괴 델타(US-KO): {out['collapse_delta']:+.3f}",flush=True)
    print(f"생태권별 mean: {state_mean}",flush=True)

if __name__=="__main__": main()
