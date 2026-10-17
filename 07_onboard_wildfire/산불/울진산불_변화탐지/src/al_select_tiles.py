"""손라벨 활성 학습(active learning) 준비 - 앙상블 불일치가 큰 타일을 골라 QGIS 라벨 묶음을 만든다.

배경: 산림청 폴리곤 라벨은 픽셀 정밀이 아니라 그 자로 재면 모델 F1 이 0.70 대에서 막힌다.
      ※ 이건 **자의 천장이지 모델의 천장이 아니다** — 정밀 GT(NIFoS 0.3m)로 재면 0.87~0.91 이다
        (FINAL_RESULTS.md 의 "천장 0.72~0.75 · 0.8 은 불가" 는 2026-09-08 에 뒤집혔다).
        손라벨이 지렛대라는 이 스크립트의 전제 자체는 그대로 유효하다.
      남은 확실한 지렛대는 사람 손라벨인데, 타일 수가 많으니 "모델들이 서로 다르게 답하고 폴리곤과도
      어긋나는 타일" 부터 라벨하면 같은 노동으로 더 많은 정보를 얻는다.

방법
  1. runs/ens2_{unet,fpn,deeplabv3plus}_s{1,2,3} 9 개 모델(ensemble_eval.py 와 같은 방식) 로 ko_refl 전체 타일 예측.
  2. 타일마다: 모델 간 확률 표준편차 평균(불일치), 앙상블 판정과 폴리곤이 다른 유효 화소 비율(어긋남),
     앙상블 대 폴리곤 F1, 폴리곤 불탄 비율, 계절, 홀드아웃 여부.
  3. 점수 = 정규화(불일치) + 정규화(어긋남). 봄 타일은 1.3 배(개엽기 반전이 문제인 계절).
  4. 상위 12 타일(화재당 최대 3, 기존 labelset/ 에 있는 타일 제외) 을 labelset_al/<화재>__<타일>/ 로 내보낸다.
     파일 구성은 export_label_package.py 와 같고 disagree.tif(불일치 지도) 와 preview 5번째 칸이 더 있다.

판정 임계값 = 9 모델 results.json 의 thr 평균.
사용: env -u LD_LIBRARY_PATH PROJ_DATA=~/miniforge3/envs/ml/share/proj python src/al_select_tiles.py
      (GPU 가 없으면 CPU 로 돈다. 타일 71 개라 몇 분.)
"""
import os, sys, glob, json, csv, time, re, numpy as np
import rasterio                      # torch 보다 먼저 (libstdc++ 충돌 방지)
from rasterio.transform import Affine
import torch
import segmentation_models_pytorch as smp

HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
RF=os.path.join(PROJ,"ko_refl"); HOLD=os.path.join(PROJ,"data","holdout_fixed.txt")
OUT=os.path.join(PROJ,"labelset_al"); OLD=os.path.join(PROJ,"labelset"); LOG=os.path.join(PROJ,"logs","al_select.log")
RUNS=[f"ens2_{d}_s{s}" for d in ("unet","fpn","deeplabv3plus") for s in (1,2,3)]
TOP_N=12; PER_FIRE=3; SPRING_W=1.3; BATCH=16
DEV="cuda" if torch.cuda.is_available() else "cpu"
if DEV=="cpu": torch.set_num_threads(32)
_ARCH={"unet":smp.Unet,"unetpp":smp.UnetPlusPlus,"deeplabv3plus":smp.DeepLabV3Plus,"fpn":smp.FPN,"manet":smp.MAnet}
TRANSFORM=Affine(20,0,0,0,-20,0); CRS="EPSG:32652"   # 더미 격자, 모든 층이 겹치게

def log(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True)
    with open(LOG,"a") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S ")+s+"\n")

def _nbr(b): return (b[3]-b[5])/(b[3]+b[5]+1e-6)
def build_x(x12,cfg):
    """ensemble_eval.py 와 같은 입력 구성."""
    pre,post=x12[:,0:6],x12[:,6:12]
    if cfg=="bitemporal": return x12
    if cfg=="rgbnir": return np.concatenate([pre[:,[0,1,2,3]],post[:,[0,1,2,3]]],1)
    if cfg=="postonly": return post
    if cfg=="postchar":
        nir,red,sw2=post[:,3],post[:,2],post[:,5]
        return np.concatenate([post,((nir-sw2)/(nir+sw2+1e-6))[:,None],((nir-red)/(nir+red+1e-6))[:,None]],1)
    if cfg=="biidx":
        nbrp=(pre[:,3]-pre[:,5])/(pre[:,3]+pre[:,5]+1e-6); nbro=(post[:,3]-post[:,5])/(post[:,3]+post[:,5]+1e-6)
        dnbr=nbrp-nbro; rbr=dnbr/(nbrp+1.001); dnbrdt=dnbr-np.median(dnbr,axis=(1,2),keepdims=True)
        ndvio=(post[:,3]-post[:,2])/(post[:,3]+post[:,2]+1e-6)
        return np.concatenate([x12,dnbr[:,None],rbr[:,None],dnbrdt[:,None],nbro[:,None],ndvio[:,None]],1)
    raise ValueError(cfg)

def load_model(run):
    cfg=json.load(open(os.path.join(PROJ,"runs",run,"results.json")))
    m=_ARCH[cfg.get("decoder","unet")](encoder_name=cfg.get("encoder","resnet34"),encoder_weights=None,in_channels=cfg["in_ch"],classes=1).to(DEV)
    m.load_state_dict(torch.load(os.path.join(PROJ,"runs",run,"model.pt"),map_location=DEV)); m.eval()
    return m,cfg["config"],float(cfg["thr"])

def f1(pred,gt,valid):
    p=(pred>0)&(valid>0); g=(gt>0)&(valid>0)
    tp=(p&g).sum(); fp=(p&~g).sum(); fn=(~p&g).sum()
    pr=tp/(tp+fp+1e-9); rc=tp/(tp+fn+1e-9); return float(2*pr*rc/(pr+rc+1e-9))

def season(fire):
    mn=int(fire.split("_")[1][4:6]); return "봄" if 3<=mn<=5 else "여름" if 6<=mn<=8 else "가을" if 9<=mn<=11 else "겨울"

def falsecolor(refl12,off):
    """SWIR 위색: R=B12(off+5) G=B8A(off+3) B=B04(off+2). 2~98 % 스트레치, uint8."""
    r,g,b=refl12[off+5],refl12[off+3],refl12[off+2]; img=np.stack([r,g,b],-1)
    lo,hi=np.percentile(img,2),np.percentile(img,98)
    return (np.clip((img-lo)/(hi-lo+1e-6),0,1)*255).astype(np.uint8)

def wtif(path,arr,count,dtype):
    with rasterio.open(path,"w",driver="GTiff",height=arr.shape[0],width=arr.shape[1],count=count,dtype=dtype,transform=TRANSFORM,crs=CRS) as d:
        if count==1: d.write(arr.astype(dtype),1)
        else:
            for i in range(count): d.write(arr[...,i].astype(dtype),i+1)

def export(rec,refl,y,valid,pmean,pstd,thr):
    fire,tile=rec["fire"],rec["tile"]; d=os.path.join(OUT,f"{fire}__{tile.replace('.npz','')}"); os.makedirs(d,exist_ok=True)
    pred=(pmean>=thr).astype(np.uint8); pre=falsecolor(refl,0); post=falsecolor(refl,6)
    dis=np.clip(pstd/0.5,0,1)*255                      # 표준편차 0~0.5 를 0~255 로
    wtif(os.path.join(d,"pre_swir.tif"),pre,3,"uint8"); wtif(os.path.join(d,"post_swir.tif"),post,3,"uint8")
    wtif(os.path.join(d,"polygon_gt.tif"),y,1,"uint8"); wtif(os.path.join(d,"model_pred.tif"),pred,1,"uint8")
    wtif(os.path.join(d,"valid.tif"),valid,1,"uint8"); wtif(os.path.join(d,"hand_gt.tif"),y.copy(),1,"uint8")
    wtif(os.path.join(d,"disagree.tif"),dis,1,"uint8")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        def over(a,c):
            o=post.copy(); o[a>0]=c; return o
        fig,ax=plt.subplots(1,5,figsize=(20,4.4))
        for a,(img,t) in zip(ax,[(pre,"pre SWIR"),(post,"post SWIR"),(over(y,[255,255,0]),"polygon (yellow)"),(over(pred,[255,0,0]),"ensemble pred (red)"),(pstd,"disagreement std")]):
            if img.ndim==2: im=a.imshow(img,cmap="magma",vmin=0,vmax=0.5); fig.colorbar(im,ax=a,fraction=0.046)
            else: a.imshow(img)
            a.set_title(t); a.axis("off")
        fig.suptitle(f"{fire} / {tile}  score={rec['score']:.3f} std={rec['mean_std']:.3f} mismatch={rec['mismatch']:.3f}")
        fig.tight_layout(); fig.savefig(os.path.join(d,"preview.png"),dpi=90); plt.close(fig)
    except Exception as ex: log("preview 실패",ex)
    return os.path.basename(d)

def main():
    t0=time.time(); os.makedirs(OUT,exist_ok=True); open(LOG,"a").close()
    hold=set(l.strip() for l in open(HOLD) if l.strip())
    models=[load_model(r) for r in RUNS]; thr=float(np.mean([t for _,_,t in models]))
    log(f"모델 {len(models)} 개 로드 ({DEV}), 판정 임계값 = thr 평균 {thr:.3f} (개별: {[round(t,3) for _,_,t in models]})")
    files=sorted(glob.glob(os.path.join(RF,"*","*.npz"))); log(f"타일 {len(files)} 개")
    recs=[]; cache={}
    for i in range(0,len(files),BATCH):
        fb=files[i:i+BATCH]; zs=[np.load(f) for f in fb]
        odd=[f for f,z in zip(fb,zs) if z["x"].shape!=(12,256,256)]
        if odd: log("격자 이상 (건너뜀):",[os.path.basename(f) for f in odd])
        keep=[(f,z) for f,z in zip(fb,zs) if z["x"].shape==(12,256,256)]
        if not keep: continue
        x12=np.stack([z["x"].astype(np.float32)/10000.0 for _,z in keep]); probs=[]
        with torch.no_grad():
            for m,cfg,_ in models:
                xt=torch.from_numpy(np.ascontiguousarray(build_x(x12,cfg))).to(DEV)
                probs.append(torch.sigmoid(m(xt)).squeeze(1).cpu().numpy())
        P=np.stack(probs); pmean=P.mean(0); pstd=P.std(0)
        for k,(f,z) in enumerate(keep):
            fire=os.path.basename(os.path.dirname(f)); tile=os.path.basename(f)
            y=z["y"].astype(np.uint8); valid=z["valid"].astype(np.uint8) if "valid" in z.files else np.ones_like(y)
            v=valid>0; pred=pmean[k]>=thr
            rec={"fire":fire,"tile":tile,"season":season(fire),"holdout":fire in hold,
                 "burn_frac":float(y[v].mean()) if v.any() else 0.0,
                 "mean_std":float(pstd[k][v].mean()) if v.any() else 0.0,
                 "mismatch":float((pred[v]!=(y[v]>0)).mean()) if v.any() else 0.0,
                 "f1_vs_polygon":f1(pred,y,valid),"valid_frac":float(v.mean())}
            recs.append(rec); cache[(fire,tile)]=(z["x"].astype(np.float32)/10000.0,y,valid,pmean[k],pstd[k])
        log(f"  진행 {min(i+BATCH,len(files))}/{len(files)}  {time.time()-t0:.0f}s")
    def norm(key):
        a=np.array([r[key] for r in recs]); lo,hi=a.min(),a.max(); return (a-lo)/(hi-lo+1e-9)
    ns,nm=norm("mean_std"),norm("mismatch")
    for r,a,b in zip(recs,ns,nm):
        r["std_norm"]=float(a); r["mismatch_norm"]=float(b); r["score"]=float((a+b)*(SPRING_W if r["season"]=="봄" else 1.0))
    recs.sort(key=lambda r:-r["score"])
    # 선택: 상위 12, 화재당 3, 기존 labelset 제외
    old=set(os.listdir(OLD)) if os.path.isdir(OLD) else set(); sel=[]; per={}
    for r in recs:
        name=f"{r['fire']}__{r['tile'].replace('.npz','')}"
        if name in old or per.get(r["fire"],0)>=PER_FIRE: continue
        sel.append(r); per[r["fire"]]=per.get(r["fire"],0)+1
        if len(sel)>=TOP_N: break
    if len(sel)<TOP_N:   # 남는 게 없으면 기존 타일도 허용
        for r in recs:
            if r in sel: continue
            if per.get(r["fire"],0)>=PER_FIRE: continue
            sel.append(r); per[r["fire"]]=per.get(r["fire"],0)+1
            if len(sel)>=TOP_N: break
    for r in recs: r["selected"]=r in sel
    json.dump({"threshold":thr,"runs":RUNS,"spring_weight":SPRING_W,"n_tiles":len(recs),"ranking":recs},open(os.path.join(OUT,"ranking.json"),"w"),ensure_ascii=False,indent=1)
    keys=["fire","tile","season","holdout","burn_frac","mean_std","mismatch","f1_vs_polygon","valid_frac","std_norm","mismatch_norm","score","selected"]
    with open(os.path.join(OUT,"ranking.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in recs: w.writerow({k:(round(r[k],4) if isinstance(r[k],float) else r[k]) for k in keys})
    manifest=[]
    for r in sel:
        refl,y,valid,pm,ps=cache[(r["fire"],r["tile"])]
        d=export(r,refl,y,valid,pm,ps,thr); manifest.append({"dir":d,**{k:r[k] for k in ("fire","tile","season","holdout","score","mean_std","mismatch","f1_vs_polygon")}})
        log(f"  선택 {d} [{r['season']}{' 홀드아웃' if r['holdout'] else ''}] score={r['score']:.3f} std={r['mean_std']:.3f} 어긋남={r['mismatch']:.3f} F1={r['f1_vs_polygon']:.3f}")
    json.dump(manifest,open(os.path.join(OUT,"manifest.json"),"w"),ensure_ascii=False,indent=2)
    log(f"완료: 타일 {len(recs)} 처리, {len(sel)} 선택 -> {OUT}  ({time.time()-t0:.0f}s)")
if __name__=="__main__": main()
