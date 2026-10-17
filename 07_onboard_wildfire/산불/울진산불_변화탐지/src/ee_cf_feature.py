"""Counterfactual anomaly feature (cheap 버전) — 봄 phenology를 분리한 순수 화재신호.

가설: 봄 dNBR반전은 pre(개엽전)→post(개엽후) 초록화 때문. pre-fire 월별 궤적을 외삽하면
"화재 없었으면 예상되는 개엽 포함 정상 post"를 예측할 수 있고, 실제 post와의 차이 = 순수 화재.
전체 ConvGRU 빌드 전, 픽셀별 선형외삽으로 싸게 가설 검증. anomaly를 AlphaEarth에 추가채널.

ko_presto_ts(월별 S2 6밴드) 재사용. 밴드순 [B02,B03,B04,B8A,B11,B12]. NBR=(B8A-B12)/(B8A+B12).
counterfactual: pre-fire 유효월의 NBR 궤적을 post월로 선형외삽 → CF_NBR. anomaly=CF_NBR - actual_NBR(post).
  봄 진짜소각: 예측정상(개엽으로 NBR↑ 기대) vs 실제(불타서 NBR↓) → anomaly 양수 큼(반전 우회).
→ ko_ae 타일에 anomaly 채널 추가해 ko_ae_cf/ 저장(x=[AE128|anomaly1]=129ch).

사용: python3 ee_cf_feature.py
"""
import os, glob, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
AE=os.path.join(PROJ,"ko_ae"); TS=os.path.join(PROJ,"ko_presto_ts"); OUT=os.path.join(PROJ,"ko_ae_cf")
B8A,B12=3,5

def nbr_series(ts):
    b8a=ts[:,B8A,:,:]; b12=ts[:,B12,:,:]
    return (b8a-b12)/(b8a+b12+1e-6)   # (T,H,W)

def cf_anomaly(ts, months, tvalid, fire_mo):
    nbr=nbr_series(ts)  # (T,H,W)
    val=[i for i,(m,v) in enumerate(zip(months,tvalid)) if v==1]
    pre=[i for i in val if months[i]<fire_mo]; post=[i for i in val if months[i]>fire_mo]
    if len(pre)<1 or len(post)<1: return None
    # post 첫 유효월 타겟
    i_post=post[0]; m_post=months[i_post]
    if len(pre)>=2:
        # pre 두 달로 픽셀별 선형외삽 → post월 CF_NBR
        i0,i1=pre[-2],pre[-1]; m0,m1=months[i0],months[i1]
        slope=(nbr[i1]-nbr[i0])/max(m1-m0,1)
        cf=nbr[i1]+slope*(m_post-m1)
    else:
        cf=nbr[pre[-1]]  # 한 달뿐이면 그대로(외삽 불가)
    anom=cf-nbr[i_post]   # 예측정상 - 실제. 소각이면 실제 NBR↓ → anom 양수 큼(봄반전 우회)
    return np.clip(anom,-1,1).astype(np.float32)

os.makedirs(OUT,exist_ok=True); n=0; skip=0
for fire in sorted(os.listdir(AE)):
    fire_mo=int(fire.split("_")[1][4:6])
    for ae_f in glob.glob(os.path.join(AE,fire,"*.npz")):
        tile=os.path.basename(ae_f); ts_f=os.path.join(TS,fire,tile)
        od=os.path.join(OUT,fire); os.makedirs(od,exist_ok=True); op=os.path.join(od,tile)
        ae=np.load(ae_f)
        if not os.path.exists(ts_f):
            # 월별 없으면 anomaly 0 채널(해당 화재는 CF 없음 — train서 대부분 ko_presto_ts 있는 대형화재)
            anom=np.zeros((256,256),np.float32); skip+=1
        else:
            tsd=np.load(ts_f); a=cf_anomaly(tsd["ts"],tsd["months"],tsd["tvalid"],fire_mo)
            anom=a if a is not None else np.zeros((256,256),np.float32)
        x=np.concatenate([ae["x"].astype(np.float32), anom[None]],0)  # 129ch
        np.savez_compressed(op, x=x, y=ae["y"].astype(np.uint8), valid=ae["valid"].astype(np.uint8)); n+=1
print(f"완료: {n}타일 (anomaly=0 폴백 {skip}) → {OUT}/ (x=129ch [AE128|CF_anomaly])",flush=True)
# 간단 진단: 홀드아웃서 anomaly가 폴리곤 소각 픽셀서 더 큰가?
hold=[l.strip() for l in open(os.path.join(PROJ,"data","holdout_fixed.txt")) if l.strip()]
din=[];dout=[]
for fire in hold:
    for f in glob.glob(os.path.join(OUT,fire,"*.npz")):
        d=np.load(f); an=d["x"][128]; y=d["y"]>0; v=d["valid"]>0
        if (y&v).sum()>50: din.append(float(an[y&v].mean())); dout.append(float(an[(~y)&v].mean()))
if din: print(f"[진단] 홀드아웃 CF_anomaly: 폴리곤 소각내 평균 {np.mean(din):.4f} vs 소각밖 {np.mean(dout):.4f} (소각내가 커야 신호 유효)",flush=True)
