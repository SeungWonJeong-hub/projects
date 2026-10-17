"""물리 proxy 무-GT 상대평가: 모델(AlphaEarth 0.677) vs 산림청 폴리곤, 누가 진짜 소각에 가까운가?

깨끗한 GT가 없으니 독립적 '물리 referee'를 만들어 둘의 일치도를 비교한다.
referee = 화재 후 다수 달 지속되는 char(SWIR B12 상승). 근거:
  - char(SWIR)는 봄 개엽 dNBR 반전을 우회(신선한 숯은 SWIR↑, 개엽은 SWIR 영향 적음)
  - 다시점 '지속성'은 진짜 소각의 독립 신호(모델의 bi-temporal 입력엔 없는 시간축)
※완전 독립은 아님(같은 광학 모달) → GT 아닌 '물리 referee'. 증거이지 증명 아님(정직).

판정: F1(model, proxy) vs F1(polygon, proxy).
  model이 proxy와 더 일치 → 모델이 폴리곤보다 물리 referee에 가까움 → 0.677은 라벨탓 과소측정.
+ 불일치 화소 재판: model_only(모델만 소각) proxy값 vs poly_only(폴리곤만) proxy값.

사용: python3 proxy_eval.py
"""
import os, sys, glob, json, numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
AE=os.path.join(PROJ,"ko_ae"); TS=os.path.join(PROJ,"ko_presto_ts")
HOLD=os.path.join(PROJ,"data","holdout_fixed.txt")
RUN=sys.argv[1] if len(sys.argv)>1 else os.path.join(PROJ,"runs","ko_ae_baseline_ha500")
CKPT=os.path.join(RUN,"ae_nal_unet.pt") if os.path.exists(os.path.join(RUN,"ae_nal_unet.pt")) else os.path.join(RUN,"ae_unet.pt")
CKJS=os.path.join(RUN,"results.json")
DEV="cuda" if torch.cuda.is_available() else "cpu"
B12=5  # ts 밴드순 [B02,B03,B04,B8A,B11,B12] → B12 index

import torch.nn as nn, torch.nn.functional as F
def gn(c): return nn.GroupNorm(min(8,c),c)
class Blk(nn.Module):
    def __init__(s,i,o): super().__init__(); s.c=nn.Sequential(nn.Conv2d(i,o,3,padding=1),gn(o),nn.GELU(),nn.Conv2d(o,o,3,padding=1),gn(o),nn.GELU())
    def forward(s,x): return s.c(x)
class UNet(nn.Module):
    def __init__(s,ic=128):
        super().__init__(); s.stem=nn.Conv2d(ic,64,1); s.d1=Blk(64,64);s.d2=Blk(64,128);s.d3=Blk(128,256);s.p=nn.MaxPool2d(2)
        s.u2=Blk(256+128,128);s.u1=Blk(128+64,64);s.o=nn.Conv2d(64,1,1)
    def up(s,x): return F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)
    def forward(s,x):
        x=s.stem(x); c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2))
        x=s.u2(torch.cat([s.up(c3),c2],1));x=s.u1(torch.cat([s.up(x),c1],1));return s.o(x)

def char_proxy(ts, months, tvalid, fire_mo):
    """화재 후 지속 char(SWIR 상승) 맵. pre=화재직전 유효월, post=화재후 최대3개 유효월 평균."""
    b12=ts[:,B12,:,:]  # (T,H,W)
    val=[i for i,(m,v) in enumerate(zip(months,tvalid)) if v==1]
    pre=[i for i in val if months[i]<fire_mo]; post=[i for i in val if months[i]>fire_mo]
    if not pre or not post: return None
    i_pre=pre[-1]                     # 화재 직전 유효월
    i_post=post[:3]                   # 화재 후 최대 3개월
    mode=os.environ.get("PROXY","mean")
    if mode=="peak":                  # 회복 전 peak char (최대 SWIR 상승)
        persist=np.max([b12[i]-b12[i_pre] for i in i_post],axis=0)
    elif mode=="first":               # 화재 직후 첫 유효월 (가장 신선)
        persist=b12[i_post[0]]-b12[i_pre]
    else:
        persist=np.mean([b12[i]-b12[i_pre] for i in i_post],axis=0)
    return persist

def f1(pred,ref,valid):
    p=(pred>0)&(valid>0); g=(ref>0)&(valid>0)
    tp=int((p&g).sum());fp=int((p&~g).sum());fn=int((~p&g).sum())
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return 2*pr*rc/(pr+rc+1e-9)

thr_json=json.load(open(CKJS))["thr"] if os.path.exists(CKJS) else 0.4
m=UNet(128).to(DEV); m.load_state_dict(torch.load(CKPT,map_location=DEV)); m.eval()
hold=[l.strip() for l in open(HOLD) if l.strip()]
rows=[]
for fire in hold:
    fire_mo=int(fire.split("_")[1][4:6])
    for ae_f in sorted(glob.glob(os.path.join(AE,fire,"*.npz"))):
        tile=os.path.basename(ae_f); ts_f=os.path.join(TS,fire,tile)
        if not os.path.exists(ts_f): continue
        ae=np.load(ae_f); tsd=np.load(ts_f)
        proxy=char_proxy(tsd["ts"],tsd["months"],tsd["tvalid"],fire_mo)
        if proxy is None: continue
        valid=ae["valid"].astype(np.uint8); poly=ae["y"].astype(np.uint8)
        with torch.no_grad():
            prob=torch.sigmoid(m(torch.from_numpy(ae["x"].astype(np.float32)).unsqueeze(0).to(DEV))).squeeze().cpu().numpy()
        pred=(prob>=thr_json).astype(np.uint8)
        # proxy 마스크: 유효화소 내 char 상위(폴리곤 소각비율에 맞춰 상대임계) — 여러 임계 평균
        vv=valid>0
        res={"tile":f"{fire}/{tile.replace('.npz','')}","season":"봄" if 3<=fire_mo<=5 else "여름" if 6<=fire_mo<=8 else "가을" if 9<=fire_mo<=11 else "겨울"}
        for q in [70,80,90]:  # char 상위 (100-q)% 를 proxy-소각으로
            t=np.percentile(proxy[vv],q)
            pmask=((proxy>=t)&vv).astype(np.uint8)
            res[f"model_vs_proxy_q{q}"]=round(f1(pred,pmask,valid),3)
            res[f"poly_vs_proxy_q{q}"]=round(f1(poly,pmask,valid),3)
        # 불일치 재판: 모델만 vs 폴리곤만, proxy(char) 평균값 (높을수록 진짜소각)
        mo=(pred>0)&(poly==0)&vv; po=(poly>0)&(pred==0)&vv
        res["model_only_proxy"]=round(float(proxy[mo].mean()),4) if mo.sum()>50 else None
        res["poly_only_proxy"]=round(float(proxy[po].mean()),4) if po.sum()>50 else None
        res["agreed_burn_proxy"]=round(float(proxy[(pred>0)&(poly>0)&vv].mean()),4) if ((pred>0)&(poly>0)&vv).sum()>50 else None
        rows.append(res)

print("=== 물리 proxy(char 지속성) 무-GT 상대평가 ===\n")
print("[모델 vs proxy] 이 값이 [폴리곤 vs proxy]보다 크면 → 모델이 물리 referee에 더 일치(=폴리곤보다 진짜소각에 가까움)\n")
def avg(k):
    v=[r[k] for r in rows if r.get(k) is not None]; return round(sum(v)/len(v),3) if v else None
for q in [70,80,90]:
    print(f"  char상위(q{q}): 모델vs проxy={avg(f'model_vs_proxy_q{q}')}  폴리곤vs proxy={avg(f'poly_vs_proxy_q{q}')}")
print(f"\n[불일치 재판] char 지속성 평균 (높을수록 진짜소각 증거):")
print(f"  모델만-소각(폴리곤 놓침): {avg('model_only_proxy')}  |  폴리곤만-소각(모델 제외): {avg('poly_only_proxy')}  |  합의-소각: {avg('agreed_burn_proxy')}")
print(f"\n계절별 모델vs폴리곤(q80):")
from collections import defaultdict
bs=defaultdict(list)
for r in rows: bs[r["season"]].append(r)
for se in ["봄","여름","가을","겨울"]:
    if se in bs:
        rr=bs[se]; mv=[x["model_vs_proxy_q80"] for x in rr]; pv=[x["poly_vs_proxy_q80"] for x in rr]
        print(f"  {se}: 모델 {round(sum(mv)/len(mv),3)} vs 폴리곤 {round(sum(pv)/len(pv),3)} ({len(rr)}타일)")
json.dump(rows,open(os.path.join(PROJ,"runs","proxy_eval.json"),"w"),ensure_ascii=False,indent=2)
print("\n해석: 모델vs проxy > 폴리곤vs проxy AND 모델만-소각 proxy > 폴리곤만-소각 proxy →")
print("      모델이 폴리곤보다 진짜 소각에 가까움 = 0.677은 라벨탓 과소측정(진짜F1 더 높음). ★단 proxy도 노이즈, 증거이지 증명 아님.")
