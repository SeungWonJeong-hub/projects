"""심각도 전이 검증 — 안동에서 학습해 **다른 사이트**에서 시험한다 (§7 남은 항목).

왜 필요한가: `sev_train.py` 는 안동 1개 사이트의 공간분리 CV 로 "RGB+NIR 학습(0.840)이
배포 dNBR 고정절단(0.707)보다 낫고, 배포 규칙은 중등급 면적을 −65% 틀린다"를 냈다.
공간분리 CV 는 **사이트 내** 일반화만 본다 — 배포 교체는 두 번째 독립 사이트가 있어야 한다.
울진은 S2 기반이라 순환이라 못 쓴다([[wildfire-uljin-gt-is-20m]]).

여기 쓰는 시험 사이트는 NIFoS **RapidEye 트랙**(5m·5밴드, S2 와 독립) GT 다.
※ 이 GT 는 소각 **범위**로는 불완전하다(화재의 일부만 라벨 — src/re_gt_completeness.py).
   그러나 라벨된 폴리곤 **안의 등급**은 온전하므로 심각도 전이 검증에는 그대로 쓸 수 있다.

사용: conda run -n ml python sev_transfer.py [--train andong] [--test gangneung goseong]
"""
import os, sys, argparse
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS","4")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import f1_score, confusion_matrix

P=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALE_LO=float(os.environ.get("SEV_SCALE_LO","0.7"))   # 기준 사이트 대비 등급별 dNBR 배율 허용폭
SCALE_HI=float(os.environ.get("SEV_SCALE_HI","1.4"))
GSD=20.0; PXA=GSD*GSD/1e4; SEED=0

def load(site, suf="_doy"):
    """(특징목록, 등급y, 마스크) — sev_train.py 와 같은 지표 정의."""
    from osgeo import gdal; gdal.UseExceptions()
    C=os.path.join(P,"nifos_cache",site+suf)
    x=np.load(os.path.join(C,"scene_20m.npz"))["x"].astype("float32")/1e4
    sev_p=os.path.join(C,"severity_20m.tif")
    if not os.path.exists(sev_p):                      # 안동은 기존 파일을 격자에 맞춰 얹는다
        # 다른 사이트에서 이 분기로 새면 **안동 라벨로 남의 사이트를 채점**하게 된다(조용한 오답).
        assert site=="andong", (f"{site}: severity_20m.tif 가 없다 — "
                                f"SITE={site} SCENE_PICK=doy python re_gt_sev20m.py 를 먼저 돌릴 것")
        g=gdal.Open(f"{P}/../nifos_burn_gt/andong/andong_severity_20m.tif")
        gt_=g.GetGeoTransform(); S=g.ReadAsArray()
        ute=np.load(os.path.join(C,"ute_20m.npy"))
        H,W=x.shape[1],x.shape[2]
        ox,oy=int(round((gt_[0]-ute[0])/GSD)),int(round((ute[3]-gt_[3])/GSD))
        S2=np.zeros((H,W),S.dtype)
        ys=slice(max(oy,0),min(oy+S.shape[0],H)); xs=slice(max(ox,0),min(ox+S.shape[1],W))
        S2[ys,xs]=S[max(-oy,0):max(-oy,0)+(ys.stop-ys.start),max(-ox,0):max(-ox,0)+(xs.stop-xs.start)]
    else:
        S2=gdal.Open(sev_p).ReadAsArray()
        H,W=min(S2.shape[0],x.shape[1]),min(S2.shape[1],x.shape[2])
        S2=S2[:H,:W]; x=x[:,:H,:W]
    pre,post=x[:6],x[6:]                               # [B02,B03,B04,B8A,B11,B12]
    idx=lambda a,b:(a-b)/(a+b+1e-6)
    dNBR  =idx(pre[3],pre[5])-idx(post[3],post[5])
    dNDVI =idx(pre[3],pre[2])-idx(post[3],post[2])
    dGNDVI=idx(pre[3],pre[1])-idx(post[3],post[1])
    dNIR  =pre[3]-post[3]
    dBlue =idx(pre[3],pre[0])-idx(post[3],post[0])
    SETS={"SWIR 포함 (dNBR+dNDVI)":[dNBR,dNDVI],
          "RGB+NIR 4지표":[dNDVI,dGNDVI,dNIR,dBlue],
          "RGB+NIR 4지표+원본반사율":[dNDVI,dGNDVI,dNIR,dBlue,*pre[:4],*post[:4]]}
    m=np.isin(S2,[1,2,3])
    return SETS,dNBR,S2[m].astype(int),m

def area_err(pred,true):
    return [(np.sum(pred==k)-np.sum(true==k))/max(np.sum(true==k),1)*100 for k in (1,2,3)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",default="andong")
    ap.add_argument("--test",nargs="+",default=["gangneung","goseong"])
    a=ap.parse_args()
    for s_ in [a.train]+list(a.test):
        ok,med,rat=gate(s_,ref=a.train)
        why=[]
        if not all(med.get(k,-9)>med.get(k+1,9) for k in (1,2) if k in med and k+1 in med):
            why.append("등급-신호 역전")
        if any(not (SCALE_LO<=v<=SCALE_HI) for k,v in rat.items() if k in (1,2)):
            why.append("눈금 불일치")
        tag="통과" if ok else "★탈락("+"·".join(why)+")"
        r=" ".join(f"{k}:{rat[k]:.2f}배" for k in sorted(rat)) if rat else "(기준)"
        print(f"[관문] {s_:10} dNBR 중앙 심 {med.get(1,float('nan')):.3f} / 중 {med.get(2,float('nan')):.3f} / "
              f"경 {med.get(3,float('nan')):.3f}  {r:28} → {tag}")
    print()
    S_tr,dnbr_tr,y_tr,m_tr=load(a.train)
    print(f"학습 {a.train}: 등급화소 {len(y_tr):,} "
          f"(심 {(y_tr==1).sum()*PXA:.0f} / 중 {(y_tr==2).sum()*PXA:.0f} / 경 {(y_tr==3).sum()*PXA:.0f} ha)\n")
    fits={}
    for name,feats in S_tr.items():
        X=np.stack([f[m_tr] for f in feats],1)
        fits[name]=HGB(max_iter=120,random_state=SEED).fit(X,y_tr)
    for site in a.test:
        S_te,dnbr_te,y_te,m_te=load(site)
        print(f"── 시험 {site}: 등급화소 {len(y_te):,} "
              f"(심 {(y_te==1).sum()*PXA:.0f} / 중 {(y_te==2).sum()*PXA:.0f} / 경 {(y_te==3).sum()*PXA:.0f} ha)")
        print(f"{'모델':<26}{'정확도':>9}{'macroF1':>9}   {'등급별 면적오차 %':>24}")
        d=dnbr_te[m_te]
        p=np.where(d>=0.44,1,np.where(d>=0.27,2,3))
        rows=[("배포 규칙 dNBR 절단",p)]
        for name,feats in S_te.items():
            X=np.stack([f[m_te] for f in feats],1)
            rows.append((name,fits[name].predict(X)))
        for name,p in rows:
            e=area_err(p,y_te)
            print(f"{name:<26}{(p==y_te).mean():>9.3f}{f1_score(y_te,p,average='macro'):>9.3f}   "
                  f"심 {e[0]:>+6.1f} 중 {e[1]:>+6.1f} 경 {e[2]:>+6.1f}")
        print(f"{'(최빈등급만 찍기)':<26}{max(np.bincount(y_te)[1:])/len(y_te):>9.3f}\n")


def gate(site, ref="andong"):
    """★심각도 GT 사전 관문 — 등급이 소각 강도 순서를 지키는가(심>중>경 dNBR 중앙값).

    왜: 고성(2019_74)에서 전이가 전 모델 붕괴(정확도 0.22-0.28 < 최빈 0.442)했는데, 원인은
    모델이 아니라 라벨이다 — 심 dNBR 0.706 < **중 0.744** 로 순서가 뒤집혀 있다. 즉 그 GT 의
    등급 축은 분광 소각강도와 다른 축이다(SWIR 을 써도 안 갈린다). 모델 출력을 안 쓰므로
    채점 **전에** 판별할 수 있다 — `cir_gt_profile.py` 가 소각 GT 에 하는 일과 같은 역할.
    ★두 조건을 본다 — 순서만으로는 부족하다(2026-09-09 인제로 확인):
      ①순서: 심>중>경 (고성 0.706<0.744, 양양 0.250<0.297 이 여기서 걸린다)
      ②눈금: 학습 사이트와 등급별 절대 수준이 비슷한가. 인제는 순서는 맞지만 심 dNBR 이
        안동의 0.51 배(0.439 vs 0.864)라 안동에서 배운 절단이 안 맞고, 전이가 최빈등급보다
        못했다(0.644 vs 0.713 — 심 면적오차 −98.7%, 심을 거의 안 찍는다).
        강릉만 심 1.05배·중 0.88배로 눈금이 맞고 유일하게 전이에 성공했다(0.743 vs 최빈 0.535).

    반환: (통과여부, 등급별 dNBR중앙값, 기준대비 배율). 통과 못 하면 심각도 자로 쓰지 않는다."""
    SETS,dnbr,y,m=load(site)
    d=dnbr[m]
    med={k:float(np.median(d[y==k])) for k in (1,2,3) if (y==k).sum()>0}
    order=all(med.get(k,-9)>med.get(k+1,9) for k in (1,2) if k in med and k+1 in med)
    ratio={}
    if ref and ref!=site:
        _,rmed,_=gate(ref,ref=None)
        ratio={k:med[k]/rmed[k] for k in med if k in rmed and rmed[k]!=0}
    scale=all(SCALE_LO<=v<=SCALE_HI for k,v in ratio.items() if k in (1,2)) if ratio else True
    return (order and scale),med,ratio

def profile(sites=("andong","gangneung","goseong")):
    """등급별 dNBR·dNDVI 분포 — 전이 실패가 '신호'가 아니라 '등급 척도' 차이인지 본다."""
    print(f"{'사이트':10} {'등급':4} {'화소':>7} {'면적ha':>8} {'dNBR중앙':>9} {'dNDVI중앙':>10} {'post NDVI':>10}")
    for s in sites:
        SETS,dnbr,y,m=load(s)
        C=os.path.join(P,"nifos_cache",s+"_doy")
        x=np.load(os.path.join(C,"scene_20m.npz"))["x"].astype("float32")/1e4
        post=x[6:][:, :m.shape[0], :m.shape[1]]
        pn=(post[3]-post[2])/(post[3]+post[2]+1e-6)
        dn=dnbr[m]; pnm=pn[m]
        for k,lab in ((1,"심"),(2,"중"),(3,"경")):
            i=y==k
            if i.sum()==0: continue
            print(f"{s:10} {lab:4} {int(i.sum()):7d} {i.sum()*PXA:8.1f} {np.median(dn[i]):9.3f} "
                  f"{'':10} {np.median(pnm[i]):10.3f}")
        print()

if __name__=="__main__": main()
