"""NIFoS 산불피해강도등급(0.3m 실측) 기준 ET-02 10m vs 20m 채점.

왜 이 평가가 필요한가:
  산림청 폴리곤은 경계 ±20-40m·under-label이라 10m 모델이 "더 잡은 것"이 진짜인지
  오탐인지 구분할 수 없다. NIFoS GT는 0.3m 실측(안동 소각 2,020ha ≈ 공식 1,944ha)이라
  10m/20m 어느 격자로 내려도 라벨 자체가 병목이 되지 않는다.

대상: 안동풍천면 2020-04-24. MIN_HA 필터로 학습셋에서 빠진 **미학습 화재** = 독립 평가.

절차: GT extent → 같은 STAC 파이프라인으로 pre/post 취득(10m·20m) → 모델 추론 →
      GT를 각 격자로 면적비율 리샘플(≥0.5=소각) → 엄격 F1 + 미터고정 경계허용 F1.

env가 갈려 있어(geo=GDAL / torch=PyTorch) 2단 실행:
  1) 준비:  GSD=10 <geo>/python nifos_eval.py prep          # GT 리샘플 + 씬 취득 → 캐시
  2) 채점:  GSD=10 TILE=512 <torch>/python nifos_eval.py runs/et02_10mT512_unet_s1
  3) 앙상블: GSD=10 TILE=512 <torch>/python nifos_eval.py --ens "runs/et02_10mT512_*"
             (배포 스펙이 9모델 앙상블이므로 단일 평균이 아니라 이 값이 진짜 성능)
"""
import os, sys, json, glob
GSD=float(os.environ.get("GSD","20"))
os.environ.setdefault("NIR","B08" if GSD==10 else "B8A")
os.environ["GSD"]=str(int(GSD)); os.environ["NO_OFFSET"]="1"
import numpy as np, datetime as dt
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
# 평가 대상 화재 — SITE env로 교체(GT 래스터 + 화재일). 값 1,2,3 = 심/중/경 = 소각.
_NB=os.path.join(PROJ,"..","nifos_burn_gt")
# ★GT 원본 해상도를 같이 적는다 — 2026-09-09 에 울진 GT 를 0.3m 로 착각해 10m 판정 근거로
#   썼다. 울진은 **원본이 20m** 라 10m 격자로 리샘플해도 10m 정보가 없고, 그 격자로 10m 모델을
#   채점하면 구조적으로 불리하다. 해상도 판정에는 native <= 평가 GSD 인 사이트만 쓸 것.
# ★ RapidEye 트랙(`re/*_severity_1m.tif`) — NIFoS 2022 작업물 중 **Rapideye(5m·5밴드) track**.
#   울진이 Sentinel2 track 이라 순환이었던 것과 달리 이쪽은 S2 와 독립이다(폴리곤이 5/10/20m
#   어느 격자에도 정렬돼 있지 않은 자유형 벡터 = S2 픽셀을 잘라 만든 게 아니다).
#   누출은 화재별로 다르다 — `src/re_gt_sites.py` 가 산림청 폴리곤과 ±3km 겹침으로 검사한다:
#     gangneung  누출 0 (**모든 학습셋에서 미학습** → 유일하게 무제한 사용 가능)
#     goseong    s50 계열만 누출(kf_20170423_484ha 속초 설악동) → 배포 20m(기존 et02+기본풀) 전용
#     yangyang   s50 계열만 누출(kf_20180419_309ha = **같은 화재**) → 배포 20m 전용
#     inje       전 학습셋 누출(kf_20190404_1053ha 등) → **채점 금지**(참고용으로만 등재)
#   2016년 3건(2016_25·30·70)은 earth-search 에 S2 씬이 0건이라 등재하지 않는다.
_RE=lambda c: os.path.join(_NB,"re",f"{c}_severity_1m.tif")
SITES={"andong":(os.path.join(_NB,"andong","안동산불피해등급.8e447d.img"),dt.date(2020,4,24)),
       "uljin": (os.path.join(_NB,"ext","Ulgin_severity.tif"),          dt.date(2022,3,4)),
       "gangneung":(_RE("2019_63"),  dt.date(2019,4,4)),   # 강릉 옥계  360.8ha  누출 0
       "goseong":  (_RE("2019_74"),  dt.date(2019,4,4)),   # 고성 토성  465.5ha  s50 오염
       "yangyang": (_RE("2018_33"),  dt.date(2018,4,19)),  # 양양 현남   12.9ha  s50 오염
       "inje":     (_RE("2019_71_1"),dt.date(2019,4,4))}   # 인제 남면   67.4ha  ★누출-채점금지
LEAKED_BASES={"gangneung":set(),"goseong":{"s50"},"yangyang":{"s50"},"inje":{"s50","base"},
              "andong":set(),"uljin":set()}
SITE=os.environ.get("SITE","andong")
GT_IMG,FDATE=SITES[SITE]
if "base" in LEAKED_BASES.get(SITE,set()) and os.environ.get("ALLOW_LEAKED")!="1":
    raise SystemExit(f"[중단] {SITE} 는 모든 학습셋에 누출된 화재다 — 홀드아웃이 아니다. "
                     f"근거: src/re_gt_sites.py. 그래도 보려면 ALLOW_LEAKED=1.")

def warn_leak(runs):
    """s50 오염 사이트를 s50 베이스 가중치로 채점하면 조용히 부풀려진다 — 이름으로 잡는다."""
    if "s50" not in LEAKED_BASES.get(SITE,set()): return
    if any("s50" in str(r) for r in (runs if isinstance(runs,(list,tuple)) else [runs])):
        print(f"[경고] {SITE} 는 s50 학습셋에 누출된 화재인데 s50 베이스 가중치로 채점하고 있다. "
              f"**이 숫자를 홀드아웃 성능으로 인용하지 말 것.**", flush=True)
# 캐시는 NIR 밴드로도 갈라야 한다 — 같은 20m라도 B8A(865nm)와 B08(842nm)은 다른 입력이다.
# 기본 밴드면 접미사 없음(기존 캐시 그대로 재사용), 벗어나면 _b08 등으로 분리.
_NIR_DEF="B08" if GSD==10 else "B8A"
_NIR_CUR=os.environ["NIR"]
# ★씬 선택 규칙 — 2026-09-09 신설. cloud(기존)=창 안에서 최저구름 1장씩 독립 선택.
#   그 규칙은 pre/post 의 **DOY 가 녹화(green-up)를 사이에 두고 갈릴 수 있다.** 강릉(2019-04-04)에서
#   pre 2018-03-26(DOY85 잎 전) · post 2019-05-10(DOY130 만엽)이 뽑혀 배경 전체가 dNDVI −0.251 로
#   "녹화"로 보였고, 모델이 GT 361ha 대비 1,316ha 를 예측했다(배경 확률중앙 0.405).
#   안동은 우연히 pre DOY123 · post DOY160 둘 다 녹화 후라 배경 −0.019 였다 — 규칙이 아니라 운이었다.
#   doy = 저구름 후보들 중 |ΔDOY| 최소 쌍을 고른다(동률이면 구름 적은 쪽).
SCENE_PICK=os.environ.get("SCENE_PICK","cloud")
assert SCENE_PICK in ("cloud","doy","fixed"), SCENE_PICK
# fixed = PRE_DATE/POST_DATE 를 그대로 쓴다(배경 수준을 의도적으로 흔드는 보정 실험용 —
# src/qc_calibrate.py). 캐시는 날짜로 갈린다.
PRE_DATE=os.environ.get("PRE_DATE"); POST_DATE=os.environ.get("POST_DATE")
if SCENE_PICK=="fixed": assert PRE_DATE and POST_DATE, "SCENE_PICK=fixed 는 PRE_DATE/POST_DATE 가 필요"
CACHE=os.path.join(PROJ,"nifos_cache",SITE+("" if _NIR_CUR==_NIR_DEF else "_"+_NIR_CUR.lower())
                   +("" if SCENE_PICK=="cloud" else
                     "_doy" if SCENE_PICK=="doy" else
                     f"_p{PRE_DATE.replace('-','')}_q{POST_DATE.replace('-','')}"))
os.makedirs(CACHE,exist_ok=True)
TILE=int(os.environ.get("TILE","256"))   # 모델 학습 타일과 맞춘다(문맥 시야 일치)
# CONFIG=rgbnir(기본, 8ch pre|post) | post_rgbnir(4ch post 만 — pre 없는 온보드 운용).
# smp_cd.build_x / step4b.to_cfg 와 같은 절단이어야 한다.
CONFIG=os.environ.get("CONFIG","rgbnir")
IN_CH={"rgbnir":8,"post_rgbnir":4,"bitemporal":12}[CONFIG]

GT_NATIVE_M=None      # GT 원본 화소크기(m) — 기록·경고용

def gt_grid():
    """NIFoS GT를 GSD 격자로 면적비율 리샘플 → ≥0.5면 소각(1). bounds도 반환.
    GT 원본이 평가 격자보다 성기면 경고한다(그 사이트로는 더 고운 격자를 채점할 수 없다)."""
    global GT_NATIVE_M
    from osgeo import gdal; gdal.UseExceptions()
    d=gdal.Open(GT_IMG); g=d.GetGeoTransform()
    GT_NATIVE_M=abs(g[1])
    if GT_NATIVE_M > GSD + 1e-6:
        print(f"[경고] {SITE} GT 원본이 {GT_NATIVE_M:g}m 인데 {GSD:g}m 격자로 채점한다 — "
              f"업샘플이라 {GSD:g}m 정보가 없다. **해상도 비교 근거로 쓰지 말 것.**", flush=True)
    x0,y1=g[0],g[3]; x1=x0+g[1]*d.RasterXSize; y0=y1+g[5]*d.RasterYSize
    # GSD 배수로 스냅 (격자 정렬 = 10m/20m 비교의 전제)
    ute=[np.floor(x0/GSD)*GSD, np.floor(y0/GSD)*GSD, np.ceil(x1/GSD)*GSD, np.ceil(y1/GSD)*GSD]
    out=os.path.join(CACHE,f"gt_{int(GSD)}m.tif")
    if not os.path.exists(out):
        # 이진화(1,2,3=소각 / 그외=비소각) 후 average → 셀 내 소각 면적비율
        tmp=os.path.join(CACHE,"gt_bin.tif")
        if not os.path.exists(tmp):
            a=d.ReadAsArray(); b=np.isin(a,[1,2,3]).astype(np.float32)
            drv=gdal.GetDriverByName("GTiff")
            t=drv.Create(tmp,d.RasterXSize,d.RasterYSize,1,gdal.GDT_Float32,
                         options=["COMPRESS=DEFLATE","TILED=YES","BIGTIFF=YES"])
            t.SetGeoTransform(g); t.SetProjection(d.GetProjection())
            t.GetRasterBand(1).WriteArray(b); t=None; del a,b
        gdal.Warp(out,tmp,xRes=GSD,yRes=GSD,outputBounds=ute,resampleAlg="average",
                  format="GTiff",creationOptions=["COMPRESS=DEFLATE"])
    frac=gdal.Open(out).ReadAsArray()
    gt=(frac>=0.5).astype(np.uint8)
    np.save(os.path.join(CACHE,f"gt_{int(GSD)}m.npy"),gt)
    np.save(os.path.join(CACHE,f"ute_{int(GSD)}m.npy"),np.array(ute))
    return gt, ute

def _ddoy(a,b):
    """연도를 무시한 DOY 거리(연말↔연초 순환 고려)."""
    d=abs(a.timetuple().tm_yday-b.timetuple().tm_yday)
    return min(d,365-d)

def _doy_pair(K,bb,pw,qw,cloud=30.0):
    """저구름 후보들 중 |ΔDOY| 최소 pre/post 쌍. 동률이면 구름 합이 작은 쪽."""
    def cand(w):
        it=K.CAT.search(collections=["sentinel-2-l2a"],bbox=bb,datetime=K.rfc(*w),
                        query={"eo:cloud_cover":{"lt":cloud}},limit=100).item_collection()
        best={}
        for i in it:
            d=i.datetime.date(); c=i.properties.get("eo:cloud_cover",99)
            best[d]=min(best.get(d,99),c)
        return sorted(best.items())
    P,Q=cand(pw),cand(qw)
    assert P and Q, "DOY 정합용 후보 없음"
    return min(((p,q) for p,_ in P for q,_ in Q),
               key=lambda pq:(_ddoy(*pq), dict(P)[pq[0]]+dict(Q)[pq[1]]))

def scene(ute):
    """GT extent에 대해 pre(전년 동시기)/post(화재후) 반사율 취득 — 학습때와 동일 로직."""
    npz=os.path.join(CACHE,f"scene_{int(GSD)}m.npz")
    if os.path.exists(npz):
        z=np.load(npz); return z["x"]
    _av=sys.argv; sys.argv=[_av[0]]      # ko_build_refl가 argv[1]을 N_FIRES로 파싱함
    import ko_build_refl as K            # GDAL 의존 — 준비 단계에서만 import
    sys.argv=_av
    lo,la=K.osr.SpatialReference(),K.osr.SpatialReference()
    la.ImportFromEPSG(4326); la.SetAxisMappingStrategy(K.osr.OAMS_TRADITIONAL_GIS_ORDER)
    lo.ImportFromEPSG(K.UTM); lo.SetAxisMappingStrategy(K.osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=K.osr.CoordinateTransformation(lo,la)
    a=ct.TransformPoint(ute[0],ute[1]); b=ct.TransformPoint(ute[2],ute[3])
    bb=[min(a[0],b[0])-0.02,min(a[1],b[1])-0.02,max(a[0],b[0])+0.02,max(a[1],b[1])+0.02]
    pc=FDATE.replace(year=FDATE.year-1)
    pw=(pc-dt.timedelta(days=30), pc+dt.timedelta(days=45))
    qw=(FDATE+dt.timedelta(days=15), FDATE+dt.timedelta(days=100))
    if SCENE_PICK=="fixed":
        pd,qd=(dt.date.fromisoformat(PRE_DATE),dt.date.fromisoformat(POST_DATE))
        print(f"[씬] 지정 pre={pd}(DOY{pd.timetuple().tm_yday}) post={qd}"
              f"(DOY{qd.timetuple().tm_yday}) ΔDOY={_ddoy(pd,qd)}",flush=True)
        pre=K.stac_refl(bb,K.rfc(pd-dt.timedelta(days=1),pd+dt.timedelta(days=1)),ute)
        post=K.stac_refl(bb,K.rfc(qd-dt.timedelta(days=1),qd+dt.timedelta(days=1)),ute)
        np.save(os.path.join(CACHE,f"scene_dates_{int(GSD)}m.npy"),
                np.array([pd.isoformat(),qd.isoformat()]))
    elif SCENE_PICK=="doy":
        pd,qd=_doy_pair(K,bb,pw,qw)
        print(f"[씬] DOY정합 pre={pd}(DOY{pd.timetuple().tm_yday}) "
              f"post={qd}(DOY{qd.timetuple().tm_yday}) ΔDOY={_ddoy(pd,qd)}",flush=True)
        pre=K.stac_refl(bb,K.rfc(pd-dt.timedelta(days=1),pd+dt.timedelta(days=1)),ute)
        post=K.stac_refl(bb,K.rfc(qd-dt.timedelta(days=1),qd+dt.timedelta(days=1)),ute)
        np.save(os.path.join(CACHE,f"scene_dates_{int(GSD)}m.npy"),
                np.array([pd.isoformat(),qd.isoformat()]))
    else:
        pre=K.stac_refl(bb,K.rfc(*pw),ute)
        post=K.stac_refl(bb,K.rfc(*qw),ute)
    assert pre and post, "장면 취득 실패"
    x=np.stack([pre[k] for k in K.ORDER]+[post[k] for k in K.ORDER],0).astype(np.int16)
    np.savez_compressed(npz,x=x); return x

_DEC={"unet":"unet","fpn":"fpn","deeplabv3plus":"deeplabv3plus"}

def _load(run):
    """run = 학습 산출 디렉터리(results.json+*.pt) 또는 개별 .pt 경로.
    배포 앙상블 runs/cir_reh_ens/*.pt 는 results.json이 없어 파일명에서 디코더를 읽고
    thr는 웹앱 스펙(wildfire_cd.json)의 0.2를 쓴다."""
    import torch, segmentation_models_pytorch as smp
    A={"unet":smp.Unet,"fpn":smp.FPN,"deeplabv3plus":smp.DeepLabV3Plus}
    dev="cuda" if torch.cuda.is_available() else "cpu"
    if run.endswith(".pt"):
        base=os.path.basename(run)
        dname=next((k for k in _DEC if f"_{k}_" in base), None)
        assert dname, f"파일명에서 디코더 판별 실패: {base}"
        r={"decoder":dname,"encoder":"resnet34","thr":0.2}
        ckpt=run
    else:
        r=json.load(open(os.path.join(run,"results.json")))
        ckpt=glob.glob(os.path.join(run,"*.pt"))[0]
    m=A[r["decoder"]](r.get("encoder","resnet34"),encoder_weights=None,in_channels=IN_CH,classes=1).to(dev).eval()
    m.load_state_dict(torch.load(ckpt,map_location=dev))
    return m,r,dev

def infer(x, runs):
    """확률맵 생성. runs가 여러 개면 확률 평균 앙상블(배포 engine과 동일 방식)."""
    import torch
    if isinstance(runs,str): runs=[runs]
    models=[];rs=[]
    for run in runs:
        m,r,dev=_load(run); models.append(m); rs.append(r)
    # 채널: [B02,B03,B04,NIR,B11,B12] x (pre,post) → pre 0..5 / post 6..11
    x8=(x if CONFIG=="bitemporal"
        else np.concatenate([x[[0,1,2,3]],x[[6,7,8,9]]],0) if CONFIG=="rgbnir"
        else x[[6,7,8,9]]).astype(np.float32)/10000.0
    _,H,W=x8.shape
    # 장면이 타일보다 작을 수 있다(20m 안동 = 249x532 < 256). 반사패딩 후 추론하고 잘라낸다
    # — 미달 타일을 건너뛰면 그 영역이 통째로 미추론(=예측 0)이 된다.
    ph,pw=max(TILE-H,0),max(TILE-W,0)
    if ph or pw:
        x8=np.pad(x8,((0,0),(0,ph),(0,pw)),mode="reflect")
    Hp,Wp=x8.shape[1],x8.shape[2]
    prob=np.zeros((Hp,Wp),np.float32); cnt=np.zeros((Hp,Wp),np.float32)
    rows_=sorted(set(list(range(0,Hp-TILE+1,TILE))+[Hp-TILE]))
    cols_=sorted(set(list(range(0,Wp-TILE+1,TILE))+[Wp-TILE]))
    with torch.no_grad():
        for r0 in rows_:
            for c0 in cols_:
                t=torch.from_numpy(x8[:,r0:r0+TILE,c0:c0+TILE]).unsqueeze(0).to(dev)
                o=np.mean([torch.sigmoid(m(t)).squeeze().cpu().numpy() for m in models],axis=0)
                prob[r0:r0+TILE,c0:c0+TILE]+=o; cnt[r0:r0+TILE,c0:c0+TILE]+=1
    return (prob/np.maximum(cnt,1))[:H,:W], float(np.mean([r.get("thr",0.5) for r in rs]))

def f1(pred,gt):
    tp=int((pred&gt).sum()); fp=int((pred&~gt).sum()); fn=int((~pred&gt).sum())
    p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); return 2*p*r/(p+r+1e-9),p,r

def tol_f1(pred,gt,tau_m):
    from scipy.ndimage import binary_dilation
    t=int(round(tau_m/GSD))
    gd=binary_dilation(gt,iterations=t) if t else gt
    pd=binary_dilation(pred,iterations=t) if t else pred
    p=int((pred&gd).sum())/(int(pred.sum())+1e-9)
    r=int((gt&pd).sum())/(int(gt.sum())+1e-9)
    return 2*p*r/(p+r+1e-9),p,r


# ── 평가 마스크 ────────────────────────────────────────────────────────────────
# 왜: 강릉(2019_63) GT 는 화재의 **일부만** 라벨돼 있다 — 북쪽에서 라벨이 직선으로 잘리고
#     그 너머의 탄 땅(dNBR 0.334, post NDVI 0.316)이 GT 에 없다. 전체 격자로 채점하면
#     그 미라벨 소각지가 전부 FP 로 잡혀 **정밀도·면적이 구조적으로 불리**하다(안동에는 없는 문제).
#     → 라벨이 신뢰되는 영역 = GT 볼록껍질을 안쪽으로 EROD E_M 만큼 깎은 곳에서만 채점한다.
#     리콜은 마스크와 무관하게 유효하므로(GT ⊂ 진실) 둘 다 보고한다.
EVAL_MASK=os.environ.get("EVAL_MASK","full")       # full(기존) | hull
HULL_ERODE_M=float(os.environ.get("HULL_ERODE_M","200"))

def eval_mask(gt):
    if EVAL_MASK=="full": return np.ones_like(gt,bool)
    import cv2
    ys,xs=np.nonzero(gt)
    pts=np.stack([xs,ys],1).astype(np.int32)
    hull=cv2.convexHull(pts)
    m=np.zeros(gt.shape,np.uint8); cv2.fillConvexPoly(m,hull,1)
    k=int(round(HULL_ERODE_M/GSD))
    if k>0:
        e=cv2.erode(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*k+1,2*k+1)))
        # 작은 사이트(양양 GT 12.9ha)는 200m 침식이면 통째로 사라진다 — 조용히 0점이 되지 않게 한다
        if (e.astype(bool)&gt).sum()==0:
            print(f"[경고] hull 침식 {HULL_ERODE_M:g}m 이면 GT 가 남지 않는다 — 침식 없이 진행한다.",flush=True)
        else:
            m=e
    return m.astype(bool)

def main():
    run=sys.argv[1]
    ens=(run=="--ens")
    if ens: run=sys.argv[2]
    if run=="prep":                                   # 1단: geo env (GDAL)
        gt,ute=gt_grid(); x=scene(ute)
        print(f"준비 완료 GSD={GSD:.0f}m | GT {gt.shape} 소각 {gt.sum()*GSD*GSD/1e4:.0f}ha | 씬 {x.shape}")
        return
    if ens:
        cand=sorted(glob.glob(os.path.join(PROJ,run)))
        runs=[d for d in cand if d.endswith(".pt") or os.path.exists(os.path.join(d,"results.json"))]
        assert runs, f"앙상블 대상 없음: {run}"
        tag=f"ENS({len(runs)}) {os.path.basename(run)}"
    else:
        runs=[run if os.path.isabs(run) else os.path.join(PROJ,run)]; tag=os.path.basename(runs[0])
    warn_leak(runs)
    gt=np.load(os.path.join(CACHE,f"gt_{int(GSD)}m.npy"))   # 2단: torch env (GDAL 불필요)
    x=np.load(os.path.join(CACHE,f"scene_{int(GSD)}m.npz"))["x"]
    H=min(gt.shape[0],x.shape[1]); W=min(gt.shape[1],x.shape[2])
    gt=gt[:H,:W].astype(bool); x=x[:,:H,:W]
    prob,thr=infer(x,runs)
    prob=prob[:H,:W]
    M=eval_mask(gt)
    if EVAL_MASK!="full":
        print(f"[마스크] {EVAL_MASK} erode={HULL_ERODE_M:g}m → 채점 화소 "
              f"{M.sum()*GSD*GSD/1e4:.0f}ha / 전체 {M.size*GSD*GSD/1e4:.0f}ha "
              f"(GT 의 {100*(gt&M).sum()/max(gt.sum(),1):.1f}% 포함)",flush=True)
        gt=gt&M; prob=np.where(M,prob,0.0)
    # 저장된 thr + 최적 thr 둘 다 보고(라벨 도메인이 달라 thr 이식성이 낮음)
    best=max(np.arange(0.05,0.95,0.025),key=lambda t:f1(prob>=t,gt)[0])
    if GT_NATIVE_M is None:                      # 캐시 재사용 경로에서도 원본 해상도를 읽는다
        try:
            from osgeo import gdal; gdal.UseExceptions()
            GT_NATIVE_M_=abs(gdal.Open(GT_IMG).GetGeoTransform()[1])
        except Exception:
            GT_NATIVE_M_=None
    else:
        GT_NATIVE_M_=GT_NATIVE_M
    if GT_NATIVE_M_ and GT_NATIVE_M_ > GSD + 1e-6:
        print(f"[경고] GT 원본 {GT_NATIVE_M_:g}m > 평가 격자 {GSD:g}m — 해상도 비교 근거로 쓰지 말 것.", flush=True)
    res={"run":tag,"site":SITE,"gsd_m":GSD,"tile":TILE,"config":CONFIG,
         "gt_native_m":GT_NATIVE_M_,
         # ★가중치 식별자. 예전엔 tag 가 전부 "ENS(9) *.pt" 라 배포·확장풀·공정베이스
         #   앙상블의 숫자가 뒤섞였다(2026-09-09 에 실제로 문서가 어긋났다).
         "weights":[os.path.relpath(r,PROJ) if not os.path.isabs(r) else r for r in runs],
         "gt_burn_ha":round(float(gt.sum())*GSD*GSD/1e4,1),
         "grid":[int(H),int(W)]}
    for tag,t in (("saved_thr",thr),("best_thr",float(best))):
        pred=prob>=t
        s,p,r=f1(pred,gt)
        res[tag]={"thr":round(float(t),3),"f1":round(s,4),"p":round(p,3),"r":round(r,3),
                  "pred_ha":round(float(pred.sum())*GSD*GSD/1e4,1)}
        for tm in (20.0,40.0):
            s2,_,_=tol_f1(pred,gt,tm); res[tag][f"f1_tau{int(tm)}m"]=round(s2,4)
    print(json.dumps(res,ensure_ascii=False,indent=2))
    out=os.path.join(PROJ,"runs","nifos_eval.jsonl")
    with open(out,"a") as f: f.write(json.dumps(res,ensure_ascii=False)+"\n")

if __name__=="__main__": main()
