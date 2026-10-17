"""사전학습 데이터 빌더 — 일본/미동부 봄산불 S2 반사율 + clean-dNBR 라벨 (GEE).

계절매칭 사전학습(절대 char)용. 폴리곤 GT 부재 → 라벨 자작:
  화재일(FIRMS peak) 기준 pre/post를 타이트 bracket(같은 leaf-off phenophase)
  → dNBR 반전 없이 깨끗 → forest & 클러스터근접 제약 → burn 마스크.
학습 땐 Stage1 승리 config(절대 char 입력)로 이 라벨을 예측시켜 char→burn 매핑 학습.
라벨 QC(실burn 신호 요구)가 야끼 2차필터 겸함.

출력 = ko_refl와 동일포맷: x=(12,256,256) int16 반사율[pre6,post6], y=(256,256) uint8.
밴드순 [B02,B03,B04,B8A,B11,B12] (build_x와 일치).
사용: REGION=japan MAX_EVENTS=200 python3 jp_build_refl.py
"""
import os, sys, json, io, urllib.request, datetime as dt
import numpy as np, ee
ee.Initialize()
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
REGION=os.environ.get("REGION","japan")
FIRE_MONTHS={"japan":("03-01","05-31"),"eastus":("03-01","04-30"),"eu":("06-01","09-30")}[REGION]  # FIRMS 탐색월(지역 fire season)
SRC=os.path.join(PROJ,f"{'jp' if REGION=='japan' else REGION}_fires.json")
OUT=os.path.join(PROJ,f"{'jp' if REGION=='japan' else REGION}_refl"); os.makedirs(OUT,exist_ok=True)
MAX_EVENTS=int(os.environ.get("MAX_EVENTS","200"))
DEDUP_DEG=float(os.environ.get("DEDUP_DEG","0.05"))   # ~5km 격자 dedup(같은 화재 중복 제거)
DNBR_THR=float(os.environ.get("DNBR_THR","0.18"))     # clean-dNBR burn 임계
BANDS=["B2","B3","B4","B8A","B11","B12"]               # GEE 명 → 저장순 [B02,B03,B04,B8A,B11,B12]
FOREST=ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(10)
SCL_BAD=[0,1,3,8,9,10,11]

def s2_clear(geom, d0, d1):
    """[d0,d1] 최저구름 S2_SR_HARMONIZED (offset 이미 보정됨) + SCL 클라우드마스크."""
    col=ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(d0,d1)\
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",40)).sort("CLOUDY_PIXEL_PERCENTAGE")
    img=ee.Image(col.first())
    scl=img.select("SCL")
    good=scl.remap(SCL_BAD,[0]*len(SCL_BAD),1)         # bad→0, 그외→1
    return img.select(BANDS).updateMask(good), img

def firms_peak_date(pt, yr):
    """해당 위치 봄 FIRMS 최고강도 탐지일 → date. 없으면 None."""
    col=ee.ImageCollection("FIRMS").filterDate(f"{yr}-{FIRE_MONTHS[0]}",f"{yr}-{FIRE_MONTHS[1]}")\
        .filterBounds(pt.buffer(2000)).select("T21")
    n=col.size().getInfo()
    if not n: return None
    # 최고 T21 이미지의 시각
    def addmax(im): return im.set("mx",im.reduceRegion(ee.Reducer.max(),pt.buffer(2000),1000).get("T21"))
    top=ee.Image(col.map(addmax).sort("mx",False).first())
    ms=top.get("system:time_start").getInfo()
    return dt.date(1970,1,1)+dt.timedelta(milliseconds=ms)

def fetch_npy(img, region):
    """getDownloadURL(NPY) → HxWxC float32."""
    url=img.getDownloadURL({"region":region,"scale":20,"format":"NPY","bands":BANDS})
    raw=urllib.request.urlopen(url,timeout=120).read()
    a=np.load(io.BytesIO(raw))                          # structured array HxW, field=band
    return np.stack([a[b] for b in BANDS],-1).astype(np.float32)

def pad_crop(a,T=256):
    H,W=a.shape[:2]
    a=a[:T,:T]
    if a.shape[0]<T or a.shape[1]<T:
        pad=[(0,T-a.shape[0]),(0,T-a.shape[1])]+[(0,0)]*(a.ndim-2)
        a=np.pad(a,pad)
    return a

def nbr(x):  # x HxWx6, [B02,B03,B04,B8A,B11,B12]: NIR=idx3, SWIR2=idx5
    nir,sw=x[...,3],x[...,5]; return (nir-sw)/(nir+sw+1e-6)

POST_NBR_MAX=float(os.environ.get("POST_NBR_MAX","0.20"))  # 절대 char gate: 탄화면 post NBR 낮음
def make_label(pre,post,fmask,valid):
    """detrend(green-up 제거) 상대 dNBR spike + 절대 char(post NBR 낮음) 둘 다 요구."""
    dnbr=nbr(pre)-nbr(post)
    fg=valid&(fmask>0.5)
    med=np.median(dnbr[fg]) if fg.sum()>50 else 0.0     # forest 배경 green-up 추정
    rel=dnbr-med                                         # detrended
    postnbr=nbr(post)
    burn=(rel>DNBR_THR)&(postnbr<POST_NBR_MAX)&(fmask>0.5)&valid
    return burn.astype(np.uint8)

def build_one(ev, idx):
    lon,lat,yr=ev["lon"],ev["lat"],ev["year"]
    pt=ee.Geometry.Point([lon,lat])
    fdate=firms_peak_date(pt,yr)
    if fdate is None: return None,"FIRMS날짜없음"
    region=pt.buffer(2560).bounds()                    # ~256px @20m
    pre_i,_=s2_clear(region,str(fdate-dt.timedelta(days=45)),str(fdate-dt.timedelta(days=6)))
    post_i,_=s2_clear(region,str(fdate+dt.timedelta(days=6)),str(fdate+dt.timedelta(days=40)))
    try:
        pre=pad_crop(fetch_npy(pre_i,region)); post=pad_crop(fetch_npy(post_i,region))
        forest6=ee.Image.cat([FOREST]*6).rename(BANDS)          # 6밴드 복제(fetch_npy 재사용)
        fmask=pad_crop(fetch_npy(forest6,region))[...,0]         # forest 0/1
    except Exception as e:
        return None,f"fetch실패:{str(e)[:30]}"
    valid=(pre[...,0]>0)&(pre[...,3]>0)&(pre[...,5]>0)&(post[...,0]>0)&(post[...,3]>0)&(post[...,5]>0)
    burn=make_label(pre,post,fmask,valid)              # detrend + 절대 char gate
    bf=burn[valid].mean() if valid.sum()>0 else 0
    if valid.mean()<0.4: return None,f"valid부족 {valid.mean():.2f}"
    if not(0.005<=bf<=0.6): return None,f"burn비율 {bf:.3f}(야끼/무신호 배제)"
    # 저장: x=[pre6,post6] int16 반사율 + fmask(오프라인 relabel용)
    x=np.concatenate([pre,post],-1).transpose(2,0,1).astype(np.int16)  # (12,256,256)
    name=f"{REGION[:2]}f_{fdate.isoformat().replace('-','')}_{idx}"
    od=os.path.join(OUT,name); os.makedirs(od,exist_ok=True)
    np.savez_compressed(os.path.join(od,f"{name}_0_0.npz"),
        x=x, y=burn, valid=valid.astype(np.uint8), fmask=(fmask>0.5).astype(np.uint8))
    return bf,f"OK burn={bf*100:.1f}%"

def main():
    evs=json.load(open(SRC))
    # dedup ~5km 격자
    seen=set(); uniq=[]
    for e in evs:
        k=(round(e["lon"]/DEDUP_DEG),round(e["lat"]/DEDUP_DEG),e["year"])
        if k in seen: continue
        seen.add(k); uniq.append(e)
    print(f"[{REGION}] 원본 {len(evs)} → dedup {len(uniq)} → 최대 {MAX_EVENTS} 빌드",flush=True)
    import glob as _g
    ok=0
    for i,e in enumerate(uniq[:MAX_EVENTS]):
        if _g.glob(os.path.join(OUT,f"*_{i}",f"*_{i}_0_0.npz")):   # 이미 빌드됨 → skip(이어하기)
            ok+=1; continue
        try: bf,msg=build_one(e,i)
        except Exception as ex: bf,msg=None,f"예외:{str(ex)[:40]}"
        if bf is not None: ok+=1
        if i%10==0 or bf is not None: print(f"  [{i+1}/{min(MAX_EVENTS,len(uniq))}] {msg} (누적 {ok})",flush=True)
    ntile=len(__import__("glob").glob(os.path.join(OUT,"*","*.npz")))
    print(f"[{REGION}] 완성 {ok}화재 / 타일 {ntile} → {OUT}",flush=True)

if __name__=="__main__": main()
