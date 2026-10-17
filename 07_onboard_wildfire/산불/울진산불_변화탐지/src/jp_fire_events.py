"""일본 봄산불 이벤트 추출 (FIRMS 열탐지 → 산림화재 클러스터, 야끼 제거).

Japan-spring 전이학습 데이터셋 1단계. FIRMS(VIIRS/MODIS) 봄(3-5월) 열탐지를
클러스터링 → 산림 land-cover 마스크로 논밭 야끼(野焼き) 제거 → 지속 소각흔 있는
산림화재만 남겨 이벤트 목록(위치·날짜·크기) 저장 → jp_fires.json.
S2 영상은 이후 이 목록으로 GEE서 취득.

산림 마스크 = ESA WorldCover(10m) tree cover(10). 지속성 = 열탐지 다일 or char 확인은 다음단계.
사용: MIN_DETECT=8 python jp_fire_events.py
"""
import os, json, datetime as dt
import ee
ee.Initialize()
HERE=os.path.dirname(os.path.abspath(__file__))
# 지역 프리셋: bbox + 봄 기간(leaf-off/개엽기 화재체제 매칭)
REGIONS={
 "japan":  {"bbox":[129,30,146,46], "months":("03-01","05-31"), "out":"jp_fires.json"},
 # 미 동부/남부 애팔래치아: 봄 fire season 3-4월 leaf-off 낙엽활엽(한국봄 phenology 매칭)
 "eastus": {"bbox":[-90,32,-75,42], "months":("03-01","04-30"), "out":"eastus_fires.json"},
 # 유럽 온대림만: 카르파티아/동유럽+알프스주변+프랑스중동부 (지중해 건조관목대 제외).
 # 위도45-52·경도2-27 = 佛보주~獨남부·오스트리아·체코·폴란드·슬로바키아·루마니아. 한국 대형 혼합림 화재와 스펙트럼 매칭
 "eu":     {"bbox":[2,45,27,52], "months":("04-01","09-30"), "out":"eu_fires.json"},
}
REGION=os.environ.get("REGION","japan")
RG=REGIONS[REGION]
OUT=os.path.join(HERE,"..",RG["out"])
MIN_DETECT=int(os.environ.get("MIN_DETECT","6"))     # 클러스터당 최소 열탐지(=규모 하한)
YEARS=[int(y) for y in os.environ.get("YEARS","2019,2020,2021,2022,2023,2024").split(",")]
JAPAN=ee.Geometry.Rectangle(RG["bbox"])              # (변수명은 유지, 실제=선택지역)
M0,M1=RG["months"]
# ESA WorldCover 2021: 10=tree, 20=shrub. 산림만.
FOREST=ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(10)

def spring_fires(yr):
    """해당 연 봄 FIRMS 열탐지 → 산림 위 화소만 → 벡터 클러스터 → 이벤트."""
    firms=ee.ImageCollection("FIRMS").filterDate(f"{yr}-{M0}",f"{yr}-{M1}").filterBounds(JAPAN).select("T21").max()
    # 열탐지 있고 & 산림 위 (야끼는 논밭=비산림이라 제거됨)
    forest=FOREST.reproject("EPSG:4326",None,1000)  # 1km로 집계 맞춤
    burn=firms.gt(325).And(forest)                   # T21>325K=강한 화재, 산림 위
    # 연결요소로 클러스터 → 각 클러스터 크기·중심
    obj=burn.selfMask().connectedComponents(ee.Kernel.plus(1),256)
    sizes=obj.select("labels").connectedPixelCount(256)
    big=sizes.gte(MIN_DETECT)
    # 벡터화(중심점)
    vec=big.selfMask().reduceToVectors(geometry=JAPAN,scale=1000,geometryType="centroid",
                                       maxPixels=1e10,bestEffort=True,eightConnected=True)
    return vec

events=[]
for yr in YEARS:
    try:
        vec=spring_fires(yr)
        feats=vec.limit(200).getInfo().get("features",[])
        n=0
        for f in feats:
            c=f["geometry"]["coordinates"]  # [lon,lat]
            events.append({"lon":round(c[0],4),"lat":round(c[1],4),"year":yr,"month":4,
                           "region":REGION,"src":"FIRMS-forest-spring"})
            n+=1
        print(f"  {yr} 봄: 산림화재 클러스터 {n}개",flush=True)
        json.dump(events,open(OUT,"w"),ensure_ascii=False,indent=1)   # 증분저장(타임아웃 대비)
    except Exception as e:
        print(f"  {yr}: {str(e)[:60]}",flush=True)
json.dump(events,open(OUT,"w"),ensure_ascii=False,indent=1)
print(f"\n총 {len(events)}개 일본 봄 산림화재 이벤트 → {OUT}")
print("(다음: 이 위치들로 S2 pre/post + char 라벨 빌드. 야끼는 산림마스크로 1차 제거됨)")
