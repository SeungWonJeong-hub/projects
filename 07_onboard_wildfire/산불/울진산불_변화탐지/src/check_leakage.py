"""누출 검사 — 평가에 쓰는 화재가 학습셋에 들어가 있지 않은가.

왜 필요한가: 학습 화재는 `kf_<날짜>_<면적>ha` 로만 이름이 붙어 **지명이 없다**. 평가 사이트는
지명·좌표로 관리된다. 이름으로는 대조가 안 되므로 **좌표로** 확인해야 한다.
실제로 안동풍천면은 산림청 폴리곤에 48.5ha 로 기록돼 `MIN_HA=50` 에 **1.5ha 차이로** 걸러졌다
— 임계값이 조금만 낮았으면 유일한 독립 자(NIFoS 안동)가 오염됐다. 우연에 기대지 않는다.

사용: python check_leakage.py [학습셋디렉터리 ...]   (기본 ko_refl_s50 ko_refl)
반환: 누출이 있으면 exit 1.
"""
import os, sys, glob
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
if not os.environ.get("PROJ_DATA"):                      # envs/geo 직접호출 함정
    os.environ["PROJ_DATA"]=os.path.join(sys.prefix,"share","proj")
from osgeo import ogr, osr
ogr.UseExceptions()

SHP=os.path.join(PROJ,"..","산불데이터","korea","mtfire","MTfire.shp")
SITES=["andong","uljin"]                                  # nifos_cache/<site>/ute_20m.npy
MARGIN_M=float(os.environ.get("LEAK_MARGIN_M","3000"))    # GT 범위 밖 여유

def main():
    dsets=sys.argv[1:] or ["ko_refl_s50","ko_refl"]
    src=osr.SpatialReference(); src.ImportFromEPSG(5186)   # MTfire = KGD2002 중부원점
    dst=osr.SpatialReference(); dst.ImportFromEPSG(32652)
    for s in (src,dst): s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(src,dst)
    trained={}
    for d in dsets:
        names={os.path.basename(x) for x in glob.glob(os.path.join(PROJ,d,"*")) if os.path.isdir(x)}
        trained[d]=names
        print(f"학습셋 {d}: {len(names)}화재")
    ds=ogr.Open(SHP); ly=ds.GetLayer()
    bad=0
    for site in SITES:
        u=os.path.join(PROJ,"nifos_cache",site,"ute_20m.npy")
        if not os.path.exists(u):
            print(f"[건너뜀] {site}: ute 캐시 없음 (nifos_eval.py prep 먼저)"); continue
        ute=np.load(u); cx0,cy0=(ute[0]+ute[2])/2,(ute[1]+ute[3])/2
        R=max(ute[2]-ute[0],ute[3]-ute[1])/2+MARGIN_M
        ly.ResetReading(); near=[]
        for f in ly:
            g=f.GetGeometryRef()
            if not g: continue
            gu=g.Clone(); gu.Transform(ct); e=gu.GetEnvelope()
            cx,cy=(e[0]+e[1])/2,(e[2]+e[3])/2
            if abs(cx-cx0)<=R and abs(cy-cy0)<=R:
                ha=gu.GetArea()/1e4
                near.append((f"kf_{str(f.GetField('OCCU_DATE')).replace('-','')[:8]}_{round(ha)}ha",
                             ha, np.hypot(cx-cx0,cy-cy0)/1000))
        hits=[(n,ha,km,d) for n,ha,km in near for d in dsets if n in trained[d]]
        print(f"\n[{site}] GT±{MARGIN_M/1000:.0f}km 안 MTfire {len(near)}건 · 누출 {len(hits)}건")
        for n,ha,km,d in hits:
            print(f"  ✗ {n} ({ha:.1f}ha, {km:.1f}km) 가 {d} 에 있다 — 이 사이트는 평가 자격이 없다")
        bad+=len(hits)
        # 아슬아슬한 것도 보고한다 — 임계값이 조금만 달랐으면 누출이었을 화재
        edge=[(n,ha,km) for n,ha,km in near if 40<=ha<60 and not any(n in trained[d] for d in dsets)]
        for n,ha,km in edge:
            print(f"  ! {n} ({ha:.1f}ha, {km:.1f}km) — MIN_HA 경계 근처다. 임계값을 낮추면 누출된다")
    print(f"\n{'누출 '+str(bad)+'건 — 평가 무효' if bad else '누출 없음 ✓'}")
    sys.exit(1 if bad else 0)

if __name__=="__main__": main()
