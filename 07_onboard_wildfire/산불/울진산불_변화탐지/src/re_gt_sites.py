"""RapidEye 5m NIFoS GT 7사이트 — 신원 확인 · 학습셋 누출 검사.

왜: 독립 자가 안동 1건뿐이라 판정이 전부 막혀 있다. `nifos_burn_gt/ext/re_*` 는
NIFoS 2022 작업물 중 **Rapideye 트랙**(5m·5밴드 Blue/Green/Red/RedEdge/NIR)이라
울진(S2 기반 → 순환)과 달리 **S2 와 독립**일 수 있는 후보다.

이 스크립트는 채점 전에 두 가지만 본다:
  ① 신원 — 각 GT 가 어느 화재인가 (산림청 이력 MTfire.shp 점과 공간·연도 대조)
  ② 누출 — 그 화재가 학습셋(`ko_refl*`)에 `kf_<날짜>_<면적>ha` 로 들어가 있는가

사용: conda run -n geo python re_gt_sites.py [--margin-m 2000]
"""
import os, sys, glob, json, argparse
if not os.environ.get("PROJ_DATA"):                      # envs/geo 직접호출 함정
    os.environ["PROJ_DATA"]=os.path.join(sys.prefix,"share","proj")
    os.environ.setdefault("GDAL_DATA",os.path.join(sys.prefix,"share","gdal"))
from osgeo import ogr, osr
ogr.UseExceptions(); osr.UseExceptions()

HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
GT_DIR=os.path.expanduser(os.environ.get("RE_GT_DIR","~/jetson_data/Steven/nifos_burn_gt/ext"))
MTFIRE=os.path.join(PROJ,"..","산불데이터","korea","mtfire","MTfire.shp")
TRAINSETS=["ko_refl","ko_refl_s50","ko_refl_10m_b08","ko_refl_10m_b08_t512",
           "ko_refl_10m_b08_t512_s50","ko_refl_10m_b08_t512_s50c"]

def trained_names():
    out={}
    for d in TRAINSETS:
        p=os.path.join(PROJ,d)
        if not os.path.isdir(p): continue
        for n in os.listdir(p):
            if n.startswith("kf_"): out.setdefault(n,[]).append(d)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--margin-m",type=float,default=3000.0)   # check_leakage.py 관례와 동일
    a=ap.parse_args()

    tr=trained_names()
    print(f"학습셋 화재 이름 {len(tr)}개 적재 ({', '.join(sorted({d for v in tr.values() for d in v}))})\n")

    utm=osr.SpatialReference(); utm.ImportFromEPSG(32652)
    kgd=osr.SpatialReference(); kgd.ImportFromEPSG(5186)
    for s in (utm,kgd): s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(kgd,utm)

    ds=ogr.Open(MTFIRE); ly=ds.GetLayer()
    pts=[]                                                # MTfire 는 폴리곤 — 형상 그대로 겹침을 본다
    for f in ly:
        g=f.GetGeometryRef()
        if g is None: continue
        g=g.Clone(); g.Transform(ct)
        pts.append((g,f.GetField("OCCU_DATE"),f.GetField("AR"),
                    f.GetField("ADRES") or f.GetField("ADDRESS1") or "", g.GetArea()/1e4))

    rows=[]
    for shp in sorted(glob.glob(f"{GT_DIR}/re_labels/*.shp")):
        code=os.path.basename(shp)[:-4]; year=code[:4]
        d=ogr.Open(shp); l=d.GetLayer()
        ha={}; env=None
        for f in l:
            g=f.GetGeometryRef(); c=f.GetField("Dam_class")
            ha[c]=ha.get(c,0.0)+g.GetArea()/1e4
            e=g.GetEnvelope()
            env=e if env is None else (min(env[0],e[0]),max(env[1],e[1]),min(env[2],e[2]),max(env[3],e[3]))
        m=a.margin_m
        box=(env[0]-m,env[1]+m,env[2]-m,env[3]+m)
        ring=ogr.Geometry(ogr.wkbLinearRing)
        for x,y in ((box[0],box[2]),(box[1],box[2]),(box[1],box[3]),(box[0],box[3]),(box[0],box[2])):
            ring.AddPoint_2D(x,y)
        poly=ogr.Geometry(ogr.wkbPolygon); poly.AddGeometry(ring)
        near=[p for p in pts if p[0].Intersects(poly)]
        near.sort(key=lambda p:-p[4])
        tot=sum(ha.values())
        print(f"■ {code}  GT {tot:.1f}ha (심 {ha.get(1,0):.1f} / 중 {ha.get(2,0):.1f} / 경 {ha.get(3,0):.1f})"
              f"  범위 {(env[1]-env[0])/1000:.1f}×{(env[3]-env[2])/1000:.1f}km")
        if not near: print("   산림청 이력 매칭 없음")
        # 누출은 절대 잘라내지 않는다 — 학습셋에 있는 것부터 전부, 그 다음 미학습 상위 몇 건
        def key_of(p): return f"kf_{p[1]}_{int(p[4])}ha"
        leaked=[p for p in near if key_of(p) in tr]
        clean=[p for p in near if key_of(p) not in tr][:5]
        hit=[]
        for g,date,ar,adr,poly_ha in leaked+clean:
            same_year = (date or "")[:4]==year
            key=f"kf_{date}_{int(poly_ha)}ha"           # 빌더 규칙(ko_build_refl.py:148)과 동일
            leak=tr.get(key)
            mark="★" if same_year else " "
            lk=f"  ⚠누출:{','.join(leak)}" if leak else ""
            print(f"   {mark} {date}  이력 {float(ar or 0):7.1f}ha · 폴리곤 {poly_ha:7.1f}ha  {adr[:34]:34} {key:22}{lk}")
            if same_year or leak: hit.append((date,float(ar or 0),poly_ha,adr,bool(leak),key))
        rows.append(dict(code=code,gt_ha=round(tot,1),
                         cls_ha={str(k):round(v,1) for k,v in sorted(ha.items())},
                         bbox_utm52n=[round(v,1) for v in env],
                         candidates=[dict(date=h[0],ar_ha=h[1],poly_ha=round(h[2],1),adres=h[3],
                                          leaked=h[4],key=h[5]) for h in hit]))
        print()
    out=os.path.join(PROJ,"data","re_gt_sites.json")
    os.makedirs(os.path.dirname(out),exist_ok=True)
    json.dump(rows,open(out,"w"),ensure_ascii=False,indent=1)
    print("기록:",out)

if __name__=="__main__": main()
