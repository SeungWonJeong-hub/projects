#!/usr/bin/env python3
"""다중시기 실험 2단계 — 젯슨에서 실행. 화재 사건 범위(ute) 만 잘라 작은 4밴드 GeoTIFF 로 저장한다.

입력: ~/s2_mt_events.json (B200 의 ko_mt/events.json 을 복사한 것), ~/Desktop/sentinel2_data/<장면>/B02,B03,B04,B08.tif
출력: ~/Desktop/s2_clips/<화재>/<장면id>.tif  (밴드 B02,B03,B04,B08, UTM 52N, 20 m, ute 격자 = ko_build_refl 과 동일)
      ~/Desktop/s2_clips/<화재>/clips.json  (장면별 날짜, 구름, 파일)
장면당 약 265 x 265 화소라 파일 하나가 수백 KB. 다시 실행하면 이미 있는 것은 건너뛴다.
사용: python3 ~/mt_jetson_clip.py [화재이름 ...]   (이름 없으면 pre, post 둘 다 있는 화재 전부)
"""
import os, sys, json, time
from osgeo import gdal
gdal.UseExceptions()
SRC=os.path.expanduser("~/Desktop/sentinel2_data"); OUT=os.path.expanduser("~/Desktop/s2_clips")
EV=json.load(open(os.path.expanduser("~/s2_mt_events.json")))
UTM=EV["utm"]; BANDS=["B02","B03","B04","B08"]
IDX={s["id"]:s for s in json.load(open(os.path.expanduser("~/s2_scenes_index.json")))}

def clip(sid, ute, dst):
    srcs=[os.path.join(SRC,sid,b+".tif") for b in BANDS]
    if not all(os.path.exists(p) for p in srcs): return "밴드없음"
    tmp=dst+".part"
    vrt=gdal.BuildVRT("", srcs, separate=True)
    gdal.Warp(tmp, vrt, dstSRS=f"EPSG:{UTM}", xRes=20, yRes=20, outputBounds=ute, resampleAlg="near",
              format="GTiff", srcNodata=0, dstNodata=0, creationOptions=["COMPRESS=DEFLATE","PREDICTOR=2"])
    vrt=None
    ds=gdal.Open(tmp)
    for i,b in enumerate(BANDS,1): ds.GetRasterBand(i).SetDescription(b)
    ds=None; os.replace(tmp,dst); return "완료"

def main():
    names=sys.argv[1:] or [n for n,e in EV["events"].items() if e["jetson_pre"] and e["jetson_post"]]
    t0=time.time(); done=skip=fail=0
    for n in names:
        e=EV["events"][n]; od=os.path.join(OUT,n); os.makedirs(od,exist_ok=True)
        meta=os.path.join(od,"clips.json"); rec=json.load(open(meta)) if os.path.exists(meta) else {}
        for role,ids in (("pre",e["jetson_pre"]),("post",e["jetson_post"])):
            for sid in ids:
                dst=os.path.join(od,sid+".tif")
                if os.path.exists(dst) and sid in rec: skip+=1; continue
                try: st=clip(sid,e["ute"],dst)
                except Exception as ex: st=f"실패 {type(ex).__name__}: {ex}"
                s=IDX.get(sid,{}); rec[sid]={"role":role,"datetime":s.get("datetime"),"cloud":s.get("cloud"),"status":st}
                if st=="완료": done+=1
                else: fail+=1
                print(f"[{n}] {role} {sid} {st}", flush=True)
            json.dump(rec,open(meta,"w"),ensure_ascii=False,indent=1)
    print(f"완료 {done} 건너뜀 {skip} 실패 {fail} | {time.time()-t0:.0f}s", flush=True)
if __name__=="__main__": main()
