"""온보드 소각면적 파이프라인 실측: clean 단일 unet_s1 vs 9-앙상블.
델타 = ①추론 지연(1x vs 9x) ②다운링크 크기(벡터 GeoJSON vs 래스터 DEFLATE) ③area_ha.
장면=영덕축산2025 실측 before/after(6밴드→RGBNIR 8ch). Jetson GPU.
사용: CIR_RAW=0 python onboard_bench_burn.py
"""
from __future__ import annotations
import os, sys, io, time, json, numpy as np, torch, rasterio
import segmentation_models_pytorch as smp
import step4_rehearsal as S
import loro_gate_tier1 as L
WEB=os.path.join(S._ROOT,"Steven","웹앱"); sys.path.insert(0,WEB)
from core import vectorize as V

SAMPLE=os.path.join(WEB,"sample_tif")
THR=float(os.environ.get("THR","0.5"))

def load_scene():
    def rd(t):
        with rasterio.open(os.path.join(SAMPLE,f"yeongdeok2025_{t}.tif")) as s:
            return s.read().astype(np.float32)/1e4, s.profile
    pre,prof=rd("before"); post,_=rd("after")
    x=np.concatenate([pre[[0,1,2,3]],post[[0,1,2,3]]],0)  # RGBNIR pre/post = 8ch
    return np.clip(x,0,1), prof

def timeit(fn,n=5):
    fn()  # warmup
    if S.DEV=="cuda": torch.cuda.synchronize()
    ts=[]
    for _ in range(n):
        if S.DEV=="cuda": torch.cuda.synchronize()
        t=time.perf_counter(); fn()
        if S.DEV=="cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter()-t)
    return float(np.median(ts))

def deflate_bytes(mask,profile):
    prof=profile.copy(); prof.update(count=1,dtype="uint8",compress="DEFLATE",nodata=0)
    with rasterio.io.MemoryFile() as mf:
        with mf.open(**prof) as d: d.write(mask.astype("uint8")[None])
        return len(mf.read())

def main():
    x,prof=load_scene(); H,W=x.shape[1:]
    print(f"장면 영덕축산2025: {H}x{W} 8ch (RGBNIR pre/post)\n")
    ens=L.load_models()                         # 9모델 (name,model)
    single=[m for n,m in ens if n=="et02_unet_s1"][0]

    # --- 지연 ---
    t_single=timeit(lambda: S.infer(single,x,H,W))
    def ens_infer():
        return np.mean([S.infer(m,x,H,W) for _,m in ens],0)
    t_ens=timeit(ens_infer,n=3)
    print(f"[지연] 단일 unet_s1 {t_single*1000:.0f} ms | 9-앙상블 {t_ens*1000:.0f} ms | 배속 {t_ens/t_single:.1f}x")

    # --- 마스크(단일) → 벡터화 & 래스터 ---
    prob=S.infer(single,x,H,W); mask=(prob>=THR).astype(np.uint8)
    mask=V.clean_mask(mask)
    polys=V.mask_to_polygons(mask,prof,min_area_m2=10000,class_name="burn")
    gj=V.to_geojson(polys,prof,class_name="burn",to_wgs84=True)
    gj_bytes=len(json.dumps(gj).encode())
    import gzip; gj_gz=len(gzip.compress(json.dumps(gj).encode()))
    ras=deflate_bytes(mask,prof)
    area_ha=gj["properties"]["total_area_ha"]; npoly=gj["properties"]["n_polygons"]
    print(f"\n[산출] 소각 {area_ha:.0f} ha · 폴리곤 {npoly}개 (thr {THR})")
    print(f"[다운링크] GeoJSON {gj_bytes/1024:.1f} KB | GeoJSON.gz {gj_gz/1024:.1f} KB | 래스터DEFLATE {ras/1024:.1f} KB")
    best = min([("GeoJSON.gz",gj_gz),("래스터DEFLATE",ras)],key=lambda a:a[1])
    print(f"  → 최소 다운링크 = {best[0]} {best[1]/1024:.1f} KB")

    out=dict(scene="yeongdeok2025",HxW=[H,W],thr=THR,
             latency_ms=dict(single=round(t_single*1000,1),ensemble=round(t_ens*1000,1),speedup=round(t_ens/t_single,2)),
             downlink_KB=dict(geojson=round(gj_bytes/1024,1),geojson_gz=round(gj_gz/1024,1),raster_deflate=round(ras/1024,1)),
             area_ha=round(float(area_ha),1),n_polygons=int(npoly))
    os.makedirs(os.path.join(S.PROJ,"runs"),exist_ok=True)
    json.dump(out,open(os.path.join(S.PROJ,"runs","onboard_bench_burn.json"),"w"),ensure_ascii=False,indent=1)
    print("\n저장: runs/onboard_bench_burn.json")

if __name__=="__main__": main()
