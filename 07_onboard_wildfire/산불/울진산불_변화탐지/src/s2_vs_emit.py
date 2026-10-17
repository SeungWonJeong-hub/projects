"""결정타: 같은 산청 화재·같은 날짜·같은 소각폴리곤에서 S2 dNBR vs EMIT dNBR 분리도 대비.
S2: B08(NIR 10m)+B12(SWIR2 20m 광대역), pre 2024-04-01 / post 2025-04-06 (EMIT과 동일).
CIR-GT UTM52 20m 격자에 직접 정합(재투영 불필요).
EMIT 분리도 +0.213 대비 S2가 얼마나 분리하는지 → SWIR 모달리티 우위 여부.
"""
import os, json, numpy as np, rasterio
from rasterio.enums import Resampling
from pystac_client import Client
CIRD="/home/syntax_kmu/syntax_inference/Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires"
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","runs","emit_sancheong")
BBOX=[127.66,35.16,127.98,35.44]

def pick(dr):
    cat=Client.open("https://earth-search.aws.element84.com/v1")
    its=[it for it in cat.search(collections=["sentinel-2-l2a"],bbox=BBOX,datetime=dr,
         query={"eo:cloud_cover":{"lt":20}}).items() if "52SDD" in it.id]
    its.sort(key=lambda x:x.properties["eo:cloud_cover"])
    return its[0]

def read_band(href, bounds, shape):
    """UTM 윈도를 target shape(20m 격자)로 읽기."""
    with rasterio.open(href) as ds:
        win=rasterio.windows.from_bounds(*bounds,transform=ds.transform)
        a=ds.read(1,window=win,out_shape=shape,resampling=Resampling.bilinear).astype(np.float32)
    return a/1e4  # earth-search 오프셋 이미 보정됨

def nbr(item, bounds, shape):
    nir=read_band(item.assets["nir"].href, bounds, shape)      # B08
    swir=read_band(item.assets["swir22"].href, bounds, shape)  # B12 2190nm
    v=(nir>0.001)&(swir>0.001)
    n=(nir-swir)/(nir+swir+1e-6); n[~v]=np.nan; return n

def main():
    d=np.load(os.path.join(CIRD,"250321_산청시천면","gt_stac.npz"))
    gt=d["gt"].astype(bool); H,W=gt.shape
    xmin,ymin,xmax,ymax=[float(v) for v in d["ute"]]
    bounds=(xmin,ymin,xmax,ymax); shape=(H,W)
    print(f"산청 CIR-GT {H}x{W} @20m, 소각 {int(gt.sum())}px")
    pre=pick("2024-03-20/2024-04-15"); post=pick("2025-03-28/2025-04-20")
    print(f"S2 pre {pre.datetime.date()} ({pre.properties['eo:cloud_cover']:.0f}%) / post {post.datetime.date()} ({post.properties['eo:cloud_cover']:.0f}%)")
    nb_pre=nbr(pre,bounds,shape); nb_post=nbr(post,bounds,shape)
    dnbr=nb_pre-nb_post
    valid=np.isfinite(dnbr)
    vi=dnbr[gt&valid]; vo=dnbr[(~gt)&valid]
    print(f"\n=== S2 광대역 B12 SWIR dNBR: 소각 안 vs 밖 ===")
    print(f"  소각 안(n={len(vi)}): p50 {np.median(vi):+.3f} mean {vi.mean():+.3f}")
    print(f"  소각 밖(n={len(vo)}): p50 {np.median(vo):+.3f} mean {vo.mean():+.3f}")
    s2_sep=np.median(vi)-np.median(vo)
    print(f"  S2 분리도 = {s2_sep:+.3f}")
    print(f"\n  {'thr':>5} {'소각내(recall)':>14} {'소각외(FPR)':>12}")
    for thr in [0.10,0.15,0.20,0.27]:
        print(f"  {thr:5.2f} {(vi>=thr).mean()*100:13.1f}% {(vo>=thr).mean()*100:11.1f}%")
    print(f"\n=== 대비 ===")
    print(f"  EMIT native SWIR(2197nm) 분리도 = +0.213 (기록)")
    print(f"  S2   광대역  SWIR(B12)     분리도 = {s2_sep:+.3f}")
    print(f"  → {'EMIT 우위' if 0.213>s2_sep+0.02 else 'S2도 동등 분리(이 화재는 S2 실패 아님)' if abs(0.213-s2_sep)<=0.05 else 'S2 우위'}")
    out=dict(fire="sancheong_2025", s2_pre=str(pre.datetime.date()), s2_post=str(post.datetime.date()),
             s2_burn_in_p50=float(np.median(vi)), s2_burn_out_p50=float(np.median(vo)),
             s2_separation=float(s2_sep), emit_separation=0.213,
             s2_recall_fpr={str(t):[float((vi>=t).mean()),float((vo>=t).mean())] for t in [0.10,0.15,0.20,0.27]})
    json.dump(out,open(os.path.join(OUT,"s2_vs_emit.json"),"w"),ensure_ascii=False,indent=1)
    print("저장: runs/emit_sancheong/s2_vs_emit.json")

if __name__=="__main__": main()
