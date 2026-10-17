"""EMIT 봄 dNBR을 산청 CIR-GT 소각 폴리곤으로 검증.
소각 안 vs 밖 dNBR 분포 비교 → 하이퍼스펙트럴 SWIR가 봄 소각을 잡는지 결정.
"""
import os, json, numpy as np, h5py
from pyproj import Transformer
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
OUT=os.path.join(PROJ,"runs","emit_sancheong")
CIRD="/home/syntax_kmu/syntax_inference/Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires"
AOI=dict(lon0=127.66,lon1=127.98,lat0=35.16,lat1=35.44); RES=0.00054223

def main():
    dnbr=np.load(os.path.join(OUT,"dnbr_sancheong.npy")); Ht,Wt=dnbr.shape
    d=np.load(os.path.join(CIRD,"250321_산청시천면","gt_stac.npz"))
    gt=d["gt"].astype(bool); H,W=gt.shape
    xmin,ymin,xmax,ymax=[float(v) for v in d["ute"]]; epsg=int(d["utm"])
    tr=Transformer.from_crs(f"EPSG:{epsg}","EPSG:4326",always_xy=True)
    # CIR-GT 소각 픽셀 UTM 중심 → lon/lat → EMIT 격자 인덱스
    ys,xs=np.where(gt)
    px=xmin+(xs+0.5)*(xmax-xmin)/W; py=ymax-(ys+0.5)*(ymax-ymin)/H
    lon,lat=tr.transform(px,py)
    cc=np.round((lon-AOI["lon0"])/RES-0.5).astype(int)
    rr=np.round((AOI["lat1"]-lat)/RES-0.5).astype(int)
    ok=(cc>=0)&(cc<Wt)&(rr>=0)&(rr<Ht)
    burn=np.zeros((Ht,Wt),bool); burn[rr[ok],cc[ok]]=True
    print(f"CIR-GT 소각 {int(gt.sum())}px(20m) → EMIT격자 {int(burn.sum())}셀(60m)")
    valid=np.isfinite(dnbr)
    bin_=burn&valid; bout=(~burn)&valid
    vi=dnbr[bin_]; vo=dnbr[bout]
    print(f"\n=== EMIT 하이퍼스펙트럴 SWIR dNBR: 소각 안 vs 밖 ===")
    print(f"  소각 안(n={len(vi)}): p50 {np.median(vi):+.3f}  p25 {np.percentile(vi,25):+.3f}  p75 {np.percentile(vi,75):+.3f}  mean {vi.mean():+.3f}")
    print(f"  소각 밖(n={len(vo)}): p50 {np.median(vo):+.3f}  p25 {np.percentile(vo,25):+.3f}  p75 {np.percentile(vo,75):+.3f}  mean {vo.mean():+.3f}")
    print(f"  분리도(안 p50 − 밖 p50) = {np.median(vi)-np.median(vo):+.3f}")
    # 임계별 소각내 검출률(recall proxy) + 소각외 오탐률
    print(f"\n  {'thr':>5} {'소각내≥thr(recall)':>18} {'소각외≥thr(FPR)':>16}")
    for thr in [0.10,0.15,0.20,0.27]:
        rec=(vi>=thr).mean(); fpr=(vo>=thr).mean()
        print(f"  {thr:5.2f} {rec*100:17.1f}% {fpr*100:15.1f}%")
    # 판정
    sep=np.median(vi)-np.median(vo)
    print(f"\n판정: {'✅ 하이퍼스펙트럴 SWIR가 봄 소각 포착(안>밖 분리)' if sep>0.05 else '△ 분리 약함'}")
    out=dict(fire="sancheong_2025_EMIT_validation",
             burn_in=dict(n=int(len(vi)),p50=float(np.median(vi)),mean=float(vi.mean())),
             burn_out=dict(n=int(len(vo)),p50=float(np.median(vo)),mean=float(vo.mean())),
             separation_p50=float(sep),
             recall_at=dict((str(t),float((vi>=t).mean())) for t in [0.10,0.15,0.20,0.27]),
             fpr_at=dict((str(t),float((vo>=t).mean())) for t in [0.10,0.15,0.20,0.27]))
    json.dump(out,open(os.path.join(OUT,"validation.json"),"w"),ensure_ascii=False,indent=1)
    print("저장: runs/emit_sancheong/validation.json")

if __name__=="__main__": main()
