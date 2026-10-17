"""EMIT 하이퍼스펙트럴로 한국 봄 소각 검증: 산청2025(화재 2025-03-21).
pre 2024-03-31(봄, 계절일치) / post 2025-04-06(2주후). native SWIR NBR.
h5py로 GLT 기하보정 직접 구현(netCDF4/xarray 없음). NBR=(NIR857-SWIR2197)/(합).
dNBR=pre-post(양수=소각). 두 씬 각자 ortho→공통 AOI 4326 격자로 정합→dNBR.
출력: runs/emit_sancheong/ (dNBR npy + quicklook png + 통계 json).
"""
from __future__ import annotations
import os, json, numpy as np, h5py
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
D=os.path.join(PROJ,"data","emit_sancheong")
OUT=os.path.join(PROJ,"runs","emit_sancheong"); os.makedirs(OUT,exist_ok=True)
PRE=os.path.join(D,"EMIT_L2A_RFL_001_20240331T035456_2409103_004.nc")
POST=os.path.join(D,"EMIT_L2A_RFL_001_20250406T010521_2509601_003.nc")
NIR_B, SWIR_B = 64, 244   # 857.6nm, 2197.1nm
# 산청 시천 화재 AOI (대략)
AOI=dict(lon0=127.66, lon1=127.98, lat0=35.16, lat1=35.44)
RES=0.00054223

def scene_nbr(path):
    h=h5py.File(path,"r")
    refl=h["reflectance"]
    nir=refl[:,:,NIR_B].astype(np.float32); swir=refl[:,:,SWIR_B].astype(np.float32)
    bad=(nir<=-0.01)|(swir<=-0.01)|(nir>1.2)|(swir>1.2)
    nbr=(nir-swir)/(nir+swir+1e-6); nbr[bad]=np.nan
    gx=h["location/glt_x"][:]; gy=h["location/glt_y"][:]
    gt=h.attrs["geotransform"];
    h.close()
    # GLT ortho: ortho[r,c] = nbr_sensor[gy-1, gx-1], glt>0
    Ho,Wo=gx.shape; ortho=np.full((Ho,Wo),np.nan,np.float32)
    m=(gx>0)&(gy>0)
    ortho[m]=nbr[gy[m]-1, gx[m]-1]
    # geotransform: [lon0, res, 0, lat0, 0, -res]
    lon0,rx,_,lat0,_,ry=[float(v) for v in gt[:6]]
    return ortho, (lon0,rx,lat0,ry)

def regrid(ortho, gt, aoi, res):
    """씬 ortho(그 자체 gt) → 공통 AOI 4326 격자(nearest)."""
    lon0,rx,lat0,ry=gt
    Wt=int(round((aoi["lon1"]-aoi["lon0"])/res)); Ht=int(round((aoi["lat1"]-aoi["lat0"])/res))
    out=np.full((Ht,Wt),np.nan,np.float32)
    Ho,Wo=ortho.shape
    # 타깃 셀 중심 lon/lat
    tl=aoi["lon0"]+(np.arange(Wt)+0.5)*res
    ta=aoi["lat1"]-(np.arange(Ht)+0.5)*res
    # 소스 인덱스
    cs=np.round((tl-lon0)/rx-0.5).astype(int)
    rs=np.round((ta-lat0)/ry-0.5).astype(int)
    cvalid=(cs>=0)&(cs<Wo); rvalid=(rs>=0)&(rs<Ho)
    for i,(r,ok_r) in enumerate(zip(rs,rvalid)):
        if not ok_r: continue
        cc=cs[cvalid]; out[i,np.where(cvalid)[0]]=ortho[r,cc]
    return out, (aoi["lon0"],res,aoi["lat1"],-res)

def main():
    print("EMIT 산청: pre 2024-03-31 / post 2025-04-06, NBR(857/2197nm)")
    o_pre,gt_pre=scene_nbr(PRE); o_post,gt_post=scene_nbr(POST)
    print(f"  ortho pre {o_pre.shape} valid {np.isfinite(o_pre).mean()*100:.0f}% | post {o_post.shape} valid {np.isfinite(o_post).mean()*100:.0f}%")
    g_pre,gt=regrid(o_pre,gt_pre,AOI,RES); g_post,_=regrid(o_post,gt_post,AOI,RES)
    both=np.isfinite(g_pre)&np.isfinite(g_post)
    print(f"  AOI 격자 {g_pre.shape} 공통유효 {both.mean()*100:.0f}%")
    dnbr=np.where(both, g_pre-g_post, np.nan)
    v=dnbr[np.isfinite(dnbr)]
    print(f"\n=== 산청 봄 dNBR (하이퍼스펙트럴 SWIR) ===")
    print(f"  dNBR 분포: p50 {np.nanpercentile(v,50):+.3f} p90 {np.nanpercentile(v,90):+.3f} p99 {np.nanpercentile(v,99):+.3f} max {v.max():+.3f}")
    for thr in [0.10,0.15,0.20,0.27]:
        frac=(v>=thr).mean(); ha=(v>=thr).sum()*(RES*111000)**2/1e4
        print(f"  dNBR>={thr}: {frac*100:.1f}% ({ha:.0f} ha 추정)")
    np.save(os.path.join(OUT,"dnbr_sancheong.npy"),dnbr)
    # quicklook
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,3,figsize=(15,5))
        ax[0].imshow(g_pre,cmap="RdYlGn",vmin=-0.2,vmax=0.7); ax[0].set_title("NBR pre 2024-03-31")
        ax[1].imshow(g_post,cmap="RdYlGn",vmin=-0.2,vmax=0.7); ax[1].set_title("NBR post 2025-04-06")
        im=ax[2].imshow(dnbr,cmap="inferno",vmin=0,vmax=0.6); ax[2].set_title("dNBR (burn=bright)"); plt.colorbar(im,ax=ax[2])
        for a in ax: a.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(OUT,"sancheong_emit_nbr.png"),dpi=90); print("\n저장: runs/emit_sancheong/sancheong_emit_nbr.png")
    except Exception as e: print("quicklook skip:",e)
    out=dict(fire="sancheong_2025", sensor="EMIT VSWIR 60m",
             pre="2024-03-31", post="2025-04-06", bands=dict(NIR=857.6,SWIR=2197.1),
             dnbr_p50=float(np.nanpercentile(v,50)),dnbr_p90=float(np.nanpercentile(v,90)),
             dnbr_p99=float(np.nanpercentile(v,99)),
             burn_ha_thr015=float((v>=0.15).sum()*(RES*111000)**2/1e4),
             common_valid_pct=float(both.mean()*100), aoi=AOI)
    json.dump(out,open(os.path.join(OUT,"stats.json"),"w"),ensure_ascii=False,indent=1)
    print("저장: runs/emit_sancheong/stats.json")

if __name__=="__main__": main()
