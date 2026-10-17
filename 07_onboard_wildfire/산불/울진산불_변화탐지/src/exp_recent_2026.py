"""완전 최근·완전 미학습 화재 실험 — 2026-03-22~24 접경A(38.12N,126.94E).
S2 pre/post(같은해 tight 브래킷) STAC 취득 → 배포 unet_s1(8ch,thr0.25) 소각면적 예측.
독립 검증용 S2 dNBR도 같이 산출. 무인증 AWS earth-search.
"""
import os, sys, json, numpy as np
from osgeo import gdal
gdal.UseExceptions()
for k,v in {"GDAL_HTTP_MAX_RETRY":"5","GDAL_HTTP_RETRY_DELAY":"5","GDAL_HTTP_TIMEOUT":"300",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS":".tif","VSI_CACHE":"TRUE",
            "GDAL_DISABLE_READDIR_ON_OPEN":"EMPTY_DIR"}.items():
    gdal.SetConfigOption(k,v)
sys.path.insert(0, os.path.dirname(__file__))
from stac import stac_search

HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
OUT=os.path.join(PROJ,"runs","exp_recent_2026"); os.makedirs(OUT,exist_ok=True)
WEB=os.path.join(PROJ,"..","..","웹앱")
DEPLOYED=os.path.join(WEB,"weights","et02_ens","et02_unet_s1.pt")

# 접경A 화재 AOI (~20x18km, 화재중심 38.11N/126.94E, UTM52N 20m 고정격자)
AOI=[126.85, 38.04, 127.03, 38.18]
UTM="EPSG:32652"; RES=20
PRE_WIN="2026-02-10T00:00:00Z/2026-03-21T00:00:00Z"
POST_WIN="2026-03-26T00:00:00Z/2026-04-25T00:00:00Z"
BANDS={"blue":"B02","green":"B03","red":"B04","nir":"B08","swir22":"B12"}  # 8ch=RGB+NIR, B12=dNBR검증

def covers(f):
    b=f["bbox"]  # [w,s,e,n]
    return b[0]<=AOI[0] and b[1]<=AOI[1] and b[2]>=AOI[2] and b[3]>=AOI[3]

def pick(win, tag):
    feats=stac_search(AOI, win, max_cloud=60, limit=80)
    feats=[f for f in feats if covers(f)] or feats  # AOI 완전덮는 씬 우선
    print(f"[{tag}] {len(feats)} scenes (AOI-covering 우선)")
    for f in feats[:6]:
        print(f"    {f['id']}  cloud={f['properties'].get('eo:cloud_cover'):.1f}  {f['properties']['datetime'][:10]}  cover={covers(f)}")
    return feats[0] if feats else None

def warp(href, resample="bilinear"):
    src="/vsicurl/"+href
    ds=gdal.Warp("", src, format="MEM", dstSRS=UTM, xRes=RES, yRes=RES,
                 outputBounds=None, resampleAlg=resample,
                 outputBoundsSRS="EPSG:4326",
                 # clip in 4326 then reproject to UTM at 20m
                 srcSRS=None)
    return ds

def warp_clip(href, resample="bilinear"):
    src="/vsicurl/"+href
    # 4326 AOI로 클립 후 UTM52N 20m 재투영 (pre/post 동일 파라미터 → 정합 보장)
    ds=gdal.Warp("", src, format="MEM", dstSRS=UTM, xRes=RES, yRes=RES,
                 outputBounds=[AOI[0],AOI[1],AOI[2],AOI[3]], outputBoundsSRS="EPSG:4326",
                 resampleAlg=resample)
    return ds.ReadAsArray().astype(np.float32)

def load_scene(feat, tag):
    a=feat["assets"]; out={}
    for key,name in BANDS.items():
        if key not in a:
            print(f"  !! {tag} missing {key}"); return None
        out[name]=warp_clip(a[key]["href"])
        print(f"  {tag} {name} {out[name].shape} mean={np.nanmean(out[name]):.0f}")
    return out

def refl(x):
    # ★배포 ET-02는 raw DN 스케일(NO_OFFSET=1): -1000 미차감, /1e4 후 clip.
    # build_cir_gt_stac.py:9,168 · build_recovered.py:21 과 동일 규약.
    return np.clip(x/1e4, 0, 1)

def main():
    pre_f=pick(PRE_WIN,"PRE"); post_f=pick(POST_WIN,"POST")
    if not pre_f or not post_f: print("no scenes"); return
    meta={"aoi":AOI,"pre":pre_f["id"],"pre_dt":pre_f["properties"]["datetime"][:10],
          "pre_cloud":pre_f["properties"].get("eo:cloud_cover"),
          "post":post_f["id"],"post_dt":post_f["properties"]["datetime"][:10],
          "post_cloud":post_f["properties"].get("eo:cloud_cover")}
    print("PRE ",meta["pre_dt"],"POST",meta["post_dt"])
    pre=load_scene(pre_f,"PRE"); post=load_scene(post_f,"POST")
    if pre is None or post is None: print("band load fail"); return
    H,W=pre["B02"].shape
    # ---- S2 dNBR 독립검증 ----
    def nbr(s):
        nir=refl(s["B08"]); swir=refl(s["B12"]); return (nir-swir)/(nir+swir+1e-6)
    dnbr=nbr(pre)-nbr(post)  # 소각이면 양수
    # ---- 모델 8ch 입력: pre[B02,B03,B04,B08]+post[...], /1e4 clip ----
    def rgbnir(s): return np.stack([refl(s["B02"]),refl(s["B03"]),refl(s["B04"]),refl(s["B08"])],0)
    x=np.concatenate([rgbnir(pre),rgbnir(post)],0).astype(np.float32)  # (8,H,W)
    # 유효마스크: 두 시기 다 0 아님
    valid=(pre["B08"]>0)&(post["B08"]>0)
    import torch, segmentation_models_pytorch as smp
    dev="cuda" if torch.cuda.is_available() else "cpu"
    m=smp.Unet("resnet34",encoder_weights=None,in_channels=8,classes=1).to(dev)
    m.load_state_dict(torch.load(DEPLOYED,map_location=dev)); m.eval()
    T=512; prob=np.zeros((H,W),np.float32)
    with torch.no_grad():
        for i in range(0,H,T):
            for j in range(0,W,T):
                p=x[:,i:i+T,j:j+T]; ph,pw=p.shape[1],p.shape[2]
                pad=np.zeros((8,T,T),np.float32); pad[:,:ph,:pw]=p
                o=torch.sigmoid(m(torch.from_numpy(pad[None]).to(dev)))[0,0].cpu().numpy()
                prob[i:i+ph,j:j+pw]=o[:ph,:pw]
    THR=0.25
    burn=(prob>=THR)&valid
    px_ha=(RES*RES)/1e4
    burn_ha=float(burn.sum()*px_ha)
    # dNBR 기준 소각(0.15) 참고
    dnbr_burn=(dnbr>=0.15)&valid
    res={**meta,"H":H,"W":W,"thr":THR,"model_burn_ha":round(burn_ha,1),
         "model_burn_px":int(burn.sum()),
         "dnbr_burn_ha_thr015":round(float(dnbr_burn.sum()*px_ha),1),
         "dnbr_p90":round(float(np.nanpercentile(dnbr[valid],90)),3),
         "dnbr_p99":round(float(np.nanpercentile(dnbr[valid],99)),3),
         "iou_model_vs_dnbr":round(float((burn&dnbr_burn).sum()/((burn|dnbr_burn).sum()+1e-9)),3)}
    json.dump(res,open(os.path.join(OUT,"result.json"),"w"),ensure_ascii=False,indent=1)
    np.savez_compressed(os.path.join(OUT,"arr.npz"),prob=prob,dnbr=dnbr,valid=valid,
                        pre_rgb=np.stack([refl(pre["B04"]),refl(pre["B03"]),refl(pre["B02"])],0),
                        post_rgb=np.stack([refl(post["B04"]),refl(post["B03"]),refl(post["B02"])],0))
    print(json.dumps(res,ensure_ascii=False,indent=1))

if __name__=="__main__": main()
