"""S2 재브래킷 라벨 복구: CIR-GT가 실패/소형(<30ha)으로 버린 화재를,
화재 직전 pre + 직후(7~40d) post 맑은 S2로 dNBR 재계산해 소각 복구.
강릉난곡(CIR-GT 0ha → S2 dNBR 206ha)처럼 브래킷 아티팩트를 되살림.
복구되면 = 신규 실측 라벨 → 학습 확대(성능↑ 정공법). 출력 runs/rebracket/.
"""
import os, glob, json, datetime as dt, numpy as np, rasterio
from rasterio.enums import Resampling
from pystac_client import Client
from pyproj import Transformer
from scipy import ndimage
PROJ=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
CIRD="/home/syntax_kmu/syntax_inference/Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires"
OUT=os.path.join(PROJ,"runs","rebracket"); os.makedirs(OUT,exist_ok=True)
cat=Client.open("https://earth-search.aws.element84.com/v1")
THR=0.15; MIN_PX=750  # 30ha@20m

def pick(clon,clat,t0,t1):
    def cov(it): x0,y0,x1,y1=it.bbox; return x0<=clon<=x1 and y0<=clat<=y1
    its=[it for it in cat.search(collections=["sentinel-2-l2a"],bbox=[clon-.01,clat-.01,clon+.01,clat+.01],datetime=f"{t0}/{t1}",query={"eo:cloud_cover":{"lt":30}}).items() if cov(it)]
    return sorted(its,key=lambda x:x.properties["eo:cloud_cover"])[0] if its else None

def nbr(it,b,shape):
    def rd(h):
        with rasterio.open(h) as ds:
            w=rasterio.windows.from_bounds(*b,transform=ds.transform); return ds.read(1,window=w,out_shape=shape,resampling=Resampling.bilinear).astype(np.float32)/1e4
    ni=rd(it.assets["nir"].href); sw=rd(it.assets["swir22"].href); v=(ni>.001)&(sw>.001); x=(ni-sw)/(ni+sw+1e-6); x[~v]=np.nan; return x

def main():
    res=[]
    for d in sorted(glob.glob(os.path.join(CIRD,"*"))):
        n=os.path.basename(d); p=os.path.join(d,"gt_stac.npz")
        if not os.path.isfile(p): continue
        try: z=np.load(p)
        except: continue
        gt=z["gt"].astype(bool); cur=int(gt.sum())
        if cur>=MIN_PX: continue   # 이미 usable 은 건너뜀
        try: fdate=dt.date(2000+int(n[:2]),int(n[2:4]),int(n[4:6]))
        except: continue
        H,W=gt.shape; xmin,ymin,xmax,ymax=[float(v) for v in z["ute"]]; b=(xmin,ymin,xmax,ymax)
        tr=Transformer.from_crs(f"EPSG:{int(z['utm'])}","EPSG:4326",always_xy=True); clon,clat=tr.transform((xmin+xmax)/2,(ymin+ymax)/2)
        pre=pick(clon,clat,(fdate-dt.timedelta(days=35)).isoformat(),(fdate-dt.timedelta(days=1)).isoformat())
        post=pick(clon,clat,(fdate+dt.timedelta(days=7)).isoformat(),(fdate+dt.timedelta(days=40)).isoformat())
        if not pre or not post: res.append((n,cur,None,"S2씬부족")); continue
        try: dnbr=nbr(pre,b,(H,W))-nbr(post,b,(H,W))
        except Exception as e: res.append((n,cur,None,f"read실패")); continue
        burn=(dnbr>THR)&np.isfinite(dnbr)
        burn=ndimage.binary_opening(burn,iterations=1)  # 노이즈 제거
        # 최대 연결성분(코히런트 scar)만
        lab,nc=ndimage.label(burn);
        if nc>0:
            sizes=ndimage.sum(burn,lab,range(1,nc+1)); big=(lab==(np.argmax(sizes)+1));
        else: big=burn
        npx=int(burn.sum()); ha=npx*400/1e4
        rec = "★복구" if npx>=MIN_PX else ("소폭" if npx>cur*2 and npx>100 else "미복구")
        res.append((n,cur,npx,f"{ha:.0f}ha pre{pre.datetime.date()} post{post.datetime.date()} {rec}"))
        if npx>=MIN_PX:
            np.savez_compressed(os.path.join(OUT,f"{n}.npz"),dnbr=dnbr.astype(np.float32),burn=burn,ute=z["ute"],utm=z["utm"])
    print(f"{'화재':22s} {'기존px':>7} {'복구px':>7}  비고")
    for n,cur,npx,note in res:
        print(f"{n:22s} {cur:7d} {str(npx):>7}  {note}")
    rec_n=sum(1 for _,_,npx,_ in res if npx and npx>=MIN_PX)
    print(f"\n★ 재브래킷으로 신규 usable(≥30ha) 복구: {rec_n}개 → 학습 화재 {10}→{10+rec_n}개 가능")
    json.dump([(n,cur,npx,note) for n,cur,npx,note in res],open(os.path.join(OUT,"summary.json"),"w"),ensure_ascii=False,indent=1)

if __name__=="__main__": main()
