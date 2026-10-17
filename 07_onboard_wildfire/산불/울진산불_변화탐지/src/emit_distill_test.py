"""증류 실행가능성: 하이퍼스펙트럴 EMIT teacher를 S2 RGB+NIR(8ch) student가 재현하나?
- teacher = EMIT post-NBR char 라벨(20m 격자로 정합)
- student feature = S2 8ch(gt_stac x: pre/post RGB+NIR) + 파생(dNIR,dGREEN...)
- 공간 홀드아웃(좌 학습/우 평가) + 교차화재(의성 학습→산청 평가)
되면=미래 하이퍼스펙트럴 태스킹 라벨로 S2 모델 성능↑ 경로 실증.
"""
import os, json, numpy as np, h5py
from pyproj import Transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
PROJ=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
CIRD="/home/syntax_kmu/syntax_inference/Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires"
NIR_B,SWIR_B=64,244
FIRES={"산청":dict(cir="250321_산청시천면",emit=PROJ+"/data/emit_sancheong/EMIT_L2A_RFL_001_20250406T010521_2509601_003.nc"),
       "의성":dict(cir="250322_의성안평면",emit=PROJ+"/data/emit_uiseong/EMIT_L2A_RFL_001_20250424T011718_2511401_011.nc")}
def emit_post_on_grid(path,bounds,shape,tr):
    h=h5py.File(path,"r"); r=h["reflectance"]; nir=r[:,:,NIR_B].astype(np.float32); sw=r[:,:,SWIR_B].astype(np.float32)
    bad=(nir<=-.01)|(sw<=-.01); nb=(nir-sw)/(nir+sw+1e-6); nb[bad]=np.nan
    gx=h["location/glt_x"][:]; gy=h["location/glt_y"][:]; gt=h.attrs["geotransform"]; h.close()
    Ho,Wo=gx.shape; o=np.full((Ho,Wo),np.nan,np.float32); m=(gx>0)&(gy>0); o[m]=nb[gy[m]-1,gx[m]-1]
    lon0,rx,_,lat0,_,ry=[float(v) for v in gt[:6]]; xmin,ymin,xmax,ymax=bounds; H,W=shape
    rr,cc=np.mgrid[0:H,0:W]; ux=xmin+(cc+.5)*(xmax-xmin)/W; uy=ymax-(rr+.5)*(ymax-ymin)/H
    lon,lat=tr.transform(ux.ravel(),uy.ravel()); ec=np.round((lon-lon0)/rx-.5).astype(int); er=np.round((lat-lat0)/ry-.5).astype(int)
    ok=(ec>=0)&(ec<Wo)&(er>=0)&(er<Ho); out=np.full(H*W,np.nan,np.float32); out[ok]=o[er[ok],ec[ok]]; return out.reshape(H,W)
def feats(x):  # x: (8,H,W) pre RGBNIR + post RGBNIR
    pre,post=x[:4],x[4:]; d=post-pre
    return np.stack([*pre,*post,*d],0)  # 12ch: pre4+post4+delta4
def load(fire):
    c=FIRES[fire]; d=np.load(os.path.join(CIRD,c["cir"],"gt_stac.npz"))
    x=d["x"].astype(np.float32); gt=d["gt"].astype(bool); vd=d["valid"].astype(bool); H,W=gt.shape
    xmin,ymin,xmax,ymax=[float(v) for v in d["ute"]]; tr=Transformer.from_crs(f"EPSG:{int(d['utm'])}","EPSG:4326",always_xy=True)
    em=emit_post_on_grid(c["emit"],(xmin,ymin,xmax,ymax),(H,W),tr)
    thr=np.nanpercentile(em,25)  # 낮은 25% NBR=char=teacher 소각
    teacher=(em<thr)&np.isfinite(em)
    F=feats(x); return F,teacher,gt,vd&np.isfinite(em)
def evalset(clf,F,y,valid,mask):
    m=valid&mask; X=F[:,m].T; yy=y[m]
    if yy.sum()<20 or (~yy).sum()<20: return None
    p=clf.predict_proba(X)[:,1]; return roc_auc_score(yy,p), f1_score(yy,p>0.5)

print("=== 증류 실행가능성: S2 8ch→EMIT teacher 재현 ===\n")
# 1) 산청 내부 랜덤홀드아웃 70/30 (EMIT 부분커버라 공간분할 불가)
F,teach,cir,val=load("산청"); H,W=cir.shape
rng=np.random.RandomState(0)
idx=np.where(val.ravel())[0]; rng.shuffle(idx)
k=int(len(idx)*0.7); trm=np.zeros(H*W,bool); trm[idx[:k]]=True; trm=trm.reshape(H,W); tem=val&(~trm)
Xtr=F[:,trm].T; ytr=teach[trm]
clf=LogisticRegression(max_iter=500,class_weight="balanced").fit(Xtr,ytr)
a,f=evalset(clf,F,teach,tem,np.ones_like(val))
print(f"[산청 랜덤홀드아웃30%] S2→EMIT-teacher 재현: AUC {a:.3f} F1 {f:.3f}")
clf2=LogisticRegression(max_iter=500,class_weight="balanced").fit(F[:,trm].T,cir[trm])
a2,f2=evalset(clf2,F,cir,tem,np.ones_like(val))
print(f"[산청 랜덤홀드아웃30%] S2→CIR-GT 재현:      AUC {a2:.3f} F1 {f2:.3f}")
# 2) 교차화재: 의성 학습 → 산청 평가 (EMIT teacher)
Fu,teachu,ciru,valu=load("의성")
clfx=LogisticRegression(max_iter=500,class_weight="balanced").fit(Fu[:,valu].T,teachu[valu])
ax,fx=evalset(clfx,F,teach,val,np.ones_like(val))
print(f"[교차화재] 의성 학습→산청 EMIT-teacher 평가: AUC {ax:.3f} F1 {fx:.3f}")
print(f"\n판정: S2 8ch가 EMIT teacher를 재현하면(AUC>0.75) = 하이퍼스펙트럴 라벨로 S2 모델 학습 가능 = 태스킹 스케일업시 성능↑ 경로 실증")
json.dump(dict(sancheong_spatial_emit=[a,f],sancheong_spatial_cirgt=[a2,f2],cross_uiseong_to_sancheong=[ax,fx]),
          open(os.path.join(PROJ,"runs","emit_distill_test.json"),"w"),indent=1)
