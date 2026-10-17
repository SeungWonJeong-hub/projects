"""앙상블 평가: runs/ko_native/ens_s*.npz(시드별 미학습8화재 원시확률)를 평균 → F1."""
import glob, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","runs","ko_native")

def mets(p,y,thr=None):
    if thr is None:
        best=(0,0.5)
        for t in np.linspace(0.02,0.9,45):
            pr=(p>=t); tp=(pr&(y>0)).sum(); fp=(pr&(y==0)).sum(); fn=((~pr)&(y>0)).sum()
            f1=2*tp/(2*tp+fp+fn+1e-9)
            if f1>best[0]: best=(f1,t)
        thr=best[1]
    pr=(p>=thr); tp=(pr&(y>0)).sum(); fp=(pr&(y==0)).sum(); fn=((~pr)&(y>0)).sum()
    f1=2*tp/(2*tp+fp+fn+1e-9); iou=tp/(tp+fp+fn+1e-9)
    prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
    return dict(thr=round(float(thr),3),f1=round(float(f1),3),iou=round(float(iou),3),
                precision=round(float(prec),3),recall=round(float(rec),3))

fs=sorted(glob.glob(os.path.join(OUT,"ens_s*.npz")))
print("멤버:",[os.path.basename(f) for f in fs])
Ps=[]; Y=None
for f in fs:
    d=np.load(f); Ps.append(d["Pt"]); Y=d["Yt"]
    print(f"  {os.path.basename(f)} 단일 best:",mets(d["Pt"],Y))
Pavg=np.mean(Ps,0)
print("\n=== 앙상블(확률평균) ===")
print("best-thr:",mets(Pavg,Y))
print("[대비] GN-UNet 단일 best 0.593 | echo 0.578")
