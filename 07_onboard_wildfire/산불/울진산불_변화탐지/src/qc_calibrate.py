"""씬 쌍 차단 임계값 보정 — 배경이 얼마나 어긋나면 **면적이 실제로** 무너지는가.

왜: `웹앱/core/qc.py` 의 차단선 0.12 는 사례 두 건(강릉 실패 −0.194 / 통과 −0.007)으로 잡은
값이라 근거가 얇다. 그런데 학습셋 245화재를 재보니 **배경 dNDVI 중앙이 −0.103, 54%가 0.12 이상**
이었다(레시피가 그렇게 만든다 — 봄 화재는 pre 가 잎 전, post 가 만엽이 되기 쉽다).
즉 지금 차단선은 **정상적인 봄 화재 분석까지 막을 수 있다.**

그래서 사례가 아니라 **곡선**으로 정한다: 안동(NIFoS 0.3m, 전체 라벨 독립 자)에서 post 를
고정하고 pre 를 바꿔 배경 수준을 훑으며, 배포 앙상블의 F1·면적오차가 어디서 꺾이는지 본다.
안동을 쓰는 이유 — 강릉·고성 GT 는 부분 라벨이라 면적오차를 절대값으로 못 쓴다(GT_USAGE.md).

사용: conda run -n ml python qc_calibrate.py [--post 2020-06-08] [--pres 2019-03-26 ...]
"""
import os, sys, json, subprocess, argparse
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
SITE=os.environ.get("SITE","andong"); GSD=20; TILE=256
RUNS="runs/cir_reh_ens/*.pt"
GEO=os.path.expanduser("~/miniforge3/envs/geo/bin/python")

def prep(pre,post):
    env=dict(os.environ, SITE=SITE, GSD=str(GSD), SCENE_PICK="fixed",
             PRE_DATE=pre, POST_DATE=post)
    env.pop("PROJ_DATA",None)                      # geo 쪽에서 스스로 채운다
    r=subprocess.run(["conda","run","-n","geo","python",os.path.join(HERE,"nifos_eval.py"),"prep"],
                     env=env, capture_output=True, text=True, cwd=PROJ)
    if r.returncode!=0:
        print(r.stdout[-800:], r.stderr[-800:]); raise SystemExit(f"prep 실패 {pre}/{post}")
    return os.path.join(PROJ,"nifos_cache",
                        f"{SITE}_p{pre.replace('-','')}_q{post.replace('-','')}")

def score(cache):
    sys.path.insert(0,HERE)
    os.environ.update(SITE=SITE,GSD=str(GSD),TILE=str(TILE),NIR="B8A")
    import importlib, nifos_eval as NE
    importlib.reload(NE)
    gt=np.load(os.path.join(cache,f"gt_{GSD}m.npy")).astype(bool)
    x=np.load(os.path.join(cache,f"scene_{GSD}m.npz"))["x"]
    H=min(gt.shape[0],x.shape[1]); W=min(gt.shape[1],x.shape[2])
    gt=gt[:H,:W]; x=x[:,:H,:W]
    import glob
    prob,_=NE.infer(x,sorted(glob.glob(os.path.join(PROJ,RUNS))))
    prob=prob[:H,:W]
    xf=x.astype(np.float32)/1e4
    idx=lambda b,i,j:(b[i]-b[j])/(b[i]+b[j]+1e-6)
    bg=float(np.median((idx(xf[:6],3,2)-idx(xf[6:],3,2))[~gt]))
    px=GSD*GSD/1e4; gt_ha=gt.sum()*px
    def m(t):
        p=prob>=t; tp=(p&gt).sum()
        f1=2*tp/max(p.sum()+gt.sum(),1)
        return f1, (p.sum()*px-gt_ha)/gt_ha*100
    f1,ae=m(0.20)
    ts=np.arange(0.05,0.95,0.025); best=max(ts,key=lambda t:m(t)[0])
    bf1,bae=m(best)
    pred_ha=float((prob>=0.20).sum()*px)
    return dict(bg=bg, f1=f1, area_err=ae, best_thr=float(best), best_f1=bf1, best_area_err=bae,
                gt_ha=float(gt_ha), pred_ha=pred_ha)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--site",default=os.environ.get("SITE","andong"))
    ap.add_argument("--post",default="2020-06-08")           # 저구름 규칙이 뽑는 post
    ap.add_argument("--pres",nargs="+",
        default=["2019-03-26","2019-04-15","2019-05-03","2019-05-23","2019-06-04"])
    a=ap.parse_args()
    global SITE; SITE=a.site
    out=[]
    print(f"{SITE} · post 고정 {a.post} · 배포 20m 앙상블")
    print(f"{'pre':12} {'ΔDOY':>5} {'배경dNDVI':>10} {'F1@0.2':>8} {'예측ha@0.2':>11} "
          f"{'면적오차@0.2':>12} {'최적thr':>7} {'최적F1':>7}")
    import datetime as dt
    for pre in a.pres:
        c=prep(pre,a.post); r=score(c); r["pre"]=pre
        d=abs(dt.date.fromisoformat(pre).timetuple().tm_yday
              -dt.date.fromisoformat(a.post).timetuple().tm_yday)
        r["ddoy"]=min(d,365-d); out.append(r)
        print(f"{pre:12} {r['ddoy']:5d} {r['bg']:+10.3f} {r['f1']:8.4f} {r['pred_ha']:11.0f} "
              f"{r['area_err']:+11.1f}% {r['best_thr']:7.3f} {r['best_f1']:7.4f}")
    json.dump(out,open(os.path.join(PROJ,"data",f"qc_calibrate_{SITE}.json"),"w"),
              ensure_ascii=False,indent=1)
    print(f"\n기록: data/qc_calibrate_{SITE}.json")

if __name__=="__main__": main()
