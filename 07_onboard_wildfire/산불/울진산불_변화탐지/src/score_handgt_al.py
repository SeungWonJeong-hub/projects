"""labelset_al 손라벨 채점 — hand_gt.tif 를 QGIS 에서 고친 타일마다 세 가지 F1 을 잰다.
  model_vs_hand   앙상블 예측(model_pred) 대 손라벨 = 진짜 F1
  model_vs_poly   앙상블 예측 대 산림청 폴리곤 = 기존 보고 방식
  poly_vs_hand    폴리곤 대 손라벨 = 라벨 노이즈 크기
hand_gt 가 polygon_gt 와 같으면(아직 안 고침) 건너뛴다. --all 이면 전부 잰다.
출력: labelset_al/handgt_result.json 과 화면 요약. 사용: python3 src/score_handgt_al.py [--all]
"""
import os, sys, glob, json, numpy as np, rasterio
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.join(HERE,"..","labelset_al")
def rd(p):
    with rasterio.open(p) as d: return d.read(1)
def f1(a,b,v):
    a=a[v]>0; b=b[v]>0; tp=(a&b).sum(); fp=(a&~b).sum(); fn=(~a&b).sum(); return round(float(2*tp/(2*tp+fp+fn+1e-9)),3)
out={}
for d in sorted(glob.glob(os.path.join(ROOT,"*__*"))):
    need=["hand_gt.tif","polygon_gt.tif","model_pred.tif","valid.tif"]
    if not all(os.path.exists(os.path.join(d,f)) for f in need): continue
    hand,poly,pred,valid=(rd(os.path.join(d,f)) for f in need); v=valid>0
    if "--all" not in sys.argv and np.array_equal(hand,poly): continue
    out[os.path.basename(d)]={"hand_px":int((hand[v]>0).sum()),"poly_px":int((poly[v]>0).sum()),
        "model_vs_hand":f1(pred,hand,v),"model_vs_poly":f1(pred,poly,v),"poly_vs_hand":f1(poly,hand,v)}
if not out: print("고친 hand_gt 가 없다. QGIS 에서 편집한 뒤 다시 실행 (--all 이면 전부)."); sys.exit(0)
json.dump(out,open(os.path.join(ROOT,"handgt_result.json"),"w"),ensure_ascii=False,indent=1)
for k,r in out.items(): print(f"{k}: 모델 대 손라벨 {r['model_vs_hand']} | 모델 대 폴리곤 {r['model_vs_poly']} | 폴리곤 대 손라벨 {r['poly_vs_hand']}")
m=np.mean([r["model_vs_hand"] for r in out.values()]); p=np.mean([r["model_vs_poly"] for r in out.values()])
print(f"평균: 모델 대 손라벨 {m:.3f}, 모델 대 폴리곤 {p:.3f} -> {'손라벨 기준이 더 높다(폴리곤이 천장을 눌렀음)' if m>p else '손라벨 기준이 더 낮다'}")
