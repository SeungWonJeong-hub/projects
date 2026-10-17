"""합성 pre 데이터로 resnet34 단독 학습(대조군 A와 동일 조건) → 단일 pre A와 비교."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ko2uljin_finetune as m

r = m.train("A_composite_pre", None, m.LR_A)  # imagenet init, LR_A, 현재 data/cd(합성 pre 타일)
out = os.path.join(os.path.dirname(m.OUT), "ko2uljin", "composite_A_results.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(r, open(out, "w"), indent=2, ensure_ascii=False)
print("\n========== 합성 pre vs 단일 pre (resnet34 단독, TEST val-thr) ==========")
print(f"  단일 pre  A : F1 0.776 / IoU 0.634")
print(f"  합성 pre  A : F1 {r['test_valthr']['f1']} / IoU {r['test_valthr']['iou']}")
print("저장:", out)
