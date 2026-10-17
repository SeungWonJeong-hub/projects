"""기존 rehearsal 앙상블을 **현재 코드**로 재채점 — 재학습 없이 평가만.

왜 필요한가: 2026-09-09 에 step4b.ens_infer 의 제로패딩을 고쳤다. 그 전에 기록된 숫자
(§6b 의 봄 0.808 / 산림청 0.713 등)와 그 후 숫자를 섞어 비교하면 안 된다 — 패딩 수정은
점수를 올리는 방향이라 비교가 한쪽으로 기운다.

사용: ENS=runs/cir_reh_ens_s50_exp CONFIG=rgbnir python reeval_rehearsal.py
"""
import os, glob, numpy as np, torch
import step4_rehearsal as S
import step4b_rehearsal_ens as B

ENS = os.environ.get("ENS", "runs/cir_reh_ens_s50_exp")
PROJ = B.PROJ
cks = sorted(glob.glob(os.path.join(PROJ, ENS, "*.pt")))
assert cks, f"가중치 없음: {ENS}"
models = []
for c in cks:
    stem = os.path.basename(c)[:-3]
    dec = next((d for d in B.ARCH if f"_{d}_" in stem or stem.endswith("_" + d)), None)
    assert dec, f"디코더 판별 실패: {stem}"
    m = B.ARCH[dec]("resnet34", encoder_weights=None, in_channels=B.IN_CH, classes=1).to(B.DEV).eval()
    m.load_state_dict(torch.load(c, map_location=B.DEV))
    models.append(m)
print(f"{ENS} · {len(models)}모델 · CONFIG={B.CONFIG}({B.IN_CH}ch) · KO={S.KO_DIR}", flush=True)
print(f"  봄 cross-fire = {B.eval_cir_ens(models):.4f}", flush=True)
print(f"  산림청 in-domain = {B.eval_forest_ens(models):.4f}", flush=True)
