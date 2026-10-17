#!/usr/bin/env bash
# post-only(pre 없음) vs bi-temporal 손실 측정 — 온보드(큐브위성) 실현성의 관문.
# bi-temporal arm 은 이미 있다(et02_20m_s50_* + cir_reh_ens_s50_exp) → post-only arm 만 만들어
# **같은 데이터셋·같은 레시피·같은 시드**로 비교한다. 다른 건 CONFIG 뿐이다.
#   bi-temporal : CONFIG=rgbnir       8ch [B02,B03,B04,NIR]×(pre,post)
#   post-only   : CONFIG=post_rgbnir  4ch [B02,B03,B04,NIR] post 만
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../src"
PY=${PY:-$HOME/envs/torch/bin/python}
LOG=${LOG:-$HOME/wildfire_run_logs}
ts(){ date +%H:%M:%S; }

echo "##### A) post-only 베이스 9모델 (ko_refl_s50, MIN_HA=50, 20m)  $(ts)"
for D in unet fpn deeplabv3plus; do for N in 1 2 3; do
  S=et02_20m_s50_post_${D}_s${N}
  [ -f ../runs/$S/results.json ] && { echo "  skip $S"; continue; }
  DS_KOREA=ko_refl_s50 MIN_HA=50 CONFIG=post_rgbnir DECODER=$D SEED=$N EPOCHS=40 \
    SAVE=$S $PY -u smp_cd.py 2>&1 | grep -E "^\[결과\]|^\[smp|Traceback|Error|out of memory"
  echo "  [$S] $(ts)"
done; done

echo "##### B) post-only rehearsal (확장풀 14화재, bi-temporal arm 과 동일)  $(ts)"
CONFIG=post_rgbnir CIR_POOL=expanded KO_DIR=ko_refl_s50 \
  BASE="runs/et02_20m_s50_post_*" OUT=cir_reh_ens_post_s50_exp \
  $PY -u step4b_rehearsal_ens.py 2>&1 | grep -vE "^  \[[0-9]"

echo "##### C) NIFoS 0.3m 실측 채점 (독립 자)  $(ts)"
for SITE in andong uljin; do
  echo "--- $SITE / bi-temporal(8ch)"
  SITE=$SITE GSD=20 TILE=256 $PY nifos_eval.py --ens "runs/cir_reh_ens_s50_exp/*.pt" 2>&1 | tail -22
  echo "--- $SITE / post-only(4ch)"
  SITE=$SITE GSD=20 TILE=256 CONFIG=post_rgbnir $PY nifos_eval.py --ens "runs/cir_reh_ens_post_s50_exp/*.pt" 2>&1 | tail -22
done
echo "@@@POSTONLY DONE@@@ $(ts)"
