set -u
# 경로는 스크립트 위치에서 유도한다(레포 이식성). 파이썬은 PY env 로 덮어쓸 수 있다.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../src"
PY=${PY:-$HOME/envs/torch/bin/python}
ts(){ date +%H:%M:%S; }

echo "##### A) 20m 베이스 9모델 (ko_refl_s50, MIN_HA=50)  $(ts)"
for D in unet fpn deeplabv3plus; do for N in 1 2 3; do
  [ -f ../runs/et02_20m_s50_${D}_s${N}/results.json ] && { echo "  skip ${D}_s${N}"; continue; }
  DS_KOREA=ko_refl_s50 MIN_HA=50 CONFIG=rgbnir DECODER=$D SEED=$N EPOCHS=40 \
    SAVE=et02_20m_s50_${D}_s${N} $PY -u smp_cd.py 2>&1 | grep -E "^\[결과\]|Traceback|Error|out of memory"
  echo "  [20m ${D}_s${N}] $(ts)"
done; done

echo "##### B) 10m@512 베이스 9모델 (공통 245화재)  $(ts)"
for D in unet fpn deeplabv3plus; do for N in 1 2 3; do
  [ -f ../runs/et02_10mT512_s50_${D}_s${N}/results.json ] && { echo "  skip ${D}_s${N}"; continue; }
  DS_KOREA=ko_refl_10m_b08_t512_s50c MIN_HA=50 CONFIG=rgbnir DECODER=$D SEED=$N EPOCHS=40 \
    SAVE=et02_10mT512_s50_${D}_s${N} $PY -u smp_cd.py 2>&1 | grep -E "^\[결과\]|Traceback|Error|out of memory"
  echo "  [10m ${D}_s${N}] $(ts)"
done; done

echo "##### C) 20m rehearsal (확장풀 14화재)  $(ts)"
CIR_POOL=expanded KO_DIR=ko_refl_s50 BASE="runs/et02_20m_s50_[udf]*" OUT=cir_reh_ens_s50_exp \
  $PY -u step4b_rehearsal_ens.py 2>&1 | grep -vE "^  \[[0-9]"

echo "##### D) 10m rehearsal (확장풀 14화재)  $(ts)"
GSD=10 T=512 KO_DIR=ko_refl_10m_b08_t512_s50c CIR_POOL=expanded \
  BASE="runs/et02_10mT512_s50_[udf]*" OUT=cir_reh_ens_10m_s50_exp \
  $PY -u step4b_rehearsal_ens.py 2>&1 | grep -vE "^  \[[0-9]"

echo "##### E) CIR 홀드아웃 채점  $(ts)"
for G in large small; do
  echo "--- 20m s50 확장풀 / $G"
  GSD=20 TILE=256 GROUP=$G $PY cir_hold_eval.py --ens "runs/cir_reh_ens_s50_exp/*.pt" 2>&1 | tail -5
  echo "--- 10m s50 확장풀 / $G"
  GSD=10 TILE=512 GROUP=$G $PY cir_hold_eval.py --ens "runs/cir_reh_ens_10m_s50_exp/*.pt" 2>&1 | tail -5
done
echo "@@@CHAIN DONE@@@ $(ts)"
