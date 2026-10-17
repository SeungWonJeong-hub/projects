#!/bin/bash
# 다중시기 실험 4단계 - 기준(bitemporal, rgbnir) 과 mt 변형을 seed 3개씩 CPU 에서 돌린다 (GPU 는 MIG 문제로 불가).
# 사용: bash src/mt_run_all.sh   (결과 runs/mt_<config>_s<seed>/results.json, 로그 logs/mt_<config>_s<seed>.log)
cd "$(dirname "$0")/.." || exit 1
PY="$HOME/miniforge3/envs/ml/bin/python"
run(){ # config seed
  env -u LD_LIBRARY_PATH OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 CUDA_VISIBLE_DEVICES= \
    CONFIG=$1 SEED=$2 DECODER=unet BIGNORE=2 EPOCHS=40 MIN_HA=500 SAVE=mt_$1_s$2 \
    "$PY" src/smp_cd_mt.py > logs/mt_$1_s$2.log 2>&1
  echo "$(date '+%T') 끝 $1 s$2: $(grep -o 'F1=[0-9.]*' logs/mt_$1_s$2.log | head -1)"
}
mkdir -p logs
for s in 1 2 3; do
  run bitemporal $s & run bitemporal_mt $s & run rgbnir $s & wait
  run rgbnir_mt $s & wait
done
echo "전부 끝 $(date '+%T')"
