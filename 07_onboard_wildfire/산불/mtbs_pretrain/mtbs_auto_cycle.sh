#!/usr/bin/env bash
# 자율 사이클: 현재 패치로 학습→안동·울진 검증→결과 로그. flock 로 중복방지.
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export LD_LIBRARY_PATH="$HOME/wildfire-seg/libs:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=""   # Orin GPU torch NVML 불안정 → CPU 고정(소규모라 충분, 진짜학습은 B200)
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1        # 로그 실시간(에폭 진행 보이게)
cd "$HOME/syntax_inference/Steven/산불/mtbs_pretrain" || exit 1
PY="$HOME/wildfire-seg/.venv/bin/python"
LOG="$HOME/.mtbs_cycle.log"; LOCK="$HOME/.mtbs_cycle.lock"
exec 9>"$LOCK"; flock -n 9 || exit 0
ts(){ TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST'; }
N=$(ls patches/*.npz 2>/dev/null | wc -l)
[ "${N:-0}" -lt 20 ] && { echo "[$(ts)] 패치 $N<20 스킵" >> "$LOG"; exit 0; }
echo "[$(ts)] ===== 사이클 시작 · 패치 $N =====" >> "$LOG"
$PY src/mtbs_pretrain_train.py --epochs 20 --bs 16 >> "$LOG" 2>&1
echo "[$(ts)] --- 검증(안동·울진) ---" >> "$LOG"
$PY src/mtbs_validate_korea.py --ckpt ckpt/mtbs_sev_best.pt >> "$LOG" 2>&1
echo "[$(ts)] ===== 사이클 완료 · 패치 $N =====" >> "$LOG"
