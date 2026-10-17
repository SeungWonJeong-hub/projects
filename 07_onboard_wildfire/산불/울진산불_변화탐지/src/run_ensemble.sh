#!/bin/bash
# --- 이식성(Jetson·B200 공통): 라이브러리·경로·인터프리터 자동선택 ---
WFS="${WILDFIRE_SEG_ROOT:-$HOME/wildfire-seg}"
[ -d "$WFS/libs" ] && export LD_LIBRARY_PATH="$WFS/libs:${LD_LIBRARY_PATH:-}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.." || exit 1
pick_py(){ local c; for c in "${PYTHON:-}" "$WFS/.venv/bin/python" "$HOME/.venv/bin/python" "$(command -v python3)" /usr/bin/python3; do [ -n "$c" ] && [ -x "$c" ] && "$c" -c 'import torch' >/dev/null 2>&1 && { echo "$c"; return; }; done; echo "${PYTHON:-$(command -v python3 || echo /usr/bin/python3)}"; }
PY="$(pick_py)"
COMMON="NORM=group MIN_TRAIN_HA=1000 KO_DATA=ko_multi_v7 LOSS=tversky TV_A=0.7 TV_B=0.3 AUG_STRONG=1 EPOCHS=40"
for S in 42 1 2; do
  echo "===== SEED $S ====="
  env $COMMON SEED=$S ENS_TAG=s$S $PY -u src/ko_train_native.py 2>&1 | grep -vE "Warning|Future" | tail -8
done
echo "ALL_ENSEMBLE_DONE"
