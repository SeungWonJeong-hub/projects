#!/bin/bash
# 손실 레버: dice_focal(균형) 3seed + balanced-Tversky(0.5/0.5) 3seed. RGB+NIR, SWIR미사용.
# --- 이식성(Jetson·B200 공통): 라이브러리·경로·인터프리터 자동선택 ---
WFS="${WILDFIRE_SEG_ROOT:-$HOME/wildfire-seg}"
[ -d "$WFS/libs" ] && export LD_LIBRARY_PATH="$WFS/libs:${LD_LIBRARY_PATH:-}"   # Jetson torch용 libcusparseLt; B200엔 없어도 무해
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.." || exit 1                 # = 울진산불_변화탐지
pick_py(){ local c; for c in "${PYTHON:-}" "$WFS/.venv/bin/python" "$HOME/.venv/bin/python" "$(command -v python3)" /usr/bin/python3; do [ -n "$c" ] && [ -x "$c" ] && "$c" -c 'import torch' >/dev/null 2>&1 && { echo "$c"; return; }; done; echo "${PYTHON:-$(command -v python3 || echo /usr/bin/python3)}"; }
PY="$(pick_py)"
C="CONFIG=rgbnir TRAIN=korea INIT=imagenet EPOCHS=40 DECODER=unet ENCODER=resnet34"
for S in 1 2 3; do
  echo "=== dice_focal seed$S ==="
  env $C SEED=$S LOSS=dice_focal SAVE=et02_df_s$S $PY src/smp_cd.py 2>&1 | grep -E "결과|F1" | tail -1
done
for S in 1 2 3; do
  echo "=== balTversky(0.5/0.5) seed$S ==="
  env $C SEED=$S LOSS=tversky_bce TA=0.5 TB=0.5 SAVE=et02_bt_s$S $PY src/smp_cd.py 2>&1 | grep -E "결과|F1" | tail -1
done
echo "=== LOSS BATCH DONE ==="
