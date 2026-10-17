#!/bin/bash
# 다양성 앙상블용 RGB+NIR 모델 3종 순차 학습 (SWIR 미사용). ET-02 baseline 0.688 넘기기용.
# --- 이식성(Jetson·B200 공통): 라이브러리·경로·인터프리터 자동선택 ---
WFS="${WILDFIRE_SEG_ROOT:-$HOME/wildfire-seg}"
[ -d "$WFS/libs" ] && export LD_LIBRARY_PATH="$WFS/libs:${LD_LIBRARY_PATH:-}"   # Jetson torch용 libcusparseLt; B200엔 없어도 무해
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.." || exit 1                 # = 울진산불_변화탐지
pick_py(){ local c; for c in "${PYTHON:-}" "$WFS/.venv/bin/python" "$HOME/.venv/bin/python" "$(command -v python3)" /usr/bin/python3; do [ -n "$c" ] && [ -x "$c" ] && "$c" -c 'import torch' >/dev/null 2>&1 && { echo "$c"; return; }; done; echo "${PYTHON:-$(command -v python3 || echo /usr/bin/python3)}"; }
PY="$(pick_py)"
COMMON="CONFIG=rgbnir TRAIN=korea INIT=imagenet EPOCHS=40 SEED=1"

echo "=== [1/3] FPN resnet34 ==="
env $COMMON DECODER=fpn ENCODER=resnet34 SAVE=et02_fpn_s1 $PY src/smp_cd.py 2>&1 | grep -vE "Warning|warn" | tail -3

echo "=== [2/3] DeepLabV3+ resnet34 ==="
env $COMMON DECODER=deeplabv3plus ENCODER=resnet34 SAVE=et02_dlv3_s1 $PY src/smp_cd.py 2>&1 | grep -vE "Warning|warn" | tail -3

echo "=== [3/3] UNet resnet50 ==="
env $COMMON DECODER=unet ENCODER=resnet50 SAVE=et02_unet_r50_s1 $PY src/smp_cd.py 2>&1 | grep -vE "Warning|warn" | tail -3

echo "=== ALL DONE ==="
