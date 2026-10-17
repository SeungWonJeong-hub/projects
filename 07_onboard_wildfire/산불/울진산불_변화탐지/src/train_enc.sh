#!/bin/bash
# 강한 인코더 레버: SegFormer(mit_b1)·EfficientNet-b3 UNet, RGB+NIR(SWIR미사용). baseline 0.688 돌파 시도.
# --- 이식성(Jetson·B200 공통): 라이브러리·경로·인터프리터 자동선택 ---
WFS="${WILDFIRE_SEG_ROOT:-$HOME/wildfire-seg}"
[ -d "$WFS/libs" ] && export LD_LIBRARY_PATH="$WFS/libs:${LD_LIBRARY_PATH:-}"   # Jetson torch용 libcusparseLt; B200엔 없어도 무해
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.." || exit 1                 # = 울진산불_변화탐지
pick_py(){ local c; for c in "${PYTHON:-}" "$WFS/.venv/bin/python" "$HOME/.venv/bin/python" "$(command -v python3)" /usr/bin/python3; do [ -n "$c" ] && [ -x "$c" ] && "$c" -c 'import torch' >/dev/null 2>&1 && { echo "$c"; return; }; done; echo "${PYTHON:-$(command -v python3 || echo /usr/bin/python3)}"; }
PY="$(pick_py)"
C="CONFIG=rgbnir TRAIN=korea INIT=imagenet EPOCHS=40 DECODER=unet SEED=1 LOSS=tversky_bce"
echo "=== mit_b1 (SegFormer enc) ==="
env $C ENCODER=mit_b1 SAVE=et02_mitb1_s1 $PY src/smp_cd.py 2>&1 | grep -E "결과|F1|Error|Traceback" | tail -2
echo "=== efficientnet-b3 ==="
env $C ENCODER=timm-efficientnet-b3 SAVE=et02_effb3_s1 $PY src/smp_cd.py 2>&1 | grep -E "결과|F1|Error|Traceback" | tail -2
echo "=== ENC BATCH DONE ==="
