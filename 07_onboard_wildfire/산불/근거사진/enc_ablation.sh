#!/bin/bash
WFS=$HOME/wildfire-seg; PY=$WFS/.venv/bin/python
cd ~/syntax_inference/Steven/산불/울진산불_변화탐지
SC=/tmp/claude-1000/-home-syntax-kmu/96dc9619-541e-4fa1-8e2e-b6cafa39bb30/scratchpad
export LD_LIBRARY_PATH="$WFS/libs"
echo "ABLATION START $(date +%H:%M:%S)"
for ENC in resnet18 resnet34 resnet50; do
  for S in 1 2 3; do
    TAG=enc_${ENC}_s${S}
    env ENCODER=$ENC DECODER=unet TRAIN=korea INIT=imagenet CONFIG=bitemporal MIN_HA=500 SEED=$S EPOCHS=40 SAVE=$TAG \
      $PY -u src/smp_cd.py 2>&1 | grep -vE "Warning|Future|warn" | grep -E "결과|봄:|여름:|가을:|겨울:|param" | sed "s/^/[$TAG] /"
    echo "[$TAG] done $(date +%H:%M:%S)"
  done
done
echo "ABLATION DONE $(date +%H:%M:%S)"
