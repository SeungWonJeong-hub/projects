#!/usr/bin/env bash
# 10m 단일 unet_s1 재브래킷 재학습 — systemd 유저 서비스로 세션 독립 실행.
set -euo pipefail
cd "$(dirname "$0")"
BASE10="$(readlink -f ../../웹앱/weights/et02_ens_10m/et02_10mT512_s50_unet_s1_reh.pt)"
export GSD=10 CIR_RAW=1 T=256 FOREST_OS=4 BS="${BS:-4}" EPOCHS="${EPOCHS:-10}"
export BASE_PT="$BASE10" RB_NPZ=gt_stac_rb_10m.npz OUT_PT="${OUT_PT:-unet_rb_fo4_10m.pt}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /usr/bin/python3 -u src/retrain_recovered.py
