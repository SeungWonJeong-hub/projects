#!/usr/bin/env bash
# --- 이식성(Jetson·B200 공통): 라이브러리·경로·인터프리터 자동선택 ---
WFS="${WILDFIRE_SEG_ROOT:-$HOME/wildfire-seg}"
[ -d "$WFS/libs" ] && export LD_LIBRARY_PATH="$WFS/libs:${LD_LIBRARY_PATH:-}"
pick_py(){ local c; for c in "${PYTHON:-}" "$WFS/.venv/bin/python" "$HOME/.venv/bin/python" "$(command -v python3)" /usr/bin/python3; do [ -n "$c" ] && [ -x "$c" ] && "$c" -c 'import torch' >/dev/null 2>&1 && { echo "$c"; return; }; done; echo "${PYTHON:-$(command -v python3 || echo /usr/bin/python3)}"; }
PY="$(pick_py)"
export PYTHONNOUSERSITE=1 CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif GDAL_HTTP_MAX_RETRY=3 GDAL_HTTP_RETRY_DELAY=2
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.." || exit 1; export PYTHONPATH=$PWD
for f in hermitspeak_nm2022 pearlhill_wa2020 bush_az2020 cameronpeak_co2020 richardspring_mt2021 moose_id2022; do
  echo "===== $f ====="; timeout 500 "$PY" src/mtbs_build.py $f 2>&1 | grep -aiE "\[$f\]" | tail -2
done
echo "=== 실측 라벨 지역 완성 ==="; for d in mtbs_data/*/; do echo "$(basename $d): $(ls $d/*.npz 2>/dev/null|wc -l)타일"; done
