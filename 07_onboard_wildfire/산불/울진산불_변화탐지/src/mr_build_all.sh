#!/usr/bin/env bash
# --- 이식성(Jetson·B200 공통): 라이브러리·경로·인터프리터 자동선택 ---
WFS="${WILDFIRE_SEG_ROOT:-$HOME/wildfire-seg}"
[ -d "$WFS/libs" ] && export LD_LIBRARY_PATH="$WFS/libs:${LD_LIBRARY_PATH:-}"
pick_py(){ local c; for c in "${PYTHON:-}" "$WFS/.venv/bin/python" "$HOME/.venv/bin/python" "$(command -v python3)" /usr/bin/python3; do [ -n "$c" ] && [ -x "$c" ] && "$c" -c 'import torch' >/dev/null 2>&1 && { echo "$c"; return; }; done; echo "${PYTHON:-$(command -v python3 || echo /usr/bin/python3)}"; }
PY="$(pick_py)"
export PYTHONNOUSERSITE=1 CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif GDAL_HTTP_MAX_RETRY=3 GDAL_HTTP_RETRY_DELAY=2
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.." || exit 1
export PYTHONPATH=$PWD
for f in australia2019 chile2024 turkey2021 california2021 uljin2022; do
  echo "===== $f ====="
  timeout 500 "$PY" src/mr_build.py $f 2>&1 | grep -aiE "\[$f\]" | tail -2
done
echo "=== 빌드 완료 ==="; for d in mr_data/*/; do echo "$(basename $d): $(ls $d/*.npz 2>/dev/null | wc -l) 타일"; done
