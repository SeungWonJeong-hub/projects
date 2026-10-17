#!/usr/bin/env bash
# 산불 CD 승격 게이트 주기 실행 (crontab 용).
# 미학습 홀드아웃 F1>=0.70 달성 시에만 weights 복사+로컬 커밋(+push 시도, 실패해도 무해).
# push 자격증명은 cron 에 없으므로 실패시 로컬 커밋만 남고 Claude 세션이 마무리한다.
#
# 이식성(젯슨/B200 공통):
#   - 경로: 하드코딩 없이 스크립트 위치(BASH_SOURCE)에서 역산 → 어느 홈에서도 동작.
#   - 파이썬: $PYTHON 오버라이드 > 후보 venv > PATH python3 > /usr/bin/python3 중
#             'import torch' 가 실제로 되는 첫 인터프리터를 자동 선택(B200 시스템파이썬 torch 부재 대비).
set -u
export PATH=/usr/local/bin:/usr/bin:/bin:${PATH:-}
export GIT_TERMINAL_PROMPT=0

# --- 경로 완전 상대화 -------------------------------------------------------
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../src
PROJ="$(cd "$SRC/.." && pwd)"                                 # 울진산불_변화탐지
LOG="$PROJ/runs/promote_gate.log"
mkdir -p "$PROJ/runs"

# --- torch 가 되는 파이썬 자동 선택 ----------------------------------------
pick_python() {
  local cands=()
  [ -n "${PYTHON:-}" ] && cands+=("$PYTHON")
  cands+=(
    "$PROJ/.venv/bin/python"
    "$HOME/wildfire-seg/.venv/bin/python"
    "$HOME/.venv/bin/python"
  )
  # PATH 상의 python3, 그리고 시스템 파이썬(젯슨 fallback)
  command -v python3 >/dev/null 2>&1 && cands+=("$(command -v python3)")
  cands+=(/usr/bin/python3)

  for py in "${cands[@]}"; do
    [ -x "$py" ] || continue
    if "$py" -c "import torch" >/dev/null 2>&1; then
      echo "$py"; return 0
    fi
  done
  # torch 되는 게 하나도 없으면 최소한 실행 가능한 python3 라도 반환(로그에 실패가 남게)
  for py in "${cands[@]}"; do [ -x "$py" ] && { echo "$py"; return 0; }; done
  return 1
}

PY="$(pick_python)" || { echo "[$(date '+%F %T')] no usable python found" >> "$LOG"; exit 1; }

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="
  echo "python: $PY"
  "$PY" "$SRC/promote_on_threshold.py"
} >> "$LOG" 2>&1
