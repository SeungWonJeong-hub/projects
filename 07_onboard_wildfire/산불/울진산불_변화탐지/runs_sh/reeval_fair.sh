#!/usr/bin/env bash
# 제로패딩 수정 전후 숫자가 섞이지 않게, 두 arm 을 **같은 코드**로 재채점한다.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); cd "$HERE/../src"
PY=${PY:-$HOME/envs/torch/bin/python}
if [ -n "${WAIT_PID:-}" ]; then
  echo "대기: PID $WAIT_PID"; while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi
echo "##### 동일 코드 재채점 (제로패딩 수정 반영)  $(date +%H:%M:%S)"
ENS=runs/cir_reh_ens_s50_exp      CONFIG=rgbnir      $PY -u reeval_rehearsal.py 2>&1 | grep -vE "Warning|warn"
ENS=runs/cir_reh_ens_post_s50_exp CONFIG=post_rgbnir $PY -u reeval_rehearsal.py 2>&1 | grep -vE "Warning|warn"
echo "@@@REEVAL DONE@@@ $(date +%H:%M:%S)"
