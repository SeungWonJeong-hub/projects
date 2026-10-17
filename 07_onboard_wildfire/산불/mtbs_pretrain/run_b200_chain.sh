#!/usr/bin/env bash
# B200 전용 체인 — 빌드 완료를 기다렸다가 학습·한국검증(·젯슨 전송)까지 자동으로 잇는다.
#   EPOCHS=60 OUT=mtbs_sev_b200.pt LOAD=memory SEND_JETSON=1 bash run_b200_chain.sh
#
# B200 고유의 함정들을 여기에 박아둔다:
#  ① PROJ_DATA — ml 환경엔 PROJ 가 둘 있고 둘 다 GDAL 과 안 맞는다.
#     ml/share/proj 는 열리지 않고 pyproj 번들은 DATABASE.LAYOUT.VERSION.MINOR=4 (>=6 필요).
#     geo 환경의 proj.db 가 ml 의 libproj 와 맞는다. (빌드도 PROJ_DATA 를 export 하고 띄울 것)
#  ② data/ 디렉터리가 없으면 화재리스트 다운로드가 FileNotFoundError 로 즉사한다.
#  ③ run_mtbs_build_all.sh 는 연도 루프가 전부 실패해도 touch $DONE 한다.
#     그래서 .done 만 믿으면 안 되고 패치 개수를 같이 본다.
#  ④ 빌드 대기를 pgrep -f 로 하면 안 된다. 이 스크립트를 heredoc 으로 만들어 띄운 부모 셸의
#     명령줄에 본문 전체가 들어 있어 pgrep 가 그 셸을 빌드로 잡는다(2026-09-11, 빌드는 04:07 에
#     끝났는데 체인이 09:08 까지 멈춤). 잠금만 봐도 안 된다 — 체인이 빌드보다 먼저 뜨면 잠금이
#     비어 있어 곧장 지나간다. **.done 이 생기고 잠금이 풀릴 때까지** 기다린다.
#  ⑤ 빌드는 patches/ 에 같은 FIRE_ID 가 있으면 건너뛴다(재개). 재빌드하려면 옛 패치를 치워야 한다.
set -u
ROOT="$HOME/syntax_inference/Steven/산불/mtbs_pretrain"
LOG="$HOME/.mtbs_chain.log"
GEO_PROJ="$HOME/miniforge3/envs/geo/share/proj"
PY_ML="$HOME/miniforge3/envs/ml/bin/python"
MIN_PATCH="${MIN_PATCH:-2000}"          # 이 밑이면 빌드가 실패한 것으로 본다
EPOCHS="${EPOCHS:-30}"; BS="${BS:-32}"; LOAD="${LOAD:-auto}"; OUT="${OUT:-mtbs_sev_best.pt}"
DONE="$HOME/.mtbs_build.done"; LOCK="$HOME/.mtbs_build.lock"
ts(){ TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST'; }
say(){ echo "[$(ts)] $*" >> "$LOG"; }

cd "$ROOT" || exit 1
say "체인 시작 (pid $$) · epochs $EPOCHS · bs $BS · load $LOAD · out $OUT"

# 1) 빌드 대기 -------------------------------------------------------------
until [ -f "$DONE" ] && flock -n "$LOCK" true; do sleep 120; done   # ④ 참고 — pgrep 금지
N=$(ls patches 2>/dev/null | wc -l)
say "빌드 종료 — 패치 $N 개"
if [ "$N" -lt "$MIN_PATCH" ]; then
  say "★패치가 $MIN_PATCH 미만 — 빌드 불완전으로 판단. 학습 중단."
  say "   (run_mtbs_build_all.sh 의 무조건 touch \$DONE 버그 때문에 .done 은 믿을 수 없다)"
  exit 1
fi

# 2) 학습 -----------------------------------------------------------------
say "학습 시작 — epochs $EPOCHS · bs $BS"
PROJ_DATA="$GEO_PROJ" "$PY_ML" src/mtbs_pretrain_train.py --epochs "$EPOCHS" --bs "$BS" --load "$LOAD" --out "$OUT" >> "$LOG" 2>&1
rc=$?; say "학습 종료 rc=$rc"
CK="$ROOT/ckpt/$OUT"
[ -f "$CK" ] || { say "★ckpt 없음 — 검증 생략"; exit 1; }
say "ckpt $(du -h "$CK" | cut -f1)"

# 3) 한국 검증 (안동·울진) --------------------------------------------------
# GT 는 젯슨 원본의 B200 미러. ~/Steven -> ~/jetson_data/Steven 심링크로 경로를 맞춘다.
say "한국 검증 시작 — 안동·울진 (미 여름 30m -> 한국 봄 20m 전이)"
PROJ_DATA="$GEO_PROJ" "$PY_ML" src/mtbs_validate_korea.py --ckpt "$CK" >> "$LOG" 2>&1
say "검증 종료 rc=$?"

# 4) 젯슨 전송 (선택) -------------------------------------------------------
# tailscale 은 userspace 모드라 system ssh 가 안 된다 — tailscale ssh 래퍼로. 같은 이름이 있으면 옆으로 치운다.
if [ "${SEND_JETSON:-0}" = 1 ]; then
  TS="$HOME/bin/tailscale --socket=$HOME/.local/run/tailscale/tailscaled.sock"
  $TS ssh syntax_kmu@jetson "D=\$HOME/syntax_inference/Steven/산불/mtbs_pretrain/ckpt; mkdir -p \$D; \
    [ -e \$D/$OUT ] && mv \$D/$OUT \$D/${OUT%.pt}_prev_\$(date +%m%d%H%M).pt; \
    cat > \$D/$OUT.part && mv \$D/$OUT.part \$D/$OUT && md5sum \$D/$OUT" < "$CK" >> "$LOG" 2>&1
  say "젯슨 전송 rc=$? · B200 md5 $(md5sum < "$CK" | cut -c1-32)"
fi
say "체인 완료"
