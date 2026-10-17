#!/usr/bin/env bash
# MTBS 전량(2015-2024 CONUS 산불) 패치 빌드 — 재개가능·견고(죽어도 이어받기).
export LC_ALL=C.UTF-8 LANG=C.UTF-8
cd "$HOME/syntax_inference/Steven/산불/mtbs_pretrain" || exit 1
LOG="$HOME/.mtbs_build.log"; DONE="$HOME/.mtbs_build.done"; LOCK="$HOME/.mtbs_build.lock"
# 파이썬·동시성은 오버라이드 가능(B200: MTBS_PY=... MTBS_WORKERS=16). 기본=젯슨 venv, 직렬.
PY="${MTBS_PY:-$HOME/wildfire-seg/.venv/bin/python}"
WORKERS="${MTBS_WORKERS:-6}"
[ -f "$DONE" ] && exit 0
exec 9>"$LOCK"; flock -n 9 || exit 0        # 중복 실행 방지
ts(){ TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST'; }
echo "[$(ts)] MTBS 전량 빌드 시작 (pid $$)" >> "$LOG"
for y in 2017 2018 2019 2020 2021 2022 2023 2024; do
  echo "[$(ts)] ===== YEAR $y =====" >> "$LOG"
  $PY src/mtbs_pretrain_build.py --batch --year "$y" --min_acres 1000 --workers "$WORKERS" >> "$LOG" 2>&1
done
touch "$DONE"
echo "[$(ts)] === 전량 빌드 완료 → sentinel ===" >> "$LOG"
