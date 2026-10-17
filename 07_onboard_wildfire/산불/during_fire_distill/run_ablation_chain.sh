#!/usr/bin/env bash
# ② ablation 자동 체인(Jetson): 취득 완료 대기 → 전체 래스터화 → SWIR ablation → footprint sweep.
# 세션독립 완주용. 로그 ~/.during_ablation.log. LORO 비교 일관성 위해 val=dixie 고정.
set -u
cd /home/syntax_kmu/syntax_inference/Steven/산불/during_fire_distill
PY=/usr/bin/python3
LOG=~/.during_ablation.log
ACQ_PID="${1:-}"

echo "=== [$(date -u +%FT%TZ)] ablation chain 시작 (acquire PID=$ACQ_PID) ===" >>"$LOG"

# 1) 취득 프로세스 종료 대기(있으면)
if [ -n "$ACQ_PID" ]; then
  while kill -0 "$ACQ_PID" 2>/dev/null; do sleep 30; done
  echo "[$(date -u +%FT%TZ)] 취득 완료 감지" >>"$LOG"
fi
N=$(ls data/s2_during/ | grep -v detect | wc -l)
echo "[$(date -u +%FT%TZ)] 이벤트 $N개, 전체 래스터화 시작" >>"$LOG"

# 2) 전체 teacher 라벨 래스터화(기본 반경, 재개=기존 마스크 skip)
EVENTS=all $PY src/rasterize_teacher.py >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] 래스터화 완료" >>"$LOG"

# 3) SWIR ablation(정본 — 재질의 불필요)
echo "[$(date -u +%FT%TZ)] SWIR ablation 시작" >>"$LOG"
WHICH=swir EPOCHS=20 VAL_EVENT=dixie_2021_us $PY src/ablate.py >>"$LOG" 2>&1

# 4) footprint 반경 sweep(반경별 재래스터화 포함 — 느림)
echo "[$(date -u +%FT%TZ)] footprint sweep 시작" >>"$LOG"
WHICH=footprint EPOCHS=20 VAL_EVENT=dixie_2021_us $PY src/ablate.py >>"$LOG" 2>&1

echo "=== [$(date -u +%FT%TZ)] ablation chain 완료 ===" >>"$LOG"
