# US 전량 수집(20m 배포 계약: B8A@256). 8샤드 병렬. 재개가능 — 샤드별 매니페스트에 있는 건 건너뛴다.
# 경로는 스크립트 위치에서 유도한다(레포 이식성). 파이썬은 PY env 로 덮어쓸 수 있다.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../src"
PY=${PY:-$HOME/miniforge3/envs/geo/bin/python}
L=${LOGDIR:-$HOME/wildfire_run_logs}
for I in 0 1 2 3 4 5 6 7; do
  SHARD_N=8 SHARD_I=$I $PY -u collect_events.py US > $L/collect_us_s$I.log 2>&1 &
done
wait
$PY collect_events.py US --merge > $L/collect_us_merge.log 2>&1
echo "@@@COLLECT DONE@@@ $(date +%H:%M)"
