# 실행 스크립트 — 오늘 돌린 실험을 그대로 재현한다

지금까지 이 스크립트들은 세션 스크래치패드와 `~/wildfire_run_logs/` 에만 있어 레포에 남지 않았다.
결과(SESSION 문서)는 있는데 그 결과를 만든 명령이 없으면 재현이 안 된다 → 여기로 옮겼다.
경로는 스크립트 위치에서 유도하므로 어느 클론에서도 그대로 돈다. `PY`·`LOGDIR` 로 덮어쓸 수 있다.

| 스크립트 | 하는 일 | 소요 |
|---|---|---|
| `s50_chain.sh` | 공정 베이스 18모델(20m·10m) → 양쪽 rehearsal → CIR 홀드아웃 채점 | ~1h55m |
| `cv_small.sh` | 소형 9화재 3-fold 교차검증(fold×arm = 6회 rehearsal + 채점) | ~1h55m |
| `collect_us.sh` | US 카탈로그 3,146화재 8샤드 수집 → 매니페스트 병합 | ~8h |

```
nohup bash runs_sh/s50_chain.sh > ~/wildfire_run_logs/s50_chain.log 2>&1 &
```
진행 `grep -E '^#####|@@@' ~/wildfire_run_logs/*.log` · 결과 해석은 `SESSION_2026-09-08_10m.md` §6b.

**주의**: `s50_chain.sh`·`cv_small.sh` 는 GPU 를 쓴다(MIG 1g.23gb 슬라이스 1개뿐이라 동시 실행 금지).
`collect_us.sh` 는 CPU·네트워크만 쓰므로 학습과 병행해도 된다 — 실제로 오늘 그렇게 돌렸다.
