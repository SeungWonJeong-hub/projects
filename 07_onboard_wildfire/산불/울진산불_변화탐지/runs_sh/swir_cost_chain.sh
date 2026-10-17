#!/usr/bin/env bash
# SWIR 포기 비용 — 12ch(bitemporal, SWIR 포함) vs 8ch(rgbnir, ET-02) **동일 세대 데이터** 비교.
#
# ⚠️ 2026-09-09: **이 스크립트의 결론(산림청 자로 SWIR +0.038)은 뒤집혔다.**
#    산림청 폴리곤은 경계 과대라벨이라 "넓게 칠하는 모델"이 유리하고, 순위를 믿을 수 없다.
#    rehearsal 까지 붙인 끝단 비교에서는 SWIR 이 독립 자(NIFoS 안동)에 **−0.026 으로 진다**
#    → runs_sh/swir_e2e_chain.sh · SESSION_2026-09-09 §6e. 여기 숫자는 부호도 인용하지 말 것.
#
# 왜 베이스만 비교하나: CIR-GT 타일(gt_stac*.npz)이 **8채널(RGB+NIR)** 이라 SWIR 이 없다.
# 성능을 만드는 CIR 봄 rehearsal 을 12ch 모델에 줄 수 없다 → 끝단(NIFoS) 비교를 하려면
# CIR-GT 를 12밴드로 재구축해야 한다(별도 작업). 여기서는 같은 ko_refl_s50 · 같은 레시피 ·
# 같은 시드로 **베이스 in-domain(산림청 홀드아웃)** 만 통제 비교한다.
# 기존 기록(SESSION_2026-08-25: 12밴드 0.650 vs 8ch 0.651)은 데이터 세대가 달라 통제가 없었다.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../src"
PY=${PY:-$HOME/envs/torch/bin/python}
ts(){ date +%H:%M:%S; }
# 앞선 작업이 GPU 를 쓰고 있으면 끝날 때까지 기다린다(MIG 1g.23gb 는 슬라이스가 하나뿐이다)
if [ -n "${WAIT_PID:-}" ]; then
  echo "대기: PID $WAIT_PID 종료까지  $(ts)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
  echo "선행 작업 종료 확인  $(ts)"
fi

echo "##### 12ch(bitemporal, SWIR 포함) 베이스 9모델  $(ts)"
for D in unet fpn deeplabv3plus; do for N in 1 2 3; do
  S=et02_20m_s50_bi12_${D}_s${N}
  [ -f ../runs/$S/results.json ] && { echo "  skip $S"; continue; }
  DS_KOREA=ko_refl_s50 MIN_HA=50 CONFIG=bitemporal DECODER=$D SEED=$N EPOCHS=40 \
    SAVE=$S $PY -u smp_cd.py 2>&1 | grep -E "^\[결과\]|^\[smp|Traceback|Error|out of memory"
  echo "  [$S] $(ts)"
done; done

echo "##### 비교 (산림청 in-domain 홀드아웃, 같은 데이터·레시피·시드)  $(ts)"
$PY - <<'PYEOF'
import json,glob,os,statistics as st
P=os.path.join(os.path.dirname(os.path.abspath(".")),"..")
def grab(pat):
    out=[]
    for f in sorted(glob.glob(f"../runs/{pat}/results.json")):
        d=json.load(open(f)); out.append((os.path.basename(os.path.dirname(f)),d["f1"],d["p"],d["r"]))
    return out
for name,pat in (("8ch  rgbnir (ET-02, SWIR 없음)","et02_20m_s50_[udf]*"),
                 ("12ch bitemporal (SWIR 포함)  ","et02_20m_s50_bi12_*"),
                 ("4ch  post_rgbnir (온보드)    ","et02_20m_s50_post_*")):
    v=grab(pat)
    if not v: print(f"{name}: 없음"); continue
    f=[x[1] for x in v]
    print(f"{name}  n={len(v)}  F1 평균 {st.mean(f):.4f} ± {st.pstdev(f):.4f}  (최소 {min(f):.3f} 최대 {max(f):.3f})")
PYEOF
echo "@@@SWIRCOST DONE@@@ $(ts)"
