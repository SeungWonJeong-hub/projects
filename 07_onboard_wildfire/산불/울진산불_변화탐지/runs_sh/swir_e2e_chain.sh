#!/usr/bin/env bash
# SWIR 포기 비용 **끝단** 측정 (§7 ③) — swir_cost_chain.sh 가 못 한 절반을 채운다.
#
# 그 스크립트는 베이스 in-domain(산림청 자)까지만 봤다. 성능을 만드는 CIR 봄 rehearsal 을
# 12ch 모델에 줄 수 없었기 때문이다 — CIR-GT 타일이 8채널(RGB+NIR)이었다.
# 그런데 빌더는 B11/B12 를 **이미 내려받고 있었다**(BANDS, GT 의 dNBR 계산에 쓴다). 저장만
# 안 했을 뿐이라 재취득 비용이 없다 → BANDS12=1 로 12ch 판을 따로 만든다(8ch 판 무손상).
#
# 통제: 12ch arm 과 8ch arm 은 **같은 s50 베이스 세대 · 같은 확장풀 · 같은 레시피**다.
#   12ch : BASE=runs/et02_20m_s50_bi12_*  CONFIG=bitemporal  CIR=gt_stac_20m_raw12.npz
#    8ch : runs/cir_reh_ens_s50_exp (이미 있음, s50_chain.sh C단계)
# 다른 것은 채널뿐이다. 자는 안동(독립) + CIR 홀드아웃(준순환) 둘 다 본다.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../src"
PY=${PY:-$HOME/envs/torch/bin/python}
GEO=${GEO:-$HOME/miniforge3/envs/geo/bin/python}
ts(){ date +%H:%M:%S; }

echo "##### A) CIR-GT 12밴드 재구축 (24화재)  $(ts)"
BANDS12=1 GSD=20 NO_OFFSET=1 OMP_NUM_THREADS=4 $GEO -u build_cir_gt_stac.py

echo "##### B) 통제 검사 — 8ch 판과 씬·GT 가 같은가  $(ts)"
$PY - <<'PYEOF' || { echo "!!! 통제 깨짐 — 중단"; exit 1; }
import numpy as np, os, sys, glob
F=os.path.join(os.path.dirname(os.path.abspath(".")),"..","산불데이터","국토정보플랫폼 데이터셋","CIR_실측라벨","fires")
F=os.path.abspath(F); bad=[]; n=0
for d in sorted(os.listdir(F)):
    a8=os.path.join(F,d,"gt_stac_20m_raw.npz"); a12=os.path.join(F,d,"gt_stac_20m_raw12.npz")
    if not (os.path.exists(a8) and os.path.exists(a12)): continue
    a,b=np.load(a8),np.load(a12); n+=1
    why=[]
    if b["x"].shape[0]!=12: why.append(f"채널 {b['x'].shape[0]}")
    if str(a["pre"])!=str(b["pre"]) or str(a["post"])!=str(b["post"]): why.append("씬 날짜")
    if not np.array_equal(a["gt"],b["gt"]): why.append("GT")
    if not np.array_equal(a["valid"],b["valid"]): why.append("valid")
    if not (np.allclose(a["x"][:4],b["x"][:4]) and np.allclose(a["x"][4:],b["x"][6:10])): why.append("RGB+NIR 채널")
    if why: bad.append(f"{d}: {'·'.join(why)}")
print(f"검사 {n}화재 · 불일치 {len(bad)}건")
for x in bad: print("  ", x)
sys.exit(1 if bad or n==0 else 0)
PYEOF

echo "##### C) 12ch rehearsal 앙상블 (확장풀 14화재)  $(ts)"
O=cir_reh_ens_bi12_s50_exp
[ -f ../runs/$O/.done ] || {
  BANDS12=1 CONFIG=bitemporal CIR_POOL=expanded KO_DIR=ko_refl_s50 \
    BASE="runs/et02_20m_s50_bi12_*" OUT=$O \
    $PY -u step4b_rehearsal_ens.py 2>&1 | grep -vE "^  \[[0-9]"
  touch ../runs/$O/.done
}

echo "##### D) 끝단 비교 — 안동(독립) + CIR 홀드아웃  $(ts)"
for ARM in "12ch bitemporal|bitemporal|1|$O" "8ch rgbnir(짝)|rgbnir|0|cir_reh_ens_s50_exp"; do
  IFS='|' read -r NAME CFG B12 ENS <<<"$ARM"
  echo "--- $NAME"
  SITE=andong GSD=20 TILE=256 CONFIG=$CFG BANDS12=$B12 \
    $PY -u nifos_eval.py --ens "runs/$ENS/*.pt" 2>&1 | grep -E '"f1"|"pred_ha"|"thr"|경고'
  for G in large small; do
    GSD=20 TILE=256 GROUP=$G CONFIG=$CFG BANDS12=$B12 \
      $PY cir_hold_eval.py --ens "runs/$ENS/*.pt" 2>&1 | tail -4
  done
done
echo "@@@SWIRE2E DONE@@@ $(ts)"
