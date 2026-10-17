#!/usr/bin/env bash
# post-only(pre 없음) 소형 9화재 3-fold 교차검증 — 온보드에서 소형이 특히 무너지는지 본다.
#
# bi-temporal 20m arm 은 이미 있다(runs/cv_f{1,2,3}_20m, §6b) → **같은 fold·같은 s50 베이스 계열·
# 같은 레시피**로 post-only arm 만 만들어 짝을 맞춘다. 다른 건 CONFIG 뿐이다.
#   bi-temporal : BASE=runs/et02_20m_s50_[udf]*  CONFIG=rgbnir      (8ch)
#   post-only   : BASE=runs/et02_20m_s50_post_*  CONFIG=post_rgbnir (4ch)
# fold 정의는 cv_small.sh 와 동일해야 한다(누출·비교가능성) — 여기서 다시 쓰지 말고 그대로 복사.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../src"
PY=${PY:-$HOME/envs/torch/bin/python}
ts(){ date +%H:%M:%S; }

BASE6="250325_경북영덕청송,250322_의성안평면,250325_영덕축산면,220304_울진북면,220528_울진근남면,230403_영주평은면"
ALWAYS="220305_강릉옥계면"
F1="250322_울주온양읍,250428_대구노곡동,230402_금산복수면"      #  70 · 84 · 107 ha
F2="230403_순천송광면,230308_합천용주면,220531_밀양부북면"      #  75 ·  93 · 200 ha
F3="220411_군위의흥면,230411_강릉난곡동,230403_함평대동면"      #  76 · 101 · 290 ha
ALL9="$F1,$F2,$F3"
train_pool(){ local hold=$1 out="$BASE6,$ALWAYS"
  for f in $(echo "$ALL9"|tr ',' ' '); do case ",$hold," in *",$f,"*) ;; *) out="$out,$f";; esac; done
  echo "$out"; }

for K in 1 2 3; do
  eval "HOLD=\$F$K"
  POOL=$(train_pool "$HOLD")
  echo "##### fold$K 홀드아웃=$HOLD  $(ts)"
  O=cv_f${K}_20m_post
  [ -f ../runs/$O/.done ] || {
    GSD=20 T=256 KO_DIR=ko_refl_s50 CIR_POOL="$POOL" SMALL_HOLD="$HOLD" \
      BASE="runs/et02_20m_s50_post_*" CONFIG=post_rgbnir OUT=$O \
      $PY -u step4b_rehearsal_ens.py 2>&1 | grep -vE "^  \[[0-9]"
    touch ../runs/$O/.done
  }
  for G in small large; do
    echo "--- fold$K post-only $G 채점"
    GSD=20 TILE=256 GROUP=$G SMALL_HOLD="$HOLD" CONFIG=post_rgbnir \
      $PY cir_hold_eval.py --ens "runs/$O/*.pt" 2>&1 | tail -6
    echo "--- fold$K bi-temporal $G 채점(짝)"
    GSD=20 TILE=256 GROUP=$G SMALL_HOLD="$HOLD" CONFIG=rgbnir \
      $PY cir_hold_eval.py --ens "runs/cv_f${K}_20m/*.pt" 2>&1 | tail -6
  done
done
echo "@@@CVPOST DONE@@@ $(ts)"
