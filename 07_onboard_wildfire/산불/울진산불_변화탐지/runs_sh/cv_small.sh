set -u
# 경로는 스크립트 위치에서 유도한다(레포 이식성). 파이썬은 PY env 로 덮어쓸 수 있다.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../src"
PY=${PY:-$HOME/envs/torch/bin/python}
ts(){ date +%H:%M:%S; }

# CIR-GT 소형 9화재(70~290ha, valid>=98%) 3-fold 교차검증.
# 강릉옥계면(237ha)은 valid 41%라 홀드아웃 자격 미달 → 항상 학습에만 쓴다.
BASE6="250325_경북영덕청송,250322_의성안평면,250325_영덕축산면,220304_울진북면,220528_울진근남면,230403_영주평은면"
ALWAYS="220305_강릉옥계면"
F1="250322_울주온양읍,250428_대구노곡동,230402_금산복수면"      #  70 · 84 · 107 ha
F2="230403_순천송광면,230308_합천용주면,220531_밀양부북면"      #  75 ·  93 · 200 ha
F3="220411_군위의흥면,230411_강릉난곡동,230403_함평대동면"      #  76 · 101 · 290 ha
ALL9="$F1,$F2,$F3"
train_pool(){ # 인자=홀드아웃 fold → 나머지 소형 + base6 + always
  local hold=$1 out="$BASE6,$ALWAYS"
  for f in $(echo "$ALL9"|tr ',' ' '); do case ",$hold," in *",$f,"*) ;; *) out="$out,$f";; esac; done
  echo "$out"
}
for K in 1 2 3; do
  eval "HOLD=\$F$K"
  POOL=$(train_pool "$HOLD")
  echo "##### fold$K 홀드아웃=$HOLD  $(ts)"
  for ARM in 20m 10m; do
    if [ "$ARM" = "20m" ]; then G=20; T=256; KO=ko_refl_s50;                 B="runs/et02_20m_s50_[udf]*"
    else                        G=10; T=512; KO=ko_refl_10m_b08_t512_s50c;   B="runs/et02_10mT512_s50_[udf]*"; fi
    O=cv_f${K}_${ARM}
    [ -f ../runs/$O/.done ] || {
      GSD=$G T=$T KO_DIR=$KO CIR_POOL="$POOL" SMALL_HOLD="$HOLD" BASE="$B" OUT=$O \
        $PY -u step4b_rehearsal_ens.py 2>&1 | grep -vE "^  \[[0-9]"
      touch ../runs/$O/.done
    }
    echo "--- fold$K $ARM 소형3 채점"
    GSD=$G TILE=$T GROUP=small SMALL_HOLD="$HOLD" $PY cir_hold_eval.py --ens "runs/$O/*.pt" 2>&1 | tail -8
    echo "--- fold$K $ARM 대형4 채점(비퇴행 확인)"
    GSD=$G TILE=$T GROUP=large SMALL_HOLD="$HOLD" $PY cir_hold_eval.py --ens "runs/$O/*.pt" 2>&1 | tail -3
  done
done
echo "@@@CV DONE@@@ $(ts)"
