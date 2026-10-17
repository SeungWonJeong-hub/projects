#!/bin/sh
# 계약 검사 pre-commit 훅 설치. 훅은 .git/ 안에 살아 클론될 때 따라오지 않으므로,
# 내용은 여기(버전관리)에 두고 .git/hooks 에서는 심링크로 가리킨다.
set -e
ROOT=$(git rev-parse --show-toplevel)
SRC="$ROOT/Steven/산불/울진산불_변화탐지/hooks/pre-commit"
DST="$ROOT/.git/hooks/pre-commit"
chmod +x "$SRC"
[ -e "$DST" ] && [ ! -L "$DST" ] && mv "$DST" "$DST.bak-$(date +%Y%m%d%H%M)" && echo "기존 훅 백업"
ln -sf "$SRC" "$DST"
echo "설치 완료: $DST -> $SRC"
"$DST" && echo "동작 확인 OK"
