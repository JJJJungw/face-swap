#!/usr/bin/env bash
# soup075 + --aug-mix 파인튜닝 (단일 변수).
#
# 왜 스크립트인가: 백슬래시 줄바꿈이 들어간 긴 명령은 tmux 붙여넣기에서
# 중간이 잘려 `--aug-mix: expected one argument` 로 죽는다(2026-08-06 두 번).
# 재현이 필요한 학습 명령은 붙여넣지 말고 저장소에 넣고 git pull 로 가져온다.
#
# 사용: bash run/train_augmix.sh
set -euo pipefail
cd "$(dirname "$0")/.."

INIT=${INIT:-train/soup/eq_tgt3k_a075.pt}
OUT=${OUT:-train/soup075_augmix}
MIX=${MIX:-0:0.7,1:0.2,2:0.1}

[ -f "$INIT" ] || { echo "init ckpt 없음: $INIT"; exit 1; }
echo "[run] init=$INIT out=$OUT aug_mix=$MIX"

python3 -u train/train_student.py \
  --data out/pairs_anime12_13500 \
  --localize-manifest out/localface_idx_occ65/manifest.jsonl \
  --out "$OUT" \
  --init-ckpt "$INIT" \
  --gen-arch deep8 --gen-ch 32 --size 512 --batch 8 --amp bf16 \
  --lr 1e-4 --steps 8000 --init-steps 500 --adv-ramp 1000 \
  --w-l1 1.0 --w-perc 2.5 --w-adv 1.5 --w-edge 3.0 --edge-mode sobel-ms \
  --w-equiv 10 --equiv-shift 4 --equiv-scale 0.02 --equiv-rot 2 \
  --aug-mix "$MIX" \
  --d-ch 48 --d-n 3 --val-n 128 \
  2>&1 | tee "out/train_$(basename "$OUT").log"
