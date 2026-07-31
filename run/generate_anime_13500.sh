#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python3"
SOURCE_LIST="out/sfhq_sources_13500.txt"
SOURCE_REPORT="out/sfhq_sources_13500.json"
OUTPUT_DIR="out/pairs_anime12_13500"
LOG_FILE="out/pairs_anime12_13500.log"
EXPECTED_IMAGES=13500
SPACE_REVISION="7ebfd54af78db89c60188434122c57863780abd0"

if [[ ! -x "$PYTHON" ]]; then
    echo "[error] missing virtualenv Python: $PYTHON" >&2
    exit 1
fi

selected=0
if [[ -f "$SOURCE_LIST" ]]; then
    selected="$(wc -l < "$SOURCE_LIST" | tr -d ' ')"
fi

if [[ "$selected" != "$EXPECTED_IMAGES" ]]; then
    "$PYTHON" run/select_sfhq_sources.py \
        --csv input/sfhq_t2i/SFHQ_T2I_dataset.csv \
        --images input/sfhq_t2i/images/images \
        --out "$SOURCE_LIST" \
        --report "$SOURCE_REPORT" \
        --n "$EXPECTED_IMAGES" \
        --seed 0 \
        --ratios "adult=0.80,senior=0.10,teen=0.05,child=0.05" \
        --max-texture-ratio 0.05
fi

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"
echo "[disk]"
df -h /
echo "[start] $(date --iso-8601=seconds) output=$OUTPUT_DIR"
echo "[resume] Re-run this script after an interruption; completed PNG+JSON pairs are skipped."

trap 'echo "[interrupted] Re-run run/generate_anime_13500.sh to continue."' INT TERM

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -u run/test_space_exact.py \
    --input input/sfhq_t2i/images/images \
    --include-file "$SOURCE_LIST" \
    --out "$OUTPUT_DIR" \
    --n "$EXPECTED_IMAGES" \
    --sample-mode uniform \
    --prompt "Transform into anime." \
    --seed 0 \
    --seed-mode fixed \
    --steps 4 \
    --cfg 1.0 \
    --style-scale 1.2 \
    --int8-transformer \
    --resume \
    --space-revision "$SPACE_REVISION" \
    2>&1 | tee -a "$LOG_FILE"

