#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_BIN="${MVRSS_CONDA_BIN:-/home/local/miniconda3/condabin/conda}"
CHECKPOINT_ROOT="checkpoints/object_detection/20260815_134341_021343__model_7__seq1-58"
REPORT_DIR="evaluation_plots/model7_seq1-58_axis_aligned"
LOG_DIR="runs/model7_seq1-58_eval_axis_aligned"
PHYSICAL_GPU=2

cd "$PROJECT_DIR" || exit 1
mkdir -p "$REPORT_DIR" "$LOG_DIR"

for epoch in $(seq 5 100); do
    epoch_tag=$(printf '%03d' "$epoch")
    report_path="$REPORT_DIR/epoch_${epoch_tag}.txt"
    log_path="$LOG_DIR/epoch_${epoch_tag}.log"

    if [[ -s "$report_path" ]]; then
        printf 'Skipping completed epoch %s: %s\n' "$epoch" "$report_path"
        continue
    fi

    printf 'Evaluating epoch %s on physical GPU %s\n' "$epoch" "$PHYSICAL_GPU"
    if ! CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" "$CONDA_BIN" run \
        --no-capture-output -n mvrss python evaluation.py \
        --checkpoint-root "$CHECKPOINT_ROOT" \
        --start-epoch "$epoch" \
        --end-epoch "$epoch" \
        --epoch-step 1 \
        --split-mode kradar_file \
        --split-dir experiments/controlled_splits \
        --batch-size 32 \
        --num-workers 0 \
        --gpu-ids 0 \
        --cuda cuda:0 \
        --model-type auto \
        --box-coordinate-mode auto \
        --eval-coordinate-mode auto \
        --official-eval-version revised \
        --official-eval-iou-mode easy \
        --official-eval-iou-backend axis_aligned \
        --official-detection-metrics-enabled false \
        --custom-iou-range-eval-enabled false \
        --distance-quartile-eval-enabled false \
        --nuscenes-style-eval-enabled false \
        --group-checkpoint-plot-best-only false \
        --ap-score-thresh 0.01 \
        --plot-output none \
        --table-txt-enabled true \
        --table-output-base-dir "$REPORT_DIR" \
        --eval-report-path "$report_path" \
        --evaluation-tensorboard-log-dir "$LOG_DIR/tensorboard" \
        > "$log_path" 2>&1; then
        printf 'Epoch %s failed; see %s\n' "$epoch" "$log_path" >&2
        exit 1
    fi

    printf 'Completed epoch %s: %s\n' "$epoch" "$report_path"
done

printf 'Completed all checkpoint evaluations (epochs 5-100).\n'
