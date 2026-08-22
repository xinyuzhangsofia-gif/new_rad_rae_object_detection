#!/usr/bin/env bash

# Compatibility wrapper: the former three-GPU mode exhausted host RAM during
# AP computation. Keep old launch commands safe by delegating to the sequential
# single-process evaluator.
exec "$(dirname "$0")/evaluate_model7_seq1_58_single_process.sh" "$@"
