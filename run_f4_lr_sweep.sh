#!/usr/bin/env bash
set -euo pipefail

cd /root/autoresearch-moe
mkdir -p run_logs

STEPS=43122
TORCHRUN=/opt/conda/bin/torchrun
DRIVER_LOG=run_logs/f4_lr_sweep_driver.log

: > "$DRIVER_LOG"

run_f4() {
  local lr="$1"
  local log_path="$2"

  {
    echo "[$(date -Iseconds)] starting F4 lr=${lr}"
    echo "log=${log_path}"
  } | tee -a "$DRIVER_LOG"

  AR_ADAMW_LR="$lr" \
  AR_MAX_STEPS="$STEPS" \
  AR_ESTIMATED_TOTAL_STEPS="$STEPS" \
  "$TORCHRUN" --standalone --nproc_per_node=4 train.py 2>&1 | tee "$log_path"

  echo "[$(date -Iseconds)] finished F4 lr=${lr}" | tee -a "$DRIVER_LOG"
}

run_f4 "0.001" "run_logs/computeF4_lr001_full400_db16.log"
run_f4 "0.0003" "run_logs/computeF4_lr0003_full400_db16.log"

echo "[$(date -Iseconds)] finished F4 LR sweep" | tee -a "$DRIVER_LOG"
