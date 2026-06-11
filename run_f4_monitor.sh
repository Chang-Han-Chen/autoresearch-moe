#!/usr/bin/env bash
set -euo pipefail

cd /root/autoresearch-moe

OUT=run_logs/f4_monitor.log
INTERVAL_SECONDS=600
LOGS=(
  run_logs/computeF4_lr001_full400_db16.log
  run_logs/computeF4_lr0003_full400_db16.log
)

snapshot() {
  local active_log=""
  for path in "${LOGS[@]}"; do
    if [[ -s "$path" ]]; then
      active_log="$path"
    fi
  done

  {
    echo "===== $(date -Iseconds) ====="
    tmux list-sessions 2>/dev/null || true
    ps -eo pid,ppid,stat,etime,cmd | rg 'torchrun|train.py|run_f4_lr_sweep' || true

    if [[ -n "$active_log" ]]; then
      echo "--- active_log: $active_log"
      tail -c 120000 "$active_log" \
        | sed -e 's/\r/\n/g' \
        | rg 'step [0-9]+|val_bpb|train_ce_loss|train_total_loss|router_entropy|expert_load_cv|max_expert_load|router_z_loss|peak_vram|mfu_percent|total_tokens_M|num_steps' \
        | tail -30
    else
      echo "--- active_log: none yet"
    fi

    echo "--- gpu"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits
    echo
  } >> "$OUT"
}

: > "$OUT"
while true; do
  snapshot
  sleep "$INTERVAL_SECONDS"
done
