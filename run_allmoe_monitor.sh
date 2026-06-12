#!/usr/bin/env bash
set -euo pipefail

cd /root/autoresearch-moe
mkdir -p run_logs

MONITOR_LOG=run_logs/allmoe_ablation_monitor.log
LOGS=(
  run_logs/computeS75_allmoe_fullint_lr003_h64.log
  run_logs/computeF1_allmoe_fullint_lr003.log
  run_logs/computeF2_allmoe_fullint_lr001.log
)

find_active_log() {
  local path
  for path in "${LOGS[@]}"; do
    if [[ -f "$path" ]] && ! tr '\r' '\n' < "$path" | rg -q '^val_bpb:'; then
      echo "$path"
      return
    fi
  done
}

sample_once() {
  local active_log
  active_log="$(find_active_log || true)"
  {
    echo
    echo "===== $(date -Iseconds) ====="
    tmux ls || true
    ps -eo pid,ppid,stat,etime,cmd | rg 'torchrun|train.py|run_allmoe_ablation' || true
    echo "--- driver"
    tail -n 20 run_logs/allmoe_ablation_driver.log 2>/dev/null || true
    if [[ -n "$active_log" ]]; then
      echo "--- active_log: $active_log"
      tail -c 120000 "$active_log" \
        | tr '\r' '\n' \
        | rg 'step [0-9]+|val_bpb|train_ce_loss|train_total_loss|router_entropy|expert_load_cv|max_expert_load|router_z_loss|peak_vram|mfu_percent|total_tokens_M|num_steps' \
        | tail -n 35 || true
    else
      echo "--- active_log: none"
      for path in "${LOGS[@]}"; do
        if [[ -f "$path" ]]; then
          echo "--- summary: $path"
          tr '\r' '\n' < "$path" \
            | rg 'val_bpb|train_ce_loss|train_total_loss|num_steps|total_tokens_M|active_params_M|mfu_percent|mean_expert_load_cv|max_layer_max_expert_load' \
            | tail -n 12 || true
        fi
      done
    fi
    echo "--- gpu"
    nvidia-smi --query-gpu=index,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,power.limit --format=csv,noheader,nounits || true
  } >> "$MONITOR_LOG"
}

while true; do
  sample_once
  if ! tmux has-session -t allmoe_ablation 2>/dev/null \
    && ! ps -eo cmd | rg -q '[t]orchrun|[t]rain.py'; then
    break
  fi
  sleep 600
done
