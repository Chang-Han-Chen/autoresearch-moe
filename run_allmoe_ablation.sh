#!/usr/bin/env bash
set -euo pipefail

cd /root/autoresearch-moe
mkdir -p run_logs

TORCHRUN=/opt/conda/bin/torchrun
DRIVER_LOG=run_logs/allmoe_ablation_driver.log

: > "$DRIVER_LOG"

run_case() {
  local size="$1"
  local depth="$2"
  local model_dim="$3"
  local head_dim="$4"
  local num_heads="$5"
  local num_kv_heads="$6"
  local moe_hidden_dim="$7"
  local steps="$8"
  local adamw_lr="$9"
  local device_batch_size="${10}"
  local log_path="${11}"

  {
    echo "[$(date -Iseconds)] starting ${size} allmoe_full_interventions"
    echo "log=${log_path}"
    echo "depth=${depth} model_dim=${model_dim} head_dim=${head_dim} num_heads=${num_heads} num_kv_heads=${num_kv_heads} moe_hidden_dim=${moe_hidden_dim}"
    echo "steps=${steps} adamw_lr=${adamw_lr} device_batch_size=${device_batch_size}"
    echo "dense_early_layers=0 value_mix=1 exclusive_attention=1 headwise_attention_gate=1 router_sigmoid_affinity=1 router_expert_bias=1 load_balance=0.003"
  } | tee -a "$DRIVER_LOG"

  AR_DEPTH="$depth" \
  AR_MODEL_DIM="$model_dim" \
  AR_HEAD_DIM="$head_dim" \
  AR_NUM_HEADS="$num_heads" \
  AR_NUM_KV_HEADS="$num_kv_heads" \
  AR_NUM_EXPERTS=16 \
  AR_TOP_K=2 \
  AR_MOE_HIDDEN_DIM="$moe_hidden_dim" \
  AR_DENSE_HIDDEN_DIM="$((2 * moe_hidden_dim))" \
  AR_DENSE_EARLY_LAYERS=0 \
  AR_VALUE_MIX_ENABLED=1 \
  AR_EXCLUSIVE_ATTENTION=1 \
  AR_HEADWISE_ATTENTION_GATE=1 \
  AR_ROUTER_SIGMOID_AFFINITY=1 \
  AR_ROUTER_EXPERT_BIAS=1 \
  AR_LOAD_BALANCE_LOSS_COEF=0.003 \
  AR_ROUTER_Z_LOSS_COEF=0.00075 \
  AR_ADAMW_LR="$adamw_lr" \
  AR_MAX_STEPS="$steps" \
  AR_ESTIMATED_TOTAL_STEPS="$steps" \
  AR_DEVICE_BATCH_SIZE="$device_batch_size" \
  "$TORCHRUN" --standalone --nproc_per_node=4 train.py 2>&1 | tee "$log_path"

  echo "[$(date -Iseconds)] finished ${size} allmoe_full_interventions" | tee -a "$DRIVER_LOG"
}

run_case "S75" 7  640  64 10 2 2176  5800 0.003 32 "run_logs/computeS75_allmoe_fullint_lr003_h64.log"
run_case "F1"  8  768 128  6 2 1792  7000 0.003 32 "run_logs/computeF1_allmoe_fullint_lr003.log"
run_case "F2" 10 1024 128  8 2 2304 14099 0.001 32 "run_logs/computeF2_allmoe_fullint_lr001.log"

echo "[$(date -Iseconds)] finished all-MoE full-intervention ablation suite" | tee -a "$DRIVER_LOG"
