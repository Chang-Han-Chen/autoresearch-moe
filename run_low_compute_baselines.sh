#!/usr/bin/env bash
set -euo pipefail

cd /root/autoresearch-moe
mkdir -p run_logs

TORCHRUN=/opt/conda/bin/torchrun
DRIVER_LOG=run_logs/low_compute_baseline_driver.log

: > "$DRIVER_LOG"

run_case() {
  local size="$1"
  local curve="$2"
  local depth="$3"
  local model_dim="$4"
  local head_dim="$5"
  local num_heads="$6"
  local num_kv_heads="$7"
  local moe_hidden_dim="$8"
  local steps="$9"
  local adamw_lr="${10}"
  local device_batch_size="${11}"
  local log_path="${12}"

  local dense_early_layers
  local value_mix_enabled
  local exclusive_attention
  local headwise_attention_gate
  local router_sigmoid_affinity
  local router_expert_bias
  local load_balance_loss_coef

  if [[ "$curve" == "full" ]]; then
    dense_early_layers=2
    value_mix_enabled=1
    exclusive_attention=1
    headwise_attention_gate=1
    router_sigmoid_affinity=1
    router_expert_bias=1
    load_balance_loss_coef=0.003
  elif [[ "$curve" == "simple" ]]; then
    dense_early_layers=0
    value_mix_enabled=0
    exclusive_attention=0
    headwise_attention_gate=0
    router_sigmoid_affinity=0
    router_expert_bias=0
    load_balance_loss_coef=0.0085
  else
    echo "unknown curve: $curve" >&2
    exit 2
  fi

  {
    echo "[$(date -Iseconds)] starting ${size} ${curve}"
    echo "log=${log_path}"
    echo "depth=${depth} model_dim=${model_dim} head_dim=${head_dim} num_heads=${num_heads} num_kv_heads=${num_kv_heads} moe_hidden_dim=${moe_hidden_dim}"
    echo "steps=${steps} adamw_lr=${adamw_lr} device_batch_size=${device_batch_size}"
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
  AR_DENSE_EARLY_LAYERS="$dense_early_layers" \
  AR_VALUE_MIX_ENABLED="$value_mix_enabled" \
  AR_EXCLUSIVE_ATTENTION="$exclusive_attention" \
  AR_HEADWISE_ATTENTION_GATE="$headwise_attention_gate" \
  AR_ROUTER_SIGMOID_AFFINITY="$router_sigmoid_affinity" \
  AR_ROUTER_EXPERT_BIAS="$router_expert_bias" \
  AR_LOAD_BALANCE_LOSS_COEF="$load_balance_loss_coef" \
  AR_ROUTER_Z_LOSS_COEF=0.00075 \
  AR_ADAMW_LR="$adamw_lr" \
  AR_MAX_STEPS="$steps" \
  AR_ESTIMATED_TOTAL_STEPS="$steps" \
  AR_DEVICE_BATCH_SIZE="$device_batch_size" \
  "$TORCHRUN" --standalone --nproc_per_node=4 train.py 2>&1 | tee "$log_path"

  echo "[$(date -Iseconds)] finished ${size} ${curve}" | tee -a "$DRIVER_LOG"
}

run_case "S75" "full"   7  640  64 10 2 2176  5800 0.003 32 "run_logs/computeS75_full_lr003_h64.log"
run_case "S75" "simple" 7  640  64 10 2 2176  5800 0.003 32 "run_logs/computeS75_simple_lr003_h64.log"
run_case "F1"  "simple" 8  768 128  6 2 1792  7000 0.003 32 "run_logs/computeF1_simple_lr003.log"
run_case "F2"  "simple" 10 1024 128  8 2 2304 14099 0.003 32 "run_logs/computeF2_simple_lr003.log"

echo "[$(date -Iseconds)] finished low-compute baseline suite" | tee -a "$DRIVER_LOG"
