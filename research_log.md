# Research Log: Simple MoE Backbone Phase

This log was reset after removing unproven extras from the working stack. The old lineage remains available in raw run logs and git history. The leaderboard uses the LR-bracketed clean backbone (`02ba694`) as the quantitative zero point for this phase.

## Fixed-Wall Leaderboard

This table tracks fixed-wall runs that became the new healthy working best. Percent improvement is relative BPB reduction: `(old_bpb - new_bpb) / old_bpb`. The comparison baseline is `02ba694`, the clean simple backbone after the LR bracket (`val_bpb 0.954881`), rather than the deliberately rough init run.

| rank | commit | new fixed-wall best | val_bpb | improvement vs previous best | cumulative vs clean baseline | note |
|---:|---|---|---:|---:|---:|---|
| 0 | `02ba694` | clean simple backbone, AdamW peak LR `0.003` | `0.954881` | -- | `0.000%` | baseline for this phase |
| 1 | `ec33c7d` | fixed value mix `0.5*v1 + 0.5*vl` | `0.947597` | `0.763%` | `0.763%` | proved fixed value memory helps |
| 2 | `188138f` | fixed value mix `0.75*v1 + 0.25*vl` | `0.945791` | `0.191%` | `0.952%` | first-heavy mix became value baseline |
| 3 | `4a7630d` | sigmoid affinities + expert bias + load-balance `0.003` | `0.942848` | `0.311%` | `1.260%` | repaired zero-LB collapse; router healthy |
| 4 | `3038052` | exclusive self-attention | `0.940518` | `0.247%` | `1.504%` | first clear attention-side win |
| 5 | `5d210c5` | headwise attention gate, init `0.95` | `0.940352` | `0.018%` | `1.522%` | small but coherent gain |
| 6 | `48fa497` | headwise attention gate, init `0.98` | `0.940076` | `0.029%` | `1.550%` | best no-dense attention stack |
| 7 | `cbbe61b` | first FFN layer dense | `0.938510` | `0.167%` | `1.714%` | faster, fewer total params, clean routing |
| 8 | `f48fb15` | first two FFN layers dense | `0.938179` | `0.035%` | `1.749%` | current fixed-wall best |

Matched-step note: the two-dense stack stopped at the no-dense step count (`2515`) reached `0.939195`, beating the no-dense `0.940076` by `0.094%`. That supports a real sample-efficiency gain, but it is intentionally not a leaderboard row because it is not the fixed-wall protocol.

## Sparsity Sweep Leaderboard

| num_experts | sparsity ratio | val_bpb | steps | tokens_M | total_params_M | active_params_M | peak_vram_gb | mfu_percent | decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | `50.0%` | `0.947202` | `3114` | `816.3` | `140.8` | `91.3` | `27.6` | `27.61` | discard |
| 8 | `25.0%` | `0.938106` | `3013` | `789.8` | `239.9` | `91.3` | `28.2` | `26.71` | discard |
| 16 | `12.5%` | `0.935178` | `2746` | `719.8` | `438.2` | `91.3` | `29.3` | `24.35` | keep |
| 32 | `6.25%` | `0.938541` | `2308` | `605.0` | `834.6` | `91.4` | `31.6` | `20.46` | discard |
| 64 | `3.125%` | `0.972360` | `1756` | `460.3` | `1627.5` | `91.6` | `36.2` | `15.57` | discard |
| 128 | `1.5625%` | `1.034533` | `1183` | `310.1` | `3213.2` | `91.9` | `45.3` | `10.49` | discard |

## Phase 1 Matched-Step Diagnostics

No additional matched-step jobs were run after the fixed-wall sweep. The fixed-wall winner is the `E=16` anchor itself, no non-winner is within `0.001` BPB of the winner, and the slower higher-expert runs did not show a better same-step training-loss signal strong enough to justify a scale-candidate diagnostic.

## Selected Sparsity Ratio

Selected default:
`NUM_EXPERTS=16`, `TOP_K=2`, sparsity ratio `12.5%`.

Reason:
`E=16` has the best fixed-wall validation result at `0.935178` BPB with clean router health and acceptable throughput. Lower expert counts are faster but lose too much sparse capacity, while higher expert counts remain stable but spend wall time and memory on inactive capacity instead of updates.

Scale candidate, if any:
none

## Fine-Grained Expert Check

| geometry | num_experts | top_k | moe_hidden_dim | dense_hidden_dim | sparsity ratio | val_bpb | steps | tokens_M | total_params_M | active_params_M | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| coarse | 16 | 2 | 1792 | 3584 | `12.5%` | `0.935178` | `2746` | `719.8` | `438.2` | `91.3` | keep |
| fine | 32 | 4 | 896 | 3584 | `12.5%` | `0.953126` | `2608` | `683.7` | `438.2` | `91.4` | discard |

## Selected MoE Geometry

Selected default:
`NUM_EXPERTS=16`, `TOP_K=2`, `MOE_HIDDEN_DIM=1792`, `DENSE_HIDDEN_DIM=3584`.

Reason:
The fine-grained counterpart preserved the intended active width and sparsity ratio, but it was much worse fixed-wall: `0.953126` BPB versus the coarse anchor's `0.935178`. It also completed fewer steps and tokens, used more VRAM, and had lower MFU. Router health was safe, so this is an efficiency and optimization result rather than a collapse.

Scale candidate, if any:
none

## Scale Validation

| depth | model_dim | curve | num_experts | top_k | moe_hidden_dim | val_bpb | steps | tokens_M | total_params_M | active_params_M | peak_vram_gb | mfu_percent |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 768 | full | 16 | 2 | 1792 | `0.935178` | `2746` | `719.8` | `438.2` | `91.3` | `29.3` | `24.35` |
| 10 | 1024 | full | 16 | 2 | 2304 | `0.973316` | `1434` | `375.9` | `977.5` | `184.8` | `45.3` | `25.59` |
| 12 | 1280 | full | 16 | 2 | 3072 | `1.045977` | `751` | `196.9` | `2003.1` | `351.6` | `69.5` | `24.90` |
| 12 | 1280 | full lr=0.001 | 16 | 2 | 3072 | `1.063829` | `754` | `197.7` | `2003.1` | `351.6` | `69.5` | `24.98` |

## Architecture Scaling Deltas

| depth | model_dim | full_bpb | simple_bpb | simple_minus_full | relative_bpb_reduction | no_dense_bpb | no_dense_minus_full |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 640 | `0.902337` | `0.916017` | `0.013680` | `1.493%` | -- | -- |
| 8 | 768 | `0.892152` | `0.892165` | `0.000013` | `0.001%` | -- | -- |
| 10 | 1024 | `0.821625` | `0.844628` | `0.023003` | `2.723%` | -- | -- |

F2 note: the full recipe also beats the simple control at matched AdamW LR `0.003` (`0.822997` vs `0.844628`, delta `0.021631`, `2.561%` relative), so the F2 architecture gap is not only LR selection.

## Scale LR Follow-Up

After the fixed-LR scale/control points, sweep AdamW peak LR locally around any viable larger config with roughly 3x spacing. Use `0.003` as the current anchor, prioritize lower candidates such as `0.001` and `0.0003` as model size increases, and keep the Muon LR tied by the existing shape-aware formula.

## Compute-Optimal Protocol Pivot

The fixed-wall protocol is useful for throughput-aware iteration, but it is not the right protocol for scaling-law decisions. For the selected `~91M` active-parameter model, use a rough compute-optimal token target of `20` tokens per active parameter: `91M * 20 ~= 1.8B` tokens. With `TOTAL_BATCH_SIZE=262,144`, this is `~6866` optimizer steps, rounded to `7000`.

For these runs, use `AR_MAX_STEPS=7000` and `AR_ESTIMATED_TOTAL_STEPS=7000` so the cosine schedule matches the step budget. Sweep AdamW peak LR only on a coarse 3x log grid: `0.000333`, `0.001`, `0.003`.

Data note: from this point forward, use the full downloaded train pool (`400` train shards, pinned validation shard excluded). Exact tokenizer-count samples show about `63.3M` training tokens per shard including BOS, or about `25.3B` train tokens total, so the planned F1-F5 compute-optimal targets stay under one epoch.

### Compute-Optimal Scale Targets

Assuming `20` training tokens per active parameter and `TOTAL_BATCH_SIZE=262,144` tokens:

| size | depth | model_dim | active_params_M | target_tokens_B | target_steps |
|---|---:|---:|---:|---:|---:|
| S75 | 7 | 640 | `76.0` | `1.519` | `5800` |
| F1 | 8 | 768 | `91.3` | `1.827` | `6968` |
| F2 | 10 | 1024 | `184.8` | `3.696` | `14099` |
| F3 | 12 | 1280 | `351.6` | `7.033` | `26827` |
| F4 | 14 | 1536 | `565.2` | `11.304` | `43122` |
| F5 | 16 | 1792 | `852.2` | `17.045` | `65021` |

S75 uses `HEAD_DIM=64`, `NUM_HEADS=10`, `NUM_KV_HEADS=2`, and `MOE_HIDDEN_DIM=2176`.

## Compute-Optimal 90M LR Sweep

| adamw_lr | steps | tokens_B | val_bpb | train_ce | peak_vram_gb | mfu_percent | router note | decision |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| `0.003` | `7000` | `1.835` | `0.892152` | `2.307537` | `29.3` | `24.30` | mean CV `0.0661`, max-layer max load `0.0779` | keep pending LR sweep |
| `0.001` | `7000` | `1.835` | `0.936404` | `2.419355` | `29.3` | `24.33` | mean CV `0.0563`, max-layer max load `0.0743` | discard |

## Compute-Optimal Full-Data Scale Runs

Use this table for scale-law decisions after switching to the full `400`-shard train pool.

| size | depth | model_dim | adamw_lr | steps | tokens_B | val_bpb | train_ce | total_params_M | active_params_M | peak_vram_gb | mfu_percent | router note | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| F2 | 10 | 1024 | `0.003` | `14099` | `3.696` | `0.822997` | `2.263927` | `977.5` | `184.8` | `45.3` | `25.66` | mean CV `0.0850`, max-layer max load `0.0780` | discard; `0.001` better |
| F2 | 10 | 1024 | `0.001` | `14099` | `3.696` | `0.821625` | `2.262125` | `977.5` | `184.8` | `45.3` | `25.69` | mean CV `0.0752`, max-layer max load `0.0753` | selected F2 LR anchor |
| F2 | 10 | 1024 | `0.000333` | `14099` | `3.696` | `0.946376` | `2.610592` | `977.5` | `184.8` | `45.3` | `25.76` | mean CV `0.0404`, max-layer max load `0.0713` | discard; undertrained |
| F3 | 12 | 1280 | `0.001` | `26827` | `7.033` | `0.765974` | `2.178932` | `2003.1` | `351.6` | `69.5` | `25.19` | mean CV `0.0958`, max-layer max load `0.0833` | selected F3 LR anchor |
| F3 | 12 | 1280 | `0.003` | `2600` | `0.682` | -- | -- | `2003.1` | `351.6` | -- | `~25.1` | matched step `2599`: loss `2.7449` vs `2.6545` at LR `0.001`, load CV `0.226` vs `0.098` | aborted; do not test higher |
| F4 | 14 | 1536 | `0.001` | `43122` | `11.304` | `0.736565` | `2.020725` | `3339.7` | `565.2` | `64.9` | `24.30` | mean CV `0.1686`, max-layer max load `0.0946` | selected F4 result |
| F4 | 14 | 1536 | `0.0003` | `34232` | `8.974` | -- | -- | `3339.7` | `565.2` | `~72.4 live` | `24.30` | stopped at `79.4%`; last-300 loss `2.3119`; matched-window loss `2.3121` vs `2.1490` for LR `0.001` | aborted; undertrained |

## Low-Compute Baseline Scaling Results

Ran the simple-backbone controls at S75, F1, and F2 under the same `20` tokens/active-parameter protocol, then compared against the full-recipe curve. The simple controls disable the dense stem, fixed value mix, exclusive attention, headwise attention gate, sigmoid-affinity routing, and expert-bias routing; they use softmax top-k routing with `LOAD_BALANCE_LOSS_COEF=0.0085`.

| size | curve | depth | model_dim | head_dim | num_heads | num_kv_heads | moe_hidden_dim | adamw_lr | steps | tokens_B | val_bpb | train_ce | active_params_M | mfu_percent | router note | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| S75 | full | 7 | 640 | 64 | 10 | 2 | 2176 | `0.003` | `5800` | `1.520` | `0.902337` | `2.509520` | `76.0` | `22.69` | mean CV `0.0622`, max-layer max load `0.0755` | completed |
| S75 | simple | 7 | 640 | 64 | 10 | 2 | 2176 | `0.003` | `5800` | `1.520` | `0.916017` | `2.547789` | `75.9` | `20.23` | mean CV `0.4888`, max-layer max load `0.3366` | completed |
| F1 | simple | 8 | 768 | 128 | 6 | 2 | 1792 | `0.003` | `7000` | `1.835` | `0.892165` | `2.477874` | `91.3` | `22.81` | mean CV `0.3706`, max-layer max load `0.2076` | completed |
| F2 | simple | 10 | 1024 | 128 | 8 | 2 | 2304 | `0.003` | `14099` | `3.696` | `0.844628` | `2.324253` | `184.7` | `23.89` | mean CV `0.5187`, max-layer max load `0.2174` | completed |

Interpretation:
The low-compute controls show a non-monotone but scale-positive architecture gap. S75 full beats simple by `0.013680` BPB, F1 is effectively tied on validation (`0.000013` BPB), and F2 full beats simple by `0.023003` BPB using the selected F2 LR anchor. The F2 simple run also showed repeated transient router/loss instability around early and mid training, including loss spikes above `6`, before recovering late; the final router health remained much worse than the full F2 anchor.

### Fixed-Wall Estimate From Low-Compute Controls

Use the full-recipe wall clock as the budget, since it is the faster curve, and estimate where the slower simple baseline would be at that same elapsed time. These are estimates from observed throughput and late train-loss slope, not separate validation checkpoints at the cutoff steps.

| size | simple cutoff at full wall | simple final_bpb | estimated simple_bpb at full wall | full_bpb | estimated fixed-wall gain | note |
|---|---:|---:|---:|---:|---:|---|
| S75 | `~5350 / 5800` steps | `0.916017` | `~0.920-0.921` | `0.902337` | `~0.018-0.019` BPB | full is about `8%` faster |
| F1 | `~6570 / 7000` steps | `0.892165` | `~0.896-0.898` | `0.892152` | `~0.004-0.006` BPB | mostly speed; matched-step validation is tied |
| F2 | `~13125 / 14099` steps | `0.844628` | `~0.850-0.852` | `0.821625` selected / `0.822997` matched LR | `~0.028-0.030` BPB | strong at matched steps and stronger fixed-wall |

Interpretation:
At fixed wall clock, S75 and F2 show larger gains than the matched-step table because the full recipe reaches the target in less time. F1 remains the weakest point: the direct validation comparison is a tie, so any fixed-wall advantage there is mainly a throughput/sample-count effect and should not be overclaimed without validating a simple checkpoint near the cutoff.

## Initial Phase Baseline

Commit:
`153211f`

Status:
keep as init baseline

Configuration:
`DEPTH=8`, `MODEL_DIM=768`, `NUM_EXPERTS=16`, `TOP_K=2`, `MOE_HIDDEN_DIM=1792`, packed grouped MoE dispatch, `TOTAL_BATCH_SIZE=2**18`, learned per-layer `qk_gamma` initialized to `1.0`, QK RMS norm with FlashAttention default `1/sqrt(d_h)` scaling, no value embeddings, no value residuals, no logit softcap, softmax top-k router, `LOAD_BALANCE_LOSS_COEF=8.5e-3`, `ROUTER_Z_LOSS_COEF=7.5e-4`.

Optimizer/init:
All non-scalar matrix weights use fan-in scaled truncated normal init, `std = INIT_STD_GLOBAL / sqrt(d_in)` with `INIT_STD_GLOBAL=1.0` and clamp at `+/-3 std`. AdamW parameters share one peak LR schedule: 100-step linear warmup from zero, cosine decay to `0.1 * peak_lr`, `weight_decay=0.1`, betas `(0.9, 0.95)`, eps `1e-8`, gradient clip `1.0`. Muon uses the same schedule multiplier, momentum `0.95`, `5` Newton-Schulz steps, beta2 `0.95`, and shape-aware peak LR `adamw_lr * 0.2 * sqrt(max(d_in, d_out))`.

Baseline result:
`val_bpb 0.964088`, `2603` steps, `682.4M` tokens, `29.9GB` peak VRAM, `23.07%` MFU, `553.7M` total params, `91.3M` active params. CE-only train loss `2.691249`, total train loss `2.702158`. Learned QK gamma moved to mean `1.162029`, min `0.768930`, max `1.499097`. Mean router entropy `1.773`, min-layer entropy `1.364`, mean load CV `0.547`, max-layer load CV `1.054`, mean max expert load `0.131`, max-layer max load `0.206`, mean z-loss `0.152`, max-layer z-loss `0.297`, load-balance diagnostic `1.270`, aux `0.01091`.

Interpretation:
This is much worse than the older feature-rich stack, but that is expected after removing value embeddings/value residuals and logit softcap. It is stable, fast, uses the intended learned-QK path, and is now the clean comparison point for LR and backbone rebuild work. The immediate concern is that `AdamW_peak=0.01` appears too aggressive or poorly matched to the stripped model: validation is poor and router layer extrema are high.

Next:
Sweep AdamW peak LR on the operator's log grid while keeping architecture fixed. Start at `0.003`, then try `0.001` or `0.0003` depending on early loss and router behavior. Keep shape-aware Muon LR tied to AdamW by the current formula.

## Current Phase Best

Commit:
`f48fb15`

Status:
keep

Configuration difference from the simple init baseline:
AdamW peak LR is `0.003`. The first two FFN layers are dense SwiGLU layers with hidden size `3584`; the upper six FFN layers remain `16` expert top-2 MoE layers with expert hidden size `1792`. Later attention layers use fixed first-value mixing `v = 0.75 * v_1 + 0.25 * v_l` for layers `1-7`. Router uses sigmoid affinities and loss-free expert bias for top-k selection only; output weights use clean selected sigmoid affinities. Load-balance coefficient is `0.003`, router z-loss coefficient remains `7.5e-4`. Attention uses exclusive self-attention plus a headwise attention gate initialized to sigmoid `0.98`.

Result:
`val_bpb 0.938179`, improving the previous fixed-wall best by `0.000331` BPB and the clean comparison baseline by `1.749%`. The run reached `2778` steps and `728.2M` tokens with `29.3GB` peak VRAM and `24.63%` MFU. Total params fell to `438.2M`, while active params stayed at `91.3M`. Router health was very clean: mean load CV `0.0719`, max-layer load CV `0.0826`, mean max load `0.0723`, max-layer max load `0.0737`, mean router bias abs `0.0074`, and max router bias abs `0.0260`.

Interpretation:
The useful stack is now a dense two-layer stem plus sparse upper MoE layers, damped first-value attention mix, DeepSeek-style sigmoid/bias routing with small differentiable load-balance pressure, XSA, and a near-identity headwise attention gate. The dense stem improves fixed-wall BPB partly by increasing throughput and update count. A matched-step diagnostic still beat the no-dense attention stack (`0.939195` vs `0.940076` at `2515` steps), so the gain is not purely a speed artifact.

## Runs

### run 1: simple backbone init baseline

Kind/thread:
baseline / simple-backbone

Pre-run hypothesis:
Removing value embeddings, value residual gates, and logit softcap gives a minimal MoE backbone that is easier to reason about. Fan-in truncated init plus learned QK gamma should at least train stably, even if quality regresses.

Expected result:
Stable finite BPB, no numerical blow-up, learned `qk_gamma` moves away from 1, and router diagnostics remain non-collapsed enough to use as the phase baseline.

Observed result:
`val_bpb 0.964088`, `2603` steps, `682.4M` tokens, `29.9GB` peak VRAM, `23.07%` MFU. Mean `qk_gamma 1.162029`, range `0.768930` to `1.499097`. Mean router entropy `1.773`, max-layer load CV `1.054`, max-layer max load `0.206`, max-layer z-loss `0.297`.

Interpretation:
The run is stable and fast, but quality is poor. Since this intentionally removes several helpful-looking components, treat it as the init baseline rather than a failed intervention. The current LR may be too high for this stripped model and fan-in init.

Agrees with hypothesis:
partial

Decision:
keep as phase baseline

Next run:
Set `ADAMW_LR=0.003` and rerun the unchanged simple backbone. If it improves BPB and router extrema, continue the downward LR sweep; if it badly undertrains, bracket with `0.001` and potentially return upward only after seeing matched-step behavior.

### run 2: AdamW peak LR 0.003

Kind/thread:
lr / simple-backbone

Pre-run hypothesis:
The `0.01` peak LR is too aggressive for the stripped simple backbone and fan-in scaled init. Reducing AdamW peak LR to `0.003`, with Muon peaks tied by the shape-aware formula, should reduce router over-specialization and improve CE/BPB.

Expected result:
Lower matched-step train loss than `0.01`, cleaner router load, and better BPB without changing throughput or memory.

Observed result:
`val_bpb 0.954881`, `2634` steps, `690.5M` tokens, `29.9GB` peak VRAM, `23.35%` MFU. Train CE `2.591596`, total train loss `2.601004`. Mean `qk_gamma 1.258413`, range `0.905162` to `1.575565`. Mean router entropy `1.709`, min-layer entropy `1.534`, mean load CV `0.309`, max-layer load CV `0.696`, mean max load `0.105`, max-layer max load `0.156`, mean z-loss `0.215`, max-layer z-loss `0.295`, load-balance diagnostic `1.088`, aux `0.00941`.

Interpretation:
The hypothesis is supported. At matched steps around `1200`, `0.003` had lower loss and much higher router entropy than `0.01`; final validation improved by `0.009207` BPB. The router is still layerwise uneven, but the average load and max-load diagnostics are much healthier.

Agrees with hypothesis:
yes

Decision:
keep as current phase best

Next run:
Try the next log-grid point, `ADAMW_LR=0.001`, with no architecture changes. If it improves again, continue to `0.0003`; if it undertrains, bracket the best LR between `0.001` and `0.003`.

### run 3: AdamW peak LR 0.001

Kind/thread:
lr / simple-backbone

Pre-run hypothesis:
If `0.003` is still too sharp or too router-specializing, lowering to `0.001` could improve validation by keeping routing more uniform and reducing update noise, even if training CE is slower.

Expected result:
Some CE slowdown versus `0.003`, but potentially better BPB if cleaner routing matters more. Abort only if loss clearly stops improving or routing/loss becomes pathological.

Observed result:
`val_bpb 1.030709`, `2648` steps, `694.2M` tokens, `29.9GB` peak VRAM, `23.46%` MFU. Train CE `2.851093`, total train loss `2.860090`. Mean `qk_gamma 1.347559`, range `0.975075` to `1.646462`. Mean router entropy `1.808`, min-layer entropy `1.489`, mean load CV `0.207`, max-layer load CV `0.415`, mean max load `0.094`, max-layer max load `0.136`, mean z-loss `0.181`, max-layer z-loss `0.296`, load-balance diagnostic `1.042`, aux `0.00900`.

Interpretation:
The lower LR made routing very clean, but it undertrained badly and validation collapsed. This is useful negative evidence: for the stripped backbone, BPB is dominated by learning progress in this LR range, not by further improving average load balance. The next lower grid point `0.0003` is not worth running now.

Agrees with hypothesis:
no

Decision:
discard

Next run:
Restore `ADAMW_LR=0.003` as the simple-backbone LR baseline, then begin adding architectural components back one at a time.

### run 4: zero-init scalar first-value residual

Kind/thread:
architecture / value-residual

Pre-run hypothesis:
The stripped backbone may have lost a useful early value stream when value paths were removed. A minimal ResFormer-style first-value residual lets later attention layers mix the first layer's value tensor through one learned scalar per layer, initialized at zero so step-0 behavior is identical to the simple backbone.

Expected result:
If the missing value path matters, BPB should improve over `0.954881` without a large speed or memory hit, and learned `value_resid_alpha` should move away from zero. If this mostly adds distracting cross-layer signal, matched-step CE or router extrema should worsen and BPB should regress.

Observed result:
Aborted at operator request around `78%` of the 450-second budget, before validation. At the last sampled point, training loss was around `2.69-2.71`, router entropy around `1.76-1.77`, load CV around `0.33-0.35`, max load around `0.115-0.118`, and throughput was slightly below the simple `0.003` baseline. Earlier matched-step checks showed similar or worse CE than the baseline and higher router concentration.

Interpretation:
The minimal scalar value residual did not show an early training advantage and appeared to disturb routing. Since this path directly changes the representation seen by the same block's MoE router, it is likely creating a routing/credit-assignment interaction rather than simply improving attention memory.

Agrees with hypothesis:
no

Decision:
abort/discard this exact form

Next run:
Do not continue this exact value residual. Rework the mechanism so the first-value signal is detached, capped, delayed, or invisible to the router before testing again.

Diagnostic follow-up:
Added `AR_GRAD_DIAG=1` instrumentation and ran the same scalar first-value residual for a 120-second diagnostic job, with compile disabled only for retained activation gradients. The scalar residual is not inert: mean absolute `value_resid_alpha` grew from `0` to `0.225` by step `596`, with one layer reaching `0.968`. However, the residual path gradient into the retained first-layer value clone stayed tiny in absolute RMS, from `0` at step 0 to only `1.46e-7` at step 550, and the first-layer value projection gradient did not become a clear highway relative to later value projections (`cv0_grad` was usually below or only slightly above `cv_later_grad_mean`). The router diagnostics degraded while alpha grew: final mean load CV `0.559`, max-layer load CV `1.183`, mean max load `0.154`, and max-layer max load `0.344`.

Interpretation update:
The data argues against the "useful early value gradient highway" story for this exact form. The scalar learns a sizable residual mixture, but the visible effect is mostly representation/routing disturbance. Since dense models have no sparse router, this explains why a value residual can help dense attention while hurting this MoE: the extra value stream changes the state seen by later routers and concentrates load before it delivers a clear gradient-flow benefit.

Decision update:
Keep the diagnostic hook because it is env-gated and useful, but remove the scalar first-value residual from the working baseline. If we revisit value residuals, the repair should directly protect the router, for example with a capped/detached residual or a router-clean path; do not rerun this uncapped scalar form.

### run 5: fine-grained 32 experts top-4

Kind/thread:
architecture / fine-grained-moe

Pre-run hypothesis:
The current `16/top-2/1792` MoE may be too coarse: each token can choose from only 120 expert pairs per layer. Moving to `32/top-4/896` keeps active FFN width (`K*H = 3584`) and total expert width (`E*H = 28672`) fixed while giving much richer expert combinations, so any improvement should come from granularity rather than raw active compute.

Expected result:
If expert granularity is the limiting factor, BPB should improve at similar active and total parameter counts. Router entropy should be interpreted relative to `log(32)`; load CV can rise mildly, but max-layer max load should not explode. Tokens/sec may fall due to top-4 dispatch overhead; if matched-step CE is good but wall-time BPB is not, this becomes an acceleration question rather than an immediate keep.

Observed result:
Aborted at step `727` after the matched-step signal was clearly bad. At step `727`, `32/top-4/896` had loss `3.136761`, about `0.049` worse than the simple `16/top-2/1792` baseline at the same step (`3.087893`). It was also slower: roughly `1.41M` tok/sec and `21.5-21.7%` MFU versus baseline roughly `1.52-1.53M` tok/sec and `23.3-23.5%` MFU. Router load was cleaner in normalized terms: max load around `0.067-0.070` for uniform `1/32=0.03125`, comparable relative concentration to the baseline's `0.13` for uniform `1/16=0.0625`, and load CV was slightly lower than baseline.

Interpretation:
The hypothesis is not supported for this scale and training horizon. Finer experts gave cleaner routing, but the model paid both an optimization cost and a wall-time cost. Since active FFN width and total expert width were held fixed, the likely issue is that top-4 over smaller experts diffuses task learning and adds dispatch overhead before the extra combinations become useful.

Agrees with hypothesis:
no

Decision:
abort/discard

Next run:
Restore the `16/top-2/1792` simple MoE baseline. Do not try `64/top-8/448` now; if we revisit geometry, test the opposite direction (`8/top-1/3584`) or a router-mechanism change rather than going finer.

### run 6: sigmoid affinity router without bias

Kind/thread:
router / sigmoid-affinity

Pre-run hypothesis:
The softmax router globally normalizes expert probabilities before both load-balancing gradients and selected-output weights. Sigmoid affinities score experts independently, then normalize only the selected expert scores for output mixing. This may reduce unhelpful global competition while keeping dropless token-choice routing and the current expert geometry fixed.

Expected result:
If independent affinities help, CE and BPB should improve without worse max-load or z-loss. Since `topk(sigmoid(logits))` preserves hard routes, a large routing-load change is not required; the main signal should be lower CE/BPB or healthier selected-weight behavior. If sigmoid saturates, the saturation diagnostics should rise and the run should be discarded.

Observed result:
`val_bpb 0.954935`, essentially tied with but slightly worse than the `0.954881` softmax-router simple baseline. The run completed `2639` steps and `691.8M` tokens in `450.1s`, with `29.9GB` peak VRAM and `23.39%` MFU, so speed/memory were unchanged. Train CE was worse than baseline (`2.614560` vs `2.591596`) despite the early matched-step advantage. Routing was somewhat cleaner on average: mean load CV `0.282` vs `0.309`, mean max load `0.100` vs `0.105`, but max-layer max load was not better (`0.167` vs `0.156`). Softmax diagnostic entropy was lower (`1.579`), router z-loss was higher (`0.418`), and sigmoid saturation was asymmetric: mean low-affinity fraction `0.408`, max-layer low-affinity fraction `0.588`, but high-affinity saturation was near zero (`4e-6` mean).

Interpretation:
The hypothesis is only weakly supported diagnostically and not supported by validation. Sigmoid selected weights gave an early matched-step loss edge and slightly cleaner average load, but the edge did not survive to final CE/BPB. The router appears sharper under the softmax diagnostic and pushes many expert affinities near zero, while not saturating high. This suggests sigmoid-only changes scoring geometry but does not by itself improve the language objective.

Agrees with hypothesis:
partial

Decision:
tentative only as a dependency for one bias test; not a standalone keep

Next run:
Test the loss-free expert-bias variant from this sigmoid router once, because the intended DeepSeek-style mechanism is sigmoid plus hard-load feedback. If bias does not improve BPB clearly or produces routing-controller pathologies, revert to the softmax `16/top-2/1792` baseline and close this router thread for now.

### run 7: fixed first-value mix 0.75/0.25

Kind/thread:
architecture / value-mix

Pre-run hypothesis:
The first-layer value stream is useful, but the earlier scalar additive form was too blunt and disturbed routing. A fixed two-coefficient mix can provide a stable early-value memory while damping the local value path, avoiding learned scalar over-adaptation.

Expected result:
Better BPB than the simple softmax-router baseline, with no throughput penalty. A moderate load increase is acceptable only if BPB improves.

Observed result:
`val_bpb 0.945791`, `2616` steps, `685.8M` tokens, `29.9GB` peak VRAM, `23.19%` MFU. Train CE `2.566037`, total train loss `2.575746`. Mean load CV `0.425`, max-layer load CV `1.183`, mean max load `0.138`, max-layer max load `0.342`.

Interpretation:
The two-coefficient value mix turns the value residual idea from harmful to useful. Fixed `(0.5,0.5)` also worked (`0.947597`) with much cleaner routing, but `(0.75,0.25)` gave the better BPB. Learned and normalized learned variants did not beat it, and raw large-scale variants such as `(2,1)` and `(3,1)` degraded by matched steps or validation. The winning mechanism looks like a damped, first-heavy value mix, not an unconstrained learned residual.

Agrees with hypothesis:
yes

Decision:
keep as value-mix baseline

Next run:
Resume sigmoid/bias routing on top of this value-mix baseline.

### run 8: sigmoid affinities plus expert bias, no load-balance loss

Kind/thread:
router / sigmoid-bias

Pre-run hypothesis:
Sigmoid affinities plus non-gradient expert bias can reduce reliance on load-balance gradients and improve CE/BPB while controlling hard expert load.

Expected result:
Lower BPB than fixed value mix alone. Bias should remain below clamp, load should not collapse, and selected sigmoid affinities should not saturate high.

Observed result:
`val_bpb 0.943786`, `2606` steps, `683.1M` tokens, `29.9GB` peak VRAM, `23.10%` MFU. Train CE `2.518327`, total train loss `2.518677`. BPB improved by `0.002005` over fixed value mix. However, one layer collapsed: max-layer load CV `2.646`, max-layer max load `0.500`, min-layer router entropy `0.000884`, and max router bias hit the `0.25` clamp.

Interpretation:
The routing mechanism is promising because BPB improved clearly, but zero load-balance pressure is too weak or too slow to prevent layerwise collapse under this controller. This exactly matches the pre-registered repair condition for a single small load-balance run.

Agrees with hypothesis:
partial

Decision:
repair, not final keep

Next run:
Keep sigmoid affinities and expert bias, add `LOAD_BALANCE_LOSS_COEF=0.003`, and do not sweep the coefficient unless the result creates a new clear failure mode.

### run 9: sigmoid affinities plus expert bias, load-balance 0.003

Kind/thread:
router / sigmoid-bias

Pre-run hypothesis:
A small differentiable load-balance term should repair the layer collapse from loss-free sigmoid/bias while preserving the BPB gain from bias-controlled sigmoid routing.

Expected result:
Similar or better BPB than the zero-load-loss sigmoid/bias run, but with much lower max-layer load CV, no clamp-hitting bias, and router load-balance diagnostic near `1`.

Observed result:
`val_bpb 0.942848`, `2619` steps, `686.6M` tokens, `29.9GB` peak VRAM, `23.21%` MFU. Train CE `2.508455`, total train loss `2.511830`. Mean load CV `0.068`, max-layer load CV `0.128`, mean max load `0.071`, max-layer max load `0.0746`, router load-balance diagnostic `1.005`, and aux loss `0.003375`. Bias stayed well below clamp: mean abs `0.011`, max abs `0.056`.

Interpretation:
The repair worked and also improved BPB. The best current story is that sigmoid selected-output weighting plus hard-load expert bias helps optimization, but this small model still benefits from a light differentiable load-balance term. This is now the current phase best.

Agrees with hypothesis:
yes

Decision:
keep as current baseline

Next run:
Run a softmax-router control with the same fixed value mix and `LOAD_BALANCE_LOSS_COEF=0.003` to check whether the gain is truly from sigmoid/bias or mostly from lowering load-balance pressure.

### run 10: softmax router control with load-balance 0.003

Kind/thread:
router / control

Pre-run hypothesis:
The sigmoid/bias gain might actually come from lowering load-balance pressure from `0.0085` to `0.003`, not from the sigmoid/bias mechanism itself. A softmax-router control with the same fixed value mix and `LOAD_BALANCE_LOSS_COEF=0.003` isolates that possibility.

Expected result:
If the coefficient change is the main mechanism, softmax `0.003` should approach the sigmoid/bias `0.942848` BPB. If sigmoid/bias is essential, softmax `0.003` should regress and/or show worse load.

Observed result:
`val_bpb 0.947950`, `2554` steps, `669.5M` tokens, `29.9GB` peak VRAM, `22.63%` MFU. Train CE `2.638674`, total train loss `2.643481`. Mean load CV `0.822`, max-layer load CV `1.774`, mean max load `0.204`, max-layer max load `0.485`, router load-balance diagnostic `1.556`, and aux loss `0.004807`.

Interpretation:
Lowering the load-balance coefficient is not sufficient. Without sigmoid/bias, softmax routing becomes highly imbalanced and BPB is worse than even the fixed value-mix softmax run. The best run's improvement really depends on the sigmoid/bias router controller.

Agrees with hypothesis:
yes, for the "sigmoid/bias is essential" branch

Decision:
discard control

Next run:
Test whether sigmoid/bias can stabilize normalized learned value mix. If it both stabilizes and improves BPB, the interventions compound; otherwise keep fixed `(0.75,0.25)`.

### run 11: normalized learned value mix with sigmoid/bias

Kind/thread:
architecture / value-mix-router-interaction

Pre-run hypothesis:
Normalized learned value mix failed mainly because it created a worst-layer routing spike. Since sigmoid/bias with `0.003` load loss repaired router health, it may stabilize normalized learned mix and allow the learned ratio/scale to improve on the fixed mix.

Expected result:
Compared with normalized learned mix alone, max-layer load should improve dramatically. A true compound win also needs BPB near or below `0.942848`.

Observed result:
`val_bpb 0.947005`, `2600` steps, `681.6M` tokens, `30.2GB` peak VRAM, `23.04%` MFU. Train CE `2.594792`, total train loss `2.598185`. The learned mix ended at mean `alpha_1 0.246`, mean `alpha_2 0.423`, and mean `gamma 0.412`. Router health was good: mean load CV `0.076`, max-layer load CV `0.149`, mean max load `0.072`, max-layer max load `0.085`, router load-balance diagnostic `1.005`, and max router bias abs `0.124`.

Interpretation:
Sigmoid/bias did stabilize normalized learned mix: max-layer max load improved from `0.322` without sigmoid/bias to `0.085`. But it did not improve quality; BPB regressed well behind the fixed `(0.75,0.25)` sigmoid/bias baseline. The interaction is stabilizing, not compounding. Keep fixed value mix for now.

Agrees with hypothesis:
partial

Decision:
discard as quality baseline, keep as diagnostic evidence

Next run:
Restore fixed `(0.75,0.25)` plus sigmoid/bias plus `0.003` load balance as the current baseline. Further gains should probably come from expert geometry, router controller tuning, or a smaller fixed value-mix ratio sweep around the first-heavy solution, not from learned normalized value mix.

### run 12: fixed value mix 3/1 with sigmoid/bias repair

Kind/thread:
architecture / value-mix-router-interaction

Pre-run hypothesis:
A stronger first-value mix might become viable once sigmoid affinities, expert bias, and `LOAD_BALANCE_LOSS_COEF=0.003` protect the router from the load spikes seen in raw large-ratio value-mix runs.

Expected result:
If the earlier `(3,1)` failure was mostly router instability, matched-step CE should approach or beat the fixed `(0.75,0.25)` sigmoid/bias baseline while load CV and max load remain healthy.

Observed result:
Aborted at step `1093`. Routing stayed healthy enough, with max load mostly around `0.09-0.12`, but matched-step loss was consistently worse: about `3.145` vs `3.114` at step `600`, `2.971` vs `2.934` at step `1000`, and `2.946` at step `1093`.

Interpretation:
The stabilized router fixes the obvious load failure mode but does not make the stronger value mix useful. The problem is quality/optimization, not router collapse. Keep fixed `(0.75,0.25)` and stop value-ratio tuning for now.

Agrees with hypothesis:
no

Decision:
discard/abort

Next run:
Move to fixed-budget expert granularity from the high-priority queue.

### run 13: 32 experts top-4 hidden-896 with current router

Kind/thread:
architecture / fine-grained-moe

Pre-run hypothesis:
With active FFN width and total expert width held fixed, `32/top-4/896` gives many more expert combinations than `16/top-2/1792`. The earlier fine-grained run was tested before the current stabilized sigmoid/bias/value-mix baseline, so the new router may unlock useful specialization.

Expected result:
If expert granularity is the right structural direction, matched-step CE and final BPB should improve or at least remain close while router max load stays controlled. Throughput may fall; a quality win at matched steps would still be worth analyzing for acceleration.

Observed result:
Aborted at step `936`. The run had very healthy routing for 32 experts: max load was mostly `0.04-0.05` where uniform is `0.03125`, and load CV settled around `0.15-0.20`. However, matched-step loss was clearly behind the current best: about `3.386` vs `3.357` at step `360`, and about `2.997` near step `930` when the best 16/top-2 curve is already much closer to `2.94` by step `1000`. Throughput also fell from about `1.51M` tok/s to `1.40M` tok/s.

Interpretation:
The stabilized router makes fine-grained routing healthy, but the smaller experts/top-4 mixture are not learning better in this short run. This is not a load-collapse problem to repair; it is a quality and wall-time loss.

Agrees with hypothesis:
no

Decision:
discard/abort

Next run:
Restore `16/top-2/1792` current best stack and try an attention-side intervention from the primary literature rather than more MoE healthcare.

### run 14: exclusive self attention on current best stack

Kind/thread:
architecture / attention

Pre-run hypothesis:
Exclusive self attention may improve the division of labor between attention and the MoE FFN by forcing attention outputs to carry contextual information orthogonal to the current token's own value vector. Because residual connections already carry self information, removing the self-value direction from attention could reduce redundant pointwise transformation without touching router mechanics.

Expected result:
If XSA transfers to this MoE setting, matched-step CE and final BPB should improve with minimal throughput cost. Router load should remain similar to the current best; if router diagnostics move sharply, the attention change is perturbing the sparse FFN rather than cleanly improving context modeling.

Observed result:
`val_bpb 0.940518`, `2536` steps, `664.8M` tokens, `30.7GB` peak VRAM, and `22.47%` MFU. Matched-step loss improved early: at step `600`, XSA was `3.102649` versus the previous best `3.113824`. Final train CE was `2.511834`, slightly worse than the previous best CE `2.508455`, but validation improved by `0.002330` BPB. Router health remained acceptable: mean load CV `0.094`, max-layer load CV `0.351`, mean max load `0.077`, max-layer max load `0.122`, mean router bias abs `0.0113`, max bias abs `0.0799`.

Interpretation:
XSA transfers cleanly to this MoE setting. The early train-loss edge and validation gain support the division-of-labor hypothesis: attention benefits from removing self-value direction even when a sparse FFN/router sits after it. The worst-layer load is less pristine than the previous best, but it is not collapsed and BPB improves. Keep XSA as the new baseline.

Agrees with hypothesis:
yes

Decision:
keep as current best

Next run:
Use XSA as the baseline for the next attention-side experiment. The most natural follow-up is the primary gated-attention variant: query-dependent headwise sigmoid gate after SDPA/XSA, initialized near identity so the initial model is not shrunk.

### run 15: headwise gated attention on XSA baseline

Kind/thread:
architecture / attention

Pre-run hypothesis:
The gated-attention result may transfer to this MoE stack if each query/head can learn how much post-SDPA context to pass through. With XSA already removing the self-value direction, a near-identity headwise sigmoid gate should act as a low-risk learned attention modulator rather than a large initialization change.

Expected result:
If the gate is useful, matched-step CE should be equal or better than XSA and final BPB should improve without severe throughput loss or new router load concentration. If it mainly shrinks attention or destabilizes the MoE, early CE should fall behind XSA despite healthy routing.

Observed result:
`val_bpb 0.940352`, `2511` steps, `658.2M` tokens, `30.7GB` peak VRAM, and `22.26%` MFU. Matched-step loss was slightly better than XSA after warmup: step `360` was `3.334208` vs XSA `3.336214`, step `600` was `3.100312` vs XSA `3.102649`, and step `2000` was `2.664697` vs XSA `2.666453`. Final BPB improved only slightly, by `0.000166`. Router health improved materially versus XSA: mean load CV `0.0636`, max-layer load CV `0.104`, mean max load `0.0714`, and max-layer max load `0.0747`. The learned gate moved away from identity: mean sigmoid of the gate bias ended at `0.874`, and gate weight RMS ended at `0.0246`.

Interpretation:
Headwise gated attention transfers, but the validation gain is small. The cleaner router load suggests attention-side modulation may reduce pressure on the MoE rather than perturb it. Since the gate bias decayed from the `0.95` init toward `0.87`, the next question is whether the model benefits from actual attenuation, or whether weight decay/global shrink is making the gate too conservative.

Agrees with hypothesis:
yes

Decision:
keep as current best, but small margin

Next run:
Try a higher near-identity gate init (`0.98`) to test whether preserving more attention amplitude while retaining query-dependent head gates improves the small BPB gain.

### run 16: headwise gated attention init 0.98

Kind/thread:
architecture / attention

Pre-run hypothesis:
The `0.95` gate run improved validation and router health, but the gate bias ended with mean sigmoid around `0.874`, so some of the effect may be broad attention shrink rather than useful sparse modulation. Initializing the gate closer to identity may preserve the query-dependent headwise mechanism while avoiding excessive attenuation from AdamW decay.

Expected result:
If the useful mechanism is query/head-dependent gating, `0.98` should match or beat the `0.95` result. If global attention attenuation is part of the gain, `0.98` may regress toward plain XSA.

Observed result:
`val_bpb 0.940076`, `2515` steps, `659.3M` tokens, `30.7GB` peak VRAM, and `22.30%` MFU. Matched-step loss was essentially tied with the `0.95` gate run at steps `1000` and `1200`, slightly better by step `1600` (`2.755895` vs `2.756617`), and better on validation by `0.000276` BPB. Router health stayed strong: mean load CV `0.0573`, max-layer load CV `0.119`, mean max load `0.0700`, and max-layer max load `0.0833`. The gate stayed closer to identity than the `0.95` run: final mean bias sigmoid was `0.928` instead of `0.874`, with weight RMS `0.0266`.

Interpretation:
Higher near-identity gate init is better. This argues against the gain coming mainly from global attention attenuation; preserving more attention amplitude while allowing learned query/head modulation improves validation. The final router load remains cleaner than XSA and comparable to the `0.95` gate run.

Agrees with hypothesis:
yes

Decision:
keep as current best

Next run:
Try `ATTENTION_GATE_INIT=0.99` as one more targeted gate-init iteration. If this also improves, the right default is likely very close to identity; if it regresses, keep `0.98` and move to a different attention-side intervention.

### run 17: headwise gated attention init 0.99

Kind/thread:
architecture / attention

Pre-run hypothesis:
Since `0.98` beat `0.95`, the headwise gate may want to begin even closer to identity. `0.99` should preserve more attention amplitude while still giving the model a query-dependent per-head attenuation path.

Expected result:
If identity-preserving gating is the key, `0.99` should match or beat the `0.98` validation BPB without router regression. If the best point needs some initial attenuation, `0.99` should lose despite similar matched-step CE.

Observed result:
`val_bpb 0.940078`, `2509` steps, `657.7M` tokens, `30.7GB` peak VRAM, and `22.24%` MFU. This was essentially tied with `0.98` but fractionally worse by `0.000002` BPB. Matched-step loss was also a tie: step `1000` was `2.922022` vs `0.98` `2.922391`, step `1200` was `2.852183` vs `2.852224`, step `1600` was `2.756154` vs `2.755895`, and step `2000` was `2.664366` vs `2.664526`. Router health was very clean: mean load CV `0.0611`, max-layer load CV `0.0825`, mean max load `0.0698`, and max-layer max load `0.0735`. Final mean gate-bias sigmoid was `0.954`.

Interpretation:
The optimum is very close to identity, and pushing from `0.98` to `0.99` does not buy more BPB. The clean router suggests the run is valid, but there is no reason to replace the `0.98` baseline.

Agrees with hypothesis:
partial

Decision:
discard/tie; keep `0.98`

Next run:
Test a centered identity-preserving gate, `2*sigmoid(g(x))`, initialized with zero bias so the initial effective gate is exactly `1.0`. This removes the large positive bias and lets each head attenuate or amplify around identity.

### run 18: centered headwise gated attention

Kind/thread:
architecture / attention

Pre-run hypothesis:
The best sigmoid-gate runs want to stay close to identity. A centered gate `2*sigmoid(g(x))`, initialized at `1.0`, may preserve the useful query/head modulation while avoiding large positive gate biases, AdamW decay of those biases, and the restriction that the gate can only attenuate attention.

Expected result:
If the useful effect is dynamic modulation around the normal attention path, centered gating should beat or match the `0.98` sigmoid gate. If the one-sided attenuation of the paper gate is important, centered gating may regress despite cleaner initialization.

Observed result:
Aborted at step `600`. The centered gate had a strong early win at step `100` (`5.192110` vs `0.98` sigmoid gate `5.239129`) and step `200` (`3.906754` vs `3.933340`), but the advantage reversed after warmup: step `360` was `3.347488` vs `3.332431`, and step `600` was `3.117709` vs `3.100722`. Router load did not collapse (`max_load 0.112` at step `600`), but load CV was higher and throughput was slightly lower.

Interpretation:
The exact-identity centered gate changes optimization scale in a way that helps the very early warmup but hurts the main training regime. Since the failure is quality, not router collapse, this is not a repairable load-balancing issue. The one-sided sigmoid gate from the paper is the better attention-gate form here.

Agrees with hypothesis:
no

Decision:
discard/abort

Next run:
Restore the standard sigmoid headwise gate with `ATTENTION_GATE_INIT=0.98` and move back to the MoE high-priority queue. The next highest-signal structural test is dense early layers, because it asks whether early routing noise is still costing quality now that the attention and router stack is stable.

### run 19: first layer dense SwiGLU

Kind/thread:
architecture / dense-early-layers

Pre-run hypothesis:
Early token representations may be too raw for useful sparse routing, even with the current stabilized sigmoid/bias router. Replacing the first MoE FFN with a dense SwiGLU at matched active FFN width should remove the noisiest early router while keeping the rest of the sparse stack intact.

Expected result:
If early routing noise is a bottleneck, the run should improve BPB or match BPB with better throughput/router health. Active parameters should stay close because dense hidden `3584` matches `TOP_K * MOE_HIDDEN_DIM`; total parameters should fall because one full expert bank is removed.

Observed result:
`val_bpb 0.938510`, `2629` steps, `689.2M` tokens, `30.0GB` peak VRAM, and `23.30%` MFU. Total params fell from `553.8M` to `496.0M`, while active params stayed matched at `91.3M`. The run was initially behind at matched steps: step `200` was `3.991079` vs the previous best `3.933340`, step `360` was `3.347321` vs `3.332431`, and step `600` was `3.112301` vs `3.100722`. It caught up late: step `2000` was `2.663486` vs `2.664526`, and final BPB improved by `0.001566`. Router health was excellent: mean load CV `0.0616`, max-layer load CV `0.0691`, mean max load `0.0705`, max-layer max load `0.0726`, mean router bias abs `0.0079`, and max bias abs `0.0706`.

Interpretation:
Dense first layer is a strong win. It starts slower in CE at matched steps but runs faster and catches up late, giving both better wall-time BPB and cleaner routing with far fewer total parameters. This supports the early-routing-noise hypothesis: the model benefits from doing one dense transformation before sparse expert routing.

Agrees with hypothesis:
yes

Decision:
keep as current best

Next run:
Try the natural extension, first two layers dense at the same active width. If it keeps improving or stays close with more throughput and fewer total params, dense stem plus sparse upper layers is a real architecture direction. If it loses, keep one dense layer as the best compromise.

### run 20: first two layers dense SwiGLU

Kind/thread:
architecture / dense-early-layers

Pre-run hypothesis:
The first dense layer result suggests early sparse routing is costly. Making the first two FFN layers dense may further reduce early routing noise and improve throughput/parameter efficiency, while leaving six upper sparse MoE layers for specialization.

Expected result:
If dense stem is the right direction, two dense layers should match or beat the one-dense-layer BPB, or at least remain close with higher throughput and lower total params. If the second dense layer removes too much expert capacity/specialization, matched-step CE and final BPB should regress.

Observed result:
`val_bpb 0.938179`, `2778` steps, `728.2M` tokens, `29.3GB` peak VRAM, and `24.63%` MFU. Total params fell again to `438.2M`, while active params remained matched at `91.3M`. The gain over one dense layer is small but clean: `0.000331` BPB, about `0.035%` relative. Matched-step CE was mixed but close early: step `100` was effectively tied (`5.249734` vs `5.248774`), step `200` was slightly worse (`3.998785` vs `3.991079`), then step `360` and `600` were slightly better (`3.344945` vs `3.347321`, `3.109956` vs `3.112301`). Router health stayed excellent: mean load CV `0.0719`, max-layer load CV `0.0826`, mean max load `0.0723`, max-layer max load `0.0737`, mean router bias abs `0.0074`, and max router bias abs `0.0260`.

Interpretation:
Two dense early layers are better than one at fixed wall time, with higher throughput, fewer total parameters, and still-clean routing. The result does not prove a large sample-efficiency gain at fixed steps, but it does support the practical dense-stem hypothesis: the model benefits from pushing early routing later while preserving sparse upper-layer capacity.

Agrees with hypothesis:
yes

Decision:
keep as current best

Next run:
Try first three layers dense. This tests where the dense-stem benefit saturates: if the third dense layer improves or stays close with more speed and clean routing, the useful sparse specialization may mostly live in the upper half of the network; if it regresses, two dense layers is likely the right compromise.

### run 21: first three layers dense SwiGLU

Kind/thread:
architecture / dense-early-layers

Pre-run hypothesis:
Two dense early layers improved fixed-wall BPB while keeping the upper MoE layers healthy. A third dense layer may further reduce premature routing noise and improve throughput, but it also removes another full expert bank and leaves only five sparse layers for specialization.

Expected result:
If the dense-stem benefit has not saturated, three dense layers should match or beat the two-dense BPB at the same 450s budget, with clean load and higher throughput. If sparse expert capacity is now too shallow, matched-step CE and final BPB should regress despite speed.

Observed result:
Two diagnostic attempts were aborted before final validation. With the old `ESTIMATED_TOTAL_STEPS=2390` schedule, three dense layers was healthy and fast but not better at matched steps: step `600` was `3.110812` vs two-dense `3.109956`, step `1000` was `2.927728` vs two-dense `2.927467`, and around step `1968` it was `2.664536` vs two-dense step `2000` `2.663244`. After noticing that the LR schedule was stale for faster dense runs, I tried `AR_ESTIMATED_TOTAL_STEPS=2960`; that stretched the cosine decay, but it made optimization clearly worse: step `600` was `3.116307` and step `1000` was `2.945162`, with healthy routing.

Interpretation:
Three dense layers did not show a sample-efficiency gain, and the corrected longer decay schedule was worse. The dense-stem benefit so far looks mostly like a throughput/wall-time effect, not a clear per-step architecture improvement. Two dense layers remains the best practical wall-time result for now, but the step-count confound needs direct validation.

Agrees with hypothesis:
partial

Decision:
discard three dense for now

Next run:
Add an explicit max-step stop and evaluate the two-dense model at exactly the no-dense baseline step count. This directly tests whether the two-dense BPB gain survives matched updates/tokens or mostly comes from getting more steps in 450 seconds.

### run 22: two dense layers matched to no-dense step count

Kind/thread:
diagnostic / dense-early-layers

Pre-run hypothesis:
The two-dense model's wall-time win may come mostly from getting `2778` updates versus the no-dense attention-gate baseline's `2515` updates. If dense stem is intrinsically better, it should still beat or closely match the no-dense BPB when stopped and validated at `2515` steps under the same step-based LR schedule.

Expected result:
If the dense benefit is mostly extra steps, matched-step validation should regress toward the no-dense `0.940076` BPB and may no longer beat it materially. If dense stem has real architecture value, it should remain clearly below `0.940076` at the same `2515` steps.

Observed result:
`val_bpb 0.939195` at exactly `2515` steps, `659.3M` tokens, `407.9s` training time, `453.5s` total time, `29.3GB` peak VRAM, and `24.59%` MFU. This beats the no-dense attention-gate baseline at the same step count (`0.940076`) by `0.000881` BPB, about `0.094%` relative. It is worse than the full fixed-wall two-dense run (`0.938179`) because the fixed-wall run reached `2778` steps. Router health was very clean: mean load CV `0.0560`, max-layer load CV `0.0600`, max load `0.0690`, max-layer max load `0.0703`, mean router bias abs `0.0080`, and max router bias abs `0.0356`.

Interpretation:
The two-dense gain is not purely an artifact of getting more updates in the 450s budget. There is a real matched-step benefit over the no-dense XSA+gate stack. However, the fixed-wall improvement is larger than the matched-step improvement, so the dense stem is doing two things at once: it slightly improves sample/update efficiency and materially improves throughput by replacing early expensive expert banks with dense FFNs. The three-dense diagnostics suggest this effect saturates quickly; pushing dense depth further does not look high-signal right now.

Agrees with hypothesis:
yes, with qualification

Decision:
keep two dense layers as the current baseline; stop dense-depth tuning for now

Next run:
Move back to the high-priority architecture queue instead of continuing small dense-depth/LR-schedule tweaks. Use both fixed-wall and matched-step diagnostics for any intervention that changes throughput.

### run 23: depth-scaled pre-norm

Kind/thread:
architecture / residual-scaling

Pre-run hypothesis:
The current pre-norm blocks feed every layer's attention and FFN with unit-RMS activations, so deeper layers may inject updates at roughly the same scale as shallow layers even though the residual stream has accumulated more transformations. Multiplying each residual-stream pre-norm by `1/sqrt(ell)` with one-based layer index may make deeper layers act more like residual refinements, improving stability without changing QK norm or the final LM head path.

Expected result:
If the scale is helpful, BPB should improve or matched-step CE should be lower with little throughput or memory change. QK gamma may compensate upward because Q/K are still explicitly RMS-normalized after projection, while V, attention gates, routers, and FFNs see the reduced deeper-layer input scale. Router health should stay clean; if max load rises or BPB worsens, the scale is probably starving upper MoE layers.

Observed result:
Aborted at step `887` after the post-warmup comparison became clearly worse. The run looked good during warmup: step `100` was `5.187956` vs the two-dense baseline `5.249734`, and step `200` was `3.989570` vs `3.998785`. After warmup it reversed hard: step `360` was `3.388878` vs baseline `3.344945`, step `600` was `3.153001` vs `3.109956`, and step `800` was `3.047116` vs `3.004887`. Router load was healthy rather than collapsed: at step `800`, load CV was `0.107` and max load was `0.076`.

Interpretation:
Depth-scaled pre-norm improves very early warmup but starves the useful upper-layer computation once the main training regime starts. Because Q/K are explicitly re-normalized inside attention, the harmful effect is probably through V, attention gates, routers, and FFNs, especially in upper layers where the scale is as low as `1/sqrt(8)`. This is not a router-health failure; it is an underpowered residual branch failure.

Agrees with hypothesis:
no

Decision:
discard/abort; restore the prior two-dense baseline behavior

Next run:
Do not use raw `1/sqrt(ell)` pre-norm scaling. If this family is revisited, try a much gentler learned or floor-clamped scale, but it is lower priority than the shared-expert idea.

### run 24: branch-only depth scaling

Kind/thread:
architecture / residual-scaling

Pre-run hypothesis:
The previous `1/sqrt(ell)` pre-norm run likely failed because it scaled too many control paths: Q/K inputs before their projections, router logits, and attention gates, in addition to the residual branch content. A cleaner test is to scale only branch content: attention V after value mixing, dense MLP inputs, and MoE expert inputs after routing. Router selection, QK norm, head gates, and the final LM head remain unscaled.

Expected result:
If the useful mechanism is residual-branch magnitude control rather than global feature shrinkage, this should preserve the early warmup benefit without the large post-warmup regression. Router health should stay comparable to the baseline because routing receives unscaled `norm(x)`. If BPB or matched-step CE still regresses while routing stays healthy, the issue is simply that upper-layer attention values and FFN/expert branches need full scale in this small model.

Observed result:
Aborted at step `474` after the matched-step loss was clearly worse than the two-dense baseline. Step `100` was `5.235683` vs baseline `5.249734`, so there was only a tiny warmup gain. By step `200` it had already lost (`4.023578` vs `3.998785`), and by step `360` the gap was large (`3.378140` vs `3.344945`). At the last sampled step `421`, loss was `3.286112`; the run was still healthy from a routing perspective, with router entropy `1.156`, load CV `0.136`, and max load `0.081`.

Interpretation:
This more surgical scaling avoided directly shrinking router inputs, but it still hurt once training left the earliest warmup. The failure therefore seems to be the branch scale itself: upper-layer attention values and FFN/expert activations need close to full scale in this model. The router stayed healthy because it was intentionally unscaled, which confirms the implementation isolated the content branch rather than breaking route selection.

Agrees with hypothesis:
no

Decision:
discard/abort; restore prior two-dense baseline behavior

Next run:
Close raw `1/sqrt(ell)` scaling for now. Any future residual-branch scaling should be much gentler or learned, but this is not as promising as the shared-expert direction.

### run 25: MLP-only quarter-power depth scaling

Kind/thread:
architecture / residual-scaling

Pre-run hypothesis:
The previous branch-only `1/sqrt(ell)` input scale likely over-shrank SwiGLU MLPs because SwiGLU is roughly quadratic in its input scale: both the gate and up projections see the scaled input. Scaling dense MLP and post-route MoE expert inputs by `ell**(-1/4)` should make the resulting MLP branch closer to `ell**(-1/2)` while leaving attention values, router logits, QK norm, gates, and the LM head unchanged.

Expected result:
If the quadratic-scale argument is right, this run should avoid the immediate post-warmup regression seen in run 24. Router health should stay close to baseline because router inputs are unscaled. The run must continue to at least step `1000` before decision, so the comparison should use step `1000` and later diagnostics rather than only early warmup.

Observed result:
Aborted after the required `1000`-step minimum; the run reached step `1194`. Early warmup was improved: step `100` was `5.192789` vs baseline `5.249734`, and step `200` was `3.968512` vs `3.998785`. The advantage did not survive. Step `360` was `3.357584` vs baseline `3.344945`, step `600` was `3.122031` vs `3.109956`, step `800` was `3.017799` vs `3.004887`, step `1000` was `2.941547` vs `2.927467`, and near step `1194` it was `2.880050` vs baseline step `1200` `2.856051`. Routing remained healthy: at step `1000`, router entropy `1.142`, load CV `0.089`, and max load `0.075`.

Interpretation:
Quarter-power input scaling is less damaging than raw `1/sqrt(ell)` input scaling, and it preserves early warmup gains longer, but it still underpowers MLP/expert branches in the main training regime. The clean router diagnostics confirm this is not a routing failure; reducing MLP branch magnitude itself hurts sample efficiency at this scale.

Agrees with hypothesis:
no

Decision:
discard/abort; restore prior two-dense baseline behavior

Next run:
Close fixed depth-dependent MLP input scaling for now. If residual branch scaling is revisited, it should be learned, floor-clamped close to one, or output-side rather than input-side. Move back to larger architecture ideas such as shared experts.

### run 26: MLP quarter-power plus V depth scaling

Kind/thread:
architecture / residual-scaling

Pre-run hypothesis:
The MLP-only quarter-power run preserved early warmup gains longer than raw branch scaling, but still underpowered the post-warmup model. Adding the previous attention value scale may reduce late-layer attention branch magnitude at the same time as the SwiGLU-aware MLP/expert input scale, potentially improving residual-branch balance without touching router logits, Q/K normalization, head gates, or the LM head.

Expected result:
Run to at least step `1000` before decision. If the two scaling mechanisms compound constructively, it should keep the step `100`/`200` warmup benefit from MLP quarter-power and narrow or reverse the step `600`/`1000` regression. Router health should stay comparable to baseline because the router still sees unscaled `norm(x)`.

Observed result:
Aborted after the required `1000`-step minimum; the run reached step `1200`. Unlike MLP-only quarter-power, this combined scale did not even preserve the warmup gain: step `100` was `5.251647` vs baseline `5.249734`, and step `200` was `4.014865` vs `3.998785`. The post-warmup gap remained clearly bad: step `360` was `3.358780` vs `3.344945`, step `600` was `3.124423` vs `3.109956`, step `800` was `3.020642` vs `3.004887`, step `1000` was `2.944048` vs `2.927467`, and step `1200` was `2.872284` vs `2.856051`. Routing stayed healthy: at step `1000`, router entropy was `1.205`, load CV `0.076`, and max load `0.072`.

Interpretation:
Adding attention V depth scaling to MLP quarter-power scaling did not repair the failure. It made the early behavior worse than MLP-only scaling and kept the same late underpowered-branch shape. The router was clean, so this is still not a MoE-health problem; the fixed depth scale is reducing useful residual-branch computation.

Agrees with hypothesis:
no

Decision:
discard/abort; iterate once more by also scaling the headwise attention gate as requested

Next run:
Scale the headwise attention gate by one-based `1/sqrt(ell)` in addition to the current MLP quarter-power and attention V scaling, while still leaving router logits, Q/K normalization, and the LM head unscaled.

### run 27: MLP quarter-power plus V and gate depth scaling

Kind/thread:
architecture / residual-scaling

Pre-run hypothesis:
The combined MLP+V scale may still have mismatched the attention branch because the learned headwise gate remained near identity while V was depth-scaled. Multiplying the headwise attention gate by one-based `1/sqrt(ell)` makes the post-attention modulation explicitly depth-aware too. This is a stronger attention-branch downscale, so it is likely risky, but it is a clean test of whether the gate should participate in the same residual-depth schedule.

Expected result:
If the attention gate was the missing control knob, the regression from run 26 should narrow by step `360`/`600` without hurting router health. If the model is simply underpowered by fixed depth scaling, this should be worse than run 26, especially in deeper-layer sample efficiency. Router logits, Q/K normalization, and the LM head remain unscaled.

Observed result:
Aborted at step `300` because the early regression was much larger than run 26 and the fixed-wall baseline. Step `100` was `5.285228` vs run 26 `5.251647` and baseline `5.249734`; step `200` was `4.123603` vs run 26 `4.014865` and baseline `3.998785`; step `300` was `3.523197`. Routing was not the failure mode: at step `300`, router entropy was `1.329`, load CV `0.154`, and max load `0.082`.

Interpretation:
Scaling the attention gate by `1/sqrt(ell)` made the already-bad MLP+V depth-scaling variant much worse. The gate was not the missing correction; it further reduced useful attention branch amplitude. This closes the fixed deterministic depth-scaling family for now.

Agrees with hypothesis:
no

Decision:
discard/abort; restore the two-dense fixed-wall best stack

Next run:
Return to larger architecture ideas rather than fixed depth scaling. Shared experts are a natural next candidate because they may add reusable dense capacity without reintroducing unstable learned value residuals or router-health tuning.

### run 28: mixed-V-only depth scaling

Kind/thread:
architecture / residual-scaling

Pre-run hypothesis:
The previous deterministic depth-scaling failures may have been dominated by scaling SwiGLU/experts. A cleaner isolation is to scale only the attention value content after the fixed first-value mix by one-based `1/sqrt(ell)`, leaving dense MLP inputs, MoE expert inputs, router logits, Q/K normalization, the headwise attention gate, and the LM head unchanged.

Expected result:
If attention value amplitude is the part that benefits from depth-dependent damping, this should improve or at least avoid the large early regression seen when MLP scaling was included. Router health should remain normal because no router input or expert computation is scaled. If it is clearly worse by step `360`, the fixed V scale is probably also underpowering useful attention content.

Observed result:
Aborted at step `400` after it was clearly worse than the two-dense baseline. Step `100` was `5.300067` vs baseline `5.249734`, step `200` was `4.089912` vs `3.998785`, and step `360` was `3.366658` vs `3.344945`. Router load stayed healthy: at step `360`, router entropy was `1.337`, load CV `0.124`, and max load `0.076`.

Interpretation:
Removing MLP/expert scaling did not rescue deterministic attention-value depth scaling. The failure is still not router health; it is underpowered attention value content. This also explains why adding V scaling to the MLP quarter-power run removed the earlier warmup advantage.

Agrees with hypothesis:
no

Decision:
discard/abort; try the mixed-V plus attention-gate version once as the requested paired attention-only variant

Next run:
Scale both mixed V and the headwise attention gate by one-based `1/sqrt(ell)`, still with no MLP or expert-input scaling.

### run 29: mixed-V plus attention-gate depth scaling

Kind/thread:
architecture / residual-scaling

Pre-run hypothesis:
Mixed-V-only scaling was bad, but pairing it with headwise attention-gate scaling may make the entire attention branch more internally consistent. This is likely to reduce attention amplitude even more, so the main question is whether it repairs the branch mismatch or simply worsens underpowering.

Expected result:
If the gate was the missing paired control, step `100`/`200` should improve relative to V-only and the step `360` gap should narrow. If it is worse by step `200` or `360`, abort and close fixed attention-depth scaling.

Observed result:
Aborted at step `300` because it was clearly worse than both V-only and the two-dense baseline. Step `100` was `5.325982` vs V-only `5.300067` and baseline `5.249734`; step `200` was `4.229460` vs V-only `4.089912` and baseline `3.998785`; step `300` was `3.545792`. Router load was healthy enough: at step `300`, router entropy was `1.442`, load CV `0.237`, and max load `0.091`.

Interpretation:
Pairing mixed-V scaling with attention-gate scaling did not repair the attention branch. It made the already-bad V-only run much worse. The deterministic scale is likely in the wrong location: shrinking internals reduces useful computation before projections/gating can adapt.

Agrees with hypothesis:
no

Decision:
discard/abort; try output-side layer scaling instead

Next run:
Move the `1/sqrt(ell+1)` scale to the residual merge: multiply attention output and FFN output just before adding back to the residual stream, for attention, MoE, and dense MLP, with no V/gate/MLP input scaling.

### run 30: output-side residual branch scaling

Kind/thread:
architecture / residual-scaling

Pre-run hypothesis:
The fixed layer scale may have failed because it was applied inside the branch computation, starving V, gates, dense MLP inputs, or expert inputs before the branch could form a useful update. Applying `1/sqrt(layer_idx+1)` only to completed branch outputs immediately before residual addition preserves full internal attention, MoE, and dense MLP computation while controlling how much each branch writes into the residual stream.

Expected result:
If residual update magnitude is the real issue, this should avoid the severe warmup regression of V/gate/internal scaling and may improve post-warmup stability. Router health should remain normal because router inputs and expert computations are unscaled. If it is worse by step `360`, fixed deterministic residual output scaling is probably not helpful at this scale.

Observed result:
Aborted at step `800` because the gap persisted through post-warmup. It was much less bad than internal scaling, but still worse than the two-dense baseline at every checkpoint: step `100` was `5.254192` vs baseline `5.249734`, step `200` was `4.017246` vs `3.998785`, step `360` was `3.362111` vs `3.344945`, step `600` was `3.126910` vs `3.109956`, and step `800` was `3.023749` vs `3.004887`. Router health stayed clean: at step `600`, router entropy was `1.200`, load CV `0.092`, and max load `0.074`.

Interpretation:
Output-side scaling confirms the placement hypothesis partially: scaling completed branch outputs is far less destructive than scaling V, gates, dense MLP inputs, or expert inputs. However, fixed `1/sqrt(layer)` still underpowers this small model's residual branches enough to lose sample efficiency. The issue is not routing or instability; the scale is simply too aggressive or not useful as a fixed deterministic rule.

Agrees with hypothesis:
partial

Decision:
discard/abort; restore the two-dense fixed-wall best stack

Next run:
Close fixed deterministic layer scaling for now. If revisited, use a gentler or learnable output-side scale rather than hard `1/sqrt(layer)`. Move back to higher-level architecture changes, with shared experts as the next high-signal candidate.

### run 31: expert-count E=16

Kind/thread:
sparsity / expert-count

Pre-run hypothesis:
At fixed `TOP_K=2` and `MOE_HIDDEN_DIM=1792`, changing `NUM_EXPERTS` changes total sparse capacity and sparsity ratio while keeping active expert compute roughly fixed.

Expected result:
The current `E=16` best-stack geometry should reproduce the known healthy anchor near `0.938179` BPB under the corrected 450-second budget, with about `91.3M` active params and clean router load.

Observed result:
`val_bpb 0.935178`, `2746` steps, `719.8M` tokens, `29.3GB` peak VRAM, and `24.35%` MFU.
Total params `438.2M`, active params `91.3M`.
Router health: mean load CV `0.0554`, max-layer load CV `0.0676`, mean max load `0.0696`, max-layer max load `0.0718`, mean router bias abs `0.0081`, max router bias abs `0.0232`.

Interpretation:
The corrected 450-second setup reproduces the best architecture cleanly and actually beats the previously logged BPB while landing close on steps, tokens, memory, and MFU. Router health is at least as good as the logged best, so this run is a sound `E=16` anchor for the sparsity sweep.

Agrees with hypothesis:
yes

Decision:
keep

Next run:
Run the lower-capacity `E=4` expert-count sweep point.

### run 32: expert-count E=4

Kind/thread:
sparsity / expert-count

Pre-run hypothesis:
At fixed `TOP_K=2` and `MOE_HIDDEN_DIM=1792`, changing `NUM_EXPERTS` changes total sparse capacity and sparsity ratio while keeping active expert compute roughly fixed.

Expected result:
`E=4` should be faster and use less memory because total sparse capacity is much smaller. The risk is that cutting total expert capacity from `16` to `4` experts may hurt BPB even with more fixed-wall steps.

Observed result:
`val_bpb 0.947202`, `3114` steps, `816.3M` tokens, `27.6GB` peak VRAM, and `27.61%` MFU.
Total params `140.8M`, active params `91.3M`.
Router health: mean load CV `0.0334`, max-layer load CV `0.0628`, mean max load `0.2611`, max-layer max load `0.2725`, mean router bias abs `0.0045`, max router bias abs `0.0118`.

Interpretation:
The low-expert run is much faster and leaner, but the capacity loss dominates: BPB is worse than the `E=16` anchor by `0.012024` despite `368` extra steps and `96.5M` extra tokens. Router behavior is healthy for `E=4`; max load is close to the `0.25` uniform baseline and well below the ratio-aware collapse threshold. This is a quality/capacity failure, not a routing failure.

Agrees with hypothesis:
yes

Decision:
discard

Next run:
Run the intermediate `E=8` expert-count sweep point.

### run 33: expert-count E=8

Kind/thread:
sparsity / expert-count

Pre-run hypothesis:
At fixed `TOP_K=2` and `MOE_HIDDEN_DIM=1792`, changing `NUM_EXPERTS` changes total sparse capacity and sparsity ratio while keeping active expert compute roughly fixed.

Expected result:
`E=8` should land between `E=4` and `E=16`: less memory and higher throughput than the anchor, but enough total expert capacity to recover much of the quality lost by `E=4`.

Observed result:
`val_bpb 0.938106`, `3013` steps, `789.8M` tokens, `28.2GB` peak VRAM, and `26.71%` MFU.
Total params `239.9M`, active params `91.3M`.
Router health: mean load CV `0.0429`, max-layer load CV `0.0557`, mean max load `0.1338`, max-layer max load `0.1387`, mean router bias abs `0.0065`, max router bias abs `0.0222`.

Interpretation:
The intermediate expert count behaves as expected on speed, memory, and router health, but it still trails the `E=16` anchor by `0.002928` BPB despite `267` extra steps and `70.0M` extra tokens. It recovers most of the `E=4` quality loss, so the capacity curve is real, but the best fixed-wall quality remains at `E=16` among tested smaller counts.

Agrees with hypothesis:
yes

Decision:
discard

Next run:
Run the higher-capacity `E=32` expert-count sweep point.

### run 34: expert-count E=32

Kind/thread:
sparsity / expert-count

Pre-run hypothesis:
At fixed `TOP_K=2` and `MOE_HIDDEN_DIM=1792`, changing `NUM_EXPERTS` changes total sparse capacity and sparsity ratio while keeping active expert compute roughly fixed.

Expected result:
`E=32` has twice the total expert capacity of the `E=16` anchor, so it may improve BPB if capacity is still limiting. The risk is that the much larger total parameter and optimizer state footprint reduces fixed-wall token count enough to erase any quality gain.

Observed result:
`val_bpb 0.938541`, `2308` steps, `605.0M` tokens, `31.6GB` peak VRAM, and `20.46%` MFU.
Total params `834.6M`, active params `91.4M`.
Router health: mean load CV `0.0906`, max-layer load CV `0.1338`, mean max load `0.0378`, max-layer max load `0.0440`, mean router bias abs `0.0098`, max router bias abs `0.1247`.

Interpretation:
The higher-capacity run is healthy but not competitive. It is worse than the `E=16` anchor by `0.003363` BPB and also slightly worse than `E=8`, while using more memory and completing `438` fewer steps than the anchor. Routing is not the problem: max load is low relative to the `1/32` uniform baseline and far below the collapse threshold, though the maximum router bias is larger than the smaller-expert-count runs. This is a fixed-wall efficiency failure for this scale.

Agrees with hypothesis:
partial

Decision:
discard

Next run:
Run the larger `E=64` expert-count sweep point unless it fails memory or initialization.

### run 35: expert-count E=64

Kind/thread:
sparsity / expert-count

Pre-run hypothesis:
At fixed `TOP_K=2` and `MOE_HIDDEN_DIM=1792`, changing `NUM_EXPERTS` changes total sparse capacity and sparsity ratio while keeping active expert compute roughly fixed.

Expected result:
`E=64` tests whether much larger total sparse capacity can compensate for a major throughput and memory penalty. It may become a scale candidate only if its sample-efficiency signal is clearly better despite worse fixed-wall speed.

Observed result:
`val_bpb 0.972360`, `1756` steps, `460.3M` tokens, `36.2GB` peak VRAM, and `15.57%` MFU.
Total params `1627.5M`, active params `91.6M`.
Router health: mean load CV `0.1289`, max-layer load CV `0.1371`, mean max load `0.0233`, max-layer max load `0.0248`, mean router bias abs `0.0121`, max router bias abs `0.1777`.

Interpretation:
The run is stable and routing is healthy, but the fixed-wall tradeoff is very poor. The model completes only `1756` steps and `460.3M` tokens, producing BPB far worse than the `E=16` anchor. The max expert load is below the ratio-aware collapse threshold for `E=64`, and expert bias stays under the clamp, so this is not a router failure. It is an optimizer-state and throughput penalty overwhelming any benefit from extra inactive capacity.

Agrees with hypothesis:
partial

Decision:
discard

Next run:
Try the largest planned `E=128` expert-count sweep point and log the fixed-wall result or the exact memory failure.

### run 36: expert-count E=128

Kind/thread:
sparsity / expert-count

Pre-run hypothesis:
At fixed `TOP_K=2` and `MOE_HIDDEN_DIM=1792`, changing `NUM_EXPERTS` changes total sparse capacity and sparsity ratio while keeping active expert compute roughly fixed.

Expected result:
`E=128` is the extreme-capacity point in the sweep. It may fail from memory or, if it fits, should expose whether very high inactive capacity gives any sample-efficiency signal strong enough to overcome a large optimizer-state and dispatch penalty.

Observed result:
`val_bpb 1.034533`, `1183` steps, `310.1M` tokens, `45.3GB` peak VRAM, and `10.49%` MFU.
Total params `3213.2M`, active params `91.9M`.
Router health: mean load CV `0.1844`, max-layer load CV `0.2020`, mean max load `0.0128`, max-layer max load `0.0137`, mean router bias abs `0.0165`, max router bias abs `0.2211`.

Interpretation:
The largest planned expert count fits and trains stably, but it is decisively noncompetitive. It reaches only `1183` steps and `310.1M` tokens under the fixed wall clock, and validation is much worse than every smaller expert count. Routing does not collapse: final max load is low and max-layer load CV is far below the safety cutoff, though the expert-bias controller comes close to its clamp. This confirms the high-expert regime is limited by optimizer state, memory traffic, and wall-clock progress, not by active compute or router instability.

Agrees with hypothesis:
yes

Decision:
discard

Next run:
Close Phase 1 by selecting the `E=16`, top-2 sparsity ratio, then run the fine-grained Phase 2 counterpart.

### run 37: fine-grained experts E=32 top_k=4 hidden=896

Kind/thread:
sparsity / fine-grained-experts

Pre-run hypothesis:
At the selected sparsity ratio, replacing each coarse expert with two half-width experts and activating four experts per token may improve specialization while preserving active FFN width and total expert width.

Expected result:
Better fixed-wall BPB or better matched-step CE/BPB than the selected coarse `E=16/top-2/1792` run, without severe throughput loss or router instability.

Observed result:
`val_bpb 0.953126`, `2608` steps, `683.7M` tokens, `31.6GB` peak VRAM, and `23.14%` MFU.
Total params `438.2M`, active params `91.4M`.
Router health: mean load CV `0.0910`, max-layer load CV `0.1093`, mean max load `0.0383`, max-layer max load `0.0422`, mean router bias abs `0.0100`, max router bias abs `0.0543`.

Interpretation:
The fine-grained geometry is router-safe, but it is decisively worse than the coarse anchor. It loses `0.017948` BPB, completes `138` fewer steps and `36.1M` fewer tokens, uses more VRAM, and has lower MFU despite nearly identical total and active parameter counts. Train CE is also worse (`2.593632` vs `2.563341`), so there is no matched-step or scale-candidate signal to pursue.

Agrees with hypothesis:
no

Decision:
discard

Next run:
Use the coarse `E=16/top-2/1792` geometry for scaling. Treat run 31 as the depth-8 full-architecture F1 anchor, then run F2 at `DEPTH=10`, `MODEL_DIM=1024`, and `MOE_HIDDEN_DIM=2304`.

### run 38: scale depth=10 dim=1024 curve=full

Kind/thread:
scaling / full

Pre-run hypothesis:
Scaling the selected full architecture from depth `8`, dim `768` to depth `10`, dim `1024` should test whether the architecture bundle remains stable and whether larger active capacity can improve validation under the same fixed wall-clock budget.

Expected result:
The model should fit and train without router collapse. BPB may improve if the larger active model's sample efficiency compensates for fewer wall-clock updates, but the main risk is undertraining from the lower token count and higher memory footprint.

Observed result:
`val_bpb 0.973316`, `1434` steps, `375.9M` tokens, `45.3GB` peak VRAM, and `25.59%` MFU.
Total params `977.5M`, active params `184.8M`.
Router health: mean load CV `0.0923`, max-layer load CV `0.1366`, mean max load `0.0733`, max-layer max load `0.0770`, mean router bias abs `0.0098`, max router bias abs `0.0482`.

Interpretation:
The full architecture scales mechanically and remains router-safe at this size, but the fixed-wall result is not competitive with the depth-8 anchor. The larger active model reaches only `1434` steps and `375.9M` tokens, so it is severely undertrained under the unchanged 450-second wall. This is a wall-clock/sample-budget scaling failure so far, not a routing failure.

Agrees with hypothesis:
partial

Decision:
keep as scale datapoint

Next run:
Run F3 at `DEPTH=12`, `MODEL_DIM=1280`, and `MOE_HIDDEN_DIM=3072`, or log the exact OOM/failure if it does not fit.

### run 39: scale depth=12 dim=1280 curve=full

Kind/thread:
scaling / full

Pre-run hypothesis:
Scaling the selected full architecture to depth `12`, dim `1280` tests whether the bundle still fits and trains cleanly near the memory ceiling, and whether the larger active model can compensate for substantially fewer fixed-wall updates.

Expected result:
The run may fit but is likely heavily wall-clock limited. Router safety is the main non-BPB check, since larger depth and width can expose layerwise expert concentration that was not visible at smaller sizes.

Observed result:
`val_bpb 1.045977`, `751` steps, `196.9M` tokens, `69.5GB` peak VRAM, and `24.90%` MFU.
Total params `2003.1M`, active params `351.6M`.
Router health: mean load CV `0.2301`, max-layer load CV `0.9632`, mean max load `0.1035`, max-layer max load `0.2745`, mean router bias abs `0.0182`, max router bias abs `0.0993`.

Interpretation:
The model fits and runs through validation, but it is not competitive under the fixed 450-second wall clock. It sees only `196.9M` tokens, less than one third of the depth-8 anchor, and validation degrades sharply. Average routing remains usable, but the final max-layer max expert load crosses the `E=16` safety threshold of `0.25`, so scale has introduced a layerwise router warning even without a full collapse.

Agrees with hypothesis:
partial

Decision:
keep as scale datapoint with router warning

Next run:
Sweep F3 learning rate downward first. Larger F4/F5 points should be revisited later with smaller `DEVICE_BATCH_SIZE` and gradient accumulation, not treated as meaningful architecture results.

### run 40: scale depth=12 dim=1280 curve=full lr=0.001

Kind/thread:
scaling-lr / full

Pre-run hypothesis:
The F3 `0.003` run may be too aggressive: it trained faster but ended with a layerwise router warning. Lowering AdamW peak LR to `0.001` could improve validation if router concentration or update noise is the main problem at this size.

Expected result:
Cleaner router load and possibly better BPB. The risk is undertraining, since F3 gets only about `750` optimizer steps under the fixed wall clock.

Observed result:
`val_bpb 1.063829`, `754` steps, `197.7M` tokens, `69.5GB` peak VRAM, and `24.98%` MFU.
Total params `2003.1M`, active params `351.6M`.
Router health: mean load CV `0.0889`, max-layer load CV `0.1192`, mean max load `0.0735`, max-layer max load `0.0806`, mean router bias abs `0.0058`, max router bias abs `0.0451`.

Interpretation:
Lowering LR fixed the router warning decisively, but it did not improve the model. Validation worsened by `0.017852` BPB versus F3 at `0.003`, and train CE was higher (`3.083678` vs `2.986518`). The F3 problem is not solved by going this low; `0.0003` is not worth running now.

Agrees with hypothesis:
partial

Decision:
discard as default, keep as LR datapoint

Next run:
Pivot away from fixed-wall scaling and run the selected 90M-active model to the compute-optimal token target with a coarse LR grid.

### run 41: compute-optimal 90M lr=0.003

Kind/thread:
compute-optimal-90m / lr-sweep

Pre-run hypothesis:
The selected `~91M` active-parameter architecture should improve substantially when trained to the rough compute-optimal target (`~1.8B` tokens) rather than the 450-second fixed-wall budget. LR `0.003` is the fixed-wall winner and should be the high end of the coarse 3x LR grid.

Expected result:
Reach `7000` optimizer steps with the cosine schedule horizon also set to `7000`. Validation should beat the fixed-wall `0.935178` anchor if the compute target is useful. Router load should remain healthy because the selected E16 geometry was stable in the shorter runs.

Observed result:
`val_bpb 0.892152`, `7000` steps, `1.835B` tokens, `29.3GB` peak VRAM, and `24.30%` MFU.
Total params `438.2M`, active params `91.3M`.
Train CE `2.307537`, train total `2.310098`.
Router health: mean load CV `0.0661`, max-layer load CV `0.1012`, mean max load `0.0720`, max-layer max load `0.0779`, mean router bias abs `0.0080`, max router bias abs `0.0277`.

Interpretation:
The compute-optimal protocol is sane for the 90M anchor: validation improves by `0.043026` BPB over the fixed-wall E16 result while keeping the same memory footprint and stable routing. LR `0.003` is a strong candidate, but it still needs the coarse lower-LR comparisons before selecting the compute-optimal default.

Agrees with hypothesis:
yes

Decision:
keep pending LR sweep

Next run:
Run the same 90M compute-optimal setup at `ADAMW_LR=0.001`.

### run 42: compute-optimal 90M lr=0.001

Kind/thread:
compute-optimal-90m / lr-sweep

Pre-run hypothesis:
Lowering AdamW peak LR by 3x may improve the long-run validation result if `0.003` is still too aggressive at the compute-optimal token target, though the risk is slower optimization over the fixed `7000` steps.

Expected result:
Router load should remain healthy. The key question is whether the lower LR can recover validation despite slower train-loss progress.

Observed result:
`val_bpb 0.936404`, `7000` steps, `1.835B` tokens, `29.3GB` peak VRAM, and `24.33%` MFU.
Total params `438.2M`, active params `91.3M`.
Train CE `2.419355`, train total `2.421871`.
Router health: mean load CV `0.0563`, max-layer load CV `0.0821`, mean max load `0.0707`, max-layer max load `0.0743`, mean router bias abs `0.0054`, max router bias abs `0.0222`.

Interpretation:
LR `0.001` is much too slow for the 90M compute-optimal run. It slightly cleans up already-healthy router metrics, but validation worsens by `0.044252` BPB versus LR `0.003`, and train CE is higher by `0.111818`. This also underperforms the older fixed-wall E16 anchor, so the problem is optimization speed, not routing safety.

Agrees with hypothesis:
no

Decision:
discard

Next run:
Switch to the full downloaded train pool and run the F2 compute-optimal target at `ADAMW_LR=0.003`.

### run 43: compute-optimal F2 full-data lr=0.003

Kind/thread:
compute-optimal-full-data-scale / lr-sweep

Pre-run hypothesis:
Scaling the selected full architecture to F2 (`DEPTH=10`, `MODEL_DIM=1024`, `MOE_HIDDEN_DIM=2304`, about `184.8M` active params) and training to the rough compute-optimal target should improve validation if the scaling protocol is meaningful. LR `0.003` is the high end of the coarse grid; it may still work because it won the 90M run, but the larger active model could eventually prefer a lower LR.

Expected result:
Run `14099` steps, matching `20` tokens per active parameter (`~3.696B` tokens) and the cosine schedule horizon. The run should remain under one full-data epoch, fit near the fixed-wall F2 memory footprint, and avoid router concentration. Validation should beat the 90M compute-optimal result if scale is paying off.

Observed result:
`val_bpb 0.822997`, `14099` steps, `3.696B` tokens, `45.3GB` peak VRAM, and `25.66%` MFU.
Total params `977.5M`, active params `184.8M`.
Train CE `2.263927`, train total `2.266572`.
Router health: mean load CV `0.0850`, max-layer load CV `0.1108`, mean max load `0.0736`, max-layer max load `0.0780`, mean router bias abs `0.0119`, max router bias abs `0.0401`.

Interpretation:
This is the first full-data compute-optimal scale result and it is a strong positive scaling datapoint. It improves over the 90M `0.003` compute-optimal result by `0.069155` BPB and over the old fixed-wall F2 result by `0.150319` BPB, while using the expected memory footprint and staying router-safe. The run consumed only about `14.6%` of the full train pool, so data exhaustion is not a concern. LR `0.003` is not too aggressive at F2, but the local LR sweep is still needed before treating it as the default for larger models.

Agrees with hypothesis:
yes

Decision:
keep pending LR sweep

Next run:
Run the same F2 full-data compute-optimal setup at `ADAMW_LR=0.001`. Skip `0.000333` unless `0.001` is competitive enough to suggest the optimum moved below `0.003`.

### run 44: compute-optimal F2 full-data lr=0.001

Kind/thread:
compute-optimal-full-data-scale / lr-sweep

Pre-run hypothesis:
Lowering AdamW peak LR by 3x from the F2 high-LR point may improve the full-data compute-optimal result if the larger active model prefers a gentler schedule than the 90M anchor. The risk is slower optimization over the same fixed `14099` steps.

Expected result:
Run the same full `400`-shard train pool, `14099` optimizer steps, and `~3.696B` tokens. Router load should remain healthy. If validation worsens clearly, keep `0.003`; if it matches or improves, finish the coarse lower point at `0.000333`.

Observed result:
`val_bpb 0.821625`, `14099` steps, `3.696B` tokens, `45.3GB` peak VRAM, and `25.69%` MFU.
Total params `977.5M`, active params `184.8M`.
Train CE `2.262125`, train total `2.264702`.
Router health: mean load CV `0.0752`, max-layer load CV `0.1062`, mean max load `0.0716`, max-layer max load `0.0753`, mean router bias abs `0.0086`, max router bias abs `0.0459`.

Interpretation:
LR `0.001` is the current F2 full-data best. It improves validation by `0.001372` BPB versus LR `0.003`, with nearly identical memory and throughput, slightly lower train CE, and cleaner mean router balance. The gain is small but directionally consistent enough that the F2 optimum may be around or below `0.001`; the lower coarse point is now worth running.

Agrees with hypothesis:
yes

Decision:
keep as current F2 best pending lower LR

Next run:
Run the same F2 full-data compute-optimal setup at `ADAMW_LR=0.000333`. If it worsens, select `0.001` as the F2 LR anchor and move to F3 full-data compute-optimal scaling.

### run 45: compute-optimal F2 full-data lr=0.000333

Kind/thread:
compute-optimal-full-data-scale / lr-sweep

Pre-run hypothesis:
Because LR `0.001` narrowly beat `0.003`, the lower coarse point should be tested to see whether the F2 optimum moved farther down. The main risk is severe undertraining over the fixed `14099` step budget.

Expected result:
Run the same full `400`-shard train pool and `~3.696B` token budget. Router load should be at least as clean as the higher-LR runs. The run is useful only if validation approaches or beats the `0.001` result; cleaner routing alone is not enough.

Observed result:
`val_bpb 0.946376`, `14099` steps, `3.696B` tokens, `45.3GB` peak VRAM, and `25.76%` MFU.
Total params `977.5M`, active params `184.8M`.
Train CE `2.610592`, train total `2.613160`.
Router health: mean load CV `0.0404`, max-layer load CV `0.0585`, mean max load `0.0679`, max-layer max load `0.0713`, mean router bias abs `0.0051`, max router bias abs `0.0395`.

Interpretation:
LR `0.000333` is much too slow for F2 at this compute target. Validation is worse than LR `0.001` by `0.124751` BPB and train CE is higher by `0.348467`. The router is very balanced, but the quality loss is decisive; the F2 full-data LR anchor is `0.001`.

Agrees with hypothesis:
no

Decision:
discard; select F2 LR `0.001`

Next run:
Move to F3 full-data compute-optimal scaling with `DEPTH=12`, `MODEL_DIM=1280`, `MOE_HIDDEN_DIM=3072`, `ADAMW_LR=0.001`, and `AR_MAX_STEPS=26827` / `AR_ESTIMATED_TOTAL_STEPS=26827`.

### run 46: compute-optimal F3 full-data lr=0.001

Kind/thread:
compute-optimal-full-data-scale / lr-sweep

Pre-run hypothesis:
Scaling the selected full architecture to F3 (`DEPTH=12`, `MODEL_DIM=1280`, `MOE_HIDDEN_DIM=3072`, about `351.6M` active params) and training to the rough compute-optimal target should improve validation if the full-data scaling protocol is meaningful. LR `0.001` is the selected F2 anchor and a plausible starting point for the larger active model.

Expected result:
Run `26827` steps, matching `20` tokens per active parameter (`~7.033B` tokens) and the cosine schedule horizon. The run should fit on 80GB H100s based on the fixed-wall F3 memory footprint, remain under one full-data epoch, and avoid sustained router concentration.

Observed result:
`val_bpb 0.765974`, `26827` steps, `7.033B` tokens, `69.5GB` peak VRAM, and `25.19%` MFU.
Total params `2003.1M`, active params `351.6M`.
Train CE `2.178932`, train total `2.181612`.
Router health: mean load CV `0.0958`, max-layer load CV `0.1619`, mean max load `0.0751`, max-layer max load `0.0833`, mean router bias abs `0.0094`, max router bias abs `0.0315`.

Interpretation:
F3 is a strong positive full-data scaling point. It improves over the selected F2 LR `0.001` result by `0.055651` BPB while using the expected memory footprint and only about `27.8%` of the full train pool. Router load is somewhat noisier than F2 but not collapsed; max-layer max load stays close to uniform for top-2 over 16 experts.

Agrees with hypothesis:
yes

Decision:
selected F3 LR anchor

Next run:
Start the high-LR F3 check at `ADAMW_LR=0.003` only as an early diagnostic. Stop it if matched-step loss and router balance are clearly worse than LR `0.001`; do not try higher LR unless the diagnostic unexpectedly reverses.

### run 47: compute-optimal F3 full-data lr=0.003 early stop

Kind/thread:
compute-optimal-full-data-scale / lr-sweep

Pre-run hypothesis:
The F3 optimum might still be above `0.001`, but this is unlikely because F2 already preferred `0.001` over `0.003`, and larger active models generally do not require higher peak LR. A short matched-step diagnostic can reject the high side without spending the full `26827` steps.

Expected result:
If LR `0.003` is viable at F3, it should not trail LR `0.001` by matched-step training loss, and router concentration should be no worse than the completed LR `0.001` run.

Observed result:
Aborted around step `2600` (`~0.682B` tokens). MFU was normal at about `25.1%`, so throughput was not the issue. At matched step `2599`, LR `0.003` had loss `2.744902` versus LR `0.001` loss `2.654509`. Router balance was also worse: load CV `0.226` versus `0.098`, max load `0.103` versus `0.075`, and router entropy `1.174` versus `1.360`.

Interpretation:
The high-LR side is not promising at F3. The run is already substantially worse on optimization and routing before `10%` of the schedule, matching the F2 signal that `0.003` is above the useful range for larger full-data compute-optimal runs.

Agrees with hypothesis:
yes

Decision:
abort; do not test LR above `0.003`

Next run:
Move to F4 compute-optimal scaling from the F3/F2 LR anchor. Start with `ADAMW_LR=0.001`; use gradient accumulation if the F4 batch does not fit directly.
