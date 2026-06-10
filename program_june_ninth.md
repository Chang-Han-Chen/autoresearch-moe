# June 9 Program: Sparsity Sweep, Then Scale Validation

This file is an implementation plan. An agent should be able to read it, run the experiments, update `research_log.md`, and leave `train.py` in a sensible final state.

## Goal

We have a current fixed-wall best architecture at 91.3M active params:

- `DEPTH=8`
- `MODEL_DIM=768`
- `NUM_EXPERTS=16`
- `TOP_K=2`
- `MOE_HIDDEN_DIM=1792`
- `DENSE_EARLY_LAYERS=2`
- `DENSE_HIDDEN_DIM=3584`
- fixed value mix `v = 0.75*v_1 + 0.25*v_l`
- sigmoid-affinity router plus expert bias
- `LOAD_BALANCE_LOSS_COEF=0.003`
- `ROUTER_Z_LOSS_COEF=7.5e-4`
- exclusive self-attention
- headwise attention gate initialized to sigmoid `0.98`
- AdamW peak LR `0.003`

Known result:
`val_bpb 0.938179`, `2778` steps, `728.2M` tokens, `29.3GB` peak VRAM, `24.63%` MFU, `438.2M` total params, `91.3M` active params.

We need to answer three questions:

1. What is the best sparsity ratio when active expert compute is held fixed?
2. At that sparsity ratio, is a fine-grained top-4 geometry better than the coarse top-2 geometry?
3. After choosing sparsity and expert granularity, do the architecture interventions scale with model size, or do their gains diminish?

## Run Command

Use the repo's normal command:

```bash
uv run torchrun --standalone --nproc_per_node=4 train.py
```

For fixed-wall runs, use the default 5-minute budget from `prepare.py`.

For matched-step diagnostics, use:

```bash
AR_MAX_STEPS=<steps> uv run torchrun --standalone --nproc_per_node=4 train.py
```

Do not change `prepare.py`. Edit the constants in `train.py` for each run.

## Phase 1: Expert-Count Sparsity Sweep

Purpose:
choose the best `TOP_K / NUM_EXPERTS` sparsity ratio while keeping active expert compute fixed.

Run these six configs:

| run | `NUM_EXPERTS` | `TOP_K` | sparsity ratio | `MOE_HIDDEN_DIM` | `DENSE_EARLY_LAYERS` |
|---:|---:|---:|---:|---:|---:|
| S1 | 4 | 2 | `50.0%` | 1792 | 2 |
| S2 | 8 | 2 | `25.0%` | 1792 | 2 |
| S3 | 16 | 2 | `12.5%` | 1792 | 2 |
| S4 | 32 | 2 | `6.25%` | 1792 | 2 |
| S5 | 64 | 2 | `3.125%` | 1792 | 2 |
| S6 | 128 | 2 | `1.5625%` | 1792 | 2 |

Keep all other current-best settings unchanged.

Approximate expected parameter counts:

| `NUM_EXPERTS` | active params | total params |
|---:|---:|---:|
| 4 | `91.3M` | `140.9M` |
| 8 | `91.3M` | `240.0M` |
| 16 | `91.3M` | `438.2M` |
| 32 | `91.3M` | `834.6M` |
| 64 | `91.3M` | `1.63B` |
| 128 | `91.3M` | `3.21B` |

Use the printed `num_params_M` and `active_params_M` from each actual run as authoritative.

### Phase 1 Logging

After every run, append a run entry to `research_log.md` under the existing `## Runs` section.

Use this exact shape:

```text
### run N: expert-count E=<NUM_EXPERTS>

Kind/thread:
sparsity / expert-count

Pre-run hypothesis:
At fixed `TOP_K=2` and `MOE_HIDDEN_DIM=1792`, changing `NUM_EXPERTS` changes total sparse capacity and sparsity ratio while keeping active expert compute roughly fixed.

Expected result:
<one or two sentences specific to this expert count>

Observed result:
`val_bpb ...`, `...` steps, `...M` tokens, `...GB` peak VRAM, and `...%` MFU.
Total params `...M`, active params `...M`.
Router health: mean load CV `...`, max-layer load CV `...`, mean max load `...`, max-layer max load `...`, mean router bias abs `...`, max router bias abs `...`.

Interpretation:
<explain quality, speed, memory, and router behavior>

Agrees with hypothesis:
yes/no/partial

Decision:
keep/discard/repair/scale-candidate

Next run:
<the next expert count, matched-step diagnostic, or final sparsity decision>
```

Also add a compact table near the top of `research_log.md` after the current fixed-wall leaderboard:

```text
## Sparsity Sweep Leaderboard

| num_experts | sparsity ratio | val_bpb | steps | tokens_M | total_params_M | active_params_M | peak_vram_gb | mfu_percent | decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
...
```

### Phase 1 Decision Rule

Primary decision:
select the `NUM_EXPERTS` with the lowest fixed-wall `val_bpb`.

Tie rule:
if two runs are within `0.0003` BPB, prefer the one with lower total params, higher throughput, and cleaner router load.

Scale-candidate rule:
if a higher-expert run has worse fixed-wall BPB but clearly better matched-step CE or BPB, mark it `scale-candidate` rather than default.

Router safety rule:
do not select a run as default if it has severe router collapse, defined as any of:

- max-layer max expert load `>= 0.25`
- max-layer load CV `>= 1.0`
- min-layer router entropy near zero
- expert bias hitting the clamp

If such a run has the best BPB, mark it `repair`, not `keep`.

### Phase 1 Matched-Step Diagnostics

Run matched-step diagnostics only after all six fixed-wall runs are complete.

Run diagnostics for:

- the fixed-wall winner;
- any run within `0.001` BPB of the winner;
- any slower high-expert run that looks better by train CE at similar steps.

Use this procedure:

1. If the contender reaches at least `2778` steps under fixed wall time, run it with `AR_MAX_STEPS=2778` and compare against the existing `E=16` anchor.
2. If the contender reaches fewer than `2778` steps, use its fixed-wall step count as `MATCH_STEPS`.
3. For a precise comparison, run both the contender and `E=16` with `AR_MAX_STEPS=$MATCH_STEPS`.
4. Log matched-step results in the same run entry or in a short diagnostic entry immediately after it.

After diagnostics, add:

```text
## Selected Sparsity Ratio

Selected default:
`NUM_EXPERTS=<E>`, `TOP_K=2`, sparsity ratio `<2/E>`.

Reason:
...

Scale candidate, if any:
...
```

Do not begin Phase 2 until this section exists.

## Phase 2: Fine-Grained Expert Test

Purpose:
test whether smaller experts with more active experts improve quality after the best coarse sparsity ratio has been chosen.

Do not test this by only halving `MOE_HIDDEN_DIM`. That would also halve active FFN width and, because `DENSE_HIDDEN_DIM = TOP_K * MOE_HIDDEN_DIM`, would halve the dense stem. The fine-grained test must preserve active FFN width and total expert width.

Let the selected Phase 1 expert count be `E`.

Run one fixed-wall fine-grained counterpart:

```python
NUM_EXPERTS = 2 * E
TOP_K = 4
MOE_HIDDEN_DIM = 896
DENSE_HIDDEN_DIM = TOP_K * MOE_HIDDEN_DIM  # 3584
DENSE_EARLY_LAYERS = 2
```

Keep all attention, router, value-mix, optimizer, batch, and time-budget settings from the current best stack.

This preserves:

- sparsity ratio: `TOP_K / NUM_EXPERTS = 4 / (2E) = 2 / E`
- active FFN width: `TOP_K * MOE_HIDDEN_DIM = 4 * 896 = 3584`
- total expert width: `NUM_EXPERTS * MOE_HIDDEN_DIM = (2E) * 896 = E * 1792`

The router has twice as many output logits, so total params and optimizer state will not be bit-identical, but the difference should be small relative to expert matrices.

If `2 * E` is too large for memory or grouped dispatch, log the OOM/failure and select the coarse geometry by default.

### Phase 2 Logging

Append a run entry to `research_log.md`:

```text
### run N: fine-grained experts E=<2E> top_k=4 hidden=896

Kind/thread:
sparsity / fine-grained-experts

Pre-run hypothesis:
At the selected sparsity ratio, replacing each coarse expert with two half-width experts and activating four experts per token may improve specialization while preserving active FFN width and total expert width.

Expected result:
Better fixed-wall BPB or better matched-step CE/BPB than the selected coarse `E/top-2/1792` run, without severe throughput loss or router instability.

Observed result:
`val_bpb ...`, `...` steps, `...M` tokens, `...GB` peak VRAM, and `...%` MFU.
Total params `...M`, active params `...M`.
Router health: mean load CV `...`, max-layer load CV `...`, mean max load `...`, max-layer max load `...`, mean router bias abs `...`, max router bias abs `...`.

Interpretation:
...

Agrees with hypothesis:
yes/no/partial

Decision:
keep/discard/repair/scale-candidate

Next run:
...
```

Add this compact table near the top of `research_log.md` after the selected sparsity section:

```text
## Fine-Grained Expert Check

| geometry | num_experts | top_k | moe_hidden_dim | dense_hidden_dim | sparsity ratio | val_bpb | steps | tokens_M | total_params_M | active_params_M | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| coarse | E | 2 | 1792 | 3584 | 2/E | ... | ... | ... | ... | ... | ... |
| fine | 2E | 4 | 896 | 3584 | 2/E | ... | ... | ... | ... | ... | ... |
```

### Phase 2 Decision Rule

Select the fine-grained geometry if it:

- beats the selected coarse geometry on fixed-wall `val_bpb`;
- is router-safe under the same safety criteria as Phase 1;
- does not lose enough throughput that the fixed-wall win is likely noise.

If fixed-wall BPB is within `0.0003`, prefer the coarse geometry unless fine-grained also has clearly better matched-step CE/BPB.

If fine-grained is worse fixed-wall but better matched-step, mark it `scale-candidate` and use the coarse geometry for the main scaling curve unless the user explicitly chooses to scale the candidate.

Run a matched-step diagnostic if either geometry is within `0.001` BPB of the other or if fine-grained is slower but has better train CE at similar steps. Use the lower fixed-wall step count as `MATCH_STEPS`, then run both coarse and fine with `AR_MAX_STEPS=$MATCH_STEPS`.

After the decision, add:

```text
## Selected MoE Geometry

Selected default:
`NUM_EXPERTS=...`, `TOP_K=...`, `MOE_HIDDEN_DIM=...`, `DENSE_HIDDEN_DIM=...`.

Reason:
...

Scale candidate, if any:
...
```

Do not begin Phase 3 until this section exists.

## Phase 3: Scale the Selected Architecture

Purpose:
test whether the current architecture interventions keep helping as model size grows.

Use the selected MoE geometry from Phase 2. If coarse wins, scale `TOP_K=2` with the coarse hidden sizes. If fine-grained wins, scale `TOP_K=4`, double the selected coarse expert count, and use half the coarse hidden size at each scale.

Run the full recipe at these sizes:

| run | `DEPTH` | `MODEL_DIM` | `NUM_HEADS` | coarse `MOE_HIDDEN_DIM` | fine `MOE_HIDDEN_DIM` | active FFN width / `DENSE_HIDDEN_DIM` |
|---:|---:|---:|---:|---:|---:|---:|
| F1 | 8 | 768 | 6 | 1792 | 896 | 3584 |
| F2 | 10 | 1024 | 8 | 2304 | 1152 | 4608 |
| F3 | 12 | 1280 | 10 | 3072 | 1536 | 6144 |
| F4 | 14 | 1536 | 12 | 3584 | 1792 | 7168 |
| F5 | 16 | 1792 | 14 | 4096 | 2048 | 8192 |

For each run:

- set `DEPTH`;
- set `MODEL_DIM`;
- set `HEAD_DIM=128`;
- set `NUM_HEADS = MODEL_DIM // HEAD_DIM`;
- keep `NUM_KV_HEADS=2`;
- if coarse geometry wins, set `TOP_K=2`, `NUM_EXPERTS=E`, and `MOE_HIDDEN_DIM` from the coarse column;
- if fine geometry wins, set `TOP_K=4`, `NUM_EXPERTS=2*E`, and `MOE_HIDDEN_DIM` from the fine column;
- keep `DENSE_HIDDEN_DIM = TOP_K * MOE_HIDDEN_DIM`;
- keep `DENSE_EARLY_LAYERS=2`;
- keep all current-best attention/router/value settings.

If a run OOMs, log it as OOM with the config and do not silently shrink the model.

## Phase 4: Paired Controls for Scaling

Purpose:
measure whether the architecture intervention gain grows, stays flat, or shrinks with scale.

### Required Control: Simple Scaled Backbone

Run a simple-backbone control at each of the five Phase 3 sizes.

Use the same:

- `DEPTH`
- `MODEL_DIM`
- `NUM_HEADS`
- `NUM_KV_HEADS`
- selected `NUM_EXPERTS`
- selected `TOP_K`
- `MOE_HIDDEN_DIM`
- `DENSE_HIDDEN_DIM = TOP_K * MOE_HIDDEN_DIM`
- AdamW peak LR `0.003`

Change these settings:

```python
DENSE_EARLY_LAYERS = 0
VALUE_MIX_ENABLED = False
EXCLUSIVE_ATTENTION = False
HEADWISE_ATTENTION_GATE = False
ROUTER_SIGMOID_AFFINITY = False
ROUTER_EXPERT_BIAS = False
LOAD_BALANCE_LOSS_COEF = 8.5e-3
ROUTER_Z_LOSS_COEF = 7.5e-4
```

This curve answers:
does the full architecture bundle beat the clean scaled MoE backbone by a similar or larger margin as model size grows?

### Optional Control: No-Dense Stabilized Stack

Run this only if the full-vs-simple delta is confusing or if we need to isolate the dense-stem contribution.

Start with depths `8`, `12`, and `16`. Fill depths `10` and `14` only if needed.

Use the full recipe except:

```python
DENSE_EARLY_LAYERS = 0
```

Keep value mix, sigmoid-bias router, XSA, and headwise gate enabled.

This curve answers:
is the dense stem itself scaling, or is the rest of the architecture bundle carrying the gain?

## Phase 3 and 4 Logging

For every scale run, append a run entry to `research_log.md`:

```text
### run N: scale depth=<DEPTH> dim=<MODEL_DIM> curve=<full|simple|no-dense>

Kind/thread:
scaling / <full|simple-control|dense-stem-control>

Pre-run hypothesis:
...

Expected result:
...

Observed result:
`val_bpb ...`, `...` steps, `...M` tokens, `...GB` peak VRAM, and `...%` MFU.
Total params `...M`, active params `...M`.
Router health: ...

Interpretation:
...

Agrees with hypothesis:
yes/no/partial

Decision:
keep/discard/repair/scale-candidate/control

Next run:
...
```

Also add this summary table near the top of `research_log.md`:

```text
## Scale Validation

| depth | model_dim | curve | num_experts | top_k | moe_hidden_dim | val_bpb | steps | tokens_M | total_params_M | active_params_M | peak_vram_gb | mfu_percent |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
...
```

Then add a paired-delta table:

```text
## Architecture Scaling Deltas

| depth | model_dim | full_bpb | simple_bpb | simple_minus_full | relative_bpb_reduction | no_dense_bpb | no_dense_minus_full |
|---:|---:|---:|---:|---:|---:|---:|---:|
...
```

`relative_bpb_reduction = (simple_bpb - full_bpb) / simple_bpb`.

## Final Decision Rules

### Sparsity

Select one default coarse expert count:

- fixed-wall winner if it is router-safe;
- lower-param/faster option if the BPB difference is within `0.0003`;
- scale-candidate label for slower high-expert options that only win matched-step.

### Expert Granularity

Select one default MoE geometry:

- coarse: `NUM_EXPERTS=E`, `TOP_K=2`, `MOE_HIDDEN_DIM=1792`;
- fine: `NUM_EXPERTS=2E`, `TOP_K=4`, `MOE_HIDDEN_DIM=896`.

Prefer fine only if it wins fixed-wall BPB safely or has such a strong matched-step advantage that the project deliberately chooses to scale a slower candidate.

Leave `train.py` set to the selected MoE geometry after Phase 2 unless Phase 3 or 4 is actively in progress.

### Scaling

Call the architecture interventions "scaling" if:

- `simple_minus_full` is positive at every completed depth; and
- the relative BPB reduction is flat or increasing, or only mildly decreasing.

Call the gains "diminishing" if:

- `simple_minus_full` shrinks monotonically with depth; or
- the depth-16 relative BPB reduction is less than half the depth-8 relative BPB reduction.

Call the result "scale-dependent retuning needed" if:

- full beats simple at small depth but loses at large depth;
- router-safe status changes with size;
- fixed two-layer dense stem loses but no-dense or fractional-dense controls suggest a different dense-stem depth.

At the end, add a final section to `research_log.md`:

```text
## Sparsity and Scale Decision

Selected sparsity:
...

Selected MoE geometry:
...

Do the interventions scale?
...

Recommended default config:
...

Open follow-up:
...
```

Leave `train.py` in the recommended default config from this final section. Do not leave it in a temporary control config.

## Do Not Change During This Program

- Do not change `TOP_K` except for the planned fine-grained geometry: Phase 2 intentionally changes `TOP_K` from `2` to `4`, and Phase 3/4 may use that selected geometry if it wins.
- Do not change the tokenizer, dataset, `prepare.py`, or validation metric.
- Do not introduce new architecture interventions beyond the planned sparsity and fine-grained expert tests.
- Do not change the optimizer or LR schedule unless a run fails numerically, and if that happens, log it as a failure rather than quietly repairing it.
- Do not compare scale curves without paired controls at the same depth and width.
