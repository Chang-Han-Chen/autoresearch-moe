# autoresearch: 100M-active MoE architecture search

This run studies architecture interventions for a small but real MoE Transformer. The benchmark is intentionally narrow: fixed data, fixed tokenizer, fixed validation metric, fixed 5-minute training budget, and a 4-GPU H100 launch. The goal is not to win by random fiddling. The goal is to build a sequence of small experiments where each run tests a clear hypothesis about why an architectural change should improve validation BPB.

## Setup

To set up a new experiment, work with the user/operator to:

1. **Agree on a run tag.** Propose a tag based on today's date, for example `may4-moe`. The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch.** Start from current master:

   ```bash
   git checkout -b autoresearch/<tag>
   ```

3. **Read the in-scope files.** The repo is small; read these before changing anything:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, tokenizer, dataloader, validation metric. Do not modify.
   - `train.py` — the only source file you modify during experiments.
   - `program.md` — this protocol.

4. **Verify data exists.** Check that `~/.cache/autoresearch/` contains training shards and a tokenizer. If not, tell the operator to run:

   ```bash
   uv run prepare.py --num-shards 8
   ```

   For 4-GPU DDP, at least 4 training shards are needed; 8 or more is better.

5. **Initialize logs.** Create `results.tsv` and `research_log.md` if they do not exist. These files are experiment records and should remain untracked unless the operator explicitly asks otherwise.

6. **Run the baseline first.** The first recorded run is the unmodified 100M-active top-2 MoE baseline in `train.py`.

## Launch command

Each experiment runs on 4 GPUs by default:

```bash
AR_TIME_BUDGET=300 uv run torchrun --standalone --nproc_per_node=4 train.py > run.log 2>&1
```

The training loop runs for 5 minutes of measured training time, excluding early warmup steps. Startup and validation add extra wall-clock time. If the full command exceeds 15 minutes, kill it and treat it as a failed run unless the log clearly shows that validation is still progressing and the operator has allowed a longer timeout.

## Fixed constraints

### You CAN change

- `train.py` only.
- Model architecture: attention, normalization, residual paths, MoE structure, routing, expert layout, value paths, gating, local/full attention schedule, model size, batch size, and activation checkpointing if useful.
- Learning-rate values.
- MoE-specific coefficients and architecture hyperparameters, such as `NUM_EXPERTS`, `TOP_K`, `MOE_HIDDEN_DIM`, router z-loss coefficient, load-balance coefficient, shared experts, MoE layer frequency, and local/global attention pattern.
- DDP implementation details required to keep training correct and stable.

### You CANNOT change

- `prepare.py`.
- `evaluate_bpb` or the validation data path.
- The tokenizer or data source.
- Installed packages or dependencies.
- The optimizer algorithm. Do not change optimizer family, Adam/Muon betas, eps, momentum schedule, Muon orthogonalization, weight-decay rule, or parameter grouping. You may change learning-rate values only.
- The main score. The main score is always `val_bpb` printed by `prepare.py:evaluate_bpb`.

## Current baseline architecture

The starting `train.py` is a 4-GPU DDP MoE Transformer with approximately 100M active parameters and roughly 298M total parameters. It uses:

- top-2 token-choice MoE FFNs,
- 8 experts,
- GQA attention,
- RoPE,
- QK norm,
- value embeddings/value residuals,
- alternating local/full attention by pattern,
- router z-loss and load-balancing loss,
- MuonAdamW from the dense baseline.

Treat this as the MoE baseline. Do not compare a later MoE intervention against the old dense model as if it were the same benchmark.

## Quantitative output format

At the end of a successful run, `train.py` prints a summary like:

```text
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    430.0
startup_seconds:  22.0
peak_vram_mb:     62000.0
mfu_percent:      31.5
total_tokens_M:   900.0
num_steps:        1700
num_params_M:     297.8
active_params_M:  99.7
world_size:       4
depth:            8
model_dim:        768
num_experts:      8
top_k:            2
router_entropy:   2.01
expert_load_cv:   0.05
max_expert_load:  0.14
router_z_loss:    4.30
router_lb_loss:   1.00
router_aux_loss:  0.0143
```

Extract the key fields with:

```bash
grep "^val_bpb:\|^peak_vram_mb:\|^total_tokens_M:\|^active_params_M:\|^num_params_M:\|^router_entropy:\|^expert_load_cv:\|^max_expert_load:" run.log
```

If no `val_bpb:` appears, inspect the failure:

```bash
tail -n 80 run.log
```

## `results.tsv`

Use tab-separated values, not comma-separated values. Initialize with this header:

```text
commit	val_bpb	memory_gb	status	hypothesis	expected_result	interpretation	agrees_with_hypothesis	next_hypothesis	description
```

Field meanings:

1. `commit`: short git hash, 7 characters.
2. `val_bpb`: validation BPB, lower is better. Use `0.000000` for crashes.
3. `memory_gb`: peak VRAM in GB, rounded to one decimal. Use `0.0` for crashes.
4. `status`: `keep`, `discard`, or `crash`.
5. `hypothesis`: the causal claim tested by the run.
6. `expected_result`: what you expected before running, including at least one predicted diagnostic when possible.
7. `interpretation`: what the result means after seeing BPB, memory, throughput, and router diagnostics.
8. `agrees_with_hypothesis`: `yes`, `no`, `partial`, or `unclear`.
9. `next_hypothesis`: how your belief changed and what should be tested next.
10. `description`: short mechanical description of the code change.

Example:

```text
commit	val_bpb	memory_gb	status	hypothesis	expected_result	interpretation	agrees_with_hypothesis	next_hypothesis	description
a1b2c3d	0.997900	60.5	keep	baseline establishes stable router and throughput	load_cv below 0.10 and no BPB regression to crash	stable baseline; router balanced and memory acceptable	yes	test whether lower load-balance coefficient reduces objective conflict	baseline 100M-active top-2 MoE
b2c3d4e	0.996800	60.7	keep	load-balance loss is slightly over-regularizing experts	same load_cv below 0.15 and lower val_bpb	BPB improved while load stayed healthy; belief increases that aux loss was too strong	yes	test smaller z-loss only if router_z_loss remains controlled	reduce load balance coefficient 1e-2 to 3e-3
c3d4e5f	1.004000	60.6	discard	removing z-loss will let router specialize faster	lower router entropy and better BPB	BPB worsened and z-loss grew; specialization became instability, not useful routing	no	restore z-loss and test shared expert instead	set router_z_loss_coef to 0
```

Keep entries concise, but not empty. A run without an interpretation is not a scientific run; it is just a CUDA-powered coin toss.

## `research_log.md`

Use this for longer reasoning than fits in `results.tsv`. For every run, write a short entry with this structure:

```md
## run <N>: <short name>

Pre-run hypothesis:
<one or two sentences explaining the mechanism>

Expected result:
<what BPB, memory, speed, and router diagnostics should do if the hypothesis is right>

Observed result:
<actual val_bpb, memory, speed, router entropy/load_cv/max_load, and crash notes if any>

Interpretation:
<why the result happened; include alternative explanations and confounders>

Agrees with hypothesis:
<yes | no | partial | unclear>

Next run:
<the next hypothesis and intervention>
```

The key discipline is: **pre-register the hypothesis before launching the run, then update the belief after reading the result**. Do not simply try changes because they are fashionable or because they appeared in a speedrun.

## Experiment loop

Repeat this loop until manually stopped by the operator:

1. Check git state and current best result.
2. Read the last few entries in `results.tsv` and `research_log.md`.
3. Form one explicit hypothesis. It should name a problem and a mechanism, for example: “router z-loss is too large and suppresses useful specialization,” or “local attention is too aggressive and hurts document-level dependencies.”
4. Predict what should happen if the hypothesis is right. Include at least one diagnostic besides BPB when possible: expert load CV, max expert load, router entropy, tokens/sec, MFU, or memory.
5. Edit `train.py` with the smallest principled change that tests the hypothesis.
6. Commit the change.
7. Run:

   ```bash
   AR_TIME_BUDGET=300 uv run torchrun --standalone --nproc_per_node=4 train.py > run.log 2>&1
   ```

8. Extract results from `run.log`.
9. Write the quantitative row to `results.tsv`.
10. Write the interpretation, whether the result agreed with the hypothesis, and the next hypothesis to `research_log.md`.
11. If `val_bpb` improved and the added complexity is justified, keep the commit and continue from it.
12. If `val_bpb` is worse or the improvement is too small for the complexity cost, reset back to the previous best commit.
13. If the run crashes, decide whether it was a trivial implementation bug or a broken idea. Fix trivial bugs; otherwise log `crash`, reset, and move on.

## Decision rule

The primary metric is lower `val_bpb`. But do not keep ugly complexity for noise-level gains.

Keep a change when at least one of the following is true:

- It improves BPB clearly.
- It gives similar BPB with simpler code, lower memory, or higher throughput.
- It improves router stability enough to enable later experiments, and the interpretation explains why.

Discard a change when:

- BPB worsens and diagnostics do not reveal an obvious fix.
- BPB improves by only noise-level amount while adding substantial complexity.
- It destabilizes routing: high expert load CV, high max expert load, exploding z-loss, NaNs, or repeated crashes.
- It changes the optimizer algorithm rather than testing architecture.

## MoE diagnostic interpretation

Use these as rough guides, not absolute laws:

- `expert_load_cv` near 0 is balanced. A moderate increase can be useful specialization; a large increase is often collapse.
- `max_expert_load` much above uniform load suggests routing concentration. For 8 experts, uniform hard load is about 0.125.
- High router entropy means diffuse routing; low entropy means specialization or collapse. Interpret it with load CV and BPB, not alone.
- `router_z_loss` growing rapidly usually means router logits are becoming too large.
- Lower BPB with slightly worse load balance can be real specialization. Worse BPB with worse load balance is probably collapse.
- Higher tokens/sec with worse BPB is not an architecture win unless the BPB/compute tradeoff is explicitly the hypothesis.

## Suggested intervention queue

Start with MoE stability and routing before exotic attention changes. Otherwise every result is confounded by router collapse.

1. **Router coefficients.** Tune load-balance coefficient and router z-loss coefficient. Hypothesis examples: “aux loss over-constrains specialization,” “z-loss is needed to prevent logit explosion.”
2. **Expert shape.** Try `MOE_HIDDEN_DIM`, `NUM_EXPERTS`, and `TOP_K` changes while keeping active params near 100M when possible.
3. **MoE frequency.** Test MoE every layer versus alternating dense/MoE layers or a shared dense expert. Hypothesis: not every layer needs sparse specialization.
4. **Shared expert.** Add one always-on expert or dense FFN path. Hypothesis: common knowledge should not consume routed expert capacity.
5. **Attention schedule.** Compare `SSSL`, `SL`, and more local-heavy patterns. Hypothesis: short 2048-token training may not benefit from frequent full attention.
6. **Gated attention.** Add a query-dependent gate after attention. Hypothesis: it reduces useless context injection and attention-sink behavior.
7. **Value residual variants.** Change where value embeddings are used or how strongly they are gated. Hypothesis: value residuals help deep information flow but may be overused.
8. **Depth-scaled RMSNorm or residual scaling.** Hypothesis: later blocks are either too identity-like or too noisy; scaling should improve useful depth.
9. **XSA / IHA / selective attention.** Only after the baseline router is stable. These are promising but more confounded.
10. **Learning-rate retunes.** For a plausible architecture that narrowly fails, try one or two LR adjustments before discarding it. Do not change optimizer internals.

## Simplicity criterion

Prefer mechanisms that have a clear problem-solution story. A small BPB improvement from a simple, interpretable change is valuable. A small BPB improvement from a tangled stack of hacks is suspect. A simplification with equal BPB is a win.

A good experiment has this form:

> Problem: router load balancing may be fighting useful specialization.  
> Intervention: reduce load-balance coefficient.  
> Expected: lower BPB, slightly higher load CV, stable z-loss.  
> Result: compare BPB and diagnostics.  
> Belief update: decide whether to keep, retune, or revert.

That is the standard for every run.