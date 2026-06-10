# autoresearch: algorithm queue for 100M-active MoE pretraining

This program replaces the old architecture-first queue. The architecture record remains in `research_log.md`; the new phase is about training algorithms, loss functions, target weighting, curricula, and auxiliary objectives that can be reproduced with small `train.py` changes.

The benchmark stays fixed: same data, tokenizer, validation BPB, optimizer family, 5-minute training budget, and current MoE backbone. The research filter is brutal: prefer ideas that can be stated in two sentences, implemented in a few lines, and tested with one clear diagnostic besides BPB.

## Core Hypothesis

Pretraining efficiency is mostly finite gradient-bandwidth allocation. Standard next-token CE spends equal gradient on targets that are easy, unscored, too local, or weakly useful; good algorithms reallocate gradient toward byte-bearing, uncertain, future-constraining targets while keeping MoE routing stable.

For this repo, the first practical implication is not MTP. It is objective alignment: `evaluate_bpb` sums per-token CE over target bytes and ignores zero-byte special tokens, while `GPT.forward` currently trains mean CE over all target tokens. The cleanest first experiment is therefore byte-weighted CE with zero-byte targets masked out.

## Research Loop Record

I ran 20 search/reflection iterations. In each iteration I scanned 10 candidate papers or idea families, selected 2 one-sentence ideas, reflected on why those 2 were better than the other 8, and updated the next keyword direction. The 40 selected ideas are collected in the next section.

### iteration 1: seed MTP/TOP search

Keywords:
`MTP`, `DeepSeek MTP`, `token order prediction`, `auxiliary loss`, `future token prediction`.

Scan-10:
DeepSeek-V3 uses MTP as an auxiliary objective on a large MoE and reports stronger performance; Gloeckle et al. train multiple future-token heads on a shared trunk and find better sample efficiency; TOP replaces exact future-token prediction with a ranking loss over upcoming-token proximity; MTP curriculum finds small models need the auxiliary horizon phased in or out; Future Summary Prediction argues MTP is too short-horizon and predicts compact future summaries instead; How Transformers Learn to Plan via MTP attributes MTP gains to cleaner reverse-planning gradients; MTP-D self-distills MTP heads to improve future-head acceptance; FastMTP is mostly an inference acceleration follow-up; Medusa shows the serving value of future heads but not base pretraining; ProphetNet is an older n-gram prediction ancestor.

Selected:
1. MTP is a future-pressure regularizer: extra future heads force hidden states to encode constraints beyond the immediate next token.
2. TOP is a cheaper and possibly cleaner version of MTP: rank the true future tokens by closeness instead of asking the model to exactly classify every future token.

Why these two:
They preserve the simple "future supervision" insight without requiring data labels or a new optimizer. TOP wins the first follow-up slot because it can be implemented by gathering logits for future targets and adding a small pairwise ranking loss.

Next keywords:
`MTP curriculum`, `future summaries`, `ranking auxiliary objective`.

### iteration 2: MTP follow-ups

Keywords:
`MTP curriculum`, `DeepSeek MTP follow-up`, `future summary prediction`, `multi-token prediction small models`.

Scan-10:
MTP curriculum shows forward and reverse schedules behave differently; FSP predicts bag or learned summaries of future sequence content; TOP reports DS-MTP underperforming its rank loss on standard tasks; DeepSeek MTP uses additional modules rather than only independent linear heads; Gloeckle MTP finds larger models benefit more; How Transformers Learn to Plan links MTP to planning circuits; MTP-D improves head training but focuses on inference; training-free MTP probes latent future ability without pretraining; FastMTP compresses vocab for drafting; Future Token Prediction tries semantic state vectors.

Selected:
3. Ramp auxiliary horizon after NTP warmup, because small models can be hurt by hard future losses before the local predictor is competent.
4. Predict a compact future summary, not just the next few exact tokens, when the goal is long-horizon representation pressure.

Why these two:
They explain when naive MTP fails and how to make the same idea gentler. The schedule is nearly free; the summary loss is a second-stage test after TOP.

Next keywords:
`gradient decoupling`, `planning MTP`, `future hidden representation`.

### iteration 3: mechanism behind future objectives

Keywords:
`MTP planning`, `gradient decoupling`, `star graph TOP`, `future representation prediction`.

Scan-10:
How Transformers Learn to Plan claims MTP decouples gradients and induces reverse path tracing; TOP's star-graph result suggests ranking is enough to shape pathfinding; FSP says exact future CE is too local; CoCoMix predicts continuous concepts and interleaves them with tokens; LLM-JEPA predicts embedding-space views instead of reconstructing tokens; latent CoT papers spend extra hidden computation before token output; pause-token papers give the model extra positions to compute; MTP-D uses self-distillation to align heads; Gloeckle MTP reports induction-head benefits; DeepSeek MTP ties future prediction to speculative decoding utility.

Selected:
5. A minimal TOP loss can use current main logits: gather logits for targets at `t+1..t+k` and enforce `logit(t+1) > logit(t+2) > ...`.
6. A cheap JEPA-style future loss can predict detached future hidden states, avoiding extra vocabulary softmaxes while still injecting future information.

Why these two:
Both are lower compute than full MTP and directly test whether future structure, rather than extra parameters, is the useful ingredient.

Next keywords:
`selective language modeling`, `token loss selection`, `hard token CE`.

### iteration 4: selective token modeling

Keywords:
`Rho-1`, `selective language modeling`, `ESLM`, `token-level loss selection`, `CVaR`.

Scan-10:
Rho-1 trains only useful tokens scored by a reference model; ESLM selects high-risk tokens online by loss or entropy; Dynamic Loss-Based Sample Reweighting adjusts instance weights from current loss; Irreducible Curriculum prioritizes learnable examples using a proxy trajectory; LFR revisits forgotten hard blocks; PreSelect selects data whose compression predicts downstream skill; BLISS estimates long-term influence with a small proxy; LongAttn scores long-dependency examples from attention; Min-K style detection shows low-probability tail tokens carry membership signal; basic focal loss is a simpler ancestor for hard-token emphasis.

Selected:
7. Online selective CE can mask or upweight the high-loss half of tokens using detached per-token CE, keeping every token in context but not every token in the loss.
8. Hard-token selection should exclude extreme outliers, because the highest-loss tokens may be noise, truncation artifacts, or rare strings rather than useful curriculum.

Why these two:
They are implementable without external reference models. Rho-1 is conceptually strong, but its reference scorer is too heavy for the first local reproduction.

Next keywords:
`dynamic reweighting`, `row loss`, `learn focus review`.

### iteration 5: online reweighting instead of offline filtering

Keywords:
`dynamic loss reweighting`, `learn focus review`, `batch selection`, `sample weights`.

Scan-10:
Dynamic loss reweighting finds deprioritizing redundant low-loss samples often works best; ESLM frames high-loss token selection as CVaR; LFR stores hard blocks and reviews them later; Rho-1 uses reference excess loss rather than raw loss; BLISS learns an influence scorer from scratch; Irreducible Curriculum simulates main loss with a proxy model; DeepSpeed Data Efficiency routes samples by difficulty; PreSelect uses a cheap scorer rather than online losses; curriculum learning orders easy-to-hard; holdout-loss selection uses validation influence.

Selected:
9. Row-level loss reweighting is the simplest sample algorithm: compute per-row CE and reduce the weight of the easiest rows in the current batch.
10. A small hard-example replay buffer is the minimal LFR reproduction: occasionally replay rows with high recent loss if throughput and implementation complexity remain acceptable.

Why these two:
They avoid changing the dataset or running an offline scorer. Token-level selection should be tested first; row-level replay is next if token selection improves BPB without router collapse.

Next keywords:
`curriculum LLM pretraining`, `data ordering`, `sequence length curriculum`.

### iteration 6: curriculum and data order

Keywords:
`curriculum learning LLM pretraining`, `data ordering`, `dataset decomposition`, `sequence length curriculum`.

Scan-10:
Beyond Random Sampling finds compression ratio, lexical diversity, and readability curricula accelerate early and mid training; Dataset Decomposition trains variable sequence lengths and avoids concat-and-chunk waste; In-context Pretraining orders related documents to make cross-document context useful; Irreducible Curriculum emphasizes learnability; LFR revisits forgotten blocks; How LR Decay Wastes Your Best Data warns that high-value examples should appear while LR is still high; PreSelect and BLISS select before ordering; LongAttn selects long-dependency examples; DeepSpeed Data Efficiency samples/routes data by quality; classic self-paced learning starts easy then broadens.

Selected:
11. Use a short warmup curriculum for auxiliary losses and maybe data difficulty: early training should learn the local distribution before hard future losses dominate.
12. A short-to-long sequence-length schedule could improve fixed-wall training by buying more early updates, then returning to the fixed 2048-token validation regime.

Why these two:
Both fit the existing 5-minute fixed-wall setting. Full document reordering is attractive but touches the data pipeline more deeply than the first algorithm pass should.

Next keywords:
`BPB objective alignment`, `number token loss`, `token byte loss`.

### iteration 7: metric and token semantics

Keywords:
`bits per byte`, `token byte loss`, `number token loss`, `regression-like token loss`.

Scan-10:
Number Token Loss says CE treats numbers as nominal classes and misses ordinal closeness; BPB evaluation weights by byte length, not token count; Rho-1 says not every token deserves equal loss; ESLM gives an online masking framework; Dynamic Reweighting supports loss-dependent weights; Output Embedding Centering stabilizes large output logits; FIM changes target structure without architecture; CoCoMix adds continuous semantic targets; Source-aware training adds document-id targets; Conditional preference pretraining conditions on reward labels.

Selected:
13. Train CE should be byte-weighted to match BPB: multiply per-token CE by target byte length and divide by total target bytes.
14. Zero-byte special-token targets should receive zero CE weight, because validation explicitly excludes them and training them wastes capacity.

Why these two:
They are not just inspired by papers; they fall directly out of this repo's evaluator. They are the highest-priority queue items.

Next keywords:
`output stability`, `z-loss`, `embedding centering`.

### iteration 8: output geometry and stability

Keywords:
`output embedding centering`, `z-loss`, `logit divergence`, `stable LLM pretraining`.

Scan-10:
Output Embedding Centering argues anisotropic output embeddings cause logit divergence and proposes deterministic centering or mu-loss; z-loss suppresses logit magnitude but treats a symptom; logit soft-capping caps logits directly; PLaMo and OLMo-style recipes use z-loss for stability; DeepMind small-proxy instability work motivates studying stability cheaply; DeepSeek uses router z-loss but not necessarily output centering; OLMoE uses router aux losses; CCE implementations expose z-loss hooks; label smoothing can reduce overconfidence but changes calibration; confidence penalties are older entropy regularizers.

Selected:
15. Deterministically center the output embedding rows before logits as a no-hyperparameter stability test.
16. If centering helps or allows higher LR, compare it against a tiny mu-loss rather than adding a sensitive logit z-loss.

Why these two:
They are simple and target training stability, not benchmark-specific hacks. Centering should be tried before adding another coefficient.

Next keywords:
`embedding-space objective`, `JEPA language`, `continuous concepts`.

### iteration 9: embedding-space objectives

Keywords:
`LLM-JEPA`, `continuous concepts`, `future hidden loss`, `embedding prediction`.

Scan-10:
LLM-JEPA trains predictive embedding views and reports gains over reconstruction objectives; CoCoMix predicts sparse-autoencoder concepts and interleaves them into hidden states; FSP learned summaries use a reverse LM embedding target; pause-token work adds latent compute via extra tokens; adaptive latent CoT recurrently reasons before emission; MTP predicts future token IDs; TOP ranks future IDs; JEPA avoids exact token reconstruction; semantic fusion reconstructs interpretable features; METRO/ELECTRA-style replaced-token objectives use model-generated signals.

Selected:
17. Future-hidden prediction is the minimal JEPA reproduction: predict `stop_grad(norm(h_{t+k}))` from `h_t` with a small cosine or MSE loss.
18. Future-summary prediction can be approximated by predicting a bag of upcoming token IDs with sampled negatives instead of a full-vocab BCE.

Why these two:
They test "future representation" without additional transformer blocks. CoCoMix is intriguing but needs a pretrained SAE, so it is not first-pass simple.

Next keywords:
`pause tokens`, `adaptive compute`, `latent CoT`.

### iteration 10: pause tokens and adaptive compute

Keywords:
`pause tokens`, `dynamic inserting tokens`, `adaptive latent CoT`, `adaptive computation`.

Scan-10:
Pause Tokens show pretrained models can benefit from delayed output positions; DIT inserts pause tokens at low-confidence positions; Adaptive Latent CoT allocates variable latent trajectories per token; AdaPonderLM adapts recurrent depth per token; Thoughtbubbles explores unsupervised parallel latent thinking; SPOT uses span-level pause-of-thought; COCONUT-like latent reasoning feeds hidden states as thoughts; early-exit work adapts inference compute; ACT is the older halting framework; pause-tuning improves long-context attention recalibration.

Selected:
19. Dynamic pause insertion is a data-transform version of selective compute: insert reserved pause tokens before low-confidence targets and ignore pause-token loss.
20. Token-wise adaptive latent compute is conceptually powerful but too invasive for the first queue; approximate it first with loss weighting or future-hidden auxiliary losses.

Why these two:
They expose a different mechanism, "hard tokens need more computation," but the clean local proxy is not actual recurrence yet.

Next keywords:
`FIM`, `denoising`, `bidirectional context`.

### iteration 11: FIM and denoising

Keywords:
`fill-in-the-middle pretraining`, `AST-FIM`, `UL2`, `prefix LM`.

Scan-10:
FIM rearranges prefix-suffix-middle targets and is used heavily for code; AST-FIM masks syntactic code units instead of random spans; SAFIM shows FIM pretraining can improve left-to-right code inference; Horizon-Length Prediction teaches FIM models to plan how much to generate; UL2 mixes denoising modes; prefix-LM gives bidirectional prefix context with causal target; any-order generation relaxes strict left-to-right order; diffusion language models exploit denoising; source-aware infilling teaches attribution; classic BART/T5 denoising is encoder-decoder rather than decoder-only.

Selected:
21. A small-rate FIM transform using reserved tokens is a simple data-side objective that may help code-like spans without architecture changes.
22. Horizon-length prediction for infilling suggests an auxiliary "how many tokens until boundary" target, but this is lower priority for general BPB.

Why these two:
FIM is simple and proven in code settings, but this benchmark is general BPB, so it should wait until BPB-aligned and TOP-style losses are tested.

Next keywords:
`document ordering`, `in-context pretraining`, `long dependency data`.

### iteration 12: context structure and document order

Keywords:
`in-context pretraining`, `related documents`, `LongAttn`, `long context data selection`.

Scan-10:
In-context Pretraining orders related documents to make previous documents predictive; LongAttn scores examples by token-level long-range dependency strength; Dataset Decomposition avoids cross-document attention waste; Beyond Random Sampling orders by text difficulty; Source-aware training adds document identifiers; PreSelect scores predictive data; BLISS estimates influence; LFR revisits hard blocks; Long-context curricula choose sequence lengths; code data mixtures improve compositional tasks.

Selected:
23. Related-document ordering is powerful but out of scope for first-pass `train.py` edits because this repo streams fixed shards without document metadata indexes.
24. Long-dependency selection is a later data-pipeline project; for now, TOP/FSP are cleaner ways to ask hidden states to represent longer futures.

Why these two:
They are important research directions but not minimal reproductions here. They helped update the search toward loss-side long-horizon proxies.

Next keywords:
`conditioning pretraining`, `source tags`, `human preferences`.

### iteration 13: conditioning and metadata

Keywords:
`pretraining with human preferences`, `conditional training`, `source-aware training`, `document ids`.

Scan-10:
Pretraining with Human Preferences finds conditional training on reward labels is Pareto-strong for alignment; Source-aware Training adds document IDs so models can later cite pretraining sources; Ctrl-style control codes show simple conditioning can steer generation; source reliability tags may teach trust; code/natural-language mixture tags can protect domain specialization; multilingual tags help low-resource balance; timestamp tags can model temporal drift; document-quality scores can condition generation; synthetic data tags can prevent contamination; instruction tags during pretraining can reduce post-training burden.

Selected:
25. If labels exist, condition rather than filter: prepend a compact quality/domain tag so the model can learn multiple distributions without pretending they are one.
26. Source/document tags are useful for attribution, but they are not a BPB-first experiment unless the dataset already exposes stable source IDs.

Why these two:
Conditioning is simple and principled, but this repo's current fixed data interface does not expose the needed labels cleanly.

Next keywords:
`distillation`, `soft targets`, `MTP-D`.

### iteration 14: distillation and soft targets

Keywords:
`MTP-D`, `self-distillation MTP`, `knowledge distillation pretraining`, `soft labels`.

Scan-10:
MTP-D self-distills future heads to preserve main-head performance; Multi-token Prediction via Self-Distillation converts pretrained AR models to MTP; METRO uses model-generated denoising signals; ELECTRA trains a discriminator on generated replacements; knowledge distillation can smooth targets; online self-distillation uses current or EMA teacher logits; DPO/RLHF are post-training preference distillation; speculative decoding distills draft models; label smoothing is a fixed soft target; Min-K methods show tail probabilities carry useful signal.

Selected:
27. If MTP heads are added, distill them from the main head or an EMA teacher so the auxiliary heads do not fight the next-token head.
28. A tiny online soft-target term can be tested only after CE-aligned losses, because self-distillation may preserve current mistakes in a 5-minute run.

Why these two:
They are plausible repairs for full MTP, not first interventions. Distillation is more useful after a future-head experiment shows throughput or BPB promise.

Next keywords:
`MoE router algorithm`, `auxiliary-loss-free balancing`, `loss interactions`.

### iteration 15: MoE-specific training algorithms

Keywords:
`auxiliary-loss-free load balancing`, `MoE router loss`, `expert bias`, `Sinkhorn routing`.

Scan-10:
DeepSeek-V3 uses auxiliary-loss-free load balancing through expert bias updates; OLMoE uses router z-loss and load-balance coefficients; selective Sinkhorn routing avoids some auxiliary loss/objective conflict; expert-choice routing allocates capacity from experts to tokens; load-balance losses can conflict with language CE; sigmoid affinity routing removes softmax competition; router z-loss controls logit magnitude; loss-free bias controllers can drift if unclamped; top-k routing makes hard-token weighting affect expert load; conditional-depth routing papers warn auxiliary utility labels can be off-policy.

Selected:
29. Keep router losses out of the main algorithm search unless diagnostics break; this code already has sigmoid affinity plus expert bias and small load-balance pressure.
30. Any selective or future auxiliary loss must be judged with router diagnostics because hard-token gradients may increase expert specialization or collapse.

Why these two:
The architecture phase already harvested the main DeepSeek routing idea. The new risk is interaction: algorithm losses can indirectly destabilize routing.

Next keywords:
`tokenization`, `byte-level`, `BPE objective`.

### iteration 16: tokenization and byte-level training

Keywords:
`byte-level simulation`, `subword tokenization`, `tokenization training efficiency`, `BPB`.

Scan-10:
Byte-level simulation decouples subword tokenization benefits in controlled training; Number Token Loss repairs CE's nominal-class issue; BPB scoring makes byte length a first-class weight; BPE token frequency skews target distribution; long tokens carry more bytes per CE term; special tokens carry zero bytes; digit tokens have ordinal structure; FIM uses reserved tokens as task delimiters; source/preference conditioning also consumes reserved tokens; tokenizer changes are out of scope here.

Selected:
31. Byte-weighted CE is also a tokenizer correction: it prevents short BPE tokens from dominating a metric measured per byte.
32. Number-aware loss is a narrow but clean semantic correction: if a target and alternatives are numeric tokens, nearby numeric values should be less wrong than distant ones.

Why these two:
The first is universal for BPB. The second is elegant but likely only moves math/code evaluations, so it is lower priority for the current validation metric.

Next keywords:
`influence data selection`, `validation-aligned scoring`, `BLISS`.

### iteration 17: influence and validation-aligned data selection

Keywords:
`BLISS`, `PreSelect`, `holdout loss`, `data influence`, `validation aligned data selection`.

Scan-10:
BLISS learns a score model for long-term influence from scratch; PreSelect chooses data whose compression predicts downstream skills; holdout-loss selection uses validation alignment; TSDS selects task-specific data; Irreducible Curriculum approximates learnability with a proxy; Rho uses reference excess loss; ESLM uses online loss/entropy; DCLM and FineWeb-Edu are corpus filters; LongAttn selects long-dependency text; LFR reviews forgotten examples.

Selected:
33. Validation-aligned online weighting is the ideal but too expensive form: choose tokens whose gradients reduce held-out BPB, not merely tokens with high loss.
34. A cheap proxy is "excess loss over a unigram or byte baseline," which may separate informative hard tokens from inherently high-entropy junk.

Why these two:
They refine selective CE: raw high loss is not enough. Excess loss is a possible second-generation selector after simple byte-weighted and percentile CE tests.

Next keywords:
`planning`, `certainty`, `loss schedule`.

### iteration 18: planning, certainty, and schedule

Keywords:
`planning via MTP`, `latent CoT limits`, `decisional certainty`, `curriculum necessary`.

Scan-10:
How Transformers Learn to Plan says MTP gradients teach reverse planning; TOP's graph result suggests ranking future proximity is a planning signal; latent CoT limits argue certainty controls exploration versus execution; DIT inserts pauses where confidence is low; ESLM selects high-risk tokens; FSP summarizes long-term future; MTP curriculum shows direct hard objectives can fail; CoCoMix concept channels may externalize latent state; LLM-JEPA predicts embeddings rather than tokens; output centering keeps logits from unstable certainty.

Selected:
35. Use confidence/loss only as a detached controller, never as a differentiable target, so the model cannot game the selector directly.
36. Schedule auxiliary objectives by certainty: start with local BPB-aligned CE, then add TOP/future losses after the model's CE has fallen.

Why these two:
They make the queue less brittle. Good algorithms are not just new losses; they are loss timing plus detached control.

Next keywords:
`compare 40 ideas`, `minimal code priority`.

### iteration 19: implementation filter

Keywords:
`few line implementation`, `no new hyperparameter`, `fixed optimizer`, `MoE diagnostics`.

Scan-10:
Byte-weight CE is a few lines and no new coefficient; zero-byte masking comes for free with byte weights; selective CE needs one selection fraction or band; TOP with main logits needs one coefficient and no new params; future-hidden prediction needs one linear head and coefficient; OEC can be deterministic; MTP heads add params and softmax compute; FSP sampled BCE needs negative sampling; pause insertion changes data length; sequence-length curriculum changes dataloader shape.

Selected:
37. Prioritize no-parameter/no-extra-softmax losses before adding auxiliary heads: byte CE, special mask, online selective CE, main-logit TOP, deterministic OEC.
38. Treat added vocabulary softmaxes as expensive in a fixed-wall benchmark; require a cheap proxy win before full MTP or FSP.

Why these two:
They prevent the algorithm phase from recreating architecture sprawl. The fastest useful experiment is the one that preserves throughput.

Next keywords:
`final hypothesis`, `practical interventions`.

### iteration 20: final distillation

Keywords:
`gradient bandwidth`, `metric alignment`, `future constraints`, `MoE stability`.

Scan-10:
MTP, TOP, FSP, Rho-1, ESLM, Dynamic Reweighting, LFR, Dataset Decomposition, OEC, and NTL all point away from uniform token CE; some reallocate horizon, some reallocate token weight, some reallocate data order, and some stabilize output geometry. The common failure mode is adding an auxiliary task whose gradient is either too hard, too late, too expensive, or misaligned with the actual score.

Selected:
39. The unified theory is gradient-bandwidth allocation under finite compute: weight the targets that matter, add future constraints only when cheap, and keep routing stable.
40. The practical algorithm queue should start with BPB-aligned CE, then selective CE, then TOP, then future representation losses, then heavier MTP/FSP/data curricula.

Why these two:
They turn 40 paper-level ideas into an experiment order. They also explain why MTP is interesting but not necessarily the first run in this specific codebase.

Next keywords:
none; search stopped here before synthesis.

## The 40 One-Sentence Ideas

1. MTP is a future-pressure regularizer: extra future heads force hidden states to encode constraints beyond the immediate next token.
2. TOP is a cheaper and possibly cleaner MTP: rank true future tokens by closeness instead of exactly classifying every future token.
3. Auxiliary future objectives should ramp in after NTP warmup because small models can be hurt by hard horizons at step 1.
4. Future Summary Prediction suggests predicting compact future content rather than only the next few exact tokens.
5. A minimal TOP loss can gather current main logits for targets at `t+1..t+k` and enforce a decreasing order.
6. A cheap JEPA-style future loss can predict detached future hidden states instead of running extra vocab softmaxes.
7. Online selective CE can upweight high-loss tokens while keeping every token in the context stream.
8. Hard-token selection should avoid the extreme loss tail, where noise and truncation artifacts live.
9. Row-level loss reweighting is the simplest online sample-weighting algorithm.
10. A small replay buffer for high-loss rows is the minimal Learn-Focus-Review reproduction.
11. Curriculum should be used as warmup, not dogma: learn local NTP first, then add harder objectives.
12. Short-to-long sequence length training may buy more early updates before returning to fixed 2048-token validation.
13. Byte-weight CE aligns training with BPB by weighting each target CE by target byte length.
14. Zero-byte special-token targets should receive zero CE weight because validation excludes them.
15. Output embedding centering is a no-hyperparameter stability test for output logit geometry.
16. If centering helps, compare against a small mu-loss rather than a sensitive z-loss.
17. Future-hidden prediction is the minimal language-JEPA reproduction.
18. Future-summary prediction can be approximated with sampled positives and negatives over upcoming token bags.
19. Dynamic pause insertion is a data-transform version of selective compute.
20. Adaptive latent compute is promising but too invasive; approximate it first with loss weighting or future-hidden losses.
21. Small-rate FIM is a simple data-side objective using reserved tokens.
22. Horizon-length prediction is a lower-priority infilling planner.
23. Related-document ordering is powerful but not a first-pass `train.py` edit.
24. Long-dependency data selection should wait behind loss-side long-horizon proxies.
25. If quality/domain labels exist, condition on them rather than filtering them away.
26. Source/document tags are useful for attribution but not BPB-first without exposed source IDs.
27. If MTP heads are added, self-distill or EMA-distill them so they do not fight the main head.
28. Soft-target self-distillation should wait because it may preserve early mistakes in a 5-minute run.
29. Do not reopen router algorithms first; the current backbone already has DeepSeek-style expert bias plus small load pressure.
30. Judge every selective/future loss with router diagnostics because hard-token gradients can destabilize MoE load.
31. Byte-weighting is also a tokenizer correction: it stops short BPE tokens from dominating a per-byte metric.
32. Number-aware loss is elegant but probably lower priority for this general BPB benchmark.
33. Ideal data weighting is validation-influence weighting, but it is too expensive for the first pass.
34. Excess loss over a unigram or byte baseline may separate informative hard tokens from high-entropy junk.
35. Confidence/loss should act as a detached controller so the model cannot game the selector.
36. Auxiliary objectives should be scheduled by certainty: local CE first, TOP/future losses later.
37. Prioritize no-parameter/no-extra-softmax losses before adding auxiliary heads.
38. Full MTP/FSP should require a cheap proxy win because extra softmaxes are costly under fixed wall time.
39. The common mechanism is finite gradient-bandwidth allocation.
40. The practical queue is BPB-aligned CE, selective CE, TOP, future representation losses, then heavier MTP/FSP/data curricula.

## Five Timed Synthesis Passes

Each pass was timed for 4 minutes after internet search stopped.

### pass 1: loss allocation

The first hypothesis was that most gains come from not treating all tokens as equally useful gradient sources. Rho-1, ESLM, dynamic reweighting, and LFR all disagree on scoring details but agree on the mechanism: easy/redundant tokens should not consume the same loss budget as informative tokens.

Practical consequence:
try byte-weighted CE first, then a conservative selective CE with a floor or hard-but-not-outlier band.

### pass 2: metric alignment

The strongest repo-specific insight is that validation is BPB, not token CE. Since validation excludes zero-byte specials and divides by target bytes, training should probably weight CE by byte length before trying more exotic objectives.

Practical consequence:
make byte-weighted CE and special-token masking the phase-opening experiment.

### pass 3: future constraints

MTP/TOP/FSP seem less about "more labels" and more about making hidden states represent plausible futures. Exact far-token CE is expensive and sometimes too hard; ranking or representation prediction is a cleaner first probe.

Practical consequence:
try main-logit TOP before adding MTP heads; try future-hidden prediction before full future-summary BCE.

### pass 4: curriculum and timing

Many auxiliary objectives are useful only after the base predictor has learned enough local structure. A bad schedule can make a good objective look bad, especially in a small model or short fixed-wall run.

Practical consequence:
any auxiliary loss after byte CE should have a simple warmup ramp and a CE-only diagnostic log.

### pass 5: MoE interaction

Algorithmic losses are not router-neutral. Hard-token weighting and future objectives can concentrate gradients on rare/content-heavy tokens, which may improve specialization or trigger expert collapse.

Practical consequence:
every run must record `expert_load_cv`, `max_expert_load`, `router_entropy`, router z-loss, tokens/sec, and CE-only loss; discard BPB gains that come with unstable routing unless the next run has a clear repair.

## High-Priority Queue

### P0: byte-weighted CE and zero-byte target mask

Hypothesis:
The current training objective optimizes mean token CE, while validation optimizes BPB and ignores zero-byte specials. Weighting token CE by `token_bytes[target]` should align the training gradient with the score and stop wasting updates on BOS targets.

Minimal implementation:
load `get_token_bytes(device)` or an equivalent buffer in `train.py`; when `reduction="mean"`, compute `ce_flat = F.cross_entropy(..., reduction="none")`, `w = token_bytes[targets.view(-1)].float()`, and `ce_loss = (ce_flat * w).sum() / w.sum().clamp_min(1)`. Keep `reduction="none"` behavior unchanged for evaluation unless deliberately reproducing evaluator logic.

Expected result:
Lower `val_bpb`; `train_ce_loss` may become less comparable, so log a CE-only unweighted diagnostic if needed. Router load may shift slightly toward content-heavy tokens but should stay healthy.

### P1: conservative online selective CE

Hypothesis:
After metric alignment, many low-loss tokens are redundant for a 5-minute run. Upweighting the detached mid-high loss band should improve sample efficiency without throwing away context.

Minimal implementation:
compute per-token CE, detach it for selection, select tokens between roughly median and 95th percentile within the batch, and blend with a floor weight such as `0.25 + 0.75 * mask`. Normalize by sum of weights.

Expected result:
Lower BPB if easy tokens were wasting gradient; possible router specialization. Discard or soften if `expert_load_cv` or max expert load jumps.

### P2: main-logit TOP

Hypothesis:
The current hidden state should know not only the next token but the relative order of the next few true tokens. A pairwise ranking loss over gathered future-target logits injects this signal with almost no extra compute.

Minimal implementation:
for offsets `1..K`, gather `logits[:, :-K, :]` at targets shifted by each offset; add `-logsigmoid(score_i - score_j)` for `i < j`, with a small ramped coefficient.

Expected result:
Similar throughput, slightly better BPB, and perhaps lower loss on long-dependency validation examples. If BPB is flat but CE diagnostics improve late, test a small coefficient or auxiliary unembedding.

### P3: future-hidden JEPA-lite

Hypothesis:
Exact future token CE may be too noisy, but predicting a detached future hidden state gives a dense, cheap future representation target. This tests the FSP/JEPA mechanism without another full-vocab head.

Minimal implementation:
add one small projection from `h_t` to `h_{t+k}` dimension and minimize cosine distance to `stop_grad(norm(h_{t+k}))`, ramped after warmup.

Expected result:
Small BPB gain without large throughput loss. Watch for representation collapse or reduced router entropy.

### P4: output embedding centering

Hypothesis:
Output embedding anisotropy can waste training stability budget and force conservative LR/logit controls. Deterministically centering output embeddings before logits may improve stability or allow later LR tests.

Minimal implementation:
compute logits with `weight = lm_head.weight - lm_head.weight.mean(dim=0, keepdim=True)`; do not add a coefficient in the first test.

Expected result:
Similar or slightly better BPB and cleaner logit behavior. If it helps, revisit LR or compare against `mu-loss`.

### P5: MTP aux head with forward curriculum

Hypothesis:
If TOP helps, exact future CE may add more useful future pressure when isolated in separate heads and ramped after warmup.

Minimal implementation:
add one or two untied linear MTP heads for `t+2` and `t+3`; compute CE on shifted targets with a coefficient ramp. Keep the main head as the only validation head.

Expected result:
Potential BPB gain but lower throughput; keep only if fixed-wall BPB beats TOP enough to justify extra softmax cost.

### P6: future bag summary

Hypothesis:
Longer-horizon content prediction is better represented as "what appears soon" than "exact token k." A sampled bag-of-future loss can pressure planning without full sequence prediction.

Minimal implementation:
for each position, sample positives from the next `W` targets and negatives from the batch or vocab; apply a small logistic loss to an auxiliary projection.

Expected result:
Lower BPB only if long-horizon representation is limiting. Defer until TOP or JEPA-lite shows future pressure is useful.

### P7: row-level dynamic reweighting

Hypothesis:
Some packed rows are already easy or redundant in the current stage. Per-row loss weighting is a lower-variance version of token selection.

Minimal implementation:
average per-token CE by row, detach row losses, and reduce easy-row weights with a floor.

Expected result:
Possibly cleaner than token selection, but weaker. Try only after P1.

### P8: short-to-long sequence-length warmup

Hypothesis:
Early training does not need full 2048-token attention on every step. Shorter sequences early may improve update count and local modeling before returning to validation length.

Minimal implementation:
start with shorter `T` for a fixed warmup fraction and then switch to `MAX_SEQ_LEN`; keep total batch tokens comparable if possible.

Expected result:
Higher early steps/sec and similar or better BPB. This is a dataloader/training-loop change, so it is lower priority than pure loss edits.

### P9: number-token loss

Hypothesis:
CE treats all wrong numeric tokens as equally wrong. Adding a small ordinal distance loss on numeric-token targets may improve arithmetic/code-like data with little runtime cost.

Minimal implementation:
build a token-to-number table for tokens that decode cleanly as numbers; add an ordinal or Wasserstein-like auxiliary loss only when the target is numeric.

Expected result:
Probably little BPB movement on general validation, but useful if later experiments add math/code diagnostics.

### P10: WSD schedule replacing cosine

Hypothesis:
Cosine LR decays continuously throughout training; WSD's long stable phase keeps LR at peak until a final linear decay. Documented to match-or-beat cosine at fixed budget across MiniCPM/DeepSeek/Qwen reports while making mid-training checkpoints representative and enabling clean continued training. Treat as infrastructure: apply to all subsequent runs once verified neutral on this codebase.

Minimal implementation:
replace cosine schedule with ~3% warmup (or 2000 steps, whichever is larger), ~82% stable at peak LR, ~15% linear decay to 10% of peak. Single function change in the scheduler. Zero new hyperparameters; split fractions are robust across reports.

Expected result:
similar or marginally better `val_bpb` at fixed budget; mid-training checkpoints become usable for cheap ablations; future continued-training experiments become trivial. Watch for instability at the warmup-to-stable transition if warmup is shortened too aggressively for MoE routing to settle.

### P11: Marin-style initialization (std proportional to 1/d_model)

Hypothesis:
Initialization std should scale inversely with hidden dimension rather than be set to a fixed constant. This is the init-only piece of mu-P and is sufficient at fixed scale to keep activations stable while transferring more cleanly to other widths than the OLMoE constant-0.02 default. Full mu-P (with LR scaling and activation scaling) is not worth the implementation surface for this benchmark; the init piece alone captures most of the within-scale benefit.

Minimal implementation:
replace constant init std (e.g., 0.02) with `std = c / d_model`, with `c` calibrated so initial activation magnitudes match the current baseline at this width. One-line change per module init. Embedding and output head may need separate calibration.

Expected result:
within-noise change in `val_bpb` at this scale; principled cross-scale transfer for any LR/WD tuning done here, which makes future width sweeps cheaper. Treat as infrastructure: apply once and forget.

### P12: Constant applied weight decay across LR schedule

Hypothesis:
PyTorch AdamW applies weight decay as `lr * weight_decay * theta` per step, so the *applied* WD scales with LR even when the configured `weight_decay` coefficient is fixed. During WSD's decay phase or any LR ramp-down, applied WD shrinks proportionally, which is a candidate contributor to gradient norm climbing in late training. Holding applied WD constant in absolute terms decouples regularization from schedule.

Minimal implementation:
in the optimizer step (or via a coefficient schedule), multiply the `weight_decay` coefficient by `peak_lr / current_lr` so that applied WD equals `peak_lr * weight_decay_coef` independent of current LR. Compatible with current Muon-based setup as long as the weight-decay path uses the same correction.

Expected result:
small `val_bpb` change (likely <1%) but cleaner gradient norm trajectory through the WSD decay phase. Diagnostic value: confirms or refutes the LR-coupled-WD contribution to gradient norm climbing. Log gradient norm by phase to make the comparison interpretable.

### P13: Embedding noise ablation (NEFTune-for-pretraining)

Hypothesis:
Three competing explanations for why NEFTune helps SFT, with sharply different predictions for pretraining: (H1) noise at the discrete-to-continuous boundary forces the model to use embedding direction rather than exact identity, predicting only embedding-output noise helps; (H2) noise acts as a generic Lipschitz-smoothness regularizer, predicting noise at any internal representation helps similarly; (H3) noise prevents overfitting to repeated inputs, predicting no benefit in pretraining regardless of noise location because each example is seen approximately once. Prior is highest on H3.

Minimal implementation:
four short runs at the same fixed budget: (1) baseline, no noise; (2) additive Gaussian noise at embedding output, magnitude `alpha / sqrt(d)` with `alpha = 5`; (3) equivalent-magnitude noise on the value projection outputs in attention; (4) equivalent-magnitude noise on the router pre-softmax inputs. All other settings identical. Run as a clean four-cell ablation, not a six-month parameter sweep.

Expected result:
H3 most likely → no run beats baseline; delete this entire idea family from the queue. H2 → all three noise variants help similarly; adopt small noise as a generic regularizer at multiple sites. H1 → only run 2 helps; embedding-output noise becomes a tool specific to the discrete-token boundary, not generalizable elsewhere. High-signal ablation regardless of outcome, and falsifies in roughly 20 minutes of total compute.

## Defer For Now

- Full external reference-model Rho-1 scoring: strong idea, too much setup for first pass.
- BLISS/PreSelect offline data selection: useful for a data-pipeline phase, not minimal `train.py`.
- Dynamic pause-token insertion: promising but changes sequence distribution and natural-token throughput.
- Adaptive latent CoT or recurrent ponder blocks: too architectural for this algorithm phase.
- Source/preference conditioning: needs labels not exposed by the current fixed dataloader.
- FIM/AST-FIM: simple for code-heavy runs, but lower priority for general BPB.

## Run Discipline

For every algorithm run, record:

- `val_bpb` as the decision metric.
- CE-only loss under the original unweighted token CE when possible.
- The actual optimized loss and each auxiliary component.
- `total_tokens_M`, `num_steps`, `mfu_percent`, and `peak_vram_mb`.
- `router_entropy`, `expert_load_cv`, `max_expert_load`, router z-loss, load-balance loss, and bias magnitudes.

Decision rule:
keep only changes that clearly improve BPB, preserve throughput enough to matter under fixed wall time, and do not destabilize routing. A tiny BPB gain from a large extra-softmax auxiliary head is not worth keeping unless it opens a clearer next experiment.

## Key Sources

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- [Pre-Training Curriculum for Multi-Token Prediction in Language Models](https://arxiv.org/abs/2505.22757)
- [Predicting the Order of Upcoming Tokens Improves Language Modeling](https://arxiv.org/abs/2508.19228)
- [Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries](https://arxiv.org/abs/2510.14751)
- [How Transformers Learn to Plan via Multi-Token Prediction](https://arxiv.org/abs/2604.11912)
- [Rho-1: Not All Tokens Are What You Need](https://arxiv.org/abs/2404.07965)
- [ESLM: Risk-Averse Selective Language Modeling for Efficient Pretraining](https://arxiv.org/abs/2505.19893)
- [Dynamic Loss-Based Sample Reweighting for Improved Large Language Model Pretraining](https://arxiv.org/abs/2502.06733)
- [Accelerating Large Language Model Pretraining via LFR Pedagogy](https://arxiv.org/abs/2409.06131)
- [Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning](https://arxiv.org/abs/2506.11300)
- [Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum](https://arxiv.org/abs/2405.13226)
- [LLM Pretraining with Continuous Concepts](https://arxiv.org/abs/2502.08524)
- [LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures](https://arxiv.org/abs/2509.14252)
- [Pretraining with Token-Level Adaptive Latent Chain-of-Thought](https://arxiv.org/abs/2602.08220)
- [Learning to Insert PAUSE Tokens for Better Reasoning](https://arxiv.org/abs/2506.03616)
- [Regress, Don't Guess: A Regression-like Loss on Number Tokens for Language Models](https://arxiv.org/abs/2411.02083)
- [Output Embedding Centering for Stable LLM Pretraining](https://arxiv.org/abs/2601.02031)
- [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255)
- [Structure-Aware Fill-in-the-Middle Pretraining for Code](https://arxiv.org/abs/2506.00204)
- [In-context Pretraining: Language Modeling Beyond Document Boundaries](https://arxiv.org/abs/2310.10638)
- [LongAttn: Selecting Long-context Training Data via Token-level Attention](https://arxiv.org/abs/2502.16860)
- [Pretraining Language Models with Human Preferences](https://arxiv.org/abs/2302.08582)
- [Source-Aware Training Enables Knowledge Attribution in Language Models](https://arxiv.org/abs/2404.01019)
