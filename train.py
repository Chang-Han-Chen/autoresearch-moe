"""
Autoresearch pretraining script: 4xH100 DDP, 100M-active top-2 MoE.

Usage:
    uv run torchrun --standalone --nproc_per_node=4 train.py

Design constraints:
- prepare.py owns MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, and evaluate_bpb. The
  evaluation contract is unchanged.
- The optimizer algorithm remains MuonAdamW + AdamW from the dense baseline.
  Experiment agents may change learning-rate values, but not the optimizer family.
- The model is intentionally a scale-down of GPT-OSS-style sparse FFN models:
  GQA attention, alternating local/full attention, top-2 token-choice MoE FFNs.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import importlib.util
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------------------------------------------------------
# Distributed setup comes before loading CUDA kernels, so each process binds to
# the intended local GPU.
# ---------------------------------------------------------------------------

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_DISTRIBUTED = WORLD_SIZE > 1
IS_MASTER = RANK == 0

if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)

device = torch.device(f"cuda:{LOCAL_RANK}")

if IS_DISTRIBUTED:
    dist.init_process_group(backend="nccl")

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only; use kernels-community on non-Hopper GPUs.
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
try:
    fa3 = get_kernel(repo).flash_attn_interface
except Exception as primary_kernel_error:
    from pathlib import Path

    fallback_repo = "kernels-community/flash-attn3"
    try:
        fa3 = get_kernel(fallback_repo, revision="main", trust_remote_code=True).flash_attn_interface
    except Exception as fallback_kernel_error:
        hub_kernel_root = Path.home() / ".cache" / "huggingface" / "hub" / "kernels--kernels-community--flash-attn3"
        build_paths = sorted(
            hub_kernel_root.glob("snapshots/*/build/torch*-cu*-x86_64-linux"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not build_paths:
            raise fallback_kernel_error from primary_kernel_error
        sys.path.insert(0, str(build_paths[0]))
        import flash_attn3

        fa3 = flash_attn3.flash_attn_interface

from prepare import (
    MAX_SEQ_LEN,
    TIME_BUDGET,
    Tokenizer,
    evaluate_bpb,
    DATA_DIR,
    VAL_FILENAME,
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def master_print(*args, **kwargs):
    if IS_MASTER:
        print(*args, **kwargs)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# ---------------------------------------------------------------------------
# Rank-sharded train dataloader. Evaluation still uses prepare.evaluate_bpb.
# ---------------------------------------------------------------------------

def _list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def _document_batches_sharded(rank, world_size, tokenizer_batch_size=128):
    """Infinite iterator over train documents, with parquet shards split by rank."""
    parquet_paths = _list_parquet_files()
    assert parquet_paths, "No parquet files found. Run `uv run prepare.py` first."

    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    train_paths = [p for p in parquet_paths if p != val_path]
    assert train_paths, "No training shards found. Run `uv run prepare.py --num-shards 8` or more."
    assert len(train_paths) >= world_size, (
        f"Need at least {world_size} train shards for {world_size} ranks; found {len(train_paths)}. "
        "Run `uv run prepare.py --num-shards 8` or more."
    )

    rank_paths = train_paths[rank::world_size]
    epoch = 1
    while True:
        for filepath in rank_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i + tokenizer_batch_size], epoch
        epoch += 1


def make_sharded_dataloader(tokenizer, B, T, rank, world_size, device, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing, copied from prepare.py but
    rank-sharded across train parquet files so DDP ranks do not consume the same
    examples. This does not modify the evaluation dataloader or metric.
    """
    row_capacity = T + 1
    batches = _document_batches_sharded(rank, world_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# GPT-MoE model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 6
    n_kv_head: int = 2
    n_embd: int = 768
    window_pattern: str = "SSSL"
    num_experts: int = 8
    top_k: int = 2
    moe_hidden_dim: int = 1792
    router_z_loss_coef: float = 1.0e-3
    load_balance_loss_coef: float = 1.0e-2
    router_sigmoid_affinity: bool = False
    router_expert_bias: bool = False
    router_bias_ema_beta: float = 0.9
    router_bias_eta: float = 1.0e-3
    router_bias_clamp: float = 0.25
    exclusive_attention: bool = False
    headwise_attention_gate: bool = False
    attention_gate_init: float = 0.95
    attention_gate_scale: float = 1.0
    value_mix_enabled: bool = False
    value_mix_learned: bool = True
    value_mix_start_layer: int = 1
    value_mix_normalized: bool = False
    value_mix_first_init: float = 0.5
    value_mix_local_init: float = 0.5
    value_mix_gamma_init: float = 1.0
    dense_early_layers: int = 0
    dense_hidden_dim: int = 3584


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_attn_gate = nn.Linear(self.n_embd, self.n_head, bias=True) if config.headwise_attention_gate else None
        self.attention_gate_scale = config.attention_gate_scale
        self.qk_gamma = nn.Parameter(torch.ones(()))
        self.exclusive_attention = config.exclusive_attention
        self.value_mix_enabled = config.value_mix_enabled and layer_idx >= config.value_mix_start_layer
        self.value_mix_learned = config.value_mix_learned
        self.value_mix_normalized = config.value_mix_normalized
        if self.value_mix_enabled:
            if self.value_mix_learned:
                self.value_mix_first = nn.Parameter(torch.empty(()))
                self.value_mix_local = nn.Parameter(torch.empty(()))
                if self.value_mix_normalized:
                    self.value_mix_gamma = nn.Parameter(torch.empty(()))
                else:
                    self.value_mix_gamma = None
            else:
                self.register_buffer("value_mix_first", torch.empty(()), persistent=True)
                self.register_buffer("value_mix_local", torch.empty(()), persistent=True)
                if self.value_mix_normalized:
                    self.register_buffer("value_mix_gamma", torch.empty(()), persistent=True)
                else:
                    self.value_mix_gamma = None
        else:
            self.value_mix_first = None
            self.value_mix_local = None
            self.value_mix_gamma = None

    def forward(self, x, first_v, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        layer_v = v
        if self.value_mix_enabled and first_v is not None:
            first_coef = self.value_mix_first
            local_coef = self.value_mix_local
            mixed_v = first_coef.to(dtype=v.dtype) * first_v.to(dtype=v.dtype) + local_coef.to(dtype=v.dtype) * v
            if self.value_mix_normalized:
                denom = torch.sqrt(first_coef.float().square() + local_coef.float().square()).clamp_min(1e-6)
                mixed_v = self.value_mix_gamma.to(dtype=v.dtype) * mixed_v / denom.to(dtype=v.dtype)
            v = mixed_v

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm for stable attention logits.
        q = q * self.qk_gamma

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        if self.exclusive_attention:
            self_v = v
            if self.n_kv_head != self.n_head:
                kv_groups = self.n_head // self.n_kv_head
                self_v = (
                    self_v[:, :, :, None, :]
                    .expand(B, T, self.n_kv_head, kv_groups, self.head_dim)
                    .reshape(B, T, self.n_head, self.head_dim)
                )
            self_v_dir = F.normalize(self_v.float(), dim=-1).to(dtype=y.dtype)
            y = y - (y * self_v_dir).sum(dim=-1, keepdim=True) * self_v_dir
        if self.c_attn_gate is not None:
            gate = self.attention_gate_scale * torch.sigmoid(self.c_attn_gate(x)).view(B, T, self.n_head, 1)
            y = y * gate.to(dtype=y.dtype)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y, layer_v


class SwiGLUExpert(nn.Module):
    def __init__(self, n_embd, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_up = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DenseSwiGLU(nn.Module):
    def __init__(self, n_embd, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.w_up = nn.Linear(n_embd, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        dense_x = x.to(dtype=self.w_gate.weight.dtype)
        out = self.w_down(F.silu(self.w_gate(dense_x)) * self.w_up(dense_x))
        return out.to(dtype=x.dtype)

    def param_count(self):
        return self.w_gate.weight.numel() + self.w_up.weight.numel() + self.w_down.weight.numel()


class TokenChoiceMoE(nn.Module):
    """Dropless top-k token-choice MoE with small router regularization."""

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.moe_hidden_dim = config.moe_hidden_dim
        self.router_z_loss_coef = config.router_z_loss_coef
        self.load_balance_loss_coef = config.load_balance_loss_coef
        self.router_sigmoid_affinity = config.router_sigmoid_affinity
        self.router_expert_bias = config.router_expert_bias
        self.router_bias_ema_beta = config.router_bias_ema_beta
        self.router_bias_eta = config.router_bias_eta
        self.router_bias_clamp = config.router_bias_clamp
        assert 1 <= self.top_k <= self.num_experts
        self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)
        if self.router_expert_bias:
            self.register_buffer("router_bias", torch.empty(config.num_experts), persistent=True)
            self.register_buffer("router_load_ema", torch.empty(config.num_experts), persistent=True)
            self.register_buffer("router_load_accum", torch.empty(config.num_experts), persistent=False)
            self.register_buffer("router_load_count", torch.empty((), dtype=torch.float32), persistent=False)
        else:
            self.router_bias = None
            self.router_load_ema = None
            self.router_load_accum = None
            self.router_load_count = None
        expert_dtype = torch.bfloat16
        self.w_gate = nn.Parameter(torch.empty(
            config.num_experts * config.n_embd, config.moe_hidden_dim, dtype=expert_dtype,
        ))
        self.w_up = nn.Parameter(torch.empty(
            config.num_experts * config.n_embd, config.moe_hidden_dim, dtype=expert_dtype,
        ))
        self.w_down = nn.Parameter(torch.empty(
            config.num_experts * config.moe_hidden_dim, config.n_embd, dtype=expert_dtype,
        ))

    @torch.no_grad()
    def reset_router_state(self):
        if not self.router_expert_bias:
            return
        self.router_bias.zero_()
        self.router_load_ema.fill_(1.0 / self.num_experts)
        self.router_load_accum.zero_()
        self.router_load_count.zero_()

    @torch.no_grad()
    def accumulate_router_load(self, load_frac):
        if not self.router_expert_bias:
            return
        self.router_load_accum.add_(load_frac.detach().to(device=self.router_load_accum.device))
        self.router_load_count.add_(1.0)

    @torch.no_grad()
    def update_router_bias(self, load_frac):
        if not self.router_expert_bias:
            return
        self.router_load_ema.mul_(self.router_bias_ema_beta).add_(load_frac, alpha=1.0 - self.router_bias_ema_beta)
        uniform = 1.0 / self.num_experts
        delta = (uniform - self.router_load_ema) / uniform
        self.router_bias.add_(self.router_bias_eta * delta)
        self.router_bias.sub_(self.router_bias.mean())
        self.router_bias.clamp_(-self.router_bias_clamp, self.router_bias_clamp)

    def forward(self, x):
        B, T, C = x.shape
        flat_x = x.reshape(B * T, C)
        N = flat_x.size(0)
        E = self.num_experts
        K = self.top_k

        # Router math is small but numerically important; keep it in fp32 even
        # under the outer bf16 autocast context.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            router_logits = self.router(flat_x.float())      # [N, E]
            router_probs = F.softmax(router_logits, dim=-1)  # [N, E]
            if self.router_sigmoid_affinity:
                affinity = torch.sigmoid(router_logits)
                score_mass = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                route_scores = affinity + self.router_bias if self.router_expert_bias else affinity
                _, top_idx = torch.topk(route_scores, K, dim=-1)
                clean_scores = affinity.gather(1, top_idx)
                top_weight = clean_scores / clean_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                prob_mean = score_mass.mean(dim=0)
            else:
                top_logits, top_idx = torch.topk(router_logits, K, dim=-1)
                top_weight = F.softmax(top_logits, dim=-1)       # [N, K]
                prob_mean = router_probs.mean(dim=0)

            # Router regularization. The hard load fraction is intentionally
            # detached; gradients flow through mean router probability, not
            # through the non-differentiable top-k indices.
            selected_one_hot = F.one_hot(top_idx, num_classes=E).float()  # [N, K, E]
            load_frac = selected_one_hot.sum(dim=(0, 1)) / float(N * K)
            load_balance_loss = E * torch.sum(load_frac.detach() * prob_mean)
            z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
            aux_loss = self.load_balance_loss_coef * load_balance_loss + self.router_z_loss_coef * z_loss

            entropy = -(router_probs * router_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            if self.router_sigmoid_affinity:
                sigmoid_mass_entropy = -(score_mass * score_mass.clamp_min(1e-9).log()).sum(dim=-1).mean()
                sigmoid_low_frac = (affinity < 0.01).float().mean()
                sigmoid_high_frac = (affinity > 0.99).float().mean()
            else:
                sigmoid_mass_entropy = router_logits.new_zeros(())
                sigmoid_low_frac = router_logits.new_zeros(())
                sigmoid_high_frac = router_logits.new_zeros(())
            load_cv = load_frac.std(unbiased=False) / load_frac.mean().clamp_min(1e-9)
            max_load = load_frac.max()
            if self.router_expert_bias:
                bias_abs = self.router_bias.detach().abs()
                router_bias_abs = bias_abs.mean()
                router_bias_max_abs = bias_abs.max()
                load_ema = self.router_load_ema.detach()
                load_ema_cv = load_ema.std(unbiased=False) / load_ema.mean().clamp_min(1e-9)
            else:
                router_bias_abs = router_logits.new_zeros(())
                router_bias_max_abs = router_logits.new_zeros(())
                load_ema_cv = router_logits.new_zeros(())

        choice_expert = top_idx.reshape(-1)
        choice_token = torch.arange(N, device=flat_x.device).repeat_interleave(K)
        sort_order = torch.argsort(choice_expert)
        sorted_expert = choice_expert.index_select(0, sort_order)
        sorted_token = choice_token.index_select(0, sort_order)
        sorted_x = flat_x.index_select(0, sorted_token).to(dtype=self.w_gate.dtype)
        sorted_weight = top_weight.reshape(-1).index_select(0, sort_order).to(dtype=sorted_x.dtype)
        expert_counts = torch.bincount(sorted_expert, minlength=E).to(torch.int32)
        expert_offsets = torch.cumsum(expert_counts, dim=0, dtype=torch.int32)

        w_gate = self.w_gate.view(E, C, self.moe_hidden_dim)
        w_up = self.w_up.view(E, C, self.moe_hidden_dim)
        w_down = self.w_down.view(E, self.moe_hidden_dim, C)
        gate = torch._grouped_mm(sorted_x, w_gate, expert_offsets)
        up = torch._grouped_mm(sorted_x, w_up, expert_offsets)
        hidden = F.silu(gate) * up
        expert_out = torch._grouped_mm(hidden, w_down, expert_offsets)
        expert_out = expert_out * sorted_weight.unsqueeze(-1)

        flat_out = torch.zeros(N, C, device=flat_x.device, dtype=expert_out.dtype)
        flat_out.index_add_(0, sorted_token, expert_out)

        stats = {
            "router_entropy": entropy.detach(),
            "expert_load_cv": load_cv.detach(),
            "max_expert_load": max_load.detach(),
            "router_z_loss": z_loss.detach(),
            "router_lb_loss": load_balance_loss.detach(),
            "router_aux_loss": aux_loss.detach(),
            "expert_load_frac": load_frac.detach(),
            "router_sigmoid_mass_entropy": sigmoid_mass_entropy.detach(),
            "router_sigmoid_low_frac": sigmoid_low_frac.detach(),
            "router_sigmoid_high_frac": sigmoid_high_frac.detach(),
            "router_bias_abs": router_bias_abs.detach(),
            "router_bias_max_abs": router_bias_max_abs.detach(),
            "router_load_ema_cv": load_ema_cv.detach(),
        }
        return flat_out.to(dtype=flat_x.dtype).view(B, T, C), aux_loss, stats

    def expert_param_count(self):
        return self.w_gate.numel() + self.w_up.numel() + self.w_down.numel()

    def active_expert_param_count(self):
        one_expert = 3 * self.n_embd * self.moe_hidden_dim
        return self.top_k * one_expert


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.moe = None
        self.ffn = None
        if layer_idx < config.dense_early_layers:
            self.ffn = DenseSwiGLU(config.n_embd, config.dense_hidden_dim)
        else:
            self.moe = TokenChoiceMoE(config)

    def forward(self, x, first_v, cos_sin, window_size):
        attn_out, layer_v = self.attn(norm(x), first_v, cos_sin, window_size)
        x = x + attn_out
        if self.moe is not None:
            ffn_out, aux_loss, stats = self.moe(norm(x))
        else:
            ffn_out = self.ffn(norm(x))
            aux_loss = x.new_zeros(())
            stats = {}
        x = x + ffn_out
        return x, aux_loss, stats, layer_v


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        head_dim = config.n_embd // config.n_head
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.last_router_stats = {}
        self.last_ce_loss = None
        self.last_total_loss = None
        self.grad_diag_enabled = False
        self.last_first_v_resid_for_diag = None

    @torch.no_grad()
    def init_weights(self):
        def init_weight(weight, d_in):
            init_std = INIT_STD_GLOBAL / math.sqrt(d_in)
            torch.nn.init.trunc_normal_(
                weight,
                mean=0.0,
                std=init_std,
                a=-3.0 * init_std,
                b=3.0 * init_std,
            )

        init_weight(self.transformer.wte.weight, self.config.n_embd)
        init_weight(self.lm_head.weight, self.config.n_embd)

        for block in self.transformer.h:
            init_weight(block.attn.c_q.weight, self.config.n_embd)
            init_weight(block.attn.c_k.weight, self.config.n_embd)
            init_weight(block.attn.c_v.weight, self.config.n_embd)
            init_weight(block.attn.c_proj.weight, self.config.n_embd)
            if block.attn.c_attn_gate is not None:
                block.attn.c_attn_gate.weight.zero_()
                gate_init = min(max(self.config.attention_gate_init, 1e-6), 1.0 - 1e-6)
                block.attn.c_attn_gate.bias.fill_(math.log(gate_init / (1.0 - gate_init)))
            block.attn.qk_gamma.fill_(1.0)
            if block.attn.value_mix_enabled:
                block.attn.value_mix_first.fill_(self.config.value_mix_first_init)
                block.attn.value_mix_local.fill_(self.config.value_mix_local_init)
                if block.attn.value_mix_gamma is not None:
                    block.attn.value_mix_gamma.fill_(self.config.value_mix_gamma_init)

            if block.moe is not None:
                init_weight(block.moe.router.weight, self.config.n_embd)
                block.moe.reset_router_state()
                init_weight(block.moe.w_gate, self.config.n_embd)
                init_weight(block.moe.w_up, self.config.n_embd)
                init_weight(block.moe.w_down, self.config.moe_hidden_dim)
            else:
                init_weight(block.ffn.w_gate.weight, self.config.n_embd)
                init_weight(block.ffn.w_up.weight, self.config.n_embd)
                init_weight(block.ffn.w_down.weight, self.config.dense_hidden_dim)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Match the dense baseline: embedding tables in bf16, compute under autocast.
        self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def expert_param_counts(self):
        total_expert = sum(
            block.moe.expert_param_count()
            for block in self.transformer.h
            if block.moe is not None
        )
        active_expert = sum(
            block.moe.active_expert_param_count()
            for block in self.transformer.h
            if block.moe is not None
        )
        return total_expert, active_expert

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters() if p.ndim >= 2)
        scalars = sum(p.numel() for p in self.transformer.h.parameters() if p.ndim < 2)
        total = wte + lm_head + transformer_matrices + scalars
        total_expert, active_expert = self.expert_param_counts()
        active = total - total_expert + active_expert
        return {
            "wte": wte,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "expert_total": total_expert,
            "expert_active": active_expert,
            "scalars": scalars,
            "total": total,
            "active": active,
        }

    def estimate_flops(self):
        """Estimated active FLOPs per token (forward + backward)."""
        counts = self.num_scaling_params()
        active_params = counts["active"]
        nparams_exclude = counts["wte"] + counts["scalars"]
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (active_params - nparams_exclude) + attn_flops

    def setup_optimizer(self, adamw_lr=0.003,
                        weight_decay=0.1, adam_betas=(0.9, 0.95), adam_eps=1e-8):
        attention_scalar_params = [block.attn.qk_gamma for block in self.transformer.h]
        attention_scalar_params.extend(
            param
            for block in self.transformer.h
            if block.attn.value_mix_enabled and block.attn.value_mix_learned
            for param in (block.attn.value_mix_first, block.attn.value_mix_local, block.attn.value_mix_gamma)
            if param is not None
        )
        attention_gate_params = [
            param
            for block in self.transformer.h
            if block.attn.c_attn_gate is not None
            for param in block.attn.c_attn_gate.parameters()
        ]
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=adamw_lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay),
            dict(kind="adamw", params=embedding_params, lr=adamw_lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay),
            dict(kind="adamw", params=attention_scalar_params, lr=adamw_lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay),
        ]
        if attention_gate_params:
            param_groups.append(dict(
                kind="adamw", params=attention_gate_params, lr=adamw_lr,
                betas=adam_betas, eps=adam_eps, weight_decay=weight_decay,
            ))

        muon_params = []

        def add_muon_group(name, params, d_in, d_out):
            params = list(params)
            if not params:
                return
            lr = adamw_lr * MUON_LR_WIDTH_FACTOR * math.sqrt(max(d_in, d_out))
            muon_params.extend(params)
            param_groups.append(dict(
                kind="muon", name=name, params=params, lr=lr,
                momentum=MUON_MOMENTUM, ns_steps=MUON_NS_STEPS, beta2=MUON_BETA2,
                weight_decay=weight_decay,
            ))

        head_dim = self.config.n_embd // self.config.n_head
        kv_dim = self.config.n_kv_head * head_dim
        blocks = list(self.transformer.h)
        moe_blocks = [block for block in blocks if block.moe is not None]
        dense_blocks = [block for block in blocks if block.ffn is not None]
        add_muon_group("attn_q", (block.attn.c_q.weight for block in blocks), self.config.n_embd, self.config.n_embd)
        add_muon_group("attn_k", (block.attn.c_k.weight for block in blocks), self.config.n_embd, kv_dim)
        add_muon_group("attn_v", (block.attn.c_v.weight for block in blocks), self.config.n_embd, kv_dim)
        add_muon_group("attn_proj", (block.attn.c_proj.weight for block in blocks), self.config.n_embd, self.config.n_embd)
        add_muon_group("router", (block.moe.router.weight for block in moe_blocks), self.config.n_embd, self.config.num_experts)
        add_muon_group("expert_gate", (block.moe.w_gate for block in moe_blocks), self.config.n_embd, self.config.moe_hidden_dim)
        add_muon_group("expert_up", (block.moe.w_up for block in moe_blocks), self.config.n_embd, self.config.moe_hidden_dim)
        add_muon_group("expert_down", (block.moe.w_down for block in moe_blocks), self.config.moe_hidden_dim, self.config.n_embd)
        add_muon_group("dense_gate", (block.ffn.w_gate.weight for block in dense_blocks), self.config.n_embd, self.config.dense_hidden_dim)
        add_muon_group("dense_up", (block.ffn.w_up.weight for block in dense_blocks), self.config.n_embd, self.config.dense_hidden_dim)
        add_muon_group("dense_down", (block.ffn.w_down.weight for block in dense_blocks), self.config.dense_hidden_dim, self.config.n_embd)

        assert len(list(self.parameters())) == (
            len(muon_params) + len(embedding_params) + len(lm_head_params) +
            len(attention_scalar_params) + len(attention_gate_params)
        )
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    @torch.no_grad()
    def accumulate_router_loads(self):
        layer_loads = self.last_router_stats.get("layer_expert_load_frac")
        if layer_loads is None:
            return
        moe_blocks = [block for block in self.transformer.h if block.moe is not None]
        for block, load_frac in zip(moe_blocks, layer_loads):
            block.moe.accumulate_router_load(load_frac)

    @torch.no_grad()
    def update_router_biases(self):
        for block in self.transformer.h:
            if block.moe is None:
                continue
            moe = block.moe
            if not moe.router_expert_bias:
                continue
            count = moe.router_load_count.clamp_min(1.0)
            load_frac = moe.router_load_accum / count
            if IS_DISTRIBUTED:
                dist.all_reduce(load_frac, op=dist.ReduceOp.SUM)
                load_frac.div_(WORLD_SIZE)
            moe.update_router_bias(load_frac)
            moe.router_load_accum.zero_()
            moe.router_load_count.zero_()

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        if self.grad_diag_enabled:
            self.last_first_v_resid_for_diag = None
        first_v = None
        aux_loss = x.new_zeros(())
        stats_by_key = {}

        for i, block in enumerate(self.transformer.h):
            x, block_aux, block_stats, layer_v = block(x, first_v, cos_sin, self.window_sizes[i])
            if first_v is None:
                if self.grad_diag_enabled and torch.is_grad_enabled():
                    first_v = layer_v.clone()
                    first_v.retain_grad()
                    self.last_first_v_resid_for_diag = first_v
                else:
                    first_v = layer_v
            aux_loss = aux_loss + block_aux
            for key, value in block_stats.items():
                stats_by_key.setdefault(key, []).append(value)

        aux_loss = aux_loss / self.config.n_layer
        self.last_router_stats = {}
        for key, values in stats_by_key.items():
            layer_values = torch.stack(values).detach()
            self.last_router_stats[f"layer_{key}"] = layer_values
            self.last_router_stats[key] = layer_values.mean()
        self.last_router_stats["router_aux_loss"] = aux_loss.detach()

        x = norm(x)
        logits = self.lm_head(x).float()

        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            # The fixed BPB evaluator calls reduction='none'. Never add router
            # auxiliary losses to token-level eval CE.
            if reduction == "mean" and self.training:
                total_loss = ce_loss + aux_loss
                self.last_ce_loss = ce_loss.detach()
                self.last_total_loss = total_loss.detach()
                return total_loss
            return ce_loss
        return logits


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW). Same algorithm as the dense baseline.
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(
                p, grad, state["exp_avg"], state["exp_avg_sq"],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device_, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device_)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device_)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # If an expert is unused in a rare collapsed-routing step, keep optimizer
        # shape stable with a zero gradient rather than crashing DDP/Muon.
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        stacked_grads = torch.stack(grads)
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"])
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(
            stacked_grads, stacked_params,
            state["momentum_buffer"], state["second_momentum_buffer"],
            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
            self._muon_beta2_t, group["ns_steps"], red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)


# ---------------------------------------------------------------------------
# Hyperparameters. Agents may edit these directly; env overrides are used for
# scale sweeps so one tmux driver can run several sizes without source edits.
# ---------------------------------------------------------------------------

def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Model architecture.
DEPTH = int(os.environ.get("AR_DEPTH", "14"))
MODEL_DIM = int(os.environ.get("AR_MODEL_DIM", "1536"))
HEAD_DIM = int(os.environ.get("AR_HEAD_DIM", "128"))
NUM_HEADS = int(os.environ.get("AR_NUM_HEADS", str(MODEL_DIM // HEAD_DIM)))
NUM_KV_HEADS = int(os.environ.get("AR_NUM_KV_HEADS", "2"))
WINDOW_PATTERN = os.environ.get("AR_WINDOW_PATTERN", "SSSL")
NUM_EXPERTS = int(os.environ.get("AR_NUM_EXPERTS", "16"))
TOP_K = int(os.environ.get("AR_TOP_K", "2"))
MOE_HIDDEN_DIM = int(os.environ.get("AR_MOE_HIDDEN_DIM", "3584"))
DENSE_EARLY_LAYERS = int(os.environ.get("AR_DENSE_EARLY_LAYERS", "2"))
DENSE_HIDDEN_DIM = int(os.environ.get("AR_DENSE_HIDDEN_DIM", str(TOP_K * MOE_HIDDEN_DIM)))
ROUTER_Z_LOSS_COEF = float(os.environ.get("AR_ROUTER_Z_LOSS_COEF", "7.5e-4"))
LOAD_BALANCE_LOSS_COEF = float(os.environ.get("AR_LOAD_BALANCE_LOSS_COEF", "0.003"))
ROUTER_SIGMOID_AFFINITY = env_bool("AR_ROUTER_SIGMOID_AFFINITY", True)
ROUTER_EXPERT_BIAS = env_bool("AR_ROUTER_EXPERT_BIAS", True)
ROUTER_BIAS_EMA_BETA = float(os.environ.get("AR_ROUTER_BIAS_EMA_BETA", "0.9"))
ROUTER_BIAS_ETA = float(os.environ.get("AR_ROUTER_BIAS_ETA", "1.0e-3"))
ROUTER_BIAS_CLAMP = float(os.environ.get("AR_ROUTER_BIAS_CLAMP", "0.25"))
EXCLUSIVE_ATTENTION = env_bool("AR_EXCLUSIVE_ATTENTION", True)
HEADWISE_ATTENTION_GATE = env_bool("AR_HEADWISE_ATTENTION_GATE", True)
ATTENTION_GATE_INIT = float(os.environ.get("AR_ATTENTION_GATE_INIT", "0.98"))
ATTENTION_GATE_SCALE = float(os.environ.get("AR_ATTENTION_GATE_SCALE", "1.0"))
VALUE_MIX_ENABLED = env_bool("AR_VALUE_MIX_ENABLED", True)
VALUE_MIX_LEARNED = env_bool("AR_VALUE_MIX_LEARNED", False)
VALUE_MIX_START_LAYER = int(os.environ.get("AR_VALUE_MIX_START_LAYER", "1"))
VALUE_MIX_NORMALIZED = env_bool("AR_VALUE_MIX_NORMALIZED", False)
VALUE_MIX_FIRST_INIT = float(os.environ.get("AR_VALUE_MIX_FIRST_INIT", "0.75"))
VALUE_MIX_LOCAL_INIT = float(os.environ.get("AR_VALUE_MIX_LOCAL_INIT", "0.25"))
VALUE_MIX_GAMMA_INIT = math.sqrt(VALUE_MIX_FIRST_INIT ** 2 + VALUE_MIX_LOCAL_INIT ** 2)

# Optimization.
INIT_STD_GLOBAL = 1.0
TOTAL_BATCH_SIZE = 2**18       # global tokens per optimizer step, across all ranks
DEVICE_BATCH_SIZE = int(os.environ.get("AR_DEVICE_BATCH_SIZE", "16"))  # per-rank microbatch
EVAL_BATCH_SIZE = int(os.environ.get("AR_EVAL_BATCH_SIZE", "64"))      # rank-0 eval only; no gradients
ADAMW_LR = float(os.environ.get("AR_ADAMW_LR", "0.001"))
MUON_LR_WIDTH_FACTOR = 0.2
MUON_MOMENTUM = 0.95
MUON_NS_STEPS = 5
MUON_BETA2 = 0.95
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
GRAD_CLIP_NORM = 1.0
WARMUP_STEPS = 100
ESTIMATED_TOTAL_STEPS = int(os.environ.get("AR_ESTIMATED_TOTAL_STEPS", "2390"))
MIN_LR_FRAC = 0.1
ENABLE_COMPILE = True          # speed experiment: compile static model regions if sparse MoE permits it

# Optional actual-run gradient diagnostics. Disabled for normal benchmark runs.
GRAD_DIAG = os.environ.get("AR_GRAD_DIAG", "0") == "1"
GRAD_DIAG_EVERY = int(os.environ.get("AR_GRAD_DIAG_EVERY", "100"))
GRAD_DIAG_EXTRA_STEPS = {0, 1, 2, 5, 10, 20, 50}

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12
TRAIN_TIME_BUDGET = int(os.environ.get("AR_TIME_BUDGET", str(TIME_BUDGET)))
MAX_TRAIN_STEPS = int(os.environ.get("AR_MAX_STEPS", "0"))
USE_STEP_BUDGET = MAX_TRAIN_STEPS > 0

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
master_print(f"Vocab size: {vocab_size:,}")
master_print(f"Distributed: world_size={WORLD_SIZE}, rank={RANK}, local_rank={LOCAL_RANK}")

config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=NUM_HEADS,
    n_kv_head=NUM_KV_HEADS,
    n_embd=MODEL_DIM,
    window_pattern=WINDOW_PATTERN,
    num_experts=NUM_EXPERTS,
    top_k=TOP_K,
    moe_hidden_dim=MOE_HIDDEN_DIM,
    dense_early_layers=DENSE_EARLY_LAYERS,
    dense_hidden_dim=DENSE_HIDDEN_DIM,
    router_z_loss_coef=ROUTER_Z_LOSS_COEF,
    load_balance_loss_coef=LOAD_BALANCE_LOSS_COEF,
    router_sigmoid_affinity=ROUTER_SIGMOID_AFFINITY,
    router_expert_bias=ROUTER_EXPERT_BIAS,
    router_bias_ema_beta=ROUTER_BIAS_EMA_BETA,
    router_bias_eta=ROUTER_BIAS_ETA,
    router_bias_clamp=ROUTER_BIAS_CLAMP,
    exclusive_attention=EXCLUSIVE_ATTENTION,
    headwise_attention_gate=HEADWISE_ATTENTION_GATE,
    attention_gate_init=ATTENTION_GATE_INIT,
    attention_gate_scale=ATTENTION_GATE_SCALE,
    value_mix_enabled=VALUE_MIX_ENABLED,
    value_mix_learned=VALUE_MIX_LEARNED,
    value_mix_start_layer=VALUE_MIX_START_LAYER,
    value_mix_normalized=VALUE_MIX_NORMALIZED,
    value_mix_first_init=VALUE_MIX_FIRST_INIT,
    value_mix_local_init=VALUE_MIX_LOCAL_INIT,
    value_mix_gamma_init=VALUE_MIX_GAMMA_INIT,
)
master_print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    raw_model = GPT(config)
raw_model.to_empty(device=device)
raw_model.init_weights()
raw_model.train()
raw_model.grad_diag_enabled = GRAD_DIAG

param_counts = raw_model.num_scaling_params()
if IS_MASTER:
    master_print("Parameter counts:")
    for key, value in param_counts.items():
        master_print(f"  {key:24s}: {value:,}")
num_params = param_counts["total"]
active_params = param_counts["active"]
num_flops_per_token = raw_model.estimate_flops()
master_print(f"Estimated active FLOPs per token: {num_flops_per_token:e}")

assert TOTAL_BATCH_SIZE % (WORLD_SIZE * DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0, (
    "TOTAL_BATCH_SIZE must divide WORLD_SIZE * DEVICE_BATCH_SIZE * MAX_SEQ_LEN"
)
grad_accum_steps = TOTAL_BATCH_SIZE // (WORLD_SIZE * DEVICE_BATCH_SIZE * MAX_SEQ_LEN)

optimizer = raw_model.setup_optimizer(
    adamw_lr=ADAMW_LR,
    adam_betas=ADAM_BETAS,
    adam_eps=ADAM_EPS,
    weight_decay=WEIGHT_DECAY,
)
if IS_MASTER:
    master_print(f"AdamW peak lr: {ADAMW_LR:.6g}")
    for group in optimizer.param_groups:
        if group["kind"] == "muon":
            master_print(f"Muon peak lr ({group['name']}): {group['initial_lr']:.6g}")

model = raw_model
if ENABLE_COMPILE and not GRAD_DIAG:
    model = torch.compile(model, dynamic=False)
elif GRAD_DIAG:
    master_print("Gradient diagnostics enabled; torch.compile disabled for activation gradient capture.")
if IS_DISTRIBUTED:
    model = DDP(
        model,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        find_unused_parameters=False,
    )

train_loader = make_sharded_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, RANK, WORLD_SIZE, device)
x, y, epoch = next(train_loader)  # prefetch first batch

master_print(f"Time budget: {TRAIN_TIME_BUDGET}s")
master_print(f"Estimated total steps for LR schedule: {ESTIMATED_TOTAL_STEPS}")
if MAX_TRAIN_STEPS > 0:
    master_print(f"Max train steps: {MAX_TRAIN_STEPS}")
master_print(f"Global batch tokens: {TOTAL_BATCH_SIZE:,}")
master_print(f"Device batch size: {DEVICE_BATCH_SIZE}")
master_print(f"Gradient accumulation steps: {grad_accum_steps}")
master_print(f"Eval batch size: {EVAL_BATCH_SIZE}")


def get_lr_multiplier(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS if WARMUP_STEPS > 0 else 1.0
    decay_steps = max(1, ESTIMATED_TOTAL_STEPS - WARMUP_STEPS)
    decay_progress = min(max((step - WARMUP_STEPS) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return MIN_LR_FRAC + (1.0 - MIN_LR_FRAC) * cosine


def maybe_all_reduce_mean(tensor):
    if IS_DISTRIBUTED:
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(WORLD_SIZE)
    return tensor


def reduce_router_stats(stats):
    reduced = {}
    for key, value in stats.items():
        t = value.detach().float().clone()
        if IS_DISTRIBUTED:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t.div_(WORLD_SIZE)
        reduced[key] = t

    summary = {}
    for key, value in reduced.items():
        if value.numel() == 1 and not key.startswith("layer_"):
            summary[key] = float(value.item())

    def add_layer_summary(layer_key, mean_key=None, min_key=None, max_key=None, old_key=None):
        values = reduced.get(layer_key)
        if values is None:
            return
        mean_value = float(values.mean().item())
        if old_key is not None:
            summary[old_key] = mean_value
        if mean_key is not None:
            summary[mean_key] = mean_value
        if min_key is not None:
            summary[min_key] = float(values.min().item())
        if max_key is not None:
            summary[max_key] = float(values.max().item())

    add_layer_summary(
        "layer_router_entropy",
        mean_key="mean_router_entropy",
        min_key="min_layer_router_entropy",
        old_key="router_entropy",
    )
    add_layer_summary(
        "layer_expert_load_cv",
        mean_key="mean_expert_load_cv",
        max_key="max_layer_expert_load_cv",
        old_key="expert_load_cv",
    )
    add_layer_summary(
        "layer_max_expert_load",
        mean_key="mean_max_expert_load",
        max_key="max_layer_max_expert_load",
        old_key="max_expert_load",
    )
    add_layer_summary(
        "layer_router_z_loss",
        mean_key="mean_router_z_loss",
        max_key="max_layer_router_z_loss",
        old_key="router_z_loss",
    )
    add_layer_summary(
        "layer_router_sigmoid_mass_entropy",
        mean_key="mean_router_sigmoid_mass_entropy",
        min_key="min_layer_router_sigmoid_mass_entropy",
    )
    add_layer_summary(
        "layer_router_sigmoid_low_frac",
        mean_key="mean_router_sigmoid_low_frac",
        max_key="max_layer_router_sigmoid_low_frac",
    )
    add_layer_summary(
        "layer_router_sigmoid_high_frac",
        mean_key="mean_router_sigmoid_high_frac",
        max_key="max_layer_router_sigmoid_high_frac",
    )
    add_layer_summary(
        "layer_router_bias_abs",
        mean_key="mean_router_bias_abs",
        max_key="max_layer_router_bias_abs",
    )
    add_layer_summary(
        "layer_router_bias_max_abs",
        max_key="max_router_bias_abs",
    )
    add_layer_summary(
        "layer_router_load_ema_cv",
        mean_key="mean_router_load_ema_cv",
        max_key="max_layer_router_load_ema_cv",
    )

    summary.setdefault("mean_router_bias_abs", 0.0)
    summary.setdefault("max_router_bias_abs", 0.0)
    summary.setdefault("max_layer_router_bias_abs", 0.0)
    summary.setdefault("mean_router_load_ema_cv", 0.0)
    summary.setdefault("max_layer_router_load_ema_cv", 0.0)
    return summary


def collect_grad_diag(model):
    blocks = list(model.transformer.h)

    def nan_tensor():
        return torch.tensor(float("nan"), dtype=torch.float32, device=device)

    def grad_rms(param):
        if param is None or param.grad is None:
            return nan_tensor()
        return param.grad.detach().float().square().mean().sqrt()

    def tensor_grad_rms(tensor):
        if tensor is None or tensor.grad is None:
            return nan_tensor()
        return tensor.grad.detach().float().square().mean().sqrt()

    def tensor_grad_max_abs(tensor):
        if tensor is None or tensor.grad is None:
            return nan_tensor()
        return tensor.grad.detach().float().abs().max()

    def mean_of(values):
        values = [v for v in values if bool(torch.isfinite(v).all().item())]
        return torch.stack(values).mean() if values else nan_tensor()

    def max_of(values):
        values = [v for v in values if bool(torch.isfinite(v).all().item())]
        return torch.stack(values).max() if values else nan_tensor()

    def total_grad_norm():
        sq_sum = torch.zeros((), dtype=torch.float32, device=device)
        for param in model.parameters():
            if param.grad is not None:
                sq_sum = sq_sum + param.grad.detach().float().square().sum()
        return sq_sum.sqrt()

    alpha_params = [
        param
        for block in blocks
        if block.attn.value_mix_enabled and block.attn.value_mix_learned
        for param in (block.attn.value_mix_first, block.attn.value_mix_local, block.attn.value_mix_gamma)
        if param is not None
    ]
    alpha_values = [p.detach().float().abs() for p in alpha_params]
    alpha_grads = [p.grad.detach().float().abs() for p in alpha_params if p.grad is not None]

    qk_gamma_params = [block.attn.qk_gamma for block in blocks]
    qk_gamma_grads = [p.grad.detach().float().abs() for p in qk_gamma_params if p.grad is not None]

    first_v = model.last_first_v_resid_for_diag
    cv_grad_rms = [grad_rms(block.attn.c_v.weight) for block in blocks]
    moe_blocks = [block for block in blocks if block.moe is not None]
    router_grad_rms = [grad_rms(block.moe.router.weight) for block in moe_blocks]
    expert_gate_grad_rms = [grad_rms(block.moe.w_gate) for block in moe_blocks]
    expert_up_grad_rms = [grad_rms(block.moe.w_up) for block in moe_blocks]
    expert_down_grad_rms = [grad_rms(block.moe.w_down) for block in moe_blocks]
    attn_proj_grad_rms = [grad_rms(block.attn.c_proj.weight) for block in blocks]

    diag = {
        "total_grad_norm": total_grad_norm(),
        "first_v_resid_grad_rms": tensor_grad_rms(first_v),
        "first_v_resid_grad_rank_max_abs": tensor_grad_max_abs(first_v),
        "value_alpha_abs_mean": mean_of(alpha_values),
        "value_alpha_abs_max": max_of(alpha_values),
        "value_alpha_grad_abs_mean": mean_of(alpha_grads),
        "value_alpha_grad_abs_max": max_of(alpha_grads),
        "qk_gamma_grad_abs_mean": mean_of(qk_gamma_grads),
        "cv0_grad_rms": cv_grad_rms[0] if cv_grad_rms else nan_tensor(),
        "cv_later_grad_rms_mean": mean_of(cv_grad_rms[1:]),
        "cv_later_grad_rms_max": max_of(cv_grad_rms[1:]),
        "router_grad_rms_mean": mean_of(router_grad_rms),
        "router_grad_rms_max": max_of(router_grad_rms),
        "expert_gate_grad_rms_mean": mean_of(expert_gate_grad_rms),
        "expert_up_grad_rms_mean": mean_of(expert_up_grad_rms),
        "expert_down_grad_rms_mean": mean_of(expert_down_grad_rms),
        "attn_proj_grad_rms_mean": mean_of(attn_proj_grad_rms),
    }

    keys = sorted(diag)
    values = torch.stack([diag[key].float().reshape(()) for key in keys])
    if IS_DISTRIBUTED:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values.div_(WORLD_SIZE)
    return {key: float(value.item()) for key, value in zip(keys, values)}


def should_log_grad_diag(step):
    return GRAD_DIAG and (step in GRAD_DIAG_EXTRA_STEPS or (GRAD_DIAG_EVERY > 0 and step % GRAD_DIAG_EVERY == 0))


def format_grad_diag(step, train_ce, train_total, lrm, diag, router_summary):
    fields = [
        f"\nGRAD_DIAG step {step:05d}",
        f"ce={train_ce:.6f}",
        f"total={train_total:.6f}",
        f"lrm={lrm:.3f}",
        f"grad_norm={diag['total_grad_norm']:.3e}",
        f"alpha_abs_mean={diag['value_alpha_abs_mean']:.3e}",
        f"alpha_grad_mean={diag['value_alpha_grad_abs_mean']:.3e}",
        f"first_v_resid_grad={diag['first_v_resid_grad_rms']:.3e}",
        f"cv0_grad={diag['cv0_grad_rms']:.3e}",
        f"cv_later_grad_mean={diag['cv_later_grad_rms_mean']:.3e}",
        f"router_grad_mean={diag['router_grad_rms_mean']:.3e}",
        f"expert_down_grad_mean={diag['expert_down_grad_rms_mean']:.3e}",
        f"attn_proj_grad_mean={diag['attn_proj_grad_rms_mean']:.3e}",
    ]
    if router_summary:
        fields.extend([
            f"router_ent={router_summary.get('router_entropy', float('nan')):.3f}",
            f"load_cv={router_summary.get('expert_load_cv', float('nan')):.3f}",
            f"max_load={router_summary.get('max_expert_load', float('nan')):.3f}",
        ])
    return " | ".join(fields)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0.0
total_training_time = 0.0
step = 0
last_router_stats_for_summary = {}
last_train_ce_loss_for_summary = float("nan")
last_train_total_loss_for_summary = float("nan")

while True:
    torch.cuda.synchronize(device)
    t0 = time.time()

    train_loss_for_log = None
    train_ce_loss_for_log = None
    train_total_loss_for_log = None
    for micro_step in range(grad_accum_steps):
        sync_ctx = model.no_sync() if IS_DISTRIBUTED and micro_step < grad_accum_steps - 1 else nullcontext()
        with sync_ctx:
            with autocast_ctx:
                loss = model(x, y)
            train_loss_for_log = loss.detach()
            train_ce_loss_for_log = raw_model.last_ce_loss
            train_total_loss_for_log = raw_model.last_total_loss
            (loss / grad_accum_steps).backward()
            raw_model.accumulate_router_loads()
        x, y, epoch = next(train_loader)

    train_loss_tensor = maybe_all_reduce_mean(train_loss_for_log.float())
    train_ce_loss_tensor = maybe_all_reduce_mean(train_ce_loss_for_log.float())
    train_total_loss_tensor = maybe_all_reduce_mean(train_total_loss_for_log.float())

    # Progress and schedules.
    if USE_STEP_BUDGET:
        progress = min(step / max(1, MAX_TRAIN_STEPS), 1.0)
    else:
        progress = min(total_training_time / TRAIN_TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    if should_log_grad_diag(step):
        grad_diag = collect_grad_diag(raw_model)
        diag_router_stats = reduce_router_stats(raw_model.last_router_stats) if raw_model.last_router_stats else {}
        if IS_MASTER:
            print(
                format_grad_diag(
                    step,
                    float(train_ce_loss_tensor.item()),
                    float(train_total_loss_tensor.item()),
                    lrm,
                    grad_diag,
                    diag_router_stats,
                ),
                flush=True,
            )

    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    raw_model.update_router_biases()
    model.zero_grad(set_to_none=True)

    train_loss_f = float(train_loss_tensor.item())
    train_ce_loss_f = float(train_ce_loss_tensor.item())
    train_total_loss_f = float(train_total_loss_tensor.item())
    bad = torch.tensor(
        [not math.isfinite(train_loss_f) or train_loss_f > 100],
        dtype=torch.int32,
        device=device,
    )
    if IS_DISTRIBUTED:
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bool(bad.item()):
        master_print("FAIL")
        if IS_DISTRIBUTED:
            dist.destroy_process_group()
        sys.exit(1)

    torch.cuda.synchronize(device)
    t1 = time.time()
    dt = t1 - t0

    # Rank 0 owns the wall-clock training budget; broadcast it so all ranks
    # use identical LR schedules and stop on the same iteration.
    if IS_MASTER and step > 10:
        total_training_time += dt
    if IS_DISTRIBUTED:
        _time_tensor = torch.tensor([total_training_time], dtype=torch.float64, device=device)
        dist.broadcast(_time_tensor, src=0)
        total_training_time = float(_time_tensor.item())

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / (H100_BF16_PEAK_FLOPS * WORLD_SIZE)
    if USE_STEP_BUDGET:
        if step > 10:
            avg_step_time = total_training_time / max(1, step - 10)
        else:
            avg_step_time = dt
        remaining = max(0, MAX_TRAIN_STEPS - step - 1) * avg_step_time
    else:
        remaining = max(0, TRAIN_TIME_BUDGET - total_training_time)

    stats = raw_model.last_router_stats
    reduced_router_stats = reduce_router_stats(stats) if stats else {}

    if IS_MASTER:
        last_train_ce_loss_for_summary = train_ce_loss_f
        last_train_total_loss_for_summary = train_total_loss_f
        last_router_stats_for_summary = reduced_router_stats
        ent = last_router_stats_for_summary.get("router_entropy", float("nan"))
        cv = last_router_stats_for_summary.get("expert_load_cv", float("nan"))
        mx = last_router_stats_for_summary.get("max_expert_load", float("nan"))
        aux = last_router_stats_for_summary.get("router_aux_loss", float("nan"))
        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | "
            f"loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | "
            f"dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
            f"mfu: {mfu:.1f}% | epoch: {epoch} | "
            f"router_ent: {ent:.3f} | load_cv: {cv:.3f} | max_load: {mx:.3f} | aux: {aux:.4f} | "
            f"remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    reached_time_budget = not USE_STEP_BUDGET and step > 10 and total_training_time >= TRAIN_TIME_BUDGET
    reached_step_budget = USE_STEP_BUDGET and step >= MAX_TRAIN_STEPS
    done = torch.tensor(
        [1 if (IS_MASTER and (reached_time_budget or reached_step_budget)) else 0],
        dtype=torch.int32,
        device=device,
    )
    if IS_DISTRIBUTED:
        dist.broadcast(done, src=0)
    if bool(done.item()):
        break

if IS_MASTER:
    print()

total_tokens = step * TOTAL_BATCH_SIZE

# Final evaluation on rank 0 only. Other ranks wait so NCCL is cleaned up safely.
if IS_DISTRIBUTED:
    dist.barrier()

val_bpb = None
if IS_MASTER:
    raw_model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(raw_model, tokenizer, EVAL_BATCH_SIZE)

if IS_DISTRIBUTED:
    dist.barrier()

# Final summary.
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = (
    100 * num_flops_per_token * TOTAL_BATCH_SIZE * max(step - 10, 0) /
    total_training_time / (H100_BF16_PEAK_FLOPS * WORLD_SIZE)
) if total_training_time > 0 else 0
peak_vram_mb_local = torch.cuda.max_memory_allocated(device) / 1024 / 1024
peak_vram_tensor = torch.tensor([peak_vram_mb_local], device=device)
if IS_DISTRIBUTED:
    dist.all_reduce(peak_vram_tensor, op=dist.ReduceOp.MAX)
peak_vram_mb = float(peak_vram_tensor.item())
qk_gamma_tensor = torch.stack([block.attn.qk_gamma.detach().float() for block in raw_model.transformer.h])
value_mix_first_values = [
    block.attn.value_mix_first.detach().float()
    for block in raw_model.transformer.h
    if block.attn.value_mix_enabled
]
value_mix_local_values = [
    block.attn.value_mix_local.detach().float()
    for block in raw_model.transformer.h
    if block.attn.value_mix_enabled
]
value_mix_gamma_values = [
    block.attn.value_mix_gamma.detach().float()
    for block in raw_model.transformer.h
    if block.attn.value_mix_enabled and block.attn.value_mix_gamma is not None
]
value_mix_first_tensor = torch.stack(value_mix_first_values) if value_mix_first_values else None
value_mix_local_tensor = torch.stack(value_mix_local_values) if value_mix_local_values else None
value_mix_gamma_tensor = torch.stack(value_mix_gamma_values) if value_mix_gamma_values else None
attention_gate_bias_sigmoid_values = [
    torch.sigmoid(block.attn.c_attn_gate.bias.detach().float()).mean()
    for block in raw_model.transformer.h
    if block.attn.c_attn_gate is not None
]
attention_gate_weight_rms_values = [
    block.attn.c_attn_gate.weight.detach().float().square().mean().sqrt()
    for block in raw_model.transformer.h
    if block.attn.c_attn_gate is not None
]
attention_gate_bias_sigmoid_tensor = (
    torch.stack(attention_gate_bias_sigmoid_values) if attention_gate_bias_sigmoid_values else None
)
attention_gate_weight_rms_tensor = (
    torch.stack(attention_gate_weight_rms_values) if attention_gate_weight_rms_values else None
)

if IS_MASTER:
    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"startup_seconds:  {startup_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"active_params_M:  {active_params / 1e6:.1f}")
    print(f"world_size:       {WORLD_SIZE}")
    print(f"depth:            {DEPTH}")
    print(f"model_dim:        {MODEL_DIM}")
    print(f"num_experts:      {NUM_EXPERTS}")
    print(f"top_k:            {TOP_K}")
    print(f"moe_hidden_dim:   {MOE_HIDDEN_DIM}")
    print(f"dense_early_layers: {DENSE_EARLY_LAYERS}")
    print(f"dense_hidden_dim: {DENSE_HIDDEN_DIM}")
    print(f"mean_qk_gamma:    {qk_gamma_tensor.mean().item():.6f}")
    print(f"min_qk_gamma:     {qk_gamma_tensor.min().item():.6f}")
    print(f"max_qk_gamma:     {qk_gamma_tensor.max().item():.6f}")
    print(f"exclusive_attention: {int(EXCLUSIVE_ATTENTION)}")
    print(f"headwise_attention_gate: {int(HEADWISE_ATTENTION_GATE)}")
    print(f"attention_gate_scale: {ATTENTION_GATE_SCALE:.6f}")
    if attention_gate_bias_sigmoid_tensor is not None:
        print(f"mean_attention_gate_bias_sigmoid: {attention_gate_bias_sigmoid_tensor.mean().item():.6f}")
        print(f"min_attention_gate_bias_sigmoid:  {attention_gate_bias_sigmoid_tensor.min().item():.6f}")
        print(f"max_attention_gate_bias_sigmoid:  {attention_gate_bias_sigmoid_tensor.max().item():.6f}")
        print(f"mean_attention_gate_bias_effective: {(ATTENTION_GATE_SCALE * attention_gate_bias_sigmoid_tensor).mean().item():.6f}")
        print(f"mean_attention_gate_weight_rms:   {attention_gate_weight_rms_tensor.mean().item():.6f}")
    print(f"value_mix_enabled: {int(VALUE_MIX_ENABLED)}")
    print(f"value_mix_learned: {int(VALUE_MIX_LEARNED)}")
    print(f"value_mix_start_layer: {VALUE_MIX_START_LAYER}")
    print(f"value_mix_normalized: {int(VALUE_MIX_NORMALIZED)}")
    if value_mix_first_tensor is not None:
        print(f"mean_value_mix_first: {value_mix_first_tensor.mean().item():.6f}")
        print(f"min_value_mix_first:  {value_mix_first_tensor.min().item():.6f}")
        print(f"max_value_mix_first:  {value_mix_first_tensor.max().item():.6f}")
        print(f"mean_value_mix_local: {value_mix_local_tensor.mean().item():.6f}")
        print(f"min_value_mix_local:  {value_mix_local_tensor.min().item():.6f}")
        print(f"max_value_mix_local:  {value_mix_local_tensor.max().item():.6f}")
    if value_mix_gamma_tensor is not None:
        print(f"mean_value_mix_gamma: {value_mix_gamma_tensor.mean().item():.6f}")
        print(f"min_value_mix_gamma:  {value_mix_gamma_tensor.min().item():.6f}")
        print(f"max_value_mix_gamma:  {value_mix_gamma_tensor.max().item():.6f}")
    print(f"train_ce_loss:    {last_train_ce_loss_for_summary:.6f}")
    print(f"train_total_loss: {last_train_total_loss_for_summary:.6f}")
    for key in [
        "router_entropy",
        "mean_router_entropy",
        "min_layer_router_entropy",
        "expert_load_cv",
        "mean_expert_load_cv",
        "max_layer_expert_load_cv",
        "max_expert_load",
        "mean_max_expert_load",
        "max_layer_max_expert_load",
        "router_z_loss",
        "mean_router_z_loss",
        "max_layer_router_z_loss",
        "mean_router_sigmoid_mass_entropy",
        "min_layer_router_sigmoid_mass_entropy",
        "mean_router_sigmoid_low_frac",
        "max_layer_router_sigmoid_low_frac",
        "mean_router_sigmoid_high_frac",
        "max_layer_router_sigmoid_high_frac",
        "mean_router_bias_abs",
        "max_router_bias_abs",
        "max_layer_router_bias_abs",
        "mean_router_load_ema_cv",
        "max_layer_router_load_ema_cv",
        "router_lb_loss",
        "router_aux_loss",
    ]:
        if key in last_router_stats_for_summary:
            print(f"{key + ':':17s} {last_router_stats_for_summary[key]:.6f}")

if IS_DISTRIBUTED:
    dist.destroy_process_group()
