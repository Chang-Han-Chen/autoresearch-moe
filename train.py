"""
Autoresearch pretraining script: 4xH100 DDP, 100M-active top-2 MoE.

Usage:
    uv run torchrun --standalone --nproc_per_node=4 train.py

Design constraints:
- prepare.py is read-only. We import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, and
  evaluate_bpb, but the evaluation contract is unchanged.
- The optimizer algorithm remains MuonAdamW + AdamW from the dense baseline.
  Experiment agents may change learning-rate values, but not the optimizer family.
- The model is intentionally a scale-down of GPT-OSS-style sparse FFN models:
  GQA attention, alternating local/full attention, top-2 token-choice MoE FFNs.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
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
fa3 = get_kernel(repo).flash_attn_interface

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


def has_ve(layer_idx, n_layer):
    """Use value embeddings on alternating layers, always including the last."""
    return layer_idx % 2 == (n_layer - 1) % 2


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
        self.qk_gamma = nn.Parameter(torch.ones(()))
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual / value embedding: preserve early token value information.
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm for stable attention logits.
        q = q * self.qk_gamma

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class SwiGLUExpert(nn.Module):
    def __init__(self, n_embd, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_up = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


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
        assert 1 <= self.top_k <= self.num_experts
        self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)
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
            top_logits, top_idx = torch.topk(router_logits, K, dim=-1)
            top_weight = F.softmax(top_logits, dim=-1)       # [N, K]

            # Router regularization. The hard load fraction is intentionally
            # detached; gradients flow through mean router probability, not
            # through the non-differentiable top-k indices.
            selected_one_hot = F.one_hot(top_idx, num_classes=E).float()  # [N, K, E]
            load_frac = selected_one_hot.sum(dim=(0, 1)) / float(N * K)
            prob_mean = router_probs.mean(dim=0)
            load_balance_loss = E * torch.sum(load_frac.detach() * prob_mean)
            z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
            aux_loss = self.load_balance_loss_coef * load_balance_loss + self.router_z_loss_coef * z_loss

            entropy = -(router_probs * router_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            load_cv = load_frac.std(unbiased=False) / load_frac.mean().clamp_min(1e-9)
            max_load = load_frac.max()

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
        self.moe = TokenChoiceMoE(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        moe_out, aux_loss, stats = self.moe(norm(x))
        x = x + moe_out
        return x, aux_loss, stats


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
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })

        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.last_router_stats = {}
        self.last_ce_loss = None
        self.last_total_loss = None

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
            block.attn.qk_gamma.fill_(1.0)

            init_weight(block.moe.router.weight, self.config.n_embd)
            init_weight(block.moe.w_gate, self.config.n_embd)
            init_weight(block.moe.w_up, self.config.n_embd)
            init_weight(block.moe.w_down, self.config.moe_hidden_dim)

        for ve in self.value_embeds.values():
            init_weight(ve.weight, ve.embedding_dim)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                init_weight(block.attn.ve_gate.weight, block.attn.ve_gate_channels)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Match the dense baseline: embedding tables in bf16, compute under autocast.
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

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
        total_expert = sum(block.moe.expert_param_count() for block in self.transformer.h)
        active_expert = sum(block.moe.active_expert_param_count() for block in self.transformer.h)
        return total_expert, active_expert

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = 0
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        total_expert, active_expert = self.expert_param_counts()
        active = total - total_expert + active_expert
        return {
            "wte": wte,
            "value_embeds": value_embeds,
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
        nparams_exclude = counts["wte"] + counts["value_embeds"] + counts["scalars"]
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
        qk_gamma_params = [block.attn.qk_gamma for block in self.transformer.h]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=adamw_lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay),
            dict(kind="adamw", params=embedding_params, lr=adamw_lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay),
            dict(kind="adamw", params=value_embeds_params, lr=adamw_lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay),
            dict(kind="adamw", params=qk_gamma_params, lr=adamw_lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay),
        ]

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
        add_muon_group("attn_q", (block.attn.c_q.weight for block in blocks), self.config.n_embd, self.config.n_embd)
        add_muon_group("attn_k", (block.attn.c_k.weight for block in blocks), self.config.n_embd, kv_dim)
        add_muon_group("attn_v", (block.attn.c_v.weight for block in blocks), self.config.n_embd, kv_dim)
        add_muon_group("attn_proj", (block.attn.c_proj.weight for block in blocks), self.config.n_embd, self.config.n_embd)
        add_muon_group("router", (block.moe.router.weight for block in blocks), self.config.n_embd, self.config.num_experts)
        add_muon_group("expert_gate", (block.moe.w_gate for block in blocks), self.config.n_embd, self.config.moe_hidden_dim)
        add_muon_group("expert_up", (block.moe.w_up for block in blocks), self.config.n_embd, self.config.moe_hidden_dim)
        add_muon_group("expert_down", (block.moe.w_down for block in blocks), self.config.moe_hidden_dim, self.config.n_embd)
        add_muon_group(
            "value_gate",
            (block.attn.ve_gate.weight for block in blocks if block.attn.ve_gate is not None),
            blocks[0].attn.ve_gate_channels,
            self.config.n_kv_head,
        )

        assert len(list(self.parameters())) == (
            len(muon_params) + len(embedding_params) + len(lm_head_params) +
            len(value_embeds_params) + len(qk_gamma_params)
        )
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        aux_loss = x.new_zeros(())
        stats_by_key = {}

        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x, block_aux, block_stats = block(x, ve, cos_sin, self.window_sizes[i])
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
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

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
# Hyperparameters. Agents may edit these directly; no CLI flags are needed.
# ---------------------------------------------------------------------------

# Model architecture: about 100M active params, about 300M total params.
DEPTH = 8
MODEL_DIM = 768
HEAD_DIM = 128
NUM_HEADS = MODEL_DIM // HEAD_DIM
NUM_KV_HEADS = 2
WINDOW_PATTERN = "SSSL"
NUM_EXPERTS = 16
TOP_K = 2
MOE_HIDDEN_DIM = 1792
ROUTER_Z_LOSS_COEF = 7.5e-4
LOAD_BALANCE_LOSS_COEF = 8.5e-3

# Optimization.
INIT_STD_GLOBAL = 1.0
TOTAL_BATCH_SIZE = 2**18       # global tokens per optimizer step, across all ranks
DEVICE_BATCH_SIZE = 32         # per-rank microbatch, safe default for 80GB H100
EVAL_BATCH_SIZE = 64           # rank-0 eval only; no gradients
ADAMW_LR = 0.001
MUON_LR_WIDTH_FACTOR = 0.2
MUON_MOMENTUM = 0.95
MUON_NS_STEPS = 5
MUON_BETA2 = 0.95
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
GRAD_CLIP_NORM = 1.0
WARMUP_STEPS = 100
ESTIMATED_TOTAL_STEPS = 2390
MIN_LR_FRAC = 0.1
ENABLE_COMPILE = True          # speed experiment: compile static model regions if sparse MoE permits it

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
    router_z_loss_coef=ROUTER_Z_LOSS_COEF,
    load_balance_loss_coef=LOAD_BALANCE_LOSS_COEF,
)
master_print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    raw_model = GPT(config)
raw_model.to_empty(device=device)
raw_model.init_weights()
raw_model.train()

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
if ENABLE_COMPILE:
    model = torch.compile(model, dynamic=False)
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

    summary.setdefault("mean_router_bias_abs", 0.0)
    summary.setdefault("max_router_bias_abs", 0.0)
    summary.setdefault("max_layer_router_bias_abs", 0.0)
    return summary


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
        x, y, epoch = next(train_loader)

    train_loss_tensor = maybe_all_reduce_mean(train_loss_for_log.float())
    train_ce_loss_tensor = maybe_all_reduce_mean(train_ce_loss_for_log.float())
    train_total_loss_tensor = maybe_all_reduce_mean(train_total_loss_for_log.float())

    # Progress and schedules.
    progress = min(total_training_time / TRAIN_TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
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

    done = torch.tensor(
        [1 if (IS_MASTER and step > 10 and total_training_time >= TRAIN_TIME_BUDGET) else 0],
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
    print(f"mean_qk_gamma:    {qk_gamma_tensor.mean().item():.6f}")
    print(f"min_qk_gamma:     {qk_gamma_tensor.min().item():.6f}")
    print(f"max_qk_gamma:     {qk_gamma_tensor.max().item():.6f}")
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
        "mean_router_bias_abs",
        "max_router_bias_abs",
        "max_layer_router_bias_abs",
        "router_lb_loss",
        "router_aux_loss",
    ]:
        if key in last_router_stats_for_summary:
            print(f"{key + ':':17s} {last_router_stats_for_summary[key]:.6f}")

if IS_DISTRIBUTED:
    dist.destroy_process_group()
