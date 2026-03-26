from __future__ import annotations

import math


def _require_torch():
    import torch
    import torch.nn as nn

    return torch, nn


class RMSNorm(_require_torch()[1].Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        torch, nn = _require_torch()
        super().__init__()
        self._torch = torch
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def _norm(self, hidden_states):
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        return hidden_states * self._torch.rsqrt(variance + self.eps)

    def forward(self, hidden_states):
        output = self._norm(hidden_states.float()).type_as(hidden_states)
        return output * self.weight


class RotaryEmbedding(_require_torch()[1].Module):
    def __init__(self, dim: int, base: float = 10_000.0, scaling_factor: float = 1.0):
        torch, _ = _require_torch()
        super().__init__()
        self.dim = dim
        self.base = base
        self.scaling_factor = scaling_factor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, dtype):
        torch, _ = _require_torch()
        scaled = position_ids.float() / self.scaling_factor
        freqs = torch.einsum("bs,d->bsd", scaled, self.inv_freq.to(position_ids.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)[:, None, :, :]
        sin = emb.sin().to(dtype=dtype)[:, None, :, :]
        return cos, sin


def rotate_half(hidden_states):
    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return _require_torch()[0].cat((-second_half, first_half), dim=-1)


def apply_rotary_pos_emb(query, key, cos, sin):
    query = (query * cos) + (rotate_half(query) * sin)
    key = (key * cos) + (rotate_half(key) * sin)
    return query, key


def repeat_kv(hidden_states, num_repeats: int):
    if num_repeats == 1:
        return hidden_states
    torch, _ = _require_torch()
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_repeats, seq_len, head_dim
    )
    return hidden_states.reshape(batch_size, num_key_value_heads * num_repeats, seq_len, head_dim)


def build_attention_mask(attention_mask, query_length: int, key_length: int, device, dtype):
    torch, _ = _require_torch()
    causal = torch.triu(
        torch.ones(query_length, key_length, device=device, dtype=torch.bool),
        diagonal=1 + key_length - query_length,
    )
    additive = torch.zeros((query_length, key_length), device=device, dtype=dtype)
    additive.masked_fill_(causal, torch.finfo(dtype).min)
    if attention_mask is None:
        return additive[None, None, :, :]
    padding_mask = attention_mask[:, None, None, :key_length].to(dtype=dtype)
    additive = additive[None, None, :, :].expand(attention_mask.size(0), 1, query_length, key_length)
    additive = additive.masked_fill(padding_mask == 0, torch.finfo(dtype).min)
    return additive


class SwiGLU(_require_torch()[1].Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        _, nn = _require_torch()
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        _, nn = _require_torch()
        return self.down_proj(nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
