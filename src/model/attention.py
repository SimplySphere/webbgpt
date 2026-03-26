from __future__ import annotations

import math

from config import ModelConfig
from model.cache import LayerKVCache
from model.modules import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    build_attention_mask,
    repeat_kv,
)


def _require_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


class GroupedQueryAttention(_require_torch()[1].Module):
    def __init__(self, config: ModelConfig):
        torch, nn, _ = _require_torch()
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.attention_dropout = config.attention_dropout
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            base=config.rope_theta,
            scaling_factor=config.rope_scaling_factor,
        )
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value: LayerKVCache | None = None,
        use_cache: bool = False,
    ):
        torch, _, F = _require_torch()
        batch_size, query_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_ids is None:
            past_length = 0 if past_key_value is None else past_key_value.key.size(-2)
            position_ids = (
                torch.arange(past_length, past_length + query_length, device=hidden_states.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        cos, sin = self.rotary_emb.forward(position_ids, dtype=query_states.dtype)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value.key, key_states], dim=2)
            value_states = torch.cat([past_key_value.value, value_states], dim=2)
        present_key_value = LayerKVCache(key=key_states, value=value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        key_length = key_states.size(-2)

        use_sdp = hasattr(F, "scaled_dot_product_attention")
        if use_sdp:
            if attention_mask is None and past_key_value is None:
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=None,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=True,
                )
            else:
                additive_mask = build_attention_mask(
                    attention_mask=attention_mask,
                    query_length=query_length,
                    key_length=key_length,
                    device=hidden_states.device,
                    dtype=query_states.dtype,
                )
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=additive_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=False,
                )
        else:
            scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
            scores = scores + build_attention_mask(
                attention_mask=attention_mask,
                query_length=query_length,
                key_length=key_length,
                device=hidden_states.device,
                dtype=scores.dtype,
            )
            attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = torch.dropout(attn_weights, self.attention_dropout, self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_length, self.hidden_size)
        return self.o_proj(attn_output), present_key_value
