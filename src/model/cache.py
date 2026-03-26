from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LayerKVCache:
    key: "torch.Tensor"
    value: "torch.Tensor"


KVCache = list[LayerKVCache]

