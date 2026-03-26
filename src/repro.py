from __future__ import annotations

import random
from typing import Any


def _seed_numpy(seed: int) -> bool:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return False
    np.random.seed(seed)
    return True


def _seed_torch(seed: int) -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return True


def seed_everything(seed: int) -> dict[str, Any]:
    random.seed(seed)
    numpy_seeded = _seed_numpy(seed)
    torch_seeded = _seed_torch(seed)
    return {
        "python": seed,
        "numpy": seed if numpy_seeded else None,
        "torch": seed if torch_seeded else None,
    }
