from __future__ import annotations

import os
from functools import partial

from config import TrainConfig
from model.transformer import DecoderLayer


def _require_torch():
    import torch
    import torch.distributed as dist

    return torch, dist


def init_distributed() -> tuple[int, int, int]:
    torch, dist = _require_torch()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main_process() -> bool:
    _, dist = _require_torch()
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    _, dist = _require_torch()
    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    _, dist = _require_torch()
    if dist.is_initialized():
        dist.destroy_process_group()


def maybe_wrap_fsdp(model, train_config: TrainConfig):
    torch, dist = _require_torch()
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return model

    from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    mp_policy = None
    if train_config.use_bf16 and torch.cuda.is_available():
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    strategy = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    }[train_config.fsdp_sharding_strategy]
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={DecoderLayer})
    wrapped = FullyShardedDataParallel(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        limit_all_gathers=True,
        sync_module_states=True,
        use_orig_params=True,
    )
    if train_config.activation_checkpointing:
        non_reentrant_wrapper = lambda module: checkpoint_wrapper(  # noqa: E731
            module, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        apply_activation_checkpointing(
            wrapped,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda module: isinstance(module, DecoderLayer),
        )
    return wrapped
