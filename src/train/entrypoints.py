from __future__ import annotations

from pathlib import Path

from config import DataConfig, ModelConfig, TrainConfig, save_config
from data.dataset import DatasetBuilder
from model.transformer import CausalTransformer
from repro import seed_everything
from train.checkpoint import CheckpointManager
from train.distributed import cleanup_distributed, init_distributed, is_main_process, maybe_wrap_fsdp
from train.loop import build_dataloader, run_training, save_run_metadata
from train.optim import build_optimizer, build_scheduler


def snapshot_configs(model_config: ModelConfig, data_config: DataConfig, train_config: TrainConfig) -> None:
    config_dir = Path(train_config.checkpoint.output_dir) / "configs"
    save_config(model_config, config_dir / "model.json")
    save_config(data_config, config_dir / "data.json")
    save_config(train_config, config_dir / "train.json")


def _run_stage(
    dataset,
    validation_dataset,
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
) -> None:
    if train_config.micro_batch_size < 1:
        raise ValueError("micro_batch_size must be at least 1")
    if train_config.checkpoint.initialize_from and train_config.checkpoint.resume_from:
        raise ValueError("Set either checkpoint.initialize_from or checkpoint.resume_from, not both.")
    init_distributed()
    try:
        seed_bundle = seed_everything(train_config.seed)
        model = CausalTransformer(model_config)
        model = maybe_wrap_fsdp(model, train_config)
        checkpoint_manager = CheckpointManager(
            output_dir=train_config.checkpoint.output_dir,
            keep_last_n=train_config.checkpoint.keep_last_n,
        )
        if train_config.checkpoint.initialize_from and not train_config.checkpoint.resume_from:
            checkpoint_manager.load(train_config.checkpoint.initialize_from, model, strict=True)
        optimizer = build_optimizer(model, train_config)
        scheduler = build_scheduler(optimizer, train_config)
        if is_main_process():
            snapshot_configs(model_config, data_config=data_config, train_config=train_config)
            save_run_metadata(
                train_config,
                train_config.checkpoint.output_dir,
                extra={"seed_bundle": seed_bundle},
            )
        train_loader = build_dataloader(dataset, batch_size=train_config.micro_batch_size, shuffle=True)
        val_loader = None
        if validation_dataset is not None and len(validation_dataset) > 0:
            val_loader = build_dataloader(validation_dataset, batch_size=train_config.micro_batch_size, shuffle=False)
        run_training(
            model=model,
            train_loader=train_loader,
            train_config=train_config,
            checkpoint_manager=checkpoint_manager,
            optimizer=optimizer,
            scheduler=scheduler,
            val_loader=val_loader,
            resume_from=train_config.checkpoint.resume_from,
        )
    finally:
        cleanup_distributed()


def run_pretraining(model_config: ModelConfig, data_config: DataConfig, train_config: TrainConfig) -> None:
    builder = DatasetBuilder(data_config)
    dataset = builder.build_pretrain()
    validation_dataset = builder.build_validation() if data_config.validation_sources else None
    stage_config = TrainConfig.from_dict(train_config.to_dict())
    if stage_config.token_budget is None:
        stage_config.token_budget = data_config.pretraining_token_budget
    _run_stage(dataset, validation_dataset, model_config, data_config, stage_config)


def run_continued_pretraining(
    model_config: ModelConfig, data_config: DataConfig, train_config: TrainConfig
) -> None:
    builder = DatasetBuilder(data_config)
    dataset = builder.build_continued_pretrain()
    validation_dataset = builder.build_validation() if data_config.validation_sources else None
    stage_config = TrainConfig.from_dict(train_config.to_dict())
    if stage_config.continued_learning_rate is not None:
        stage_config.learning_rate = stage_config.continued_learning_rate
    if stage_config.continued_min_learning_rate is not None:
        stage_config.min_learning_rate = stage_config.continued_min_learning_rate
    if stage_config.continued_warmup_steps is not None:
        stage_config.warmup_steps = stage_config.continued_warmup_steps
    if stage_config.continued_max_steps is not None:
        stage_config.max_steps = stage_config.continued_max_steps
    if stage_config.token_budget is None:
        stage_config.token_budget = data_config.continued_pretraining_token_budget
    _run_stage(dataset, validation_dataset, model_config, data_config, stage_config)
