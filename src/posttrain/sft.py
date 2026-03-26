from __future__ import annotations

import math
import sys
from pathlib import Path

from config import DataConfig, ModelConfig, TrainConfig
from data.dataset import DatasetBuilder
from model.transformer import CausalTransformer
from posttrain.eval import (
    ensure_no_regression_prompt_overlap,
    generate_qualitative_samples,
)
from repro import seed_everything
from train.checkpoint import CheckpointManager
from train.distributed import cleanup_distributed, init_distributed, is_main_process, maybe_wrap_fsdp
from train.entrypoints import snapshot_configs
from train.loop import EvalControl, build_dataloader, evaluate_language_model, run_training, save_run_metadata
from train.optim import build_optimizer, build_scheduler


def _apply_sft_overrides(train_config: TrainConfig) -> TrainConfig:
    stage_config = TrainConfig.from_dict(train_config.to_dict())
    if stage_config.sft_learning_rate is not None:
        stage_config.learning_rate = stage_config.sft_learning_rate
    if stage_config.sft_min_learning_rate is not None:
        stage_config.min_learning_rate = stage_config.sft_min_learning_rate
    if stage_config.sft_warmup_steps is not None:
        stage_config.warmup_steps = stage_config.sft_warmup_steps
    if stage_config.sft_max_steps is not None:
        stage_config.max_steps = stage_config.sft_max_steps
    return stage_config


def _evaluate_sft_validation(model, dataloader, _max_batches: int | None) -> dict[str, float]:
    metrics = evaluate_language_model(model, dataloader, None)
    metrics["batches_evaluated"] = len(dataloader)
    metrics["examples_evaluated"] = len(dataloader.dataset)
    return metrics


def run_sft_job(model_config: ModelConfig, data_config: DataConfig, train_config: TrainConfig) -> None:
    stage_config = _apply_sft_overrides(train_config)
    builder = DatasetBuilder(data_config)
    train_dataset, validation_dataset = builder.build_sft_split(
        seed=stage_config.seed,
        validation_fraction=stage_config.sft_validation_fraction,
        validation_min_examples=stage_config.sft_validation_min_examples,
        allow_weak_validation=stage_config.allow_weak_posttrain_validation,
        require_explicit_validation=stage_config.require_explicit_sft_validation,
    )
    train_examples = getattr(train_dataset, "examples", None)
    validation_examples = getattr(validation_dataset, "examples", None)
    if train_examples is not None and validation_examples is not None:
        ensure_no_regression_prompt_overlap(
            stage_name="sft",
            train_examples=train_examples,
            validation_examples=validation_examples,
        )
    elif is_main_process():
        print(
            "WebbGPT: skipping SFT regression-prompt overlap guard because prepared datasets do not expose raw prompt metadata in v1.",
            file=sys.stderr,
            flush=True,
        )
    if stage_config.checkpoint.initialize_from and stage_config.checkpoint.resume_from:
        raise ValueError("Set either checkpoint.initialize_from or checkpoint.resume_from, not both.")
    init_distributed()
    try:
        seed_bundle = seed_everything(stage_config.seed)
        model = CausalTransformer(model_config)
        model = maybe_wrap_fsdp(model, stage_config)
        checkpoint_manager = CheckpointManager(
            output_dir=stage_config.checkpoint.output_dir,
            keep_last_n=stage_config.checkpoint.keep_last_n,
        )
        if stage_config.checkpoint.initialize_from and not stage_config.checkpoint.resume_from:
            checkpoint_manager.load(stage_config.checkpoint.initialize_from, model, strict=True)
        optimizer = build_optimizer(model, stage_config)
        scheduler = build_scheduler(optimizer, stage_config)
        if is_main_process():
            snapshot_configs(model_config, data_config, stage_config)
            save_run_metadata(
                stage_config,
                stage_config.checkpoint.output_dir,
                extra={
                    "seed_bundle": seed_bundle,
                    "validation_policy": {
                        "require_explicit_validation": stage_config.require_explicit_sft_validation,
                        "validation_min_examples": stage_config.sft_validation_min_examples,
                        "allow_weak_posttrain_validation": stage_config.allow_weak_posttrain_validation,
                    },
                },
            )
        train_loader = build_dataloader(train_dataset, batch_size=stage_config.micro_batch_size, shuffle=True)
        val_loader = None
        if validation_dataset is not None and len(validation_dataset) > 0:
            val_loader = build_dataloader(
                validation_dataset, batch_size=stage_config.micro_batch_size, shuffle=False
            )
        steps_per_epoch = max(
            1,
            math.ceil(len(train_loader) / max(stage_config.gradient_accumulation_steps, 1)),
        )
        eval_interval = max(1, math.ceil(steps_per_epoch / max(stage_config.sft_evals_per_epoch, 1)))
        early_eval_step = min(10, eval_interval)
        eval_history_path = Path(stage_config.checkpoint.output_dir) / "eval_history.jsonl"

        def _eval_payload_callback(model, _step: int, _final_eval: bool, _state, _metrics):
            return {
                "qualitative_samples": generate_qualitative_samples(
                    model,
                    data_config.tokenizer_path,
                    regression_path="data/eval/posttrain_regression.jsonl",
                    max_new_tokens=128,
                    temperature=0.0,
                    top_p=1.0,
                )
            }

        run_training(
            model=model,
            train_loader=train_loader,
            train_config=stage_config,
            checkpoint_manager=checkpoint_manager,
            optimizer=optimizer,
            scheduler=scheduler,
            val_loader=val_loader,
            resume_from=stage_config.checkpoint.resume_from,
            best_checkpoint_name="best" if val_loader is not None else None,
            eval_payload_callback=_eval_payload_callback if val_loader is not None else None,
            eval_fn=_evaluate_sft_validation if val_loader is not None else None,
            eval_control=(
                EvalControl(
                    stage_name="sft",
                    evaluate_at_start=True,
                    early_eval_step=early_eval_step,
                    eval_interval_steps=eval_interval,
                    validation_max_batches=None,
                    best_min_delta=stage_config.sft_best_min_delta,
                    early_stopping_patience_evals=stage_config.sft_early_stopping_patience_evals,
                    overfit_train_loss_threshold=0.05,
                    overfit_worsening_patience=2,
                    train_dataset_size=len(train_dataset),
                    validation_dataset_size=len(validation_dataset),
                    steps_per_epoch=steps_per_epoch,
                    eval_history_path=str(eval_history_path),
                )
                if val_loader is not None
                else None
            ),
            save_final_checkpoint=True,
        )
    finally:
        cleanup_distributed()
