from __future__ import annotations

import contextlib
import json
import math
import sys
import time
from pathlib import Path

from config import DataConfig, ModelConfig, TrainConfig
from data.dataset import DatasetBuilder
from model.transformer import CausalTransformer
from posttrain.eval import (
    append_eval_history,
    ensure_no_regression_prompt_overlap,
    generate_qualitative_samples,
    update_topk_candidates,
    write_selection_metadata,
)
from repro import seed_everything
from train.checkpoint import CheckpointManager
from train.distributed import barrier, cleanup_distributed, init_distributed, is_main_process, maybe_wrap_fsdp
from train.entrypoints import snapshot_configs
from train.loop import build_dataloader, evaluate_language_model, save_run_metadata
from train.optim import build_optimizer, build_scheduler


def _require_torch():
    import torch

    return torch


def _sequence_log_probs(model, input_ids, attention_mask):
    torch = _require_torch()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:, 1:].to(token_log_probs.dtype)
    return (token_log_probs * mask).sum(dim=-1)


def _apply_dpo_overrides(train_config: TrainConfig) -> TrainConfig:
    stage_config = TrainConfig.from_dict(train_config.to_dict())
    if stage_config.dpo_learning_rate is not None:
        stage_config.learning_rate = stage_config.dpo_learning_rate
    if stage_config.dpo_min_learning_rate is not None:
        stage_config.min_learning_rate = stage_config.dpo_min_learning_rate
    if stage_config.dpo_warmup_steps is not None:
        stage_config.warmup_steps = stage_config.dpo_warmup_steps
    if stage_config.dpo_max_steps is not None:
        stage_config.max_steps = stage_config.dpo_max_steps
    return stage_config


def evaluate_dpo_model(policy_model, reference_model, dataloader, max_batches: int, beta: float) -> dict[str, float]:
    torch = _require_torch()
    device = next(policy_model.parameters()).device
    policy_training = bool(getattr(policy_model, "training", False))
    policy_model.eval()
    reference_model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0.0, device=device)
    total_margin = torch.tensor(0.0, device=device)
    total_examples = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)

            policy_chosen = _sequence_log_probs(policy_model, chosen_input_ids, chosen_attention_mask)
            policy_rejected = _sequence_log_probs(policy_model, rejected_input_ids, rejected_attention_mask)
            ref_chosen = _sequence_log_probs(reference_model, chosen_input_ids, chosen_attention_mask)
            ref_rejected = _sequence_log_probs(reference_model, rejected_input_ids, rejected_attention_mask)
            logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
            losses = -torch.nn.functional.logsigmoid(logits)
            margins = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
            batch_size = torch.tensor(float(chosen_input_ids.size(0)), device=device)
            total_loss += losses.sum()
            total_correct += (margins > 0).to(torch.float32).sum()
            total_margin += margins.sum()
            total_examples += batch_size
    dist = getattr(torch, "distributed", None)
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_margin, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_examples, op=dist.ReduceOp.SUM)
    total = max(float(total_examples.item()), 1.0)
    policy_model.train(policy_training)
    return {
        "val_dpo_loss": float(total_loss.item() / total),
        "preference_accuracy": float(total_correct.item() / total),
        "mean_margin": float(total_margin.item() / total),
        "examples_evaluated": float(total_examples.item()),
    }


def run_dpo_job(
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    reference_checkpoint: str,
    beta: float = 0.1,
) -> None:
    torch = _require_torch()
    stage_config = _apply_dpo_overrides(train_config)
    builder = DatasetBuilder(data_config)
    train_dataset, validation_dataset = builder.build_preference_split(
        seed=stage_config.seed,
        validation_fraction=stage_config.dpo_validation_fraction,
        validation_min_examples=stage_config.dpo_validation_min_examples,
        allow_weak_validation=stage_config.allow_weak_posttrain_validation,
        require_explicit_validation=stage_config.require_explicit_dpo_validation,
    )
    train_examples = getattr(train_dataset, "examples", None)
    validation_examples = getattr(validation_dataset, "examples", None)
    if train_examples is not None and validation_examples is not None:
        ensure_no_regression_prompt_overlap(
            stage_name="dpo",
            train_examples=train_examples,
            validation_examples=validation_examples,
        )
    elif train_examples is None or validation_examples is None:
        print(
            "WebbGPT: skipping DPO regression-prompt overlap guard because prepared datasets do not expose raw prompt metadata in v1.",
            file=sys.stderr,
            flush=True,
        )
    init_distributed()
    try:
        seed_bundle = seed_everything(stage_config.seed)
        policy_model = CausalTransformer(model_config)
        reference_model = CausalTransformer(model_config)
        checkpoint_manager = CheckpointManager(
            output_dir=stage_config.checkpoint.output_dir,
            keep_last_n=stage_config.checkpoint.keep_last_n,
        )
        checkpoint_manager.load(reference_checkpoint, reference_model, strict=True)
        policy_model = maybe_wrap_fsdp(policy_model, stage_config)
        reference_model = reference_model.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad_(False)
        optimizer = build_optimizer(policy_model, stage_config)
        scheduler = build_scheduler(optimizer, stage_config)
        if stage_config.checkpoint.resume_from:
            loaded = checkpoint_manager.load(
                stage_config.checkpoint.resume_from,
                policy_model,
                optimizer=optimizer,
                scheduler=scheduler,
                strict=True,
            )
            step = loaded.step
            train_state = loaded.payload.get("extra_state", {}).get("train_state", {})
            best_eval_loss = train_state.get("best_eval_loss", math.inf)
            best_eval_step = train_state.get("best_eval_step", -1)
            best_preference_accuracy = train_state.get("best_preference_accuracy", float("-inf"))
            best_mean_margin = train_state.get("best_mean_margin", float("-inf"))
            examples_seen = int(train_state.get("examples_seen", 0))
        else:
            checkpoint_manager.load(reference_checkpoint, policy_model, strict=True)
            step = 0
            best_eval_loss = math.inf
            best_eval_step = -1
            best_preference_accuracy = float("-inf")
            best_mean_margin = float("-inf")
            examples_seen = 0

        if is_main_process():
            snapshot_configs(model_config, data_config, stage_config)
            save_run_metadata(
                stage_config,
                stage_config.checkpoint.output_dir,
                extra={
                    "seed_bundle": seed_bundle,
                    "validation_policy": {
                        "require_explicit_validation": stage_config.require_explicit_dpo_validation,
                        "validation_min_examples": stage_config.dpo_validation_min_examples,
                        "allow_weak_posttrain_validation": stage_config.allow_weak_posttrain_validation,
                    },
                },
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_model = policy_model.to(device)
        train_loader = build_dataloader(train_dataset, batch_size=stage_config.micro_batch_size, shuffle=True)
        val_loader = None
        if validation_dataset is not None and len(validation_dataset) > 0:
            val_loader = build_dataloader(
                validation_dataset, batch_size=stage_config.micro_batch_size, shuffle=False
            )
        lm_health_loader = None
        if stage_config.dpo_enable_lm_health_eval and data_config.validation_sources:
            lm_health_loader = build_dataloader(
                builder.build_validation(),
                batch_size=stage_config.micro_batch_size,
                shuffle=False,
            )
        steps_per_epoch = max(
            1,
            math.ceil(len(train_loader) / max(stage_config.gradient_accumulation_steps, 1)),
        )
        eval_interval = max(1, math.ceil(steps_per_epoch / max(stage_config.dpo_evals_per_epoch, 1)))
        early_eval_step = min(10, eval_interval)
        eval_history_path = Path(stage_config.checkpoint.output_dir) / "eval_history.jsonl"
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if stage_config.use_bf16 and torch.cuda.is_available()
            else contextlib.nullcontext()
        )
        last_saved_step = -1
        last_eval_step = -1
        last_lm_health_loss = math.nan
        no_improvement_evals = 0
        lm_health_worsening_evals = 0
        should_stop_training = False
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0

        def _approx_epoch(current_step: int, *, initial_eval: bool, final_eval: bool) -> float:
            if initial_eval:
                return 0.0
            if final_eval:
                return step / float(steps_per_epoch)
            return (current_step + 1) / float(steps_per_epoch)

        def _is_better_dpo_checkpoint(metrics: dict[str, float]) -> tuple[bool, str]:
            loss = float(metrics["val_dpo_loss"])
            accuracy = float(metrics["preference_accuracy"])
            margin = float(metrics["mean_margin"])
            if math.isnan(loss):
                return False, "val_dpo_loss"
            if math.isinf(best_eval_loss) or loss < best_eval_loss - stage_config.dpo_best_min_delta:
                return True, "val_dpo_loss"
            if abs(loss - best_eval_loss) <= stage_config.dpo_best_min_delta:
                if accuracy > best_preference_accuracy:
                    return True, "preference_accuracy"
                if accuracy == best_preference_accuracy and margin > best_mean_margin:
                    return True, "mean_margin"
            return False, "val_dpo_loss"

        def _run_eval(current_step: int, *, final_eval: bool, initial_eval: bool = False) -> None:
            nonlocal best_eval_loss
            nonlocal best_eval_step
            nonlocal best_preference_accuracy
            nonlocal best_mean_margin
            nonlocal last_eval_step
            nonlocal last_lm_health_loss
            nonlocal no_improvement_evals
            nonlocal lm_health_worsening_evals
            nonlocal should_stop_training
            metrics = evaluate_dpo_model(
                policy_model,
                reference_model,
                val_loader,
                len(val_loader) if val_loader is not None else 0,
                beta=beta,
            )
            payload: dict[str, object] = {"step": current_step, "eval": metrics}
            if final_eval:
                payload["final_eval"] = True
            if initial_eval:
                payload["initial_eval"] = True
            payload["approx_epoch"] = _approx_epoch(
                current_step, initial_eval=initial_eval, final_eval=final_eval
            )
            payload["train_dataset_size"] = len(train_dataset)
            payload["validation_dataset_size"] = len(validation_dataset)
            payload["train_examples_seen"] = examples_seen
            payload["validation_examples_evaluated"] = int(metrics["examples_evaluated"])
            payload["qualitative_samples"] = generate_qualitative_samples(
                policy_model,
                data_config.tokenizer_path,
                regression_path="data/eval/posttrain_regression.jsonl",
                max_new_tokens=128,
                temperature=0.0,
                top_p=1.0,
            )
            lm_health_metrics = None
            if lm_health_loader is not None:
                lm_health_metrics = evaluate_language_model(
                    policy_model,
                    lm_health_loader,
                    stage_config.num_eval_batches,
                )
                payload["lm_health"] = lm_health_metrics
            improved, selection_metric = _is_better_dpo_checkpoint(metrics)
            previous_best_value = None if math.isinf(best_eval_loss) else best_eval_loss
            previous_best_step = best_eval_step if best_eval_step >= 0 else None
            previous_selection_value = {
                "val_dpo_loss": previous_best_value,
                "preference_accuracy": (
                    None if best_preference_accuracy == float("-inf") else best_preference_accuracy
                ),
                "mean_margin": None if best_mean_margin == float("-inf") else best_mean_margin,
            }.get(selection_metric)
            if improved:
                best_eval_loss = float(metrics["val_dpo_loss"])
                best_preference_accuracy = float(metrics["preference_accuracy"])
                best_mean_margin = float(metrics["mean_margin"])
                best_eval_step = current_step
                no_improvement_evals = 0
            else:
                no_improvement_evals += 1
            if lm_health_metrics is not None and not math.isnan(last_lm_health_loss):
                if float(lm_health_metrics["loss"]) > last_lm_health_loss:
                    lm_health_worsening_evals += 1
                else:
                    lm_health_worsening_evals = 0
            elif lm_health_metrics is not None:
                lm_health_worsening_evals = 0
            if lm_health_metrics is not None:
                last_lm_health_loss = float(lm_health_metrics["loss"])
            payload["best_step_so_far"] = best_eval_step
            if is_main_process():
                print(json.dumps(payload))
                append_eval_history(eval_history_path, payload)
            if improved:
                barrier()
                if is_main_process():
                    best_path = checkpoint_manager.save_named(
                        "best",
                        step=current_step,
                        model=policy_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        extra_state={
                            "dpo": {"beta": beta},
                            "train_state": {
                                "best_eval_loss": best_eval_loss,
                                "best_eval_step": best_eval_step,
                                "best_preference_accuracy": best_preference_accuracy,
                                "best_mean_margin": best_mean_margin,
                                "examples_seen": examples_seen,
                            },
                        },
                    )
                    selection_payload = {
                        "stage": "dpo",
                        "step": current_step,
                        "approx_epoch": payload["approx_epoch"],
                        "train_dataset_size": len(train_dataset),
                        "validation_dataset_size": len(validation_dataset),
                        "train_examples_seen": payload["train_examples_seen"],
                        "validation_examples_evaluated": payload["validation_examples_evaluated"],
                        "metrics": metrics,
                        "lm_health": lm_health_metrics,
                        "selection_metric": selection_metric,
                        "selection_value": float(metrics[selection_metric]),
                        "previous_best_value": previous_selection_value,
                        "previous_best_step": previous_best_step,
                        "replacement_reason": (
                            f"new best by {selection_metric}" if previous_selection_value is not None else "first best checkpoint"
                        ),
                        "improvement_delta": (
                            None
                            if previous_selection_value is None
                            else (
                                previous_selection_value - float(metrics[selection_metric])
                                if selection_metric == "val_dpo_loss"
                                else float(metrics[selection_metric]) - previous_selection_value
                            )
                        ),
                        "best_step_so_far": best_eval_step,
                        "qualitative_samples": payload["qualitative_samples"],
                    }
                    write_selection_metadata(best_path, selection_payload)
                    candidate_path = checkpoint_manager.save_named(
                        f"candidate-step-{current_step:08d}",
                        step=current_step,
                        model=policy_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        extra_state={
                            "dpo": {"beta": beta},
                            "train_state": {
                                "best_eval_loss": best_eval_loss,
                                "best_eval_step": best_eval_step,
                                "best_preference_accuracy": best_preference_accuracy,
                                "best_mean_margin": best_mean_margin,
                                "examples_seen": examples_seen,
                            },
                        },
                    )
                    write_selection_metadata(candidate_path, selection_payload)
                    update_topk_candidates(
                        checkpoint_manager.output_dir,
                        candidate_path=candidate_path,
                        candidate_payload=selection_payload,
                        metric_key="val_dpo_loss",
                        limit=stage_config.posttrain_top_k_checkpoints,
                        lower_is_better=True,
                    )
                barrier()
            if no_improvement_evals >= stage_config.dpo_early_stopping_patience_evals:
                should_stop_training = True
                if is_main_process():
                    print(
                        f"WebbGPT: stopping dpo early after {no_improvement_evals} validation evals without improvement.",
                        file=sys.stderr,
                        flush=True,
                    )
            if (
                lm_health_metrics is not None
                and no_improvement_evals >= 2
                and lm_health_worsening_evals >= 2
            ):
                should_stop_training = True
                if is_main_process():
                    print(
                        "WebbGPT: stopping dpo early because preference metrics plateaued while LM-health degraded.",
                        file=sys.stderr,
                        flush=True,
                    )
            last_eval_step = current_step

        if val_loader is not None:
            _run_eval(0, final_eval=False, initial_eval=True)

        while step < stage_config.max_steps and not should_stop_training:
            for batch in train_loader:
                if step >= stage_config.max_steps or should_stop_training:
                    break
                start_time = time.perf_counter()
                chosen_input_ids = batch["chosen_input_ids"].to(device)
                rejected_input_ids = batch["rejected_input_ids"].to(device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(device)

                with autocast_context:
                    policy_chosen = _sequence_log_probs(policy_model, chosen_input_ids, chosen_attention_mask)
                    policy_rejected = _sequence_log_probs(
                        policy_model, rejected_input_ids, rejected_attention_mask
                    )
                    with torch.no_grad():
                        ref_chosen = _sequence_log_probs(
                            reference_model, chosen_input_ids, chosen_attention_mask
                        )
                        ref_rejected = _sequence_log_probs(
                            reference_model, rejected_input_ids, rejected_attention_mask
                        )
                    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
                    loss = -torch.nn.functional.logsigmoid(logits).mean()

                batch_examples = int(chosen_input_ids.size(0))
                examples_seen += batch_examples
                micro_step += 1
                scaled_loss = loss / max(stage_config.gradient_accumulation_steps, 1)
                scaled_loss.backward()

                if micro_step % max(stage_config.gradient_accumulation_steps, 1) != 0:
                    continue

                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), stage_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if step % stage_config.log_every_steps == 0 and is_main_process():
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "loss": float(loss.item()),
                                "lr": float(scheduler.get_last_lr()[0]),
                                "train_examples_seen": examples_seen,
                                "step_time_sec": time.perf_counter() - start_time,
                            }
                        )
                    )
                if val_loader is not None and (
                    (step > 0 and step == early_eval_step)
                    or (step > 0 and step % eval_interval == 0)
                ):
                    _run_eval(step, final_eval=False)
                if (
                    stage_config.checkpoint.save_every_steps > 0
                    and step > 0
                    and step % stage_config.checkpoint.save_every_steps == 0
                ):
                    if is_main_process():
                        checkpoint_manager.save(
                            step=step,
                            model=policy_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            extra_state={
                                "dpo": {"beta": beta},
                                "train_state": {
                                    "best_eval_loss": best_eval_loss,
                                    "best_eval_step": best_eval_step,
                                    "best_preference_accuracy": best_preference_accuracy,
                                    "best_mean_margin": best_mean_margin,
                                    "examples_seen": examples_seen,
                                },
                            },
                        )
                        last_saved_step = step
                step += 1
        if val_loader is not None and step > 0 and last_eval_step != step:
            _run_eval(step, final_eval=True)
        if step > 0 and step != last_saved_step and is_main_process():
            checkpoint_manager.save(
                step=step,
                model=policy_model,
                optimizer=optimizer,
                scheduler=scheduler,
                extra_state={
                    "dpo": {"beta": beta},
                    "train_state": {
                        "best_eval_loss": best_eval_loss,
                        "best_eval_step": best_eval_step,
                        "best_preference_accuracy": best_preference_accuracy,
                        "best_mean_margin": best_mean_margin,
                        "examples_seen": examples_seen,
                    },
                },
            )
    finally:
        cleanup_distributed()
