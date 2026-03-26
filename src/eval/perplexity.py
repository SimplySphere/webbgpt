from __future__ import annotations

from train.loop import evaluate_language_model


def evaluate_perplexity(model, dataloader, max_batches: int = 50) -> dict[str, float]:
    return evaluate_language_model(model, dataloader, max_batches=max_batches)

