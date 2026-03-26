from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from config import ModelConfig
from model.transformer import CausalTransformer
from train.checkpoint import CheckpointManager


def test_checkpoint_roundtrip(tmp_path: Path):
    model = CausalTransformer(
        ModelConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(str(tmp_path))
    checkpoint_dir = manager.save(step=5, model=model, optimizer=optimizer, extra_state={"hello": "world"})
    loaded = manager.load(str(checkpoint_dir), model, optimizer=optimizer)
    assert loaded.step == 5
    assert loaded.payload["extra_state"]["hello"] == "world"


def test_checkpoint_overwrites_existing_step_directory(tmp_path: Path):
    model = CausalTransformer(
        ModelConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(str(tmp_path))
    checkpoint_dir = manager.save(step=10, model=model, optimizer=optimizer, extra_state={"run": "first"})
    stale_file = checkpoint_dir / "stale.txt"
    stale_file.write_text("stale")

    checkpoint_dir = manager.save(step=10, model=model, optimizer=optimizer, extra_state={"run": "second"})
    assert checkpoint_dir.exists()
    assert not stale_file.exists()
    loaded = manager.load(str(checkpoint_dir), model, optimizer=optimizer)
    assert loaded.payload["extra_state"]["run"] == "second"


def test_checkpoint_save_named_overwrites_existing_named_directory(tmp_path: Path):
    model = CausalTransformer(
        ModelConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(str(tmp_path))
    best_dir = manager.save_named("best", step=12, model=model, optimizer=optimizer, extra_state={"run": "first"})
    stale_file = best_dir / "stale.txt"
    stale_file.write_text("stale")

    best_dir = manager.save_named("best", step=18, model=model, optimizer=optimizer, extra_state={"run": "second"})

    assert best_dir.exists()
    assert not stale_file.exists()
    loaded = manager.load(str(best_dir), model, optimizer=optimizer)
    assert loaded.step == 18
    assert loaded.payload["extra_state"]["run"] == "second"
