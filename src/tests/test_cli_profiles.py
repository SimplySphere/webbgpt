import subprocess
import sys
from pathlib import Path

import cli
import pytest

from cli import (
    _apply_remote_run_preset,
    _can_reuse_tokenizer_corpus,
    _can_reuse_tokenizer_model,
    _prepare_manual_continue_train_config,
    _prepare_manual_dpo_train_config,
    _prepare_manual_sft_train_config,
    _profile_config_paths,
    _stage_train_config,
    _validate_profile_hardware_fit,
    _validate_remote_profile,
)
from config import DataConfig, DataSourceConfig, EvalConfig, GroundingConfig, ModelConfig, ServeConfig, TokenizerConfig, TrainConfig


def _valid_remote_data_config() -> DataConfig:
    return DataConfig(
        tokenizer_path="artifacts/tokenizer/webbgpt.model",
        pretrain_sources=[
            DataSourceConfig(
                name="pretrain",
                format="hf",
                dataset_name="HuggingFaceFW/fineweb-edu",
                dataset_config_name="sample-10BT",
                split="train",
                streaming=True,
                skip_records=2048,
            )
        ],
        validation_sources=[
            DataSourceConfig(
                name="validation",
                format="hf",
                dataset_name="HuggingFaceFW/fineweb-edu",
                dataset_config_name="sample-10BT",
                split="train",
                streaming=True,
                max_records=2048,
            )
        ],
        continued_pretrain_sources=[
            DataSourceConfig(name="education", path="data/domain/education_corpus.txt", format="text"),
            DataSourceConfig(name="advising", path="data/domain/advising_corpus.txt", format="text"),
        ],
        sft_sources=[
            DataSourceConfig(name="public_sft", path="data/posttrain/sft_public_seed.jsonl", format="jsonl"),
            DataSourceConfig(name="domain_sft", path="data/posttrain/sft_domain_synthetic.jsonl", format="jsonl"),
        ],
        preference_sources=[
            DataSourceConfig(name="public_pref", path="data/posttrain/preference_public_seed.jsonl", format="jsonl"),
            DataSourceConfig(name="domain_pref", path="data/posttrain/preference_domain_synthetic.jsonl", format="jsonl"),
        ],
    )


def _valid_remote_eval_config() -> EvalConfig:
    return EvalConfig(
        benchmark_paths=[
            "data/eval/chat_sanity.jsonl",
            "data/eval/assistant.jsonl",
            "data/eval/webb_course_present.responses",
            "data/eval/webb_course_missing.responses",
            "data/eval/webb_handbook_present.responses",
            "data/eval/webb_handbook_missing.responses",
        ],
        enforce_release_gates=True,
        grounding=GroundingConfig(
            dsn="sqlite:///artifacts/grounding/webbgpt-3b.db",
            seed_url_pack="data/webb/seed_urls_demo.json",
            handbook_url="data/webb/mock/handbook.txt",
            sync_on_start=True,
        ),
    )


def _valid_remote_serve_config() -> ServeConfig:
    return ServeConfig(
        checkpoint_path="artifacts/runs/remote-3b/export/final",
        grounding=GroundingConfig(
            dsn="sqlite:///artifacts/grounding/webbgpt-3b.db",
            seed_url_pack="data/webb/seed_urls_demo.json",
            handbook_url="data/webb/mock/handbook.txt",
        ),
    )


def test_profile_config_paths_cover_local_mvp_and_remote_7b():
    base = Path("/tmp/sample-configs")
    local = _profile_config_paths(base, "local-mvp")
    remote = _profile_config_paths(base, "remote-7b")

    assert local["model"] == base / "model-local-mvp.json"
    assert local["serve"] == base / "serve-local-mvp.json"
    assert remote["model"] == base / "model-7b.json"
    assert remote["tokenizer"] == base / "tokenizer-7b.json"


def test_validate_remote_profile_rejects_placeholder_data():
    data_config = _valid_remote_data_config()
    data_config.sft_sources[0] = DataSourceConfig(
        name="placeholder",
        path="data/local/sft.jsonl",
        format="jsonl",
    )
    with pytest.raises(RuntimeError, match="debug placeholder data"):
        _validate_remote_profile(
            TokenizerConfig(),
            ModelConfig(),
            data_config,
            _valid_remote_eval_config(),
            _valid_remote_serve_config(),
        )


def test_validate_remote_profile_rejects_overlapping_validation():
    data_config = _valid_remote_data_config()
    data_config.validation_sources[0] = DataSourceConfig(
        name="overlap",
        format="hf",
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config_name="sample-10BT",
        split="train",
        streaming=True,
        skip_records=2048,
        max_records=4096,
    )
    with pytest.raises(RuntimeError, match="must not overlap"):
        _validate_remote_profile(
            TokenizerConfig(),
            ModelConfig(),
            data_config,
            _valid_remote_eval_config(),
            _valid_remote_serve_config(),
        )


def test_stage_train_config_preserves_initialize_from_lineage():
    train_config = TrainConfig(run_name="webbgpt-3b")
    stage_config = _stage_train_config(
        train_config,
        stage_name="sft",
        output_dir="artifacts/runs/remote-3b/checkpoints/sft",
        initialize_from="artifacts/runs/remote-3b/checkpoints/continue/step-00050000",
        max_steps=20_000,
    )
    assert stage_config.run_name == "webbgpt-3b-sft"
    assert stage_config.checkpoint.output_dir.endswith("/sft")
    assert stage_config.checkpoint.initialize_from.endswith("/continue/step-00050000")
    assert stage_config.checkpoint.resume_from is None


def test_stage_train_config_applies_continue_overrides():
    train_config = TrainConfig(
        run_name="webbgpt-local-mvp",
        learning_rate=5e-4,
        min_learning_rate=5e-5,
        warmup_steps=200,
        max_steps=20_000,
        continued_learning_rate=1e-4,
        continued_min_learning_rate=1e-5,
        continued_warmup_steps=25,
        continued_max_steps=250,
    )
    stage_config = _stage_train_config(
        train_config,
        stage_name="continue",
        output_dir="artifacts/runs/local-mvp/checkpoints/continue",
        initialize_from="artifacts/runs/local-mvp/checkpoints/pretrain/step-00020000",
    )
    assert stage_config.learning_rate == 1e-4
    assert stage_config.min_learning_rate == 1e-5
    assert stage_config.warmup_steps == 25
    assert stage_config.max_steps == 250


def test_prepare_manual_continue_train_config_uses_latest_local_mvp_pretrain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    pretrain_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/pretrain/step-00020000"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    stale_continue_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/continue/step-00001050"
    stale_continue_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli.time, "strftime", lambda _: "20260324-120000")

    stage_config = _prepare_manual_continue_train_config(
        "sample-configs/model-local-mvp.json",
        "sample-configs/data-local-mvp.json",
        "sample-configs/train-local-mvp.json",
        TrainConfig(run_name="webbgpt-local-mvp"),
    )

    assert stage_config.checkpoint.initialize_from == "artifacts/runs/local-mvp/checkpoints/pretrain/step-00020000"
    assert stage_config.checkpoint.output_dir == "artifacts/runs/local-mvp/checkpoints/continue"
    assert (
        tmp_path / "artifacts/runs/local-mvp/checkpoints/continue.stale-20260324-120000/step-00001050"
    ).exists()


def test_prepare_manual_sft_train_config_uses_latest_local_mvp_continue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    continue_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/continue/step-00000145"
    continue_dir.mkdir(parents=True, exist_ok=True)
    stale_sft_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/sft/step-00000075"
    stale_sft_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli.time, "strftime", lambda _: "20260324-120100")

    stage_config = _prepare_manual_sft_train_config(
        "sample-configs/model-local-mvp.json",
        "sample-configs/data-local-mvp.json",
        "sample-configs/train-local-mvp.json",
        TrainConfig(run_name="webbgpt-local-mvp", sft_max_steps=3_000),
    )

    assert stage_config.checkpoint.initialize_from == "artifacts/runs/local-mvp/checkpoints/continue/step-00000145"
    assert stage_config.checkpoint.output_dir == "artifacts/runs/local-mvp/checkpoints/sft"
    assert stage_config.max_steps == 3_000
    assert (
        tmp_path / "artifacts/runs/local-mvp/checkpoints/sft.stale-20260324-120100/step-00000075"
    ).exists()


def test_prepare_manual_dpo_train_config_uses_stage_output_and_dpo_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    stale_dpo_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/dpo/step-00000050"
    stale_dpo_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli.time, "strftime", lambda _: "20260324-120200")

    stage_config = _prepare_manual_dpo_train_config(
        "sample-configs/model-local-mvp.json",
        "sample-configs/data-local-mvp.json",
        "sample-configs/train-local-mvp.json",
        TrainConfig(run_name="webbgpt-local-mvp", max_steps=20_000, dpo_max_steps=1_500),
        reference_checkpoint="artifacts/runs/local-mvp/checkpoints/sft/step-00003000",
    )

    assert stage_config.checkpoint.output_dir == "artifacts/runs/local-mvp/checkpoints/dpo"
    assert stage_config.max_steps == 1_500
    assert stage_config.checkpoint.initialize_from is None
    assert (
        tmp_path / "artifacts/runs/local-mvp/checkpoints/dpo.stale-20260324-120200/step-00000050"
    ).exists()


def test_apply_remote_run_preset_mvp_reduces_serious_budgets():
    data_config, train_config, eval_config = _apply_remote_run_preset(
        _valid_remote_data_config(),
        TrainConfig(run_name="webbgpt-3b", max_steps=400_000, continued_max_steps=50_000, sft_max_steps=20_000, dpo_max_steps=10_000),
        _valid_remote_eval_config(),
        preset="mvp",
    )
    assert data_config.pretraining_token_budget == 2_500_000_000
    assert data_config.continued_pretraining_token_budget == 350_000_000
    assert train_config.max_steps == 25_000
    assert train_config.continued_max_steps == 8_000
    assert train_config.sft_max_steps == 6_000
    assert train_config.dpo_max_steps == 3_000
    assert eval_config.release_gates.chat_sanity_pass_rate_min == 0.5
    assert train_config.checkpoint.save_every_steps == 250


def test_validate_profile_hardware_fit_rejects_remote_3b_on_low_memory_mac(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli, "_system_memory_bytes", lambda: 16 * 1024**3)

    with pytest.raises(RuntimeError, match="local-mvp"):
        _validate_profile_hardware_fit("remote-3b", ModelConfig())


def test_validate_profile_hardware_fit_rejects_remote_7b_on_non_linux(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli, "_system_memory_bytes", lambda: 512 * 1024**3)

    with pytest.raises(RuntimeError, match="Linux multi-GPU"):
        _validate_profile_hardware_fit(
            "remote-7b",
            ModelConfig(
                name="webbgpt-7b",
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                max_position_embeddings=4096,
            ),
        )


def test_can_reuse_tokenizer_corpus_when_config_matches(tmp_path: Path):
    config = cli.TokenizerCorpusConfig(output_path=str(tmp_path / "corpus.txt"))
    output_path = Path(config.output_path)
    output_path.write_text("hello\n")
    output_path.with_suffix(".txt.meta.json").write_text(
        cli.json.dumps({"config": config.to_dict(), "documents_written": 1})
    )

    assert _can_reuse_tokenizer_corpus(config) is True


def test_can_reuse_tokenizer_model_when_config_matches(tmp_path: Path):
    config = cli.TokenizerConfig(model_prefix=str(tmp_path / "tokenizer"))
    Path(f"{config.model_prefix}.model").write_text("stub")
    Path(f"{config.model_prefix}.vocab").write_text("stub")
    Path(f"{config.model_prefix}.tokenizer.json").write_text(cli.json.dumps(config.to_dict()))

    assert _can_reuse_tokenizer_model(config) is True


def test_main_test_command_invokes_pytest(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert check is False
        assert "env" in kwargs
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sys, "argv", ["webbgpt", "test", "-q", "src/tests/test_config.py"])
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main() == 0
    assert calls == [[sys.executable, "-m", "pytest", "-q", "src/tests/test_config.py"]]


def test_main_test_command_uses_safe_subset_when_torch_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:3] == [sys.executable, "-c", "import torch"]:
            return subprocess.CompletedProcess(cmd, 1)
        assert check is False
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sys, "argv", ["webbgpt", "test"])
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main() == 0
    assert calls[0] == [sys.executable, "-c", "import torch"]
    assert calls[1] == [sys.executable, "-m", "pytest", *cli.SAFE_DEFAULT_TEST_PATHS]
