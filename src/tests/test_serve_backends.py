from __future__ import annotations

import sys
from types import SimpleNamespace

from config import ServeConfig
from serve.backends.transformers_backend import TransformersChatBackend


class _FakeTensor:
    def __init__(self, values: list[int]):
        self.values = list(values)

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self.values))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            _row, column = key
            if isinstance(column, slice):
                return _FakeTensor(self.values[column])
            return self.values[column]
        return self.values[key]


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    def __init__(self):
        self.model_max_length = 16
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.unk_token_id = 1

    def __call__(self, _prompt, return_tensors="pt"):
        assert return_tensors == "pt"
        return _FakeBatch(
            {
                "input_ids": _FakeTensor(list(range(20))),
                "attention_mask": _FakeTensor([1] * 20),
            }
        )

    def convert_tokens_to_ids(self, _token):
        return None

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "decoded output"


class _FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=16)
        self.generation_config = SimpleNamespace(do_sample=False, temperature=0.0, top_p=1.0, top_k=50)
        self.last_generate_kwargs = None

    def to(self, _device):
        return self

    def eval(self):
        return None

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        prompt_len = kwargs["input_ids"].shape[1]
        total_len = prompt_len + kwargs["max_new_tokens"]
        return _FakeTensor(list(range(total_len)))


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorchModule:
    def __init__(self):
        self.cuda = SimpleNamespace(is_available=lambda: False)

    def device(self, name: str) -> str:
        return name

    def no_grad(self):
        return _FakeNoGrad()


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        return _FakeTokenizer()


class _FakeAutoModelForCausalLM:
    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        return _FakeModel()


def test_transformers_backend_truncates_prompt_and_sanitizes_generation_config(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorchModule())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=_FakeAutoTokenizer,
            AutoModelForCausalLM=_FakeAutoModelForCausalLM,
        ),
    )

    backend = TransformersChatBackend(ServeConfig(checkpoint_path="mock-checkpoint", max_model_len=16))
    text = backend.generate("long prompt", max_tokens=8, temperature=0.0, top_p=1.0)

    assert text == "decoded output"
    assert backend.model.last_generate_kwargs is not None
    assert backend.model.last_generate_kwargs["input_ids"].shape[1] == 8
    assert backend.model.last_generate_kwargs["max_new_tokens"] == 8
    generation_config = backend.model.last_generate_kwargs["generation_config"]
    assert generation_config.temperature is None
    assert generation_config.top_p is None
    assert generation_config.top_k is None
