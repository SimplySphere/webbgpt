import pytest


torch = pytest.importorskip("torch")

from config import ModelConfig
from model.modules import build_attention_mask
from model.transformer import CausalTransformer


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )


def test_forward_shapes():
    model = CausalTransformer(_tiny_config())
    input_ids = torch.randint(0, 128, (2, 16))
    outputs = model(input_ids=input_ids, labels=input_ids)
    assert outputs.logits.shape == (2, 16, 128)
    assert outputs.loss is not None


def test_generation_cache_extends_sequence():
    model = CausalTransformer(_tiny_config())
    input_ids = torch.randint(0, 128, (1, 8))
    output = model.generate(input_ids=input_ids, max_new_tokens=4, temperature=0.0)
    assert output.shape[-1] >= input_ids.shape[-1]


def test_build_attention_mask_shape():
    mask = torch.ones(2, 10, dtype=torch.long)
    additive = build_attention_mask(mask, query_length=4, key_length=10, device=mask.device, dtype=torch.float32)
    assert additive.shape == (2, 1, 4, 10)

