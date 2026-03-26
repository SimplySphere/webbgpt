from importlib import import_module

__all__ = ["CausalLMOutput", "CausalTransformer"]


def __getattr__(name: str):
    if name in {"CausalLMOutput", "CausalTransformer"}:
        module = import_module("model.transformer")
        return getattr(module, name)
    raise AttributeError(name)
