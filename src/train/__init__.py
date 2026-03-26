from importlib import import_module

__all__ = ["run_continued_pretraining", "run_pretraining"]


def __getattr__(name: str):
    if name in {"run_continued_pretraining", "run_pretraining"}:
        module = import_module("train.entrypoints")
        return getattr(module, name)
    raise AttributeError(name)
