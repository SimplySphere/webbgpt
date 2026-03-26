from importlib import import_module

__all__ = ["run_dpo_job", "run_sft_job"]


def __getattr__(name: str):
    if name in {"run_dpo_job", "run_sft_job"}:
        module_name = "posttrain.dpo" if name == "run_dpo_job" else "posttrain.sft"
        module = import_module(module_name)
        return getattr(module, name)
    raise AttributeError(name)
