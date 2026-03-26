__all__ = ["AssistantOrchestrator"]


def __getattr__(name: str):
    if name == "AssistantOrchestrator":
        from serve.orchestrator import AssistantOrchestrator

        return AssistantOrchestrator
    raise AttributeError(name)
