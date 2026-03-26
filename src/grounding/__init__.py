__all__ = ["CatalogStore", "GroundingProvider"]


def __getattr__(name: str):
    if name == "CatalogStore":
        from grounding.store import CatalogStore

        return CatalogStore
    if name == "GroundingProvider":
        from grounding.provider import GroundingProvider

        return GroundingProvider
    raise AttributeError(name)
