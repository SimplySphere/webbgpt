from __future__ import annotations

import json
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


def load_config(path: str | Path, cls: type[T]) -> T:
    raw_path = Path(path)
    suffix = raw_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(raw_path.read_text())
    elif suffix in {".toml", ".tml"}:
        payload = tomllib.loads(raw_path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {raw_path.suffix}")
    return cls.from_dict(payload)


def save_config(config: Any, path: str | Path) -> None:
    raw_path = Path(path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True))


def save_payload(payload: Any, path: str | Path) -> None:
    raw_path = Path(path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
