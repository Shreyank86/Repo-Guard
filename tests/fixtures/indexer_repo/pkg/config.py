"""Configuration helpers."""

import json
from pathlib import Path

DEFAULT_PATH = "config.json"


def load_config(path: str, strict: bool = False, *, encoding: str = "utf-8", **overrides) -> dict:
    """Load a JSON config file."""
    data = json.loads(Path(path).read_text(encoding=encoding))
    validate(data, strict)
    data.update(overrides)
    return data


def validate(data, strict=False):
    if strict and not data:
        raise ValueError("empty config")
    return True


def parse_args(a, b=1, /, c=2, *args, d, e: int = 3, **kwargs) -> None:
    pass
