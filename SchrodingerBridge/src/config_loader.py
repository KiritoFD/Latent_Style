from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from config_schema import ExperimentConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "_base":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: str | Path, *, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        chain = " -> ".join(str(p) for p in [*seen, path])
        raise ValueError(f"Config inheritance cycle detected: {chain}")
    seen.add(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    base_ref = raw.get("_base")
    if not base_ref:
        return raw

    base_paths = base_ref if isinstance(base_ref, list) else [base_ref]
    merged: dict[str, Any] = {}
    for item in base_paths:
        base_path = Path(item)
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        merged = _deep_merge(merged, load_config(base_path, _seen=seen.copy()))

    return _deep_merge(merged, raw)


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(load_config(config_path))
