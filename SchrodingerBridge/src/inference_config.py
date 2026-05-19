from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config_schema import ExperimentConfig


_DEFAULTS_PATH = Path(__file__).resolve().with_name("inference_config.json")


def _config_dict(config: dict[str, Any] | ExperimentConfig | None) -> dict[str, Any]:
    if isinstance(config, ExperimentConfig):
        return config.to_dict()
    if isinstance(config, dict):
        return config
    return {}


@lru_cache(maxsize=1)
def load_inference_defaults() -> dict[str, Any]:
    if not _DEFAULTS_PATH.exists():
        return {}
    try:
        return json.loads(_DEFAULTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_inference_section(config: dict[str, Any] | None) -> dict[str, Any]:
    defaults = dict(load_inference_defaults().get("inference", {}) or {})
    config_dict = _config_dict(config)
    if not config_dict:
        return defaults
    local = config_dict.get("inference", {}) or {}
    if isinstance(local, dict):
        defaults.update(local)
    return defaults


def resolve_full_eval_section(config: dict[str, Any] | None) -> dict[str, Any]:
    defaults = dict(load_inference_defaults().get("full_eval", {}) or {})
    config_dict = _config_dict(config)
    if not config_dict:
        return defaults
    training = config_dict.get("training", {}) or {}
    if isinstance(training, dict):
        mapping = {
            "num_steps": "full_eval_num_steps",
            "step_size": "full_eval_step_size",
            "style_strength": "full_eval_style_strength",
            "batch_size": "full_eval_batch_size",
            "max_src_samples": "full_eval_max_src_samples",
            "max_ref_compare": "full_eval_max_ref_compare",
            "max_ref_cache": "full_eval_max_ref_cache",
            "ref_feature_batch_size": "full_eval_ref_feature_batch_size",
        }
        for dst_key, src_key in mapping.items():
            if src_key in training and training.get(src_key) is not None:
                defaults[dst_key] = training.get(src_key)
    local = config_dict.get("full_eval", {}) or {}
    if isinstance(local, dict):
        defaults.update(local)
    return defaults


def compact_runtime_config(config: dict[str, Any] | None) -> dict[str, Any]:
    config_dict = _config_dict(config)
    if not config_dict:
        return {}

    compact = copy.deepcopy(config_dict)
    infer_defaults = dict(load_inference_defaults().get("inference", {}) or {})
    full_eval_defaults = dict(load_inference_defaults().get("full_eval", {}) or {})

    infer_local = compact.get("inference")
    if isinstance(infer_local, dict):
        pruned_infer = {k: v for k, v in infer_local.items() if infer_defaults.get(k) != v}
        if pruned_infer:
            compact["inference"] = pruned_infer
        else:
            compact.pop("inference", None)

    full_eval_local = compact.get("full_eval")
    if isinstance(full_eval_local, dict):
        pruned_full_eval = {k: v for k, v in full_eval_local.items() if full_eval_defaults.get(k) != v}
        if pruned_full_eval:
            compact["full_eval"] = pruned_full_eval
        else:
            compact.pop("full_eval", None)

    training = compact.get("training")
    if isinstance(training, dict):
        mapping = {
            "full_eval_num_steps": "num_steps",
            "full_eval_step_size": "step_size",
            "full_eval_style_strength": "style_strength",
            "full_eval_batch_size": "batch_size",
            "full_eval_max_src_samples": "max_src_samples",
            "full_eval_max_ref_compare": "max_ref_compare",
            "full_eval_max_ref_cache": "max_ref_cache",
            "full_eval_ref_feature_batch_size": "ref_feature_batch_size",
        }
        for train_key, default_key in mapping.items():
            if train_key in training and full_eval_defaults.get(default_key) == training.get(train_key):
                training.pop(train_key, None)

    return compact
