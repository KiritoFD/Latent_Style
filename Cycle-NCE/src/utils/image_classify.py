from __future__ import annotations

import importlib.util
from pathlib import Path

_THIS = Path(__file__).resolve()
_IMPL = _THIS.with_name("image-classify.py")

spec = importlib.util.spec_from_file_location("utils_image_classify_impl", _IMPL)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load classifier module from {_IMPL}")
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

build_model = _mod.build_model
EvalImageClassifier = _mod.EvalImageClassifier
load_eval_image_classifier = _mod.load_eval_image_classifier
write_report_json = _mod.write_report_json

__all__ = [
    "build_model",
    "EvalImageClassifier",
    "load_eval_image_classifier",
    "write_report_json",
]
