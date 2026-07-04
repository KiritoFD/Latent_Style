"""B2 POC eval-only launcher.

Skips training; directly invokes full eval on a given checkpoint using the
same code path as src/run.py:_run_full_eval_for_checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import importlib

run_module = importlib.import_module("run")
config_schema = importlib.import_module("config_schema")

ExperimentConfig = config_schema.ExperimentConfig
load_config = config_schema.load_config

CONFIG_PATH = ROOT / "configs" / "620_spectral_poc.json"
CKPT_PATH = ROOT / "exp" / "620_spectral_poc" / "epoch_0008.pt"


def main() -> None:
    raw = load_config(str(CONFIG_PATH))
    config = ExperimentConfig.from_mapping(raw)
    # Make sure training section reflects runtime config (merge raw model/bridge/training/data/checkpoint)
    for section_name in ("model", "bridge", "training", "data", "checkpoint"):
        raw_section = raw.get(section_name, {})
        section_obj = getattr(config, section_name, None)
        if not isinstance(raw_section, dict) or section_obj is None:
            continue
        for key, value in raw_section.items():
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
    print(f"[b2_eval] config loaded, contract_family={getattr(config.model, 'contract_family', '?')}")
    print(f"[b2_eval] invoking full eval on {CKPT_PATH}")
    result = run_module._run_full_eval_for_checkpoint(config, CKPT_PATH)
    print(f"[b2_eval] done. convergence_payload={result}")


if __name__ == "__main__":
    main()
