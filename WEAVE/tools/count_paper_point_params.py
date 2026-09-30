from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import build_model_from_config, count_parameters  # noqa: E402


def load_config_from_path(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint missing config: {path}")
    return cfg


def count_from_config(config: dict[str, Any]) -> int:
    model_cfg = config.get("model", config)
    model = build_model_from_config(model_cfg, use_checkpointing=False)
    return int(count_parameters(model))


def main() -> int:
    parser = argparse.ArgumentParser(description="Count trainable parameters for paper-facing LBM configs or checkpoints.")
    parser.add_argument(
        "--item",
        action="append",
        default=[],
        help="Entry in the form 'Label=path/to/config_or_checkpoint'.",
    )
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    if not args.item:
        raise ValueError("At least one --item is required")

    rows = []
    for raw in args.item:
        if "=" not in str(raw):
            raise ValueError(f"Invalid --item: {raw}")
        label, path_raw = str(raw).split("=", 1)
        path = Path(path_raw.strip())
        if not path.is_absolute():
            path = (ROOT.parent / path).resolve()
        cfg = load_config_from_path(path)
        params = count_from_config(cfg)
        rows.append(
            {
                "label": label.strip(),
                "source_path": str(path),
                "parameter_count": params,
                "parameter_count_millions": params / 1_000_000.0,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(args.output_csv)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
