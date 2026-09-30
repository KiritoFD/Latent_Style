from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _variant(base: dict, *, axis_source: str, exp_name: str) -> dict:
    payload = copy.deepcopy(base)
    bridge = payload.setdefault("bridge", {})
    if not isinstance(bridge, dict):
        raise TypeError("Config field 'bridge' must be a JSON object")
    bridge["terminal_swd_mode"] = "standard"
    bridge["terminal_swd_axis_source"] = axis_source

    checkpoint = payload.setdefault("checkpoint", {})
    if not isinstance(checkpoint, dict):
        raise TypeError("Config field 'checkpoint' must be a JSON object")
    checkpoint["save_dir"] = str(Path("./exp") / exp_name)

    experiment = payload.setdefault("experiment", {})
    if isinstance(experiment, dict):
        experiment["name"] = exp_name
        experiment["notes"] = (
            f"SA-SWD projection-axis ablation generated from base config. "
            f"terminal_swd_axis_source={axis_source}."
        )
    ablation = payload.setdefault("ablation", {})
    if isinstance(ablation, dict):
        ablation["name"] = exp_name
        ablation["stage"] = f"saswd_axis_{axis_source}"
        ablation["notes"] = (
            f"Matched SA-SWD projection-axis ablation generated from base config. "
            f"terminal_swd_axis_source={axis_source}."
        )
    payload["output_dir"] = str(Path("exp") / exp_name)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Create matched Random-SWD vs SA-SWD ablation configs.")
    parser.add_argument("--base", required=True, type=Path, help="Base experiment config JSON.")
    parser.add_argument("--out-dir", default=Path("configs"), type=Path, help="Directory for generated configs.")
    parser.add_argument("--prefix", default=None, help="Output name prefix. Defaults to base filename stem.")
    args = parser.parse_args()

    base = _load_json(args.base)
    prefix = args.prefix or args.base.stem
    outputs = {
        "semantic": args.out_dir / f"{prefix}_saswd_semantic.json",
        "random": args.out_dir / f"{prefix}_saswd_random.json",
    }
    for axis_source, path in outputs.items():
        _write_json(path, _variant(base, axis_source=axis_source, exp_name=path.stem))
        print(path)


if __name__ == "__main__":
    main()
