from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SERIES_NAME = "TextureTearer3"


def _load_base_config(src_dir: Path, base_config_arg: str | None) -> tuple[dict, Path]:
    if base_config_arg:
        base_path = Path(base_config_arg).expanduser()
        if not base_path.is_absolute():
            base_path = (src_dir / base_path).resolve()
    else:
        candidates = [
            (src_dir / "config.json").resolve(),
            (src_dir / "config_A02_ResOn_None_Swin4.json").resolve(),
        ]
        base_path = next((p for p in candidates if p.exists()), candidates[-1])
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    with open(base_path, "r", encoding="utf-8-sig") as f:
        return json.load(f), base_path


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


EXPERIMENTS: list[tuple[str, dict]] = [
    (
        "T01_ResOn_None_Swin4_Noise",
        {
            "model": {
                "ablation_no_residual": False,
                "skip_routing_mode": "none",
                "num_decoder_blocks": 2,
                "decoder_block_type": "window_attn",
                "input_anchor_noise_std": 0.05,
                "input_anchor_noise_eval": False,
            },
            "loss": {
                "w_identity": 5.0,
            },
            "training": {
                "batch_size": 256,
                "use_gradient_checkpointing": True,
            },
        },
    ),
    (
        "T02_ResOff_Adapt_Conv1_LowIDT",
        {
            "model": {
                "ablation_no_residual": True,
                "skip_routing_mode": "adaptive",
                "num_decoder_blocks": 1,
                "decoder_block_type": "conv",
                "residual_gain": 2.0,
            },
            "loss": {
                "w_identity": 1.0,
            },
            "training": {
                "batch_size": 288,
                "use_gradient_checkpointing": False,
            },
        },
    ),
    (
        "T03_ResOff_Adapt_Conv1_HFSWD",
        {
            "model": {
                "ablation_no_residual": True,
                "skip_routing_mode": "adaptive",
                "num_decoder_blocks": 1,
                "decoder_block_type": "conv",
            },
            "loss": {
                "w_identity": 2.0,
                "swd_patch_sizes": [3, 5, 7, 11],
                "swd_use_high_freq": True,
                "swd_hf_weight_ratio": 5.0,
            },
            "training": {
                "batch_size": 288,
                "use_gradient_checkpointing": False,
            },
        },
    ),
]


def generate(src_dir: Path, base_config_arg: str | None = None) -> list[str]:
    base, base_path = _load_base_config(src_dir, base_config_arg)
    print(f"Base config: {base_path}")
    history_dir = src_dir
    history_dir.mkdir(parents=True, exist_ok=True)

    run_ids: list[str] = []
    for name, patch in EXPERIMENTS:
        cfg = copy.deepcopy(base)
        cfg.setdefault("model", {})
        cfg.setdefault("loss", {})
        cfg.setdefault("checkpoint", {})
        cfg.setdefault("training", {})
        _deep_update(cfg, patch)
        cfg["checkpoint"]["save_dir"] = f"../{SERIES_NAME}_{name}"
        cfg_name = f"config_{name}.json"
        _write_json(history_dir / cfg_name, cfg)
        run_ids.append(name)
        print(f"generated: history_configs/{cfg_name}")

    if len(run_ids) != len(EXPERIMENTS):
        raise RuntimeError(f"Expected {len(EXPERIMENTS)} runs, got {len(run_ids)}")
    return run_ids


def _write_run_script(src_dir: Path, run_ids: list[str]) -> None:
    script_path = src_dir / f"{SERIES_NAME}.bat"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write("setlocal\n")
        f.write("cd /d %~dp0\n")
        f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")
        total = len(run_ids)
        for idx, run_id in enumerate(run_ids, start=1):
            cfg_name = f"history_configs\\config_{run_id}.json"
            f.write(f"echo [{idx}/{total}] Running {run_id}...\n")
            f.write(f"uv run run.py --config {cfg_name}\n")
            f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")
    print(f"generated: {script_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TextureTearer3 experiment suite.")
    parser.add_argument(
        "--base-config",
        type=str,
        default=None,
        help="Optional base config path. Default uses src/config.json.",
    )
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    run_ids = generate(src_dir, base_config_arg=args.base_config)
    _write_run_script(src_dir, run_ids)


if __name__ == "__main__":
    main()
