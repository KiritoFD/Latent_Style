#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal probe for 620 spatial-bridge checkpoints.

Loads a checkpoint + config, samples source/target latent pairs from the
configured training latent cache, and runs the model at t=0, 0.5, 0.875.
Records per-layer block statistics, cross-attention/gate/FiLM statistics,
endpoint-head statistics, and endpoint alpha relative to the target latent.

Example:
    python tools/run_internal_probe.py \
        --config exp/620_spatial_bridge/620_film_v5_gated_local_smoke/config.json \
        --checkpoint exp/620_spatial_bridge/620_film_v5_gated_local_smoke/epoch_0001.pt \
        --output_dir docs/620/fog/gradient_probe \
        --num_samples_per_style 1 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Add repo/src to path so we can reuse the inference loader.
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.dataset import _load_latent_file  # type: ignore
from utils.inference import LGTInference  # type: ignore


def _to_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _latent_stats(x: torch.Tensor) -> Dict[str, float]:
    xf = x.detach().float()
    return {
        "mean": float(xf.mean()),
        "std": float(xf.std(unbiased=False)),
        "channel_std": float(xf.std(dim=(2, 3), unbiased=False).mean()),
        "dynamic_range": float((xf.amax(dim=(1, 2, 3)) - xf.amin(dim=(1, 2, 3))).mean()),
    }


def _load_latent_samples(
    data_root: str,
    style_subdirs: List[str],
    num_per_style: int = 1,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples: List[Dict[str, Any]] = []
    root = Path(data_root)
    for style_id, subdir in enumerate(style_subdirs):
        style_dir = root / subdir
        files = sorted(style_dir.glob("*.pt")) + sorted(style_dir.glob("*.npy"))
        if not files:
            raise RuntimeError(f"No latent files found for style={subdir} under {style_dir}")
        chosen = rng.choice(files, size=min(num_per_style, len(files)), replace=False)
        for f in chosen:
            latent = _load_latent_file(f)
            samples.append(
                {
                    "style_id": int(style_id),
                    "style_name": str(subdir),
                    "path": str(f),
                    "latent": latent,
                }
            )
    return samples


def _build_pairs(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pair each style's source with the next style's target (cyclic)."""
    n = len(samples)
    pairs = []
    for i in range(n):
        src = samples[i]
        tgt = samples[(i + 1) % n]
        pairs.append(
            {
                "source": src,
                "target": tgt,
            }
        )
    return pairs


def _gather_block_debug(model: torch.nn.Module) -> Dict[str, Any]:
    """Collect per-layer statistics already stored by SpatialBridgeBlock620."""
    out: Dict[str, Any] = {}
    for idx, block in enumerate(model.blocks):
        key_prefix = f"block{idx}_"
        debug = getattr(block, "last_debug", {}) or {}
        for k, v in debug.items():
            out[f"{key_prefix}{k}"] = _to_float(v)
    return out


def _run_probe(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    num_samples_per_style: int = 1,
    seed: int = 42,
    device: str | None = None,
) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config to locate data paths.
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    data_cfg = raw_config.get("data", {})
    data_root = data_cfg.get("data_root", "")
    style_subdirs = data_cfg.get("style_subdirs", [])
    if not data_root or not style_subdirs:
        raise ValueError("Config must specify data.data_root and data.style_subdirs")

    # Load checkpoint and model via the same path as evaluation.
    inf = LGTInference(
        checkpoint_path,
        device=device,
        num_steps=1,
        config_override_path=config_path,
    )
    model = inf.model
    model.eval()

    samples = _load_latent_samples(data_root, style_subdirs, num_samples_per_style, seed)
    pairs = _build_pairs(samples)

    time_points = [0.0, 0.5, 0.875]

    # We will run a single batch containing all pairs.
    source_latents = torch.stack([p["source"]["latent"] for p in pairs], dim=0).to(device)
    target_latents = torch.stack([p["target"]["latent"] for p in pairs], dim=0).to(device)

    per_sample_records: List[Dict[str, Any]] = []
    per_time_records: Dict[float, List[Dict[str, Any]]] = {t: [] for t in time_points}

    for t in time_points:
        t_tensor = torch.full((source_latents.shape[0],), t, device=device, dtype=source_latents.dtype)
        with torch.no_grad():
            _ = model(
                source_latents,
                source=source_latents,
                t=t_tensor,
                style_latent=target_latents,
                target_latent=target_latents,
            )

        block_debug = _gather_block_debug(model)

        for b in range(source_latents.shape[0]):
            record: Dict[str, Any] = {
                "t": t,
                "source_style_id": int(pairs[b]["source"]["style_id"]),
                "source_style_name": str(pairs[b]["source"]["style_name"]),
                "target_style_id": int(pairs[b]["target"]["style_id"]),
                "target_style_name": str(pairs[b]["target"]["style_name"]),
                "input_stats": _latent_stats(source_latents[b : b + 1]),
                "target_stats": _latent_stats(target_latents[b : b + 1]),
                "endpoint_head_mode": "endpoint_lowhigh"
                if _to_float(model.last_debug.get("endpoint_head_mode_lowhigh", 0.0)) > 0.5
                else "velocity",
            }
            # Global debug from model.last_debug; pick scalar values for this sample.
            for k, v in model.last_debug.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    record[k] = _to_float(v)

            # Per-layer block debug.
            for k, v in block_debug.items():
                record[k] = v

            per_time_records[t].append(record)
            per_sample_records.append(record)

    # Aggregate across samples per time point.
    aggregate: Dict[str, Any] = {}
    for t in time_points:
        agg: Dict[str, Any] = {"t": t, "count": len(per_time_records[t])}
        keys = [k for k in per_time_records[t][0].keys() if k not in {
            "source_style_id", "source_style_name", "target_style_id", "target_style_name",
            "input_stats", "target_stats", "endpoint_head_mode",
        }]
        for k in keys:
            vals = [r[k] for r in per_time_records[t] if isinstance(r.get(k), (int, float))]
            if vals:
                agg[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
        aggregate[f"t_{t}"] = agg

    payload: Dict[str, Any] = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "data_root": data_root,
        "style_subdirs": style_subdirs,
        "num_samples_per_style": num_samples_per_style,
        "seed": seed,
        "time_points": time_points,
        "aggregate": aggregate,
        "samples": per_sample_records,
    }

    out_file = output_path / f"internal_probe_{Path(checkpoint_path).stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run internal probe on a 620 checkpoint.")
    parser.add_argument("--config", required=True, help="Path to experiment config.json")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--output_dir", default="docs/620/fog/gradient_probe", help="Directory to write probe JSON")
    parser.add_argument("--num_samples_per_style", type=int, default=1, help="Samples per style")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default=None, help="torch device (cuda/cpu)")
    args = parser.parse_args()

    payload = _run_probe(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples_per_style=args.num_samples_per_style,
        seed=args.seed,
        device=args.device,
    )

    out_file = Path(args.output_dir) / f"internal_probe_{Path(args.checkpoint).stem}.json"
    print(f"Probe complete. Wrote {out_file}")
    # Print a concise summary to stdout.
    for t in payload["time_points"]:
        agg = payload["aggregate"][f"t_{t}"]
        alpha = agg.get("endpoint_alpha", {})
        print(
            f"  t={t:<5} alpha_mean={alpha.get('mean', float('nan')):.4f}  "
            f"velocity_abs={agg.get('velocity_abs', {}).get('mean', float('nan')):.4f}  "
            f"endpoint_pred_abs={agg.get('endpoint_pred_abs', {}).get('mean', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
