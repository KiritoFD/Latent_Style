from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


def _repo_src_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


SRC_PATH = str(_repo_src_path())
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from config_schema import load_experiment_config  # noqa: E402
from losses import OTFlowMatchingObjective  # noqa: E402
from model import build_model_from_config  # noqa: E402
from probe_conditioning_sensitivity import (  # noqa: E402
    _apply_config_overrides,
    _git_commit,
    _random_inputs,
    _runtime_metadata,
    _write_csv,
)


def _complexity_descriptor(attn: torch.Tensor) -> torch.Tensor:
    probs = attn.float().clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1)
    max_entropy = max(math.log(float(max(int(probs.shape[-1]), 2))), 1e-8)
    entropy = entropy / max_entropy
    median = entropy.median(dim=1, keepdim=True).values
    return torch.stack(
        [
            entropy.mean(dim=1),
            entropy.std(dim=1, unbiased=False),
            (entropy > median).float().mean(dim=1),
            entropy.amax(dim=1),
        ],
        dim=1,
    ).float()


def _collect_topogate_descriptors(
    objective: OTFlowMatchingObjective,
    model: torch.nn.Module,
    latent: torch.Tensor,
    *,
    style_id: torch.Tensor,
) -> list[torch.Tensor | None]:
    style_id_t = objective._expand_style_id_tensor(style_id, batch=int(latent.shape[0]), device=latent.device)
    content_feat_16 = objective._ot_encoder_feature_map(model, latent, style_id=style_id_t).to(dtype=latent.dtype)
    body_blocks = list(getattr(model, "body_blocks", []))
    probe_map = content_feat_16
    h = content_feat_16
    for block in body_blocks:
        if hasattr(block, "last_topology_attn"):
            setattr(block, "last_topology_attn", None)
        if hasattr(block, "last_attn"):
            setattr(block, "last_attn", None)
    rows: list[torch.Tensor | None] = []
    with torch.no_grad():
        for block in body_blocks:
            h = block(h, style_map=probe_map, gate=0.0)
            attn = getattr(block, "last_topology_attn", None)
            if attn is None:
                attn = getattr(block, "last_attn", None)
            rows.append(_complexity_descriptor(attn.detach().float()) if torch.is_tensor(attn) else None)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether TopoGate OT descriptors use one block or all body blocks.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--style-id", type=int, default=1)
    parser.add_argument("--override", action="append", default=[], help="Config override in section.field=value form.")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    applied_overrides = _apply_config_overrides(cfg, list(args.override))
    device = torch.device(args.device)
    model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge).to(device)
    model.eval()
    objective = OTFlowMatchingObjective(cfg)
    inputs = _random_inputs(
        batch_size=int(args.batch_size),
        latent_channels=cfg.model.latent_channels,
        latent_size=int(args.latent_size),
        style_id=int(args.style_id),
        seed=int(args.seed),
        device=device,
    )

    x_descs = _collect_topogate_descriptors(objective, model, inputs["x"], style_id=inputs["style_id"])
    y_descs = _collect_topogate_descriptors(objective, model, inputs["lat_b"], style_id=inputs["style_id"])

    per_block_rows: list[dict[str, Any]] = []
    valid_x: list[torch.Tensor] = []
    valid_y: list[torch.Tensor] = []
    for idx, (x_desc, y_desc) in enumerate(zip(x_descs, y_descs)):
        row: dict[str, Any] = {"block_idx": idx, "descriptor_present": float(x_desc is not None and y_desc is not None)}
        if x_desc is not None and y_desc is not None:
            cost = torch.cdist(x_desc, y_desc, p=2).pow(2)
            row.update(
                {
                    "descriptor_width": int(x_desc.shape[1]),
                    "cost_mean": float(cost.mean().item()),
                    "cost_var": float(cost.var(unbiased=False).item()),
                    "content_complexity_mean": float(x_desc[:, 0].mean().item()),
                    "target_complexity_mean": float(y_desc[:, 0].mean().item()),
                }
            )
            valid_x.append(x_desc)
            valid_y.append(y_desc)
        per_block_rows.append(row)

    summary: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "config": str(args.config),
        "applied_overrides": applied_overrides,
        "git_commit": _git_commit(),
        "runtime_metadata": _runtime_metadata(args.device, device),
        "batch_size": int(args.batch_size),
        "latent_size": int(args.latent_size),
        "seed": int(args.seed),
        "style_id": int(args.style_id),
        "descriptor_blocks": len(valid_x),
        "per_block_rows": per_block_rows,
    }
    if valid_x:
        last_x = valid_x[-1]
        last_y = valid_y[-1]
        last_cost = torch.cdist(last_x, last_y, p=2).pow(2)
        agg_x = torch.cat(valid_x, dim=1)
        agg_y = torch.cat(valid_y, dim=1)
        agg_cost = torch.cdist(agg_x, agg_y, p=2).pow(2)
        summary.update(
            {
                "last_block_cost_mean": float(last_cost.mean().item()),
                "last_block_cost_var": float(last_cost.var(unbiased=False).item()),
                "aggregate_cost_mean": float(agg_cost.mean().item()),
                "aggregate_cost_var": float(agg_cost.var(unbiased=False).item()),
                "aggregate_minus_last_mean_abs": float((agg_cost - last_cost).abs().mean().item()),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_block_topogate_descriptor.csv", per_block_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
