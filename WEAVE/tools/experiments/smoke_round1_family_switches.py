from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

WORKSPACE = Path(__file__).resolve().parents[3]
ROUND1_CONFIG_ROOT = WORKSPACE / "SchrodingerBridge" / "configs" / "aaai2027" / "round1_full_sweep"
SB_SRC = WORKSPACE / "SchrodingerBridge" / "src"
if str(SB_SRC) not in sys.path:
    sys.path.insert(0, str(SB_SRC))

from config_schema import load_experiment_config
from losses import OTFlowMatchingObjective
from model import build_model_from_config


def _candidate_config_paths(config_root: Path) -> list[Path]:
    paths = sorted(config_root.glob("aaai2027_round1_*_seed42_b8a2.json"))
    return [
        path
        for path in paths
        if not path.name.endswith(".remote.launch.json")
        and not path.name.endswith(".segmented.launch.json")
    ]


def _content_hw_from_latent(latent_size: int) -> tuple[int, int]:
    side = max(1, int(latent_size) // 2)
    return side, side


def _build_synthetic_batch(
    cfg,
    *,
    batch_size: int,
    latent_size: int,
    bank_tokens: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    latent_channels = int(cfg.model.latent_channels)
    num_styles = max(2, int(cfg.model.num_styles))
    dino_dim = max(1, int(getattr(cfg.model, "tokenizer_dino_dim", 384)))
    dino_h, dino_w = _content_hw_from_latent(latent_size)
    num_patches = dino_h * dino_w

    content = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device)
    target_style = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device)
    target_style_id = torch.arange(batch_size, device=device, dtype=torch.long) % num_styles
    source_style_id = (target_style_id + 1) % num_styles

    batch = {
        "content": content,
        "target_style": target_style,
        "target_style_id": target_style_id,
        "source_style_id": source_style_id,
        "content_dino_patches": torch.randn(batch_size, num_patches, dino_dim, device=device),
        "content_dino_hw": torch.tensor([dino_h, dino_w], device=device, dtype=torch.long),
        "target_style_dino_bank_patches": torch.randn(batch_size, bank_tokens, dino_dim, device=device),
    }
    return batch


def _conditioning_payload(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "content": batch["content"],
        "target_style": batch["target_style"],
        "target_style_id": batch["target_style_id"],
        "source_style_id": batch["source_style_id"],
        "content_dino_patches": batch["content_dino_patches"],
        "content_dino_hw": batch["content_dino_hw"],
        "target_style_dino_bank_patches": batch["target_style_dino_bank_patches"],
    }


def _first_grad_stat(model: torch.nn.Module) -> tuple[str | None, float]:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_abs = float(torch.nan_to_num(param.grad.detach().float().abs().mean()).item())
        return name, grad_abs
    return None, 0.0


def _tensor_shape(x: torch.Tensor) -> list[int]:
    return [int(v) for v in x.shape]


def run_smoke_for_config(
    config_path: Path,
    *,
    batch_size: int,
    latent_size: int,
    bank_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    cfg = load_experiment_config(config_path)
    torch.manual_seed(0)

    model = build_model_from_config(cfg.model, use_checkpointing=False).to(device)
    model.train()
    objective = OTFlowMatchingObjective(cfg)

    batch = _build_synthetic_batch(
        cfg,
        batch_size=batch_size,
        latent_size=latent_size,
        bank_tokens=bank_tokens,
        device=device,
    )
    conditioning = _conditioning_payload(batch)
    t_half = torch.full((batch_size,), 0.5, device=device, dtype=batch["content"].dtype)

    model.zero_grad(set_to_none=True)
    model.set_runtime_conditioning(
        {
            "content_dino_patches": batch["content_dino_patches"],
            "content_dino_hw": batch["content_dino_hw"],
            "target_style_dino_bank_patches": batch["target_style_dino_bank_patches"],
        }
    )
    try:
        direct_velocity = model(
            batch["content"],
            t=t_half,
            style_id=batch["target_style_id"],
        )
        base_endpoint = model.predict_transport_base(
            batch["content"],
            t=t_half,
            style_id=batch["target_style_id"],
        )
        integrated = model.integrate_transport(
            batch["content"],
            style_id=batch["target_style_id"],
            num_steps=2,
            step_size=1.0,
            style_strength=1.0,
        )

        loss_dict = objective.compute(
            model,
            content=batch["content"],
            target_style=batch["target_style"],
            target_style_id=batch["target_style_id"],
            source_style_id=batch["source_style_id"],
            conditioning=conditioning,
        )
        loss = loss_dict["loss"]
        loss.backward()

        grad_name, grad_abs_mean = _first_grad_stat(model)
        result = {
            "status": "ok",
            "family_id": config_path.stem.replace("aaai2027_round1_", "").replace("_seed42_b8a2", ""),
            "config_path": str(config_path),
            "tokenizer_family": str(getattr(cfg.model, "tokenizer_family", "legacy_factorized")),
            "backbone_attention_family": str(getattr(cfg.model, "backbone_attention_family", "legacy_semantic_crossattn")),
            "solver_family": str(getattr(cfg.model, "solver_family", "euler_legacy")),
            "semantic_supervision_family": str(getattr(cfg.bridge, "semantic_supervision_family", "legacy_terminal_swd")),
            "direct_velocity_shape": _tensor_shape(direct_velocity),
            "base_endpoint_shape": _tensor_shape(base_endpoint),
            "integrated_shape": _tensor_shape(integrated),
            "loss": float(torch.nan_to_num(loss.detach().float()).item()),
            "loss_is_finite": bool(torch.isfinite(loss.detach()).item()),
            "terminal_swd": float(torch.nan_to_num(loss_dict["terminal_swd"].detach().float()).item()),
            "first_grad_name": grad_name,
            "first_grad_abs_mean": grad_abs_mean,
        }
    finally:
        model.clear_runtime_conditioning()

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round-1 family switch smoke test")
    parser.add_argument("--config-root", type=Path, default=ROUND1_CONFIG_ROOT)
    parser.add_argument("--family-id", action="append", default=[], help="Optional family id(s) to filter.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--bank-tokens", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "SchrodingerBridge" / "aaai2027" / "round1_family_switch_smoke_20260610.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_root = args.config_root.resolve()
    all_paths = _candidate_config_paths(config_root)
    wanted = {str(x).strip().lower() for x in args.family_id if str(x).strip()}
    if wanted:
        config_paths = [
            path
            for path in all_paths
            if path.stem.replace("aaai2027_round1_", "").replace("_seed42_b8a2", "").lower() in wanted
        ]
    else:
        config_paths = all_paths
    if not config_paths:
        raise FileNotFoundError(f"No round-1 config paths selected under {config_root}")

    device = torch.device(args.device)
    results: list[dict[str, Any]] = []
    failures = 0

    for path in config_paths:
        family_id = path.stem.replace("aaai2027_round1_", "").replace("_seed42_b8a2", "")
        try:
            result = run_smoke_for_config(
                path,
                batch_size=max(1, int(args.batch_size)),
                latent_size=max(8, int(args.latent_size)),
                bank_tokens=max(1, int(args.bank_tokens)),
                device=device,
            )
        except Exception as exc:
            failures += 1
            result = {
                "status": "failed",
                "family_id": family_id,
                "config_path": str(path),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        results.append(result)
        print(f"[smoke_round1_family_switches] {family_id}: {result['status']}", flush=True)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": str(device),
        "batch_size": max(1, int(args.batch_size)),
        "latent_size": max(8, int(args.latent_size)),
        "bank_tokens": max(1, int(args.bank_tokens)),
        "row_count": len(results),
        "failure_count": failures,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[smoke_round1_family_switches] wrote {output}", flush=True)

    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
