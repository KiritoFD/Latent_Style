from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch

WORKSPACE = Path(__file__).resolve().parents[2]
SB_SRC = WORKSPACE / "src"

import sys

if str(SB_SRC) not in sys.path:
    sys.path.insert(0, str(SB_SRC))

from config_schema import load_experiment_config
from losses import OTFlowMatchingObjective
from model import build_model_from_config
from style_families import runtime_conditioning_requires_dino


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
    batch = {
        "content": torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device),
        "target_style": torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device),
        "target_style_id": torch.arange(batch_size, device=device, dtype=torch.long) % num_styles,
        "source_style_id": (torch.arange(batch_size, device=device, dtype=torch.long) + 1) % num_styles,
    }
    if runtime_conditioning_requires_dino(
        tokenizer_family=str(getattr(cfg.model, "tokenizer_family", "legacy_factorized")),
        semantic_supervision_family=str(getattr(cfg.bridge, "semantic_supervision_family", "legacy_terminal_swd")),
    ):
        feature_dim = max(1, int(getattr(cfg.model, "tokenizer_dino_dim", cfg.model.base_dim)))
        dino_h, dino_w = _content_hw_from_latent(latent_size)
        num_patches = dino_h * dino_w
        batch["content_dino_patches"] = torch.randn(batch_size, num_patches, feature_dim, device=device)
        batch["content_dino_hw"] = torch.tensor([dino_h, dino_w], device=device, dtype=torch.long)
        batch["target_style_dino_bank_patches"] = torch.randn(batch_size, bank_tokens, feature_dim, device=device)
    return batch


def _conditioning_payload(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    payload = {
        "content": batch["content"],
        "target_style": batch["target_style"],
        "target_style_id": batch["target_style_id"],
        "source_style_id": batch["source_style_id"],
    }
    for key in ("content_dino_patches", "content_dino_hw", "target_style_dino_bank_patches"):
        if key in batch:
            payload[key] = batch[key]
    return payload


def _first_grad_stat(model: torch.nn.Module) -> tuple[str | None, float]:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_abs = float(torch.nan_to_num(param.grad.detach().float().abs().mean()).item())
        return name, grad_abs
    return None, 0.0


def _tensor_shape(x: torch.Tensor) -> list[int]:
    return [int(v) for v in x.shape]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if torch.is_tensor(value):
        return float(torch.nan_to_num(value.detach().float()).item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _maybe_tensor_shape(x: Any) -> list[int] | None:
    if not torch.is_tensor(x):
        return None
    return _tensor_shape(x)


def _attention_entropy(attn: torch.Tensor | None) -> float:
    if attn is None:
        return 0.0
    probs = attn.detach().float().clamp_min(1e-8)
    return float((-(probs * probs.log()).sum(dim=-1).mean()).item())


def _numeric_debug_dict(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                out[str(key)] = _safe_float(value)
        elif isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test one experiment config with build/forward/backward.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--bank-tokens", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    device = torch.device(str(args.device))
    result: dict[str, Any]
    try:
        cfg = load_experiment_config(config_path)
        torch.manual_seed(0)
        model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge, use_checkpointing=False).to(device)
        model.train()
        objective = OTFlowMatchingObjective(cfg)
        batch = _build_synthetic_batch(
            cfg,
            batch_size=max(1, int(args.batch_size)),
            latent_size=max(8, int(args.latent_size)),
            bank_tokens=max(1, int(args.bank_tokens)),
            device=device,
        )
        needs_dino_runtime = runtime_conditioning_requires_dino(
            tokenizer_family=str(getattr(cfg.model, "tokenizer_family", "legacy_factorized")),
            semantic_supervision_family=str(getattr(cfg.bridge, "semantic_supervision_family", "legacy_terminal_swd")),
        )
        model.zero_grad(set_to_none=True)
        if needs_dino_runtime:
            model.set_runtime_conditioning(
                {
                    "content_dino_patches": batch["content_dino_patches"],
                    "content_dino_hw": batch["content_dino_hw"],
                    "target_style_dino_bank_patches": batch["target_style_dino_bank_patches"],
                }
            )
        try:
            t_half = torch.full((int(args.batch_size),), 0.5, device=device, dtype=batch["content"].dtype)
            direct = model(batch["content"], t=t_half, style_id=batch["target_style_id"])
            endpoint = model.predict_transport_base(batch["content"], t=t_half, style_id=batch["target_style_id"])
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
                conditioning=_conditioning_payload(batch),
            )
            loss = loss_dict["loss"]
            loss.backward()
        finally:
            model.clear_runtime_conditioning()
        grad_name, grad_abs = _first_grad_stat(model)
        first_block = model.body_blocks[0] if getattr(model, "body_blocks", None) else None
        semantic_attn = getattr(model, "last_semantic_attn", None)
        semantic_topology_attn = getattr(model, "last_semantic_topology_attn", None)
        tokenizer_debug = _numeric_debug_dict(getattr(getattr(model, "structured_style_tokenizer", None), "last_debug", {}))
        result = {
            "status": "ok",
            "config": str(config_path),
            "objective_mode": str(getattr(cfg.bridge, "objective_mode", "omf")),
            "tokenizer_family": str(getattr(cfg.model, "tokenizer_family", "legacy_factorized")),
            "solver_family": str(getattr(cfg.model, "solver_family", "euler_legacy")),
            "transport_prediction_mode": str(getattr(cfg.model, "transport_prediction_mode", "velocity")),
            "semantic_supervision_family": str(getattr(cfg.bridge, "semantic_supervision_family", "legacy_terminal_swd")),
            "dino_runtime_required": needs_dino_runtime,
            "bridge_sigma": float(getattr(cfg.bridge, "bridge_sigma", 0.0)),
            "forward_shape": _tensor_shape(direct),
            "endpoint_shape": _tensor_shape(endpoint),
            "integrated_shape": _tensor_shape(integrated),
            "loss": float(torch.nan_to_num(loss.detach().float()).item()),
            "flow": float(torch.nan_to_num(loss_dict["flow"].detach().float()).item()),
            "terminal_swd": float(torch.nan_to_num(loss_dict["terminal_swd"].detach().float()).item()),
            "t_mean": float(torch.nan_to_num(loss_dict["t_mean"].detach().float()).item()),
            "semantic_attn_shape": _maybe_tensor_shape(semantic_attn),
            "semantic_topology_attn_shape": _maybe_tensor_shape(semantic_topology_attn),
            "semantic_topology_attn_entropy_runtime": _attention_entropy(semantic_topology_attn),
            "semantic_topology_attn_entropy_metric": _safe_float(loss_dict.get("semantic_topology_attn_entropy", 0.0)),
            "semantic_topology_attn_active_metric": _safe_float(loss_dict.get("semantic_topology_attn_active", 0.0)),
            "semantic_attn_mean_metric": _safe_float(loss_dict.get("semantic_attn_mean", 0.0)),
            "semantic_k_abs_metric": _safe_float(loss_dict.get("semantic_k_abs", 0.0)),
            "structured_style_tokenizer_debug": tokenizer_debug,
            "body_block_type": type(first_block).__name__ if first_block is not None else "",
            "first_grad_name": grad_name,
            "first_grad_abs_mean": grad_abs,
        }
    except Exception as exc:
        result = {
            "status": "failed",
            "config": str(config_path),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

    output = str(args.output).strip()
    if output:
        out_path = Path(output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(out_path)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
