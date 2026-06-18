from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_schema import BridgeConfig, ModelConfig  # noqa: E402
from model620 import SpatialBridge620  # noqa: E402


def _build_model(device: torch.device) -> SpatialBridge620:
    model_cfg = ModelConfig(
        contract_family="620_spatial_bridge",
        style_condition_source="target_dino_patches",
        solver_family="solver_i2sb",
        transport_prediction_mode="velocity",
        latent_channels=4,
        num_styles=5,
        base_dim=24,
        time_dim=32,
        num_res_blocks=2,
        style_attn_num_heads=4,
        style_attn_num_tokens=16,
        style_cross_attn_gate_init=0.08,
        tokenizer_dino_dim=32,
    )
    bridge_cfg = BridgeConfig(bridge_sigma=0.02)
    model = SpatialBridge620(model_cfg, bridge_cfg).to(device)
    model.eval()
    return model


def run_probe(*, device: str = "cpu", seed: int = 620, atol: float = 1e-7) -> dict[str, object]:
    torch.manual_seed(int(seed))
    dev = torch.device(device)
    model = _build_model(dev)
    x = torch.randn(2, 4, 8, 8, device=dev)
    t = torch.tensor([0.2, 0.7], device=dev)
    style_a = torch.randn(2, 9, 32, device=dev)
    style_b = style_a.clone()
    style_b[:, :, :8] = -style_b[:, :, :8] + 0.37
    cls_a = style_a.mean(dim=1)
    cls_b = style_b.mean(dim=1)

    with torch.inference_mode():
        v_a = model(x, t=t, style_id=torch.tensor([0, 1], device=dev), style_dino_patches=style_a, style_dino_cls=cls_a)
        debug_a = {k: float(v.detach().cpu()) for k, v in model.last_debug.items() if torch.is_tensor(v)}
        v_b = model(x, t=t, style_id=torch.tensor([0, 1], device=dev), style_dino_patches=style_b, style_dino_cls=cls_b)
        debug_b = {k: float(v.detach().cpu()) for k, v in model.last_debug.items() if torch.is_tensor(v)}
        endpoint_a = x + (1.0 - t).view(-1, 1, 1, 1) * v_a
        endpoint_b = x + (1.0 - t).view(-1, 1, 1, 1) * v_b

    velocity_delta = float((v_a - v_b).abs().mean().detach().cpu())
    endpoint_delta = float((endpoint_a - endpoint_b).abs().mean().detach().cpu())
    cross_delta = max(float(debug_a.get("cross_attn_delta_abs", 0.0)), float(debug_b.get("cross_attn_delta_abs", 0.0)))
    entropy = float(debug_b.get("cross_attn_entropy", 0.0))
    ok = velocity_delta > atol and endpoint_delta > atol and cross_delta > atol and entropy > 0.0
    return {
        "ok": bool(ok),
        "device": str(dev),
        "seed": int(seed),
        "velocity_delta_abs": velocity_delta,
        "endpoint_delta_abs": endpoint_delta,
        "cross_attn_delta_abs": cross_delta,
        "cross_attn_entropy": entropy,
        "style_gate_value": float(debug_b.get("style_gate_value", 0.0)),
        "threshold": float(atol),
        "classification": "runtime_and_training_real" if ok else "dead_or_too_small",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe 620 DINO patch conditioning path liveness.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=620)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args()
    result = run_probe(device=str(args.device), seed=int(args.seed), atol=float(args.atol))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
