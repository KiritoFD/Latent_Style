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

from config_schema import BridgeConfig, ExperimentConfig, ModelConfig  # noqa: E402
from losses620 import SpatialBridgeObjective620, _lowpass  # noqa: E402
from model620 import SpatialBridge620  # noqa: E402


def _cfg() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model = ModelConfig(
        latent_channels=2,
        num_styles=2,
        base_dim=8,
        time_dim=8,
        num_res_blocks=1,
        style_attn_num_heads=2,
        tokenizer_dino_dim=6,
        contract_family="620_spatial_bridge",
        style_condition_source="target_dino_patches",
    )
    cfg.bridge = BridgeConfig(
        w_flow=1.0,
        single_step_swd_weight=0.5,
        single_step_edge_weight=0.1,
        semantic_swd_num_projections=4,
        training_target_projection_kernel=3,
        bridge_sigma=0.02,
    )
    return cfg


def run_probe(*, device: str = "cpu", seed: int = 620) -> dict[str, object]:
    torch.manual_seed(int(seed))
    dev = torch.device(device)
    cfg = _cfg()
    objective = SpatialBridgeObjective620(cfg)
    model = SpatialBridge620(cfg.model, cfg.bridge).to(dev)
    model.eval()
    content = torch.randn(2, 2, 6, 6, device=dev)
    target = torch.randn_like(content) + 0.35
    style_patches = torch.randn(2, 4, 6, device=dev)
    out = objective.compute(
        model,
        content=content,
        target_style=target,
        target_style_id=torch.tensor([0, 1], device=dev),
        conditioning={"target_style_dino_patches": style_patches},
    )
    z_hat1 = objective.last_debug["z_hat1"].to(dev)
    z_low = _lowpass(z_hat1, objective.lowpass_kernel)
    c_low = _lowpass(content, objective.lowpass_kernel)
    t_low = _lowpass(target, objective.lowpass_kernel)
    z_high = z_hat1 - z_low
    t_high = target - t_low
    expected = {
        "endpoint_low_to_source": (z_low - c_low).float().abs().mean(),
        "endpoint_low_to_target": (z_low - t_low).float().abs().mean(),
        "endpoint_high_to_target": (z_high - t_high).float().abs().mean(),
    }
    values = {key: float(out[key].detach().cpu()) for key in expected}
    diffs = {key: abs(values[key] - float(val.detach().cpu())) for key, val in expected.items()}
    finite = all(torch.isfinite(out[key]).item() for key in (*expected.keys(), "endpoint_low_target_ratio"))
    ok = bool(finite and max(diffs.values()) < 1e-6 and float(out["style_dino_active"].detach().cpu()) == 1.0)
    return {
        "ok": ok,
        "device": str(dev),
        "seed": int(seed),
        **values,
        "endpoint_low_target_ratio": float(out["endpoint_low_target_ratio"].detach().cpu()),
        "style_dino_active": float(out["style_dino_active"].detach().cpu()),
        "max_recompute_diff": max(diffs.values()),
        "classification": "endpoint_decomposition_live" if ok else "endpoint_decomposition_failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe 620 endpoint low/high decomposition diagnostics.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=620)
    args = parser.parse_args()
    result = run_probe(device=str(args.device), seed=int(args.seed))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
