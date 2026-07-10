from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_schema import ExperimentConfig
from model import build_model_from_config
from spectral620 import dwt2_haar


STYLE_NAMES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def load_packed_latents(packed_dir: Path) -> dict[str, torch.Tensor]:
    banks: dict[str, torch.Tensor] = {}
    for path in sorted(packed_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        banks[str(payload["subdir"])] = payload["latents"].float()
    missing = [name for name in STYLE_NAMES if name not in banks]
    if missing:
        raise FileNotFoundError(f"missing packed styles: {missing}")
    return banks


def band_energy(tensor: torch.Tensor) -> torch.Tensor:
    ll, lh, hl, hh = dwt2_haar(tensor.float())
    return torch.stack([band.square().mean(dim=(1, 2, 3)) for band in (ll, lh, hl, hh)], dim=1)


def frequency_probe(banks: dict[str, torch.Tensor], pairs_per_route: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    all_energy = []
    route_rows = []
    for src_idx, src_name in enumerate(STYLE_NAMES):
        for tgt_idx, tgt_name in enumerate(STYLE_NAMES):
            if src_idx == tgt_idx:
                continue
            src_bank = banks[src_name]
            tgt_bank = banks[tgt_name]
            src_sel = torch.randint(len(src_bank), (pairs_per_route,), generator=generator)
            tgt_sel = torch.randint(len(tgt_bank), (pairs_per_route,), generator=generator)
            delta = tgt_bank[tgt_sel] - src_bank[src_sel]
            energy = band_energy(delta)
            all_energy.append(energy)
            route_mean = energy.mean(dim=0)
            route_share = route_mean / route_mean.sum()
            route_rows.append(
                {
                    "source": src_name,
                    "target": tgt_name,
                    **{f"energy_{band}": float(route_mean[i]) for i, band in enumerate(("LL", "LH", "HL", "HH"))},
                    **{f"share_{band}": float(route_share[i]) for i, band in enumerate(("LL", "LH", "HL", "HH"))},
                }
            )
    energies = torch.cat(all_energy, dim=0)
    mean_energy = energies.mean(dim=0)
    shares = mean_energy / mean_energy.sum()
    return {
        "definition": "For squared flow matching, gradient energy with respect to each predicted DWT band is proportional to the target residual energy in that band.",
        "num_pairs": int(energies.shape[0]),
        "bands": ["LL", "LH", "HL", "HH"],
        "mean_transport_energy": mean_energy.tolist(),
        "fm_gradient_energy_share": shares.tolist(),
        "route_rows": route_rows,
    }


def style_separability_probe(banks: dict[str, torch.Tensor]) -> dict:
    per_style_features: dict[str, list[torch.Tensor]] = {}
    for style_name in STYLE_NAMES:
        bands = dwt2_haar(banks[style_name].float())
        features = []
        for band in bands:
            flat = band.flatten(2)
            mean = flat.mean(dim=2)
            std = flat.std(dim=2)
            centered = flat - mean.unsqueeze(2)
            covariance = torch.bmm(centered, centered.transpose(1, 2)) / flat.shape[2]
            upper = torch.triu_indices(covariance.shape[1], covariance.shape[2])
            features.append(torch.cat([mean, std, covariance[:, upper[0], upper[1]]], dim=1))
        per_style_features[style_name] = features

    rows = []
    for band_idx, band_name in enumerate(("LL", "LH", "HL", "HH")):
        classes = [per_style_features[style][band_idx] for style in STYLE_NAMES]
        class_means = torch.stack([features.mean(dim=0) for features in classes])
        global_mean = torch.cat(classes, dim=0).mean(dim=0)
        between = (class_means - global_mean).square().mean()
        within_by_style = torch.stack(
            [(features - features.mean(dim=0)).square().mean() for features in classes]
        )
        ratio_by_style = between / within_by_style.clamp_min(1e-12)
        rows.append(
            {
                "band": band_name,
                "between_variance": float(between),
                "within_variance": float(within_by_style.mean()),
                "between_within_ratio": float(between / within_by_style.mean()),
                "ratio_sem": float(ratio_by_style.std(unbiased=True) / np.sqrt(len(STYLE_NAMES))),
            }
        )
    return {
        "definition": "Style separability is the between-style variance divided by within-style variance of per-image channel mean, standard deviation, and covariance statistics in each DWT band.",
        "num_images": int(sum(len(bank) for bank in banks.values())),
        "rows": rows,
    }


def build_model(checkpoint: dict, device: torch.device, *, endpoint_only: bool):
    raw_config = copy.deepcopy(checkpoint["config"])
    raw_config.setdefault("model", {})["endpoint_adain_only_last_step"] = bool(endpoint_only)
    config = ExperimentConfig.from_mapping(raw_config)
    model = build_model_from_config(config.model, bridge_cfg=config.bridge, use_checkpointing=False).to(device)
    state = {key.removeprefix("_orig_mod."): value for key, value in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def endpoint_probe(
    checkpoint_path: Path,
    banks: dict[str, torch.Tensor],
    device: torch.device,
    pairs_per_route: int,
    max_steps: int,
    seed: int,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    generator = torch.Generator().manual_seed(seed)
    pairs = []
    for src_idx, src_name in enumerate(STYLE_NAMES):
        tgt_idx = (src_idx + 1) % len(STYLE_NAMES)
        tgt_name = STYLE_NAMES[tgt_idx]
        src_sel = torch.randint(len(banks[src_name]), (pairs_per_route,), generator=generator)
        tgt_sel = torch.randint(len(banks[tgt_name]), (pairs_per_route,), generator=generator)
        pairs.append((src_idx, tgt_idx, banks[src_name][src_sel], banks[tgt_name][tgt_sel]))

    rows = []
    for endpoint_only in (False, True):
        model = build_model(checkpoint, device, endpoint_only=endpoint_only)
        for steps in range(1, max_steps + 1):
            content_drift = []
            target_distance = []
            lowpass_drift = []
            highpass_drift = []
            for _, tgt_idx, content_cpu, target_cpu in pairs:
                content = content_cpu.to(device)
                target = target_cpu.to(device)
                style_ids = torch.full((content.shape[0],), tgt_idx, device=device, dtype=torch.long)
                generated = model.integrate_transport(
                    content,
                    style_ids,
                    num_steps=steps,
                    step_size=1.0,
                    target_style_latent=target,
                )
                content_drift.append((generated - content).square().mean(dim=(1, 2, 3)).cpu())
                target_distance.append((generated - target).square().mean(dim=(1, 2, 3)).cpu())
                gen_bands = dwt2_haar(generated.float())
                src_bands = dwt2_haar(content.float())
                lowpass_drift.append((gen_bands[0] - src_bands[0]).square().mean(dim=(1, 2, 3)).cpu())
                highpass_drift.append(
                    torch.stack(
                        [(gen_bands[i] - src_bands[i]).square().mean(dim=(1, 2, 3)) for i in (1, 2, 3)], dim=1
                    ).mean(dim=1).cpu()
                )
            rows.append(
                {
                    "mode": "endpoint-only" if endpoint_only else "every-step",
                    "steps": steps,
                    "latent_content_mse": float(torch.cat(content_drift).mean()),
                    "latent_content_sem": float(torch.cat(content_drift).std(unbiased=True) / np.sqrt(torch.cat(content_drift).numel())),
                    "latent_target_mse": float(torch.cat(target_distance).mean()),
                    "ll_content_mse": float(torch.cat(lowpass_drift).mean()),
                    "hf_content_mse": float(torch.cat(highpass_drift).mean()),
                    "hf_content_sem": float(torch.cat(highpass_drift).std(unbiased=True) / np.sqrt(torch.cat(highpass_drift).numel())),
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "num_pairs": len(pairs) * pairs_per_route,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent / "fig_data")
    parser.add_argument("--frequency_pairs_per_route", type=int, default=250)
    parser.add_argument("--endpoint_pairs_per_route", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    banks = load_packed_latents(args.packed_dir)
    frequency = frequency_probe(banks, args.frequency_pairs_per_route, args.seed)
    endpoint = endpoint_probe(
        args.checkpoint,
        banks,
        torch.device(args.device),
        args.endpoint_pairs_per_route,
        args.max_steps,
        args.seed,
    )
    style_separability = style_separability_probe(banks)
    payload = {
        "frequency_probe": frequency,
        "style_separability_probe": style_separability,
        "endpoint_probe": endpoint,
    }
    (args.output_dir / "method_probes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (args.output_dir / "method_probe_endpoint.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint["rows"][0]))
        writer.writeheader()
        writer.writerows(endpoint["rows"])
    with (args.output_dir / "method_probe_frequency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frequency["route_rows"][0]))
        writer.writeheader()
        writer.writerows(frequency["route_rows"])
    with (args.output_dir / "method_probe_style_separability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(style_separability["rows"][0]))
        writer.writeheader()
        writer.writerows(style_separability["rows"])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
