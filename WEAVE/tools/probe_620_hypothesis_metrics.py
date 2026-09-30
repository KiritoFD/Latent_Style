from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_schema import load_experiment_config  # noqa: E402
from model import build_model_from_config  # noqa: E402
from utils.dataset import AdaCUTLatentDataset  # noqa: E402
from utils.inference import decode_latent, load_vae  # noqa: E402
from utils.training import strip_compile_prefix  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_checkpoint_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state: dict[str, torch.Tensor] | None = None
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
        if state is None and all(isinstance(k, str) for k in payload.keys()):
            state = payload
    if state is None:
        raise TypeError(f"Unsupported checkpoint payload: {checkpoint}")
    return strip_compile_prefix({str(k): v for k, v in state.items()})


def _build_dataset(
    cfg,
    device: str,
    *,
    data_root_override: str = "",
    latent_cache_dir_override: str = "",
    pairing_cache_override: str = "",
    dino_cache_override: str = "",
) -> AdaCUTLatentDataset:
    data_cfg = cfg.data
    train_cfg = cfg.training
    contract_family = str(getattr(cfg.model, "contract_family", "legacy") or "legacy").strip().lower()
    needs_dino_runtime = contract_family == "620_spatial_bridge"
    dino_cache_path = str(dino_cache_override or data_cfg.dino_cache_path) if needs_dino_runtime else ""
    dino_cache_required = bool(data_cfg.dino_cache_required) if needs_dino_runtime else False
    return AdaCUTLatentDataset(
        data_root=str(data_root_override or data_cfg.data_root),
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=False,
        identity_ratio=data_cfg.identity_ratio,
        batch_size_hint=int(train_cfg.batch_size),
        balance_target_styles_per_batch=bool(data_cfg.balance_target_styles_per_batch),
        preload_to_gpu=False,
        preload_max_vram_gb=0.0,
        preload_reserve_ratio=float(data_cfg.preload_reserve_ratio),
        virtual_length_multiplier=float(data_cfg.virtual_length_multiplier),
        content_style_sampling_weights=data_cfg.content_style_sampling_weights,
        target_style_sampling_weights=data_cfg.target_style_sampling_weights,
        pairing_cache_path=str(pairing_cache_override or data_cfg.pairing_cache_path),
        pairing_cache_topk=int(data_cfg.pairing_cache_topk),
        pairing_cache_active_topk=int(data_cfg.pairing_cache_active_topk),
        pairing_cache_sample_mode=str(data_cfg.pairing_cache_sample_mode),
        pairing_cache_rank_schedule=str(data_cfg.pairing_cache_rank_schedule),
        pairing_cache_min_topk=int(data_cfg.pairing_cache_min_topk),
        pairing_cache_curriculum_epochs=int(data_cfg.pairing_cache_curriculum_epochs),
        pairing_cache_rank_power=float(data_cfg.pairing_cache_rank_power),
        pairing_cache_explore_prob=float(data_cfg.pairing_cache_explore_prob),
        pairing_cache_explore_topk=int(data_cfg.pairing_cache_explore_topk),
        pairing_cache_dual_target_mix=float(data_cfg.pairing_cache_dual_target_mix),
        pairing_cache_dual_target_topk=int(data_cfg.pairing_cache_dual_target_topk),
        pairing_cache_aux_target_topk=int(data_cfg.pairing_cache_aux_target_topk),
        pairing_cache_cross_only=bool(data_cfg.pairing_cache_cross_only),
        latent_cache_mode=str(data_cfg.latent_cache_mode),
        latent_cache_dir=str(latent_cache_dir_override or data_cfg.latent_cache_dir),
        dino_cache_path=dino_cache_path,
        dino_cache_required=dino_cache_required,
        dino_bank_limit_per_style=int(data_cfg.dino_bank_limit_per_style),
        style_caption_path=str(getattr(data_cfg, "style_caption_path", "")),
        device=device,
    )


def _build_model(cfg, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge, use_checkpointing=False)
    state_dict = _load_checkpoint_state(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            json.dumps(
                {
                    "warning": "non_strict_checkpoint_load",
                    "missing_keys": list(missing),
                    "unexpected_keys": list(unexpected),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    model = model.to(device=device)
    model.eval()
    return model


def _prepare_batch(item: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            out[key] = value.unsqueeze(0).to(device=device)
        elif isinstance(value, int):
            out[key] = torch.tensor([value], device=device)
    return out


def _lowpass(x: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    return F.avg_pool2d(x.float(), kernel_size=k, stride=1, padding=k // 2)


def _luminance(img: torch.Tensor) -> torch.Tensor:
    xf = img.detach().float().clamp(0.0, 1.0)
    return 0.299 * xf[:, 0:1] + 0.587 * xf[:, 1:2] + 0.114 * xf[:, 2:3]


def _decode(vae, latent: torch.Tensor, device: str) -> torch.Tensor:
    with torch.no_grad():
        return decode_latent(vae, latent, device=device)


def _make_state(source: torch.Tensor, target: torch.Tensor, t: float, mode: str) -> torch.Tensor:
    t4 = torch.full((source.shape[0], 1, 1, 1), float(t), device=source.device, dtype=source.dtype)
    if mode == "content_only":
        return source
    if mode == "training_linear":
        return (1.0 - t4) * source + t4 * target
    raise ValueError(f"Unknown state mode: {mode}")


def _predict_endpoint(
    model: torch.nn.Module,
    state: torch.Tensor,
    *,
    t: float,
    style_id: torch.Tensor,
    style_patches: torch.Tensor | None,
    style_cls: torch.Tensor | None,
    style_text_tokens: torch.Tensor | None,
) -> torch.Tensor:
    t_batch = torch.full((state.shape[0],), float(t), device=state.device, dtype=state.dtype)
    with torch.no_grad():
        return model.predict_endpoint(
            state,
            t=t_batch,
            style_id=style_id,
            style_dino_patches=style_patches,
            style_dino_cls=style_cls,
            style_text_tokens=style_text_tokens,
        )


def _alpha_metrics(base: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, prefix: str) -> dict[str, float]:
    base_f = base.detach().float().reshape(base.shape[0], -1)
    target_f = target.detach().float().reshape(target.shape[0], -1)
    pred_f = pred.detach().float().reshape(pred.shape[0], -1)
    delta = target_f - base_f
    move = pred_f - base_f
    denom = delta.square().sum(dim=1).clamp_min(1e-8)
    alpha = (move * delta).sum(dim=1) / denom
    proj = alpha[:, None] * delta
    ortho = (move - proj).square().sum(dim=1).sqrt()
    delta_norm = delta.square().sum(dim=1).sqrt().clamp_min(1e-8)
    return {
        f"{prefix}_alpha_mean": float(alpha.mean().item()),
        f"{prefix}_alpha_min": float(alpha.min().item()),
        f"{prefix}_alpha_max": float(alpha.max().item()),
        f"{prefix}_orth_rms": float(ortho.mean().item()),
        f"{prefix}_orth_over_delta": float((ortho / delta_norm).mean().item()),
        f"{prefix}_shrink_gap": float((1.0 - alpha).clamp_min(0.0).mean().item()),
    }


def _style_distance(
    style_patches_a: torch.Tensor | None,
    style_cls_a: torch.Tensor | None,
    style_patches_b: torch.Tensor | None,
    style_cls_b: torch.Tensor | None,
) -> float:
    diffs: list[torch.Tensor] = []
    if style_cls_a is not None and style_cls_b is not None:
        diffs.append((style_cls_a.detach().float() - style_cls_b.detach().float()).reshape(style_cls_a.shape[0], -1))
    if style_patches_a is not None and style_patches_b is not None:
        diffs.append((style_patches_a.detach().float() - style_patches_b.detach().float()).reshape(style_patches_a.shape[0], -1))
    if not diffs:
        return 0.0
    total = torch.cat(diffs, dim=1)
    return float(total.square().mean(dim=1).sqrt().mean().item())


def _find_alt_batch(
    dataset: AdaCUTLatentDataset,
    *,
    device: torch.device,
    anchor_target_style_id: int,
    anchor_index: int,
    max_scan: int,
) -> tuple[int, dict[str, torch.Tensor]] | None:
    for offset in range(1, max_scan + 1):
        idx = (anchor_index + offset) % len(dataset)
        try:
            item = dataset[idx]
        except KeyError:
            continue
        batch = _prepare_batch(item, device)
        alt_style_id = int(batch["target_style_id"].item())
        if alt_style_id == anchor_target_style_id:
            continue
        return idx, batch
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantify 620 whitening hypotheses with endpoint shrinkage and style sensitivity metrics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--dataset-seed-epoch", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--times", type=float, nargs="+", default=[0.0, 0.125, 0.25, 0.5, 0.75, 0.875])
    parser.add_argument("--state-mode", choices=["training_linear", "content_only"], default="training_linear")
    parser.add_argument("--lowpass-kernel", type=int, default=5)
    parser.add_argument("--alt-style-scan", type=int, default=64)
    parser.add_argument("--vae-model", type=str, default="ema")
    parser.add_argument("--vae-cache-dir", type=str, default="")
    parser.add_argument("--data-root-override", type=str, default="")
    parser.add_argument("--latent-cache-dir-override", type=str, default="")
    parser.add_argument("--pairing-cache-override", type=str, default="")
    parser.add_argument("--dino-cache-override", type=str, default="")
    parser.add_argument("--max-scan-multiplier", type=int, default=20)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    device = torch.device(args.device)
    dataset = _build_dataset(
        cfg,
        str(device),
        data_root_override=str(args.data_root_override or ""),
        latent_cache_dir_override=str(args.latent_cache_dir_override or ""),
        pairing_cache_override=str(args.pairing_cache_override or ""),
        dino_cache_override=str(args.dino_cache_override or ""),
    )
    dataset.set_epoch(int(args.dataset_seed_epoch))
    model = _build_model(cfg, args.checkpoint, device)
    vae = load_vae(
        device=str(device),
        model_id=str(args.vae_model),
        cache_dir=str(args.vae_cache_dir or None) if args.vae_cache_dir else None,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    collected = 0
    scan_offset = 0
    wanted_samples = max(1, int(args.sample_count))
    max_scans = max(wanted_samples, wanted_samples * max(1, int(args.max_scan_multiplier)))
    times = [float(max(0.0, min(0.999, t))) for t in args.times]
    while collected < wanted_samples and scan_offset < max_scans:
        sample_index = int(args.start_index) + scan_offset
        scan_offset += 1
        try:
            item = dataset[sample_index % len(dataset)]
        except KeyError as exc:
            skipped.append(
                {
                    "sample_index": sample_index,
                    "reason": "dataset_keyerror",
                    "message": str(exc),
                }
            )
            continue
        batch = _prepare_batch(item, device)
        source = batch["content"]
        target = batch["target_style"]
        target_style_id = batch["target_style_id"].long()
        target_style_id_scalar = int(target_style_id.item())
        style_patches = batch.get("target_style_dino_patches")
        style_cls = batch.get("target_style_dino_cls")
        style_text_tokens = batch.get("target_style_text_tokens")
        if style_patches is None and style_cls is None:
            skipped.append(
                {
                    "sample_index": sample_index,
                    "reason": "missing_style_condition",
                    "message": "target style DINO tensors were unavailable",
                }
            )
            continue
        alt = _find_alt_batch(
            dataset,
            device=device,
            anchor_target_style_id=target_style_id_scalar,
            anchor_index=sample_index,
            max_scan=max(1, int(args.alt_style_scan)),
        )
        if alt is None:
            skipped.append(
                {
                    "sample_index": sample_index,
                    "reason": "missing_alt_style",
                    "message": "unable to find alternate target style with valid sidecars",
                }
            )
            continue
        alt_index, alt_batch = alt
        alt_style_id = alt_batch["target_style_id"].long()
        alt_style_patches = alt_batch.get("target_style_dino_patches")
        alt_style_cls = alt_batch.get("target_style_dino_cls")
        alt_style_text_tokens = alt_batch.get("target_style_text_tokens")
        style_distance = _style_distance(style_patches, style_cls, alt_style_patches, alt_style_cls)
        source_img = _decode(vae, source, str(device))
        target_img = _decode(vae, target, str(device))
        source_lum = _luminance(source_img)
        target_lum = _luminance(target_img)
        for t in times:
            state = _make_state(source, target, t, args.state_mode)
            endpoint = _predict_endpoint(
                model,
                state,
                t=t,
                style_id=target_style_id,
                style_patches=style_patches,
                style_cls=style_cls,
                style_text_tokens=style_text_tokens,
            )
            endpoint_alt = _predict_endpoint(
                model,
                state,
                t=t,
                style_id=alt_style_id,
                style_patches=alt_style_patches,
                style_cls=alt_style_cls,
                style_text_tokens=alt_style_text_tokens,
            )
            endpoint_img = _decode(vae, endpoint, str(device))
            endpoint_alt_img = _decode(vae, endpoint_alt, str(device))
            endpoint_lum = _luminance(endpoint_img)
            endpoint_alt_lum = _luminance(endpoint_alt_img)

            low_kernel = max(1, int(args.lowpass_kernel))
            source_low = _lowpass(source, low_kernel)
            target_low = _lowpass(target, low_kernel)
            endpoint_low = _lowpass(endpoint, low_kernel)
            source_high = source - source_low
            target_high = target - target_low
            endpoint_high = endpoint - endpoint_low

            row = {
                "sample_index": sample_index,
                "alt_sample_index": int(alt_index),
                "source_style_id": int(batch["source_style_id"].item()),
                "target_style_id": target_style_id_scalar,
                "alt_target_style_id": int(alt_style_id.item()),
                "t": float(t),
                "state_mode": str(args.state_mode),
                "style_distance_rms": float(style_distance),
                **_alpha_metrics(source, target, endpoint, "latent"),
                **_alpha_metrics(source_low, target_low, endpoint_low, "low"),
                **_alpha_metrics(source_high, target_high, endpoint_high, "high"),
                "endpoint_latent_std": float(endpoint.detach().float().std(unbiased=False).item()),
                "endpoint_latent_abs_mean": float(endpoint.detach().float().abs().mean().item()),
                "endpoint_to_source_rms": float((endpoint.detach().float() - source.detach().float()).square().mean().sqrt().item()),
                "endpoint_to_target_rms": float((endpoint.detach().float() - target.detach().float()).square().mean().sqrt().item()),
                "endpoint_img_std": float(endpoint_lum.std(unbiased=False).item()),
                "endpoint_img_mean": float(endpoint_lum.mean().item()),
                "endpoint_img_to_source_rms": float((endpoint_lum - source_lum).square().mean().sqrt().item()),
                "endpoint_img_to_target_rms": float((endpoint_lum - target_lum).square().mean().sqrt().item()),
                "style_swap_endpoint_rms": float((endpoint.detach().float() - endpoint_alt.detach().float()).square().mean().sqrt().item()),
                "style_swap_endpoint_img_rms": float((endpoint_lum - endpoint_alt_lum).square().mean().sqrt().item()),
            }
            row["style_sensitivity_latent"] = row["style_swap_endpoint_rms"] / max(row["style_distance_rms"], 1e-8)
            row["style_sensitivity_img"] = row["style_swap_endpoint_img_rms"] / max(row["style_distance_rms"], 1e-8)
            rows.append(row)
        collected += 1

    _write_csv(output_dir / "hypothesis_metrics.csv", rows)
    _write_csv(output_dir / "hypothesis_skipped_samples.csv", skipped)

    by_t: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_t.setdefault(f"{float(row['t']):.3f}", []).append(row)

    summary_by_t: dict[str, dict[str, float]] = {}
    mean_keys = [
        "latent_alpha_mean",
        "latent_shrink_gap",
        "latent_orth_over_delta",
        "low_alpha_mean",
        "high_alpha_mean",
        "style_sensitivity_latent",
        "style_sensitivity_img",
        "endpoint_to_source_rms",
        "endpoint_to_target_rms",
        "endpoint_img_to_source_rms",
        "endpoint_img_to_target_rms",
        "endpoint_img_std",
    ]
    for key, items in by_t.items():
        summary_by_t[key] = {
            metric: float(sum(float(item.get(metric, 0.0)) for item in items) / max(len(items), 1))
            for metric in mean_keys
        }

    t0 = summary_by_t.get("0.000", {})
    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "output_dir": str(output_dir),
        "device": str(device),
        "sample_count_requested": int(args.sample_count),
        "sample_count_collected": int(collected),
        "sample_scan_attempts": int(scan_offset),
        "state_mode": str(args.state_mode),
        "times": times,
        "summary_by_t": summary_by_t,
        "skipped_samples": skipped,
        "headline": {
            "t0_latent_alpha_mean": float(t0.get("latent_alpha_mean", 0.0)),
            "t0_latent_shrink_gap": float(t0.get("latent_shrink_gap", 0.0)),
            "t0_latent_orth_over_delta": float(t0.get("latent_orth_over_delta", 0.0)),
            "t0_low_alpha_mean": float(t0.get("low_alpha_mean", 0.0)),
            "t0_high_alpha_mean": float(t0.get("high_alpha_mean", 0.0)),
            "t0_style_sensitivity_latent": float(t0.get("style_sensitivity_latent", 0.0)),
            "t0_endpoint_img_to_source_rms": float(t0.get("endpoint_img_to_source_rms", 0.0)),
            "t0_endpoint_img_to_target_rms": float(t0.get("endpoint_img_to_target_rms", 0.0)),
            "t0_endpoint_img_std": float(t0.get("endpoint_img_std", 0.0)),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
