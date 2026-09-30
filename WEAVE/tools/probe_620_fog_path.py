from __future__ import annotations

import argparse
import csv
import json
import math
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def _lowpass(x: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    return F.avg_pool2d(x.float(), kernel_size=k, stride=1, padding=k // 2)


def _latent_metrics(x: torch.Tensor) -> dict[str, float]:
    xf = x.detach().float()
    low = _lowpass(xf, 5)
    high = xf - low
    return {
        "latent_mean": float(xf.mean().item()),
        "latent_std": float(xf.std(unbiased=False).item()),
        "latent_abs_mean": float(xf.abs().mean().item()),
        "latent_low_std": float(low.std(unbiased=False).item()),
        "latent_high_rms": float(high.square().mean().sqrt().item()),
        "latent_low_abs": float(low.abs().mean().item()),
    }


def _image_metrics(img: torch.Tensor) -> dict[str, float]:
    xf = img.detach().float().clamp(0.0, 1.0)
    lum = 0.299 * xf[:, 0:1] + 0.587 * xf[:, 1:2] + 0.114 * xf[:, 2:3]
    dx = F.pad(lum[..., :, 1:] - lum[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(lum[..., 1:, :] - lum[..., :-1, :], (0, 0, 0, 1))
    grad = torch.sqrt(dx.square() + dy.square() + 1e-8)
    low = _lowpass(lum, 9)
    high = lum - low
    return {
        "img_mean": float(lum.mean().item()),
        "img_std": float(lum.std(unbiased=False).item()),
        "img_grad_rms": float(grad.square().mean().sqrt().item()),
        "img_high_rms": float(high.square().mean().sqrt().item()),
    }


def _delta_metrics(x: torch.Tensor, ref: torch.Tensor, prefix: str) -> dict[str, float]:
    diff = (x.detach().float() - ref.detach().float())
    return {
        f"{prefix}_delta_abs": float(diff.abs().mean().item()),
        f"{prefix}_delta_rms": float(diff.square().mean().sqrt().item()),
    }


def _prepare_batch(item: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            out[key] = value.unsqueeze(0).to(device=device)
        elif isinstance(value, int):
            out[key] = torch.tensor([value], device=device)
    return out


def _stage_payload(
    *,
    model: torch.nn.Module,
    content: torch.Tensor,
    target_style_id: torch.Tensor,
    style_patches: torch.Tensor | None,
    style_cls: torch.Tensor | None,
    style_text_tokens: torch.Tensor | None,
    num_steps: int,
    sigma_override: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_sigma = None
    if sigma_override is not None and hasattr(model, "bridge_sigma"):
        original_sigma = float(getattr(model, "bridge_sigma"))
        setattr(model, "bridge_sigma", float(sigma_override))
    try:
        with torch.no_grad():
            endpoint = model.predict_endpoint(
                content,
                t=torch.zeros((content.shape[0],), device=content.device, dtype=content.dtype),
                style_id=target_style_id,
                style_dino_patches=style_patches,
                style_dino_cls=style_cls,
                style_text_tokens=style_text_tokens,
            )
            generated = model.integrate(
                content,
                style_id=target_style_id,
                num_steps=max(1, int(num_steps)),
                step_size=1.0,
                style_dino_patches=style_patches,
                style_dino_cls=style_cls,
                style_text_tokens=style_text_tokens,
            )
        return endpoint, generated
    finally:
        if original_sigma is not None:
            setattr(model, "bridge_sigma", float(original_sigma))


def _decode(vae, latent: torch.Tensor, device: str) -> torch.Tensor:
    with torch.no_grad():
        return decode_latent(vae, latent, device=device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe where 620 whitening/fog enters the real inference path.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--dataset-seed-epoch", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--sigma-override", type=float, default=None)
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
    sample_summaries: list[dict[str, Any]] = []
    skipped_samples: list[dict[str, Any]] = []
    stages = sorted({max(1, int(s)) for s in args.steps})

    wanted_samples = max(1, int(args.sample_count))
    max_scans = max(wanted_samples, wanted_samples * max(1, int(args.max_scan_multiplier)))
    collected = 0
    scan_offset = 0
    while collected < wanted_samples and scan_offset < max_scans:
        sample_index = int(args.start_index) + scan_offset
        scan_offset += 1
        try:
            item = dataset[sample_index % len(dataset)]
        except KeyError as exc:
            skipped_samples.append(
                {
                    "sample_index": sample_index,
                    "reason": "dataset_keyerror",
                    "message": str(exc),
                }
            )
            continue
        batch = _prepare_batch(item, device)
        content = batch["content"]
        target = batch["target_style"]
        target_style_id = batch["target_style_id"].long()
        style_patches = batch.get("target_style_dino_patches")
        style_cls = batch.get("target_style_dino_cls")
        style_text_tokens = batch.get("target_style_text_tokens")

        source_img = _decode(vae, content, str(device))
        target_img = _decode(vae, target, str(device))

        base_row = {
            "sample_index": sample_index,
            "source_style_id": int(batch["source_style_id"].item()),
            "target_style_id": int(target_style_id.item()),
        }
        source_stage = {
            **base_row,
            "stage": "source_latent",
            **_latent_metrics(content),
            **_image_metrics(source_img),
            **_delta_metrics(content, target, "to_target_latent"),
            **_delta_metrics(source_img, target_img, "to_target_img"),
        }
        target_stage = {
            **base_row,
            "stage": "target_latent",
            **_latent_metrics(target),
            **_image_metrics(target_img),
            **_delta_metrics(target, content, "to_source_latent"),
            **_delta_metrics(target_img, source_img, "to_source_img"),
        }
        rows.extend([source_stage, target_stage])

        endpoint, _ = _stage_payload(
            model=model,
            content=content,
            target_style_id=target_style_id,
            style_patches=style_patches,
            style_cls=style_cls,
            style_text_tokens=style_text_tokens,
            num_steps=1,
            sigma_override=args.sigma_override,
        )
        endpoint_img = _decode(vae, endpoint, str(device))
        endpoint_row = {
            **base_row,
            "stage": "predict_endpoint_t0",
            "num_steps": 0,
            "sigma_eval": float(args.sigma_override) if args.sigma_override is not None else float(getattr(model, "bridge_sigma", 0.0)),
            **_latent_metrics(endpoint),
            **_image_metrics(endpoint_img),
            **_delta_metrics(endpoint, content, "to_source_latent"),
            **_delta_metrics(endpoint, target, "to_target_latent"),
            **_delta_metrics(endpoint_img, source_img, "to_source_img"),
            **_delta_metrics(endpoint_img, target_img, "to_target_img"),
        }
        rows.append(endpoint_row)

        sample_summary = {
            "sample_index": sample_index,
            "source_style_id": int(batch["source_style_id"].item()),
            "target_style_id": int(target_style_id.item()),
            "endpoint_latent_high_vs_source_ratio": endpoint_row["latent_high_rms"] / max(source_stage["latent_high_rms"], 1e-8),
            "endpoint_img_grad_vs_source_ratio": endpoint_row["img_grad_rms"] / max(source_stage["img_grad_rms"], 1e-8),
        }

        for num_steps in stages:
            _, generated = _stage_payload(
                model=model,
                content=content,
                target_style_id=target_style_id,
                style_patches=style_patches,
                style_cls=style_cls,
                style_text_tokens=style_text_tokens,
                num_steps=num_steps,
                sigma_override=args.sigma_override,
            )
            gen_img = _decode(vae, generated, str(device))
            row = {
                **base_row,
                "stage": f"integrate_nfe_{num_steps}",
                "num_steps": int(num_steps),
                "sigma_eval": float(args.sigma_override) if args.sigma_override is not None else float(getattr(model, "bridge_sigma", 0.0)),
                **_latent_metrics(generated),
                **_image_metrics(gen_img),
                **_delta_metrics(generated, content, "to_source_latent"),
                **_delta_metrics(generated, target, "to_target_latent"),
                **_delta_metrics(gen_img, source_img, "to_source_img"),
                **_delta_metrics(gen_img, target_img, "to_target_img"),
            }
            rows.append(row)
            sample_summary[f"nfe_{num_steps}_latent_high_vs_source_ratio"] = row["latent_high_rms"] / max(
                source_stage["latent_high_rms"], 1e-8
            )
            sample_summary[f"nfe_{num_steps}_img_grad_vs_source_ratio"] = row["img_grad_rms"] / max(
                source_stage["img_grad_rms"], 1e-8
            )
            sample_summary[f"nfe_{num_steps}_img_std_vs_source_ratio"] = row["img_std"] / max(
                source_stage["img_std"], 1e-8
            )
        sample_summaries.append(sample_summary)
        collected += 1

    _write_csv(output_dir / "fog_stage_metrics.csv", rows)
    _write_csv(output_dir / "fog_sample_summary.csv", sample_summaries)
    _write_csv(output_dir / "fog_skipped_samples.csv", skipped_samples)

    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_stage.setdefault(str(row["stage"]), []).append(row)

    stage_summary: dict[str, dict[str, float]] = {}
    for stage_name, stage_rows in by_stage.items():
        keys = [
            "latent_std",
            "latent_high_rms",
            "img_std",
            "img_grad_rms",
            "to_source_latent_delta_rms",
            "to_target_latent_delta_rms",
            "to_source_img_delta_rms",
            "to_target_img_delta_rms",
        ]
        stage_summary[stage_name] = {
            key: float(sum(float(r.get(key, 0.0)) for r in stage_rows) / max(len(stage_rows), 1))
            for key in keys
        }

    source_stats = stage_summary.get("source_latent", {})
    endpoint_stats = stage_summary.get("predict_endpoint_t0", {})
    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "output_dir": str(output_dir),
        "device": str(device),
        "sample_count": int(args.sample_count),
        "sample_count_collected": int(collected),
        "sample_scan_attempts": int(scan_offset),
        "start_index": int(args.start_index),
        "max_scan_multiplier": int(args.max_scan_multiplier),
        "steps": stages,
        "sigma_eval": float(args.sigma_override) if args.sigma_override is not None else float(getattr(model, "bridge_sigma", 0.0)),
        "data_root_override": str(args.data_root_override or ""),
        "latent_cache_dir_override": str(args.latent_cache_dir_override or ""),
        "pairing_cache_override": str(args.pairing_cache_override or ""),
        "dino_cache_override": str(args.dino_cache_override or ""),
        "stage_summary": stage_summary,
        "skipped_samples": skipped_samples,
        "headline": {
            "endpoint_latent_high_vs_source_ratio": float(endpoint_stats.get("latent_high_rms", 0.0) / max(source_stats.get("latent_high_rms", 1e-8), 1e-8)),
            "endpoint_img_grad_vs_source_ratio": float(endpoint_stats.get("img_grad_rms", 0.0) / max(source_stats.get("img_grad_rms", 1e-8), 1e-8)),
            "endpoint_img_std_vs_source_ratio": float(endpoint_stats.get("img_std", 0.0) / max(source_stats.get("img_std", 1e-8), 1e-8)),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
