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
        dino_cache_path=str(dino_cache_override or data_cfg.dino_cache_path),
        dino_cache_required=bool(data_cfg.dino_cache_required),
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


def _lowpass(x: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    return F.avg_pool2d(x.float(), kernel_size=k, stride=1, padding=k // 2)


def _image_lowpass(img: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    xf = img.detach().float().clamp(0.0, 1.0)
    lum = 0.299 * xf[:, 0:1] + 0.587 * xf[:, 1:2] + 0.114 * xf[:, 2:3]
    return _lowpass(lum, kernel)


def _rms(x: torch.Tensor) -> float:
    return float(x.detach().float().square().mean().sqrt().item())


def _mean_abs(x: torch.Tensor) -> float:
    return float(x.detach().float().abs().mean().item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().float().reshape(a.shape[0], -1)
    bv = b.detach().float().reshape(b.shape[0], -1)
    cos = F.cosine_similarity(av, bv, dim=1, eps=1e-8)
    return float(cos.mean().item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace 620 solver steps to separate endpoint drift from solver re-anchoring.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--sigma-override", type=float, default=None)
    parser.add_argument("--vae-model", type=str, default="ema")
    parser.add_argument("--vae-cache-dir", type=str, default="")
    parser.add_argument("--data-root-override", type=str, default="")
    parser.add_argument("--latent-cache-dir-override", type=str, default="")
    parser.add_argument("--pairing-cache-override", type=str, default="")
    parser.add_argument("--dino-cache-override", type=str, default="")
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
    model = _build_model(cfg, args.checkpoint, device)
    vae = load_vae(
        device=str(device),
        model_id=str(args.vae_model),
        cache_dir=str(args.vae_cache_dir or None) if args.vae_cache_dir else None,
    )

    item = dataset[int(args.sample_index) % len(dataset)]
    batch = _prepare_batch(item, device)
    h = batch["content"]
    source = batch["content"]
    target = batch["target_style"]
    target_style_id = batch["target_style_id"].long()
    style_patches = batch.get("target_style_dino_patches")
    style_cls = batch.get("target_style_dino_cls")
    style_text_tokens = batch.get("target_style_text_tokens")

    source_img = decode_latent(vae, source, device=str(device))
    target_img = decode_latent(vae, target, device=str(device))
    source_low = _image_lowpass(source_img)
    target_low = _image_lowpass(target_img)

    steps = max(1, int(args.num_steps))
    sigma_eval = float(args.sigma_override) if args.sigma_override is not None else float(getattr(model, "bridge_sigma", 0.0))
    original_sigma = None
    if args.sigma_override is not None and hasattr(model, "bridge_sigma"):
        original_sigma = float(getattr(model, "bridge_sigma"))
        setattr(model, "bridge_sigma", sigma_eval)

    rows: list[dict[str, Any]] = []
    try:
        for idx in range(steps):
            t_curr = idx / float(steps)
            t_next = (idx + 1) / float(steps)
            t_batch = torch.full((h.shape[0],), t_curr, device=h.device, dtype=h.dtype)
            with torch.no_grad():
                endpoint = model.predict_endpoint(
                    h,
                    t=t_batch,
                    style_id=target_style_id,
                    style_dino_patches=style_patches,
                    style_dino_cls=style_cls,
                    style_text_tokens=style_text_tokens,
                )
            denom = max(1e-6, 1.0 - t_curr)
            c_curr = (1.0 - t_next) / denom
            c_tgt = (t_next - t_curr) / denom
            step_mean = c_curr * h + c_tgt * endpoint
            if sigma_eval > 0.0:
                var = (sigma_eval ** 2) * ((t_next - t_curr) * max(0.0, 1.0 - t_next) / denom)
                h_next = step_mean + torch.randn_like(step_mean) * (var ** 0.5) if var > 0.0 else step_mean
            else:
                var = 0.0
                h_next = step_mean

            current_img = decode_latent(vae, h, device=str(device))
            endpoint_img = decode_latent(vae, endpoint, device=str(device))
            next_img = decode_latent(vae, h_next, device=str(device))
            current_low = _image_lowpass(current_img)
            endpoint_low = _image_lowpass(endpoint_img)
            next_low = _image_lowpass(next_img)

            endpoint_delta = endpoint - h
            step_delta = h_next - h
            to_source = source - h
            to_target = target - h
            img_to_source = source_low - current_low
            img_to_target = target_low - current_low
            step_img_delta = next_low - current_low
            endpoint_img_delta = endpoint_low - current_low

            rows.append(
                {
                    "step": idx + 1,
                    "t_curr": t_curr,
                    "t_next": t_next,
                    "sigma_eval": sigma_eval,
                    "c_curr": c_curr,
                    "c_tgt": c_tgt,
                    "variance": var,
                    "endpoint_delta_rms": _rms(endpoint_delta),
                    "step_delta_rms": _rms(step_delta),
                    "step_vs_endpoint_scale": _rms(step_delta) / max(_rms(endpoint_delta), 1e-8),
                    "endpoint_low_delta_abs": _mean_abs(endpoint_img_delta),
                    "step_low_delta_abs": _mean_abs(step_img_delta),
                    "step_low_to_source_cos": _cosine(step_img_delta, img_to_source),
                    "step_low_to_target_cos": _cosine(step_img_delta, img_to_target),
                    "endpoint_low_to_source_cos": _cosine(endpoint_img_delta, img_to_source),
                    "endpoint_low_to_target_cos": _cosine(endpoint_img_delta, img_to_target),
                    "step_latent_to_source_cos": _cosine(step_delta, to_source),
                    "step_latent_to_target_cos": _cosine(step_delta, to_target),
                    "dist_source_low_before": _mean_abs(current_low - source_low),
                    "dist_target_low_before": _mean_abs(current_low - target_low),
                    "dist_source_low_after": _mean_abs(next_low - source_low),
                    "dist_target_low_after": _mean_abs(next_low - target_low),
                    "img_std_before": float(current_low.std(unbiased=False).item()),
                    "img_std_endpoint": float(endpoint_low.std(unbiased=False).item()),
                    "img_std_after": float(next_low.std(unbiased=False).item()),
                }
            )
            h = h_next
    finally:
        if original_sigma is not None:
            setattr(model, "bridge_sigma", float(original_sigma))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "solver_trace.csv", rows)

    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "output_dir": str(output_dir),
        "device": str(device),
        "sample_index": int(args.sample_index),
        "num_steps": int(steps),
        "sigma_eval": sigma_eval,
        "headline": {
            "avg_step_vs_endpoint_scale": float(sum(r["step_vs_endpoint_scale"] for r in rows) / max(len(rows), 1)),
            "avg_step_low_to_source_cos": float(sum(r["step_low_to_source_cos"] for r in rows) / max(len(rows), 1)),
            "avg_step_low_to_target_cos": float(sum(r["step_low_to_target_cos"] for r in rows) / max(len(rows), 1)),
            "low_distance_source_delta": float(rows[-1]["dist_source_low_after"] - rows[0]["dist_source_low_before"]),
            "low_distance_target_delta": float(rows[-1]["dist_target_low_after"] - rows[0]["dist_target_low_before"]),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
