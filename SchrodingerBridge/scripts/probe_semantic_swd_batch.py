from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config_schema import load_experiment_config  # noqa: E402
from trainer import SBTrainer  # noqa: E402
from utils.dataset import AdaCUTLatentDataset  # noqa: E402


def _build_dataset(config, device: torch.device, batch_size: int) -> AdaCUTLatentDataset:
    data_cfg = config.data
    return AdaCUTLatentDataset(
        data_root=data_cfg.data_root,
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=False,
        identity_ratio=data_cfg.identity_ratio,
        batch_size_hint=batch_size,
        balance_target_styles_per_batch=bool(data_cfg.balance_target_styles_per_batch),
        preload_to_gpu=False,
        preload_max_vram_gb=0.0,
        preload_reserve_ratio=float(data_cfg.preload_reserve_ratio),
        virtual_length_multiplier=1.0,
        content_style_sampling_weights=data_cfg.content_style_sampling_weights,
        target_style_sampling_weights=data_cfg.target_style_sampling_weights,
        pairing_cache_path=data_cfg.pairing_cache_path,
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
        latent_cache_dir=str(data_cfg.latent_cache_dir),
        style_caption_path=str(getattr(data_cfg, "style_caption_path", "")),
        device=str(device),
    )


def _tensor_stats(value: torch.Tensor | None) -> dict[str, object] | None:
    if value is None or not torch.is_tensor(value):
        return None
    v = value.detach().float().cpu()
    return {
        "shape": list(v.shape),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "abs_max": float(v.abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe one real batch for semantic SWD activation.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/semantic_swd_musiq/semantic_swd_guided_cons5.json"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path)
    config.training.batch_size = int(args.batch_size)
    config.training.num_workers = 0
    config.training.use_amp = False
    config.training.gpu_monitor_enabled = False
    config.training.full_eval_defer_until_training_end = True
    config.checkpoint.save_dir = str(PROJECT_ROOT / "exp/_debug_semantic_swd_probe")

    device = torch.device(args.device)
    dataset = _build_dataset(config, device, int(args.batch_size))
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, drop_last=True, num_workers=0)
    trainer = SBTrainer(config=config, device=device, config_path=str(config_path))
    raw_batch = next(iter(loader))
    batch = trainer._move_batch(raw_batch)

    trainer.model.train()
    with torch.no_grad():
        metrics = trainer.loss_fn.compute(
            trainer.model,
            content=batch["content"],
            target_style=batch["target_style"],
            target_style_id=batch["target_style_id"],
            source_style_id=batch.get("source_style_id"),
            aux_target_style=batch.get("aux_target_style"),
            aux_target_valid=batch.get("aux_target_valid"),
            conditioning=batch,
        )

    wanted = [
        "loss",
        "flow",
        "loss_swd",
        "loss_swd_ss",
        "single_step_swd",
        "swd_guidance_active",
        "swd_guidance_mean",
        "swd_guidance_std",
    ]
    selected_metrics = {
        key: float(metrics[key].detach().float().mean().cpu())
        for key in wanted
        if key in metrics and torch.is_tensor(metrics[key])
    }
    debug = {
        key: float(value.detach().float().mean().cpu())
        for key, value in getattr(trainer.model, "last_debug", {}).items()
        if torch.is_tensor(value)
    }
    report = {
        "config": str(config_path),
        "device": str(device),
        "loss_fn": type(trainer.loss_fn).__name__,
        "model": type(trainer.model).__name__,
        "batch_keys": sorted(batch.keys()),
        "content_shape": list(batch["content"].shape),
        "target_shape": list(batch["target_style"].shape),
        "target_style_ids": batch["target_style_id"].detach().cpu().tolist(),
        "metrics": selected_metrics,
        "model_debug": debug,
        "last_cross_attn_guidance": _tensor_stats(getattr(trainer.model, "last_cross_attn_guidance", None)),
        "last_pixel_entropy": _tensor_stats(getattr(trainer.model, "last_pixel_entropy", None)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
