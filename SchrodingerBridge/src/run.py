from __future__ import annotations

import argparse
import logging
import gc
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config_schema import ExperimentConfig, load_experiment_config
from trainer import SBTrainer
from utils.dataset import AdaCUTLatentDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_cpu_threads(config: ExperimentConfig) -> None:
    train_cfg = config.training
    cpu_threads = train_cfg.cpu_threads
    cpu_interop_threads = train_cfg.cpu_interop_threads
    if cpu_threads is not None:
        try:
            torch.set_num_threads(int(cpu_threads))
        except Exception:
            pass
    if cpu_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(cpu_interop_threads))
        except Exception:
            pass


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def _resolve_num_workers(requested: int) -> int:
    if requested >= 0:
        return requested
    if os.name == "nt":
        return 0
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


def _run_full_eval_for_checkpoint(config: ExperimentConfig, checkpoint_path: Path) -> None:
    train_cfg = config.training
    out_dir = checkpoint_path.parent / "full_eval" / checkpoint_path.stem
    eval_script = Path(__file__).resolve().parent / "utils" / "run_evaluation.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(out_dir),
        "--test_dir",
        str(train_cfg.test_image_dir),
        "--cache_dir",
        str(train_cfg.full_eval_cache_dir),
        "--clip_hf_cache_dir",
        str(train_cfg.full_eval_clip_hf_cache_dir),
        "--batch_size",
        str(int(train_cfg.full_eval_batch_size)),
    ]
    if train_cfg.full_eval_num_steps is not None:
        cmd += ["--num_steps", str(int(train_cfg.full_eval_num_steps))]
    if train_cfg.full_eval_step_size is not None:
        cmd += ["--step_size", str(float(train_cfg.full_eval_step_size))]
    if train_cfg.full_eval_style_strength is not None:
        cmd += ["--style_strength", str(float(train_cfg.full_eval_style_strength))]
    if train_cfg.full_eval_max_src_samples is not None:
        cmd += ["--max_src_samples", str(int(train_cfg.full_eval_max_src_samples))]
    if train_cfg.full_eval_max_ref_compare is not None:
        cmd += ["--max_ref_compare", str(int(train_cfg.full_eval_max_ref_compare))]
    if train_cfg.full_eval_max_ref_cache is not None:
        cmd += ["--max_ref_cache", str(int(train_cfg.full_eval_max_ref_cache))]
    if train_cfg.full_eval_ref_feature_batch_size is not None:
        cmd += ["--ref_feature_batch_size", str(int(train_cfg.full_eval_ref_feature_batch_size))]
    if train_cfg.full_eval_target_chunk_size is not None:
        cmd += ["--target_chunk_size", str(int(train_cfg.full_eval_target_chunk_size))]
    if train_cfg.full_eval_vae_decode_batch_size is not None:
        cmd += ["--vae_decode_batch_size", str(int(train_cfg.full_eval_vae_decode_batch_size))]
    if train_cfg.full_eval_only_lpips_clip_style is not None:
        cmd += ["--eval_only_lpips_clip_style" if bool(train_cfg.full_eval_only_lpips_clip_style) else "--no-eval_only_lpips_clip_style"]
    if bool(train_cfg.full_eval_enable_introstyle):
        cmd.append("--eval_enable_introstyle")
    else:
        cmd.append("--no-eval_enable_introstyle")
    if str(train_cfg.full_eval_introstyle_style_bank_root).strip():
        cmd += ["--introstyle_style_bank_root", str(train_cfg.full_eval_introstyle_style_bank_root)]
    if str(train_cfg.full_eval_introstyle_model_id).strip():
        cmd += ["--introstyle_model_id", str(train_cfg.full_eval_introstyle_model_id)]
    if str(train_cfg.full_eval_introstyle_modelscope_id).strip():
        cmd += ["--introstyle_modelscope_id", str(train_cfg.full_eval_introstyle_modelscope_id)]
    if str(train_cfg.full_eval_introstyle_modelscope_cache_dir).strip():
        cmd += ["--introstyle_modelscope_cache_dir", str(train_cfg.full_eval_introstyle_modelscope_cache_dir)]
    if bool(train_cfg.full_eval_introstyle_allow_network):
        cmd.append("--introstyle_allow_network")
    cmd += ["--introstyle_bank_limit_per_style", str(int(train_cfg.full_eval_introstyle_bank_limit_per_style))]
    cmd += ["--introstyle_batch_size", str(int(train_cfg.full_eval_introstyle_batch_size))]
    cmd += ["--introstyle_topk", str(int(train_cfg.full_eval_introstyle_topk))]
    cmd += ["--introstyle_t", str(int(train_cfg.full_eval_introstyle_t))]
    cmd += ["--introstyle_up_ft_index", str(int(train_cfg.full_eval_introstyle_up_ft_index))]
    cmd += ["--introstyle_ensemble_size", str(int(train_cfg.full_eval_introstyle_ensemble_size))]
    if train_cfg.full_eval_save_generated_images is not None:
        cmd += ["--save_generated_images" if bool(train_cfg.full_eval_save_generated_images) else "--no-save_generated_images"]
    if train_cfg.full_eval_save_summary_grid is not None:
        cmd += ["--save_summary_grid" if bool(train_cfg.full_eval_save_summary_grid) else "--no-save_summary_grid"]
    if bool(train_cfg.full_eval_force_regen):
        cmd.append("--force_regen")
    if bool(train_cfg.full_eval_profile_timing):
        cmd.append("--profile_timing")
    if bool(train_cfg.full_eval_disable_lpips):
        cmd.append("--eval_disable_lpips")
    if bool(train_cfg.full_eval_enable_art_fid):
        cmd.append("--eval_enable_art_fid")
    else:
        cmd.append("--no-eval_enable_art_fid")
    if bool(train_cfg.full_eval_enable_kid):
        cmd.append("--eval_enable_kid")
    else:
        cmd.append("--no-eval_enable_kid")

    logger.info("Running full eval for %s -> %s", checkpoint_path, out_dir)
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    wall = time.perf_counter() - start
    logger.info("Full eval completed for %s in %.1fs", checkpoint_path.name, wall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train latent Schrodinger bridge model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config json")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path)

    if args.resume:
        config.training.resume_checkpoint = args.resume

    train_cfg = config.training
    seed = int(train_cfg.seed)
    _set_seed(seed)
    _set_cpu_threads(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Seed: %d", seed)

    data_cfg = config.data
    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.data_root,
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=bool(data_cfg.allow_hflip),
        identity_ratio=data_cfg.identity_ratio,
        batch_size_hint=int(train_cfg.batch_size),
        balance_target_styles_per_batch=bool(data_cfg.balance_target_styles_per_batch),
        preload_to_gpu=bool(data_cfg.preload_to_gpu),
        preload_max_vram_gb=float(data_cfg.preload_max_vram_gb),
        preload_reserve_ratio=float(data_cfg.preload_reserve_ratio),
        virtual_length_multiplier=float(data_cfg.virtual_length_multiplier),
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
        dino_cache_path=str(data_cfg.dino_cache_path),
        dino_cache_required=bool(data_cfg.dino_cache_required),
        dino_bank_limit_per_style=int(data_cfg.dino_bank_limit_per_style),
        device=str(device),
    )

    style_count = len(dataset.style_subdirs)
    if int(config.model.num_styles) != style_count:
        logger.warning("model.num_styles mismatch detected; forcing to %d", style_count)
        config.model.num_styles = style_count

    batch_size = int(train_cfg.batch_size)
    num_workers = _resolve_num_workers(int(train_cfg.num_workers))
    shuffle = bool(train_cfg.shuffle)
    persistent_workers = bool(train_cfg.persistent_workers and num_workers > 0)
    pin_memory = bool(train_cfg.pin_memory)
    generator = torch.Generator().manual_seed(seed)

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = max(1, int(train_cfg.prefetch_factor))
    dataloader = DataLoader(**dataloader_kwargs)

    logger.info(
        "DataLoader | batch=%d workers=%d shuffle=%s pin_memory=%s persistent_workers=%s preload_to_gpu=%s balanced_target=%s pairing_cache=%s pairing_mode=%s rank_schedule=%s active_topk=%d min_topk=%d curriculum_epochs=%d rank_power=%.3f topk=%d explore=%.3f/%d dual_mix=%.3f/%d aux_topk=%d",
        batch_size,
        num_workers,
        shuffle,
        pin_memory,
        persistent_workers,
        bool(getattr(dataset, "preload_to_gpu", False)),
        bool(getattr(dataset, "balance_target_styles_per_batch", False)),
        bool(getattr(dataset, "offline_pairing_map", {})),
        str(getattr(dataset, "pairing_cache_sample_mode", "")),
        str(getattr(dataset, "pairing_cache_rank_schedule", "")),
        int(getattr(dataset, "pairing_cache_active_topk", 0)),
        int(getattr(dataset, "pairing_cache_min_topk", 0)),
        int(getattr(dataset, "pairing_cache_curriculum_epochs", 0)),
        float(getattr(dataset, "pairing_cache_rank_power", 1.0)),
        int(getattr(dataset, "pairing_cache_topk", 0)),
        float(getattr(dataset, "pairing_cache_explore_prob", 0.0)),
        int(getattr(dataset, "pairing_cache_explore_topk", 0)),
        float(getattr(dataset, "pairing_cache_dual_target_mix", 0.0)),
        int(getattr(dataset, "pairing_cache_dual_target_topk", 0)),
        int(getattr(dataset, "pairing_cache_aux_target_topk", 0)),
    )

    trainer = SBTrainer(config=config, device=device, config_path=str(config_path))
    deferred_eval_checkpoints: list[Path] = []

    epoch = int(trainer.start_epoch)
    while epoch <= int(trainer.num_epochs):
        dataset.set_epoch(epoch)
        if bool(getattr(dataset, "offline_pairing_map", {})):
            logger.info(
                "Pairing cache epoch=%d active_topk=%d mode=%s rank_schedule=%s explore=%.3f/%d dual_mix=%.3f/%d aux_topk=%d",
                epoch,
                int(dataset.current_pairing_active_topk()),
                str(getattr(dataset, "pairing_cache_sample_mode", "")),
                str(getattr(dataset, "pairing_cache_rank_schedule", "")),
                float(getattr(dataset, "pairing_cache_explore_prob", 0.0)),
                int(getattr(dataset, "pairing_cache_explore_topk", 0)),
                float(getattr(dataset, "pairing_cache_dual_target_mix", 0.0)),
                int(getattr(dataset, "pairing_cache_dual_target_topk", 0)),
                int(getattr(dataset, "pairing_cache_aux_target_topk", 0)),
            )
        metrics = trainer.train_epoch(dataloader, epoch)
        trainer.step_scheduler()
        trainer.log_epoch(epoch, metrics)
        logger.info(
            "Epoch %d/%d | loss=%.4f flow=%.4f kin=%.4f ot=%.4f tswd=%.4f attn=%.3f k=%.3f ent=%.3f sigma=%.3f idr=%.2f t=%.3f |v|=%.3f lr=%.2e data=%.1fs comp=%.1fs peak=%.2f/%.2fGB",
            epoch,
            trainer.num_epochs,
            metrics.get("loss", 0.0),
            metrics.get("flow", 0.0),
            metrics.get("kinetic_energy", 0.0),
            metrics.get("ot_cost", 0.0),
            metrics.get("terminal_swd", 0.0),
            metrics.get("semantic_attn_mean", 0.0),
            metrics.get("semantic_k_abs", 0.0),
            metrics.get("plan_entropy", 0.0),
            metrics.get("bridge_sigma", 0.0),
            metrics.get("identity_ratio", 0.0),
            metrics.get("t_mean", 0.0),
            metrics.get("velocity_abs", 0.0),
            metrics.get("lr", 0.0),
            metrics.get("data_time_sec", 0.0),
            metrics.get("compute_time_sec", 0.0),
            metrics.get("cuda_peak_allocated_gb", 0.0),
            metrics.get("cuda_peak_reserved_gb", 0.0),
        )
        if epoch % trainer.save_interval == 0 or epoch == trainer.num_epochs or trainer.requested_stop:
            ckpt_path = trainer.save_checkpoint(epoch, metrics)
            if bool(train_cfg.full_eval_each_epoch):
                if hasattr(trainer, "wait_for_pending_checkpoints"):
                    trainer.wait_for_pending_checkpoints()
                if bool(getattr(train_cfg, "full_eval_defer_until_training_end", False)):
                    deferred_eval_checkpoints.append(ckpt_path)
                else:
                    _run_full_eval_for_checkpoint(config, ckpt_path)
        if trainer.requested_stop:
            logger.info("Early stop requested by training.stop_after_global_steps; ending training loop after epoch %d.", epoch)
            break
        epoch += 1

    if hasattr(trainer, "wait_for_pending_checkpoints"):
        trainer.wait_for_pending_checkpoints()
    if deferred_eval_checkpoints:
        logger.info("Deferred full eval queue contains %d checkpoints.", len(deferred_eval_checkpoints))
        del trainer
        del dataloader
        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for ckpt_path in deferred_eval_checkpoints:
            _run_full_eval_for_checkpoint(config, ckpt_path)
    logger.info("Training completed.")


if __name__ == "__main__":
    main()
