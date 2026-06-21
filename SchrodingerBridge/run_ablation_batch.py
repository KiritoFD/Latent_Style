"""
run_ablation_batch.py — Single-process focused ablation runner.

Loads the shared dataset (packed latents, pairing cache, DINO cache) exactly
ONCE at startup, then runs every experiment by rebuilding only the model and
optimizer.  Eliminates ~12–15 s of per-experiment reload overhead.

Pairing cache is served from POSIX shared memory after the first load
(see src/utils/dataset.py:_load_pairing_cache).

Usage (from SchrodingerBridge/):
    python run_ablation_batch.py \\
        --names-file /tmp/ablation_names.txt \\
        --configs-dir tools/massive_ablation/configs_focused \\
        --exp-base   exp/620_focused_ablation \\
        [--resume-from EXPERIMENT_NAME]   # skip until this name
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path BEFORE any project imports.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config_schema import ExperimentConfig, load_experiment_config
from style_families import runtime_conditioning_requires_dino
from trainer import SBTrainer
from utils.dataset import AdaCUTLatentDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants copied / adapted from src/run.py
# ---------------------------------------------------------------------------
FULL_EVAL_RUNTIME_COLUMNS = [
    "checkpoint", "output_dir", "wall_sec", "summary_wall_total_sec",
    "timing_lancet_generation_sec", "timing_lpips_sec", "timing_clip_sec",
    "generated_count", "transfer_clip_style", "transfer_content_lpips",
    "allpairs_clip_style", "allpairs_content_lpips",
]


# ---------------------------------------------------------------------------
# Helpers (mirrors of src/run.py private helpers)
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_cpu_threads(config: ExperimentConfig) -> None:
    train_cfg = config.training
    cpu_threads = getattr(train_cfg, "cpu_threads", None)
    cpu_interop = getattr(train_cfg, "cpu_interop_threads", None)
    if cpu_threads is not None:
        try:
            torch.set_num_threads(int(cpu_threads))
        except Exception:
            pass
    if cpu_interop is not None:
        try:
            torch.set_num_interop_threads(int(cpu_interop))
        except Exception:
            pass


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
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


def _resolve_eval_subdir(train_cfg: object) -> str:
    eval_subdir = str(getattr(train_cfg, "full_eval_output_subdir", "full_eval") or "full_eval").strip()
    return eval_subdir or "full_eval"


def _load_json_object(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        logger.exception("Failed to parse JSON at %s", path)
        return None
    return payload if isinstance(payload, dict) else None


def _append_full_eval_runtime_row(*, checkpoint_path: Path, out_dir: Path, wall_sec: float) -> None:
    summary_path = out_dir / "summary.json"
    timings: dict = {}
    transfer: dict = {}
    allpairs: dict = {}
    generated_count = 0
    summary_wall = 0.0
    if summary_path.is_file():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            timings = dict(summary.get("timings_sec") or {})
            analysis = dict(summary.get("analysis") or {})
            transfer = dict(analysis.get("style_transfer_ability") or {})
            allpairs = dict(analysis.get("all_pairs_overview") or {})
            generated_count = int(summary.get("generated_count", 0) or 0)
            summary_wall = float(timings.get("wall_total", 0.0) or 0.0)
        except Exception:
            logger.exception("Failed to parse full-eval summary at %s", summary_path)
    log_dir = checkpoint_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "full_eval_runtime.csv"
    row = {
        "checkpoint": checkpoint_path.name,
        "output_dir": str(out_dir),
        "wall_sec": float(wall_sec),
        "summary_wall_total_sec": summary_wall,
        "timing_lancet_generation_sec": float(timings.get("lancet_generation", 0.0) or 0.0),
        "timing_lpips_sec": float(timings.get("lpips", 0.0) or 0.0),
        "timing_clip_sec": float(timings.get("clip", 0.0) or 0.0),
        "generated_count": int(generated_count),
        "transfer_clip_style": float(transfer.get("clip_style", 0.0) or 0.0),
        "transfer_content_lpips": float(transfer.get("content_lpips", 0.0) or 0.0),
        "allpairs_clip_style": float(allpairs.get("clip_style", 0.0) or 0.0),
        "allpairs_content_lpips": float(allpairs.get("content_lpips", 0.0) or 0.0),
    }
    write_header = not csv_path.is_file()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FULL_EVAL_RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Full-eval subprocess call (identical logic to src/run.py)
# ---------------------------------------------------------------------------

def _run_full_eval_for_checkpoint(config: ExperimentConfig, checkpoint_path: Path) -> dict | None:
    """Launch run_evaluation.py as a subprocess (identical to src/run.py)."""
    # Import the helper from the canonical location so we don't duplicate the
    # 100-line command-builder.  We call it through the existing src/run module.
    run_mod = importlib.import_module("run")  # src/run.py
    return run_mod._run_full_eval_for_checkpoint(config, checkpoint_path)


# ---------------------------------------------------------------------------
# Dataset fingerprint – used to detect mismatched data configs
# ---------------------------------------------------------------------------

def _data_fingerprint(config: ExperimentConfig) -> str:
    data_cfg = config.data
    return json.dumps(
        {
            "data_root": str(data_cfg.data_root),
            "style_subdirs": list(data_cfg.style_subdirs),
            "pairing_cache_path": str(getattr(data_cfg, "pairing_cache_path", "")),
            "dino_cache_path": str(getattr(data_cfg, "dino_cache_path", "")),
            "latent_cache_mode": str(getattr(data_cfg, "latent_cache_mode", "off")),
            "latent_cache_dir": str(getattr(data_cfg, "latent_cache_dir", "")),
        },
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# Build the shared dataset from a reference config
# ---------------------------------------------------------------------------

def build_shared_dataset(config: ExperimentConfig, device: torch.device) -> AdaCUTLatentDataset:
    data_cfg = config.data
    contract_family = str(getattr(config.model, "contract_family", "legacy") or "legacy").strip().lower()
    needs_dino = contract_family == "620_spatial_bridge" or runtime_conditioning_requires_dino(
        tokenizer_family=str(getattr(config.model, "tokenizer_family", "legacy_factorized")),
        semantic_supervision_family=str(getattr(config.bridge, "semantic_supervision_family", "legacy_terminal_swd")),
    )
    dino_cache_path = str(data_cfg.dino_cache_path) if needs_dino else ""
    dino_cache_required = bool(data_cfg.dino_cache_required) if needs_dino else False

    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.data_root,
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=bool(data_cfg.allow_hflip),
        identity_ratio=data_cfg.identity_ratio,
        batch_size_hint=int(config.training.batch_size),
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
        dino_cache_path=dino_cache_path,
        dino_cache_required=dino_cache_required,
        dino_bank_limit_per_style=int(data_cfg.dino_bank_limit_per_style),
        style_caption_path=str(getattr(data_cfg, "style_caption_path", "")),
        device=str(device),
    )
    logger.info(
        "Shared dataset loaded: %d styles, %d total samples (pairing_cache=%s, dino=%s).",
        len(dataset.style_subdirs),
        dataset.content_count,
        bool(dataset.offline_pairing_map),
        bool(dataset.dino_item_sidecars),
    )
    return dataset


# ---------------------------------------------------------------------------
# Per-experiment training loop
# ---------------------------------------------------------------------------

def _eval_convergence_requests_stop(train_cfg: object, payload: dict | None, *, epoch: int) -> bool:
    if not bool(getattr(train_cfg, "full_eval_stop_on_convergence", False)):
        return False
    if not isinstance(payload, dict):
        return False
    stop_ready = payload.get("stop_ready")
    if stop_ready is None:
        stop_ready = bool(payload.get("converged")) or bool(payload.get("objective_patience_converged"))
    return bool(stop_ready) and int(epoch) >= max(0, int(getattr(train_cfg, "full_eval_convergence_min_epochs", 0)))


def _load_existing_eval_convergence(config: ExperimentConfig, checkpoint_dir: Path) -> dict | None:
    eval_subdir = _resolve_eval_subdir(config.training)
    return _load_json_object(checkpoint_dir / eval_subdir / "round2_convergence.json")


def run_one_experiment(
    *,
    name: str,
    config: ExperimentConfig,
    config_path: Path,
    dataset: AdaCUTLatentDataset,
    device: torch.device,
) -> int:
    """Run training + full-eval for one ablation experiment.

    The *dataset* object is shared across experiments; only the model,
    optimizer, scheduler, and trainer state are rebuilt each time.

    Returns 0 on success, non-zero on failure.
    """
    train_cfg = config.training
    seed = int(train_cfg.seed)
    _set_seed(seed)
    _set_cpu_threads(config)

    # Validate num_styles matches dataset
    style_count = len(dataset.style_subdirs)
    if int(config.model.num_styles) != style_count:
        logger.warning("[%s] model.num_styles mismatch; forcing to %d", name, style_count)
        config.model.num_styles = style_count

    # Rebuild DataLoader (lightweight – dataset is shared)
    batch_size = int(train_cfg.batch_size)
    num_workers = _resolve_num_workers(int(train_cfg.num_workers))
    shuffle = bool(train_cfg.shuffle)
    persistent_workers = bool(train_cfg.persistent_workers and num_workers > 0)
    pin_memory = bool(train_cfg.pin_memory)
    generator = torch.Generator().manual_seed(seed)
    dataloader_kwargs: dict = dict(
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
        dataloader_kwargs["prefetch_factor"] = max(1, int(getattr(train_cfg, "prefetch_factor", 2)))
    dataloader = DataLoader(**dataloader_kwargs)

    logger.info(
        "[%s] DataLoader | batch=%d workers=%d shuffle=%s pin_memory=%s pairing=%s dino=%s",
        name, batch_size, num_workers, shuffle, pin_memory,
        bool(dataset.offline_pairing_map), bool(dataset.dino_item_sidecars),
    )

    # Build trainer (model + optimizer + scheduler)
    trainer = SBTrainer(config=config, device=device, config_path=str(config_path))

    existing_convergence = _load_existing_eval_convergence(config, trainer.checkpoint_dir)
    if _eval_convergence_requests_stop(train_cfg, existing_convergence, epoch=int(trainer.start_epoch) - 1):
        trainer.requested_stop = True
        logger.info("[%s] Already converged before training loop.", name)

    deferred_eval_checkpoints: list[Path] = []
    epoch = int(trainer.start_epoch)

    try:
        while epoch <= int(trainer.num_epochs):
            if trainer.requested_stop:
                logger.info("[%s] Eval convergence already satisfied; skipping epoch %d.", name, epoch)
                break

            dataset.set_epoch(epoch)
            if bool(getattr(dataset, "offline_pairing_map", {})):
                logger.info(
                    "[%s] Pairing cache epoch=%d active_topk=%d mode=%s",
                    name, epoch,
                    int(dataset.current_pairing_active_topk()),
                    str(getattr(dataset, "pairing_cache_sample_mode", "")),
                )

            metrics = trainer.train_epoch(dataloader, epoch)
            trainer.step_scheduler()
            trainer.log_epoch(epoch, metrics)
            logger.info(
                "[%s] Epoch %d/%d | loss=%.4f flow=%.4f "
                "data=%.1fs comp=%.1fs epoch=%.1fs sps=%.2f "
                "gpu=%.1f/%.1f%% vram=%.2f/%.2fGB",
                name, epoch, trainer.num_epochs,
                metrics.get("loss", 0.0), metrics.get("flow", 0.0),
                metrics.get("data_time_sec", 0.0), metrics.get("compute_time_sec", 0.0),
                metrics.get("epoch_time_sec", 0.0), metrics.get("samples_per_sec", 0.0),
                metrics.get("gpu_util_mean", 0.0), metrics.get("gpu_util_peak", 0.0),
                metrics.get("gpu_vram_used_gb_mean", 0.0), metrics.get("gpu_vram_used_gb_peak", 0.0),
            )

            if epoch % trainer.save_interval == 0 or epoch == trainer.num_epochs or trainer.requested_stop:
                ckpt_path = trainer.save_checkpoint(epoch, metrics)
                if hasattr(trainer, "wait_for_pending_checkpoints"):
                    trainer.wait_for_pending_checkpoints()

                if bool(getattr(train_cfg, "full_eval_defer_until_training_end", False)):
                    deferred_eval_checkpoints.append(ckpt_path)
                elif bool(train_cfg.full_eval_each_epoch):
                    eval_offloaded = False
                    convergence_payload = None
                    if hasattr(trainer, "offload_for_full_eval"):
                        trainer.offload_for_full_eval()
                        eval_offloaded = True
                    try:
                        convergence_payload = _run_full_eval_for_checkpoint(config, ckpt_path)
                    finally:
                        if eval_offloaded and hasattr(trainer, "restore_after_full_eval"):
                            trainer.restore_after_full_eval()
                    if _eval_convergence_requests_stop(train_cfg, convergence_payload, epoch=epoch):
                        trainer.requested_stop = True
                        logger.info("[%s] Early stop by eval convergence at epoch %d.", name, epoch)

            if trainer.requested_stop:
                logger.info("[%s] Early stop after epoch %d.", name, epoch)
                break
            epoch += 1

    finally:
        if hasattr(trainer, "wait_for_pending_checkpoints"):
            trainer.wait_for_pending_checkpoints()

    # Deferred eval (full_eval_defer_until_training_end=true)
    if deferred_eval_checkpoints:
        logger.info("[%s] Running %d deferred full-evals.", name, len(deferred_eval_checkpoints))
        # Offload model to CPU so the eval subprocess gets the full GPU budget.
        if hasattr(trainer, "offload_for_full_eval"):
            trainer.offload_for_full_eval()
        try:
            for ckpt_path in deferred_eval_checkpoints:
                _run_full_eval_for_checkpoint(config, ckpt_path)
        finally:
            # Trainer is about to be deleted anyway, but be tidy.
            if hasattr(trainer, "restore_after_full_eval"):
                try:
                    trainer.restore_after_full_eval()
                except Exception:
                    pass

    # Release model + optimizer memory before the next experiment.
    del trainer
    del dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-process batch ablation runner (dataset loaded once).",
    )
    p.add_argument(
        "--names-file", required=True,
        help="Path to a text file with one experiment name per line (no .json extension).",
    )
    p.add_argument(
        "--configs-dir", required=True,
        help="Directory containing <name>.json config files.",
    )
    p.add_argument(
        "--exp-base", required=True,
        help="Base directory where experiment output directories live.",
    )
    p.add_argument(
        "--resume-from", default="",
        help="Skip all experiments before this name (inclusive start). "
             "Useful for resuming a partially completed run.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    names_file = Path(args.names_file)
    configs_dir = Path(args.configs_dir)
    exp_base = Path(args.exp_base)
    resume_from = str(args.resume_from).strip()

    # -----------------------------------------------------------------------
    # Read ordered experiment names
    # -----------------------------------------------------------------------
    if not names_file.is_file():
        logger.error("Names file not found: %s", names_file)
        sys.exit(1)
    names: list[str] = []
    for raw in names_file.read_text(encoding="utf-8").splitlines():
        n = raw.strip()
        if n and not n.startswith("#"):
            names.append(n)
    if not names:
        logger.error("No experiment names found in %s", names_file)
        sys.exit(1)
    total = len(names)
    logger.info("Loaded %d experiment names from %s", total, names_file)

    # -----------------------------------------------------------------------
    # Optional resume: skip until --resume-from name
    # -----------------------------------------------------------------------
    skip_until_done = False
    if resume_from:
        if resume_from in names:
            skip_until_done = True
            logger.info("Resuming from '%s' (skipping earlier experiments).", resume_from)
        else:
            logger.warning("--resume-from '%s' not found in names list; running all.", resume_from)

    # -----------------------------------------------------------------------
    # Load first config to bootstrap the shared dataset
    # -----------------------------------------------------------------------
    first_cfg_path = configs_dir / f"{names[0]}.json"
    if not first_cfg_path.is_file():
        logger.error("Config not found: %s", first_cfg_path)
        sys.exit(1)
    first_config = load_experiment_config(first_cfg_path)
    _set_seed(int(first_config.training.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # -----------------------------------------------------------------------
    # Build shared dataset (the expensive step – done exactly once)
    # -----------------------------------------------------------------------
    t0_dataset = time.perf_counter()
    dataset = build_shared_dataset(first_config, device)
    logger.info(
        "Shared dataset ready in %.1fs.  Will be reused across all %d experiments.",
        time.perf_counter() - t0_dataset,
        total,
    )

    # Compute reference fingerprint to detect divergent data configs
    ref_fingerprint = _data_fingerprint(first_config)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    failed: list[str] = []
    ok_count = 0
    t0_total = time.perf_counter()

    for idx, name in enumerate(names):
        # Resume-from skip logic
        if skip_until_done:
            if name == resume_from:
                skip_until_done = False  # start from here
            else:
                logger.info("[%d/%d] Skipping (resume-from): %s", idx, total, name)
                continue

        cfg_path = configs_dir / f"{name}.json"
        out_dir = exp_base / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy config into exp dir (mirrors run_one in the sh script)
        import shutil
        shutil.copy2(cfg_path, out_dir / "config.json")

        print("")
        print("=" * 67)
        print(f"[{idx}/{total}] batch: {name}")
        print("=" * 67)

        if not cfg_path.is_file():
            logger.error("[%d/%d] Config missing: %s", idx, total, cfg_path)
            failed.append(name)
            continue

        try:
            config = load_experiment_config(cfg_path)
        except Exception:
            logger.exception("[%d/%d] Failed to load config: %s", idx, total, cfg_path)
            failed.append(name)
            continue

        # Warn if data config diverges from the shared dataset
        fp = _data_fingerprint(config)
        if fp != ref_fingerprint:
            logger.warning(
                "[%s] Data config differs from shared dataset; results may be incorrect. "
                "Diverging fields: %s",
                name,
                {
                    k: (json.loads(fp).get(k), json.loads(ref_fingerprint).get(k))
                    for k in json.loads(fp)
                    if json.loads(fp).get(k) != json.loads(ref_fingerprint).get(k)
                },
            )

        # Per-experiment log file (mirrors focused.log from the old sh script)
        exp_log_path = out_dir / "focused.log"
        log_handler = logging.FileHandler(exp_log_path, mode="a", encoding="utf-8")
        log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        t0 = time.perf_counter()
        rc = 1
        try:
            rc = run_one_experiment(
                name=name,
                config=config,
                config_path=cfg_path,
                dataset=dataset,
                device=device,
            )
        except KeyboardInterrupt:
            logger.warning("[%s] Interrupted by user.", name)
            root_logger.removeHandler(log_handler)
            log_handler.close()
            raise
        except Exception:
            logger.exception("[%s] Experiment raised an unhandled exception.", name)
            rc = 1
        finally:
            root_logger.removeHandler(log_handler)
            log_handler.close()

        elapsed = time.perf_counter() - t0
        if rc == 0:
            ok_count += 1
            logger.info("[%s] OK in %.1fs", name, elapsed)
        else:
            failed.append(name)
            logger.error("[%s] FAILED (rc=%d) in %.1fs", name, rc, elapsed)

        print(f"--- END {name} (rc={rc}) ---")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_elapsed = time.perf_counter() - t0_total
    ran = ok_count + len(failed)
    print("")
    print(f"BATCH DONE.  Ran={ran}/{total}  OK={ok_count}  Failed={len(failed)}"
          f"  Wall={total_elapsed:.0f}s")
    if failed:
        failed_path = exp_base / "batch_failed.txt"
        failed_path.write_text("\n".join(failed) + "\n", encoding="utf-8")
        logger.error("Failed experiments: %s", " ".join(failed))
        logger.info("Failed list written to %s", failed_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
