from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    config: Dict[str, Any],
    metrics: Dict[str, float],
    global_step: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "metrics": metrics,
    }
    path = checkpoint_dir / f"epoch_{int(epoch):04d}.pt"
    torch.save(payload, path)
    logger.info("Saved checkpoint: %s (step=%d)", path, int(global_step))
    return path


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    device: str = "cuda",
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint: %s", checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_state = _strip_compile_prefix(state["model_state_dict"])
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        logger.warning("Missing keys on load (first 12): %s", list(missing)[:12])
    if unexpected:
        logger.warning("Unexpected keys on load (first 12): %s", list(unexpected)[:12])

    try:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    except Exception as exc:
        logger.warning("Skip optimizer state restore: %s", exc)

    sched_state = state.get("scheduler_state_dict")
    if scheduler is not None and sched_state is not None:
        try:
            scheduler.load_state_dict(sched_state)
        except Exception as exc:
            logger.warning("Skip scheduler state restore: %s", exc)

    scaler_state = state.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception as exc:
            logger.warning("Skip scaler state restore: %s", exc)

    return {
        "start_epoch": int(state.get("epoch", 0)) + 1,
        "global_step": int(state.get("global_step", 0)),
        "metrics": state.get("metrics", {}),
    }


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    ckpts = sorted(Path(checkpoint_dir).glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None
