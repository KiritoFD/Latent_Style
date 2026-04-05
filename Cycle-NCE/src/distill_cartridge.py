from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .dataset import AdaCUTLatentDataset
    from .losses import AdaCUTObjective
    from .model import LatentAdaCUT, build_model_from_config
except ImportError:  # pragma: no cover
    from dataset import AdaCUTLatentDataset
    from losses import AdaCUTObjective
    from model import LatentAdaCUT, build_model_from_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DistilledStyleCartridge(nn.Module):
    """
    Lightweight dual-track style cartridge.
    - style_code: global 1D texture recipe for decoder modulation
    - style_palette: 2D palette tokens routed by semantic cross-attn in bottleneck
    """

    def __init__(self, style_dim: int, body_channels: int, num_colors: int = 64) -> None:
        super().__init__()
        self.style_code = nn.Parameter(torch.randn(1, int(style_dim)) * 0.02)
        self.style_palette = nn.Parameter(torch.randn(1, int(body_channels), max(1, int(num_colors)), 1) * 0.02)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.style_code, self.style_palette


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    src_dir = Path(__file__).resolve().parent
    candidates = [
        (base_dir / p),
        (src_dir / p),
        (src_dir.parent / p),
        (Path.cwd() / p),
    ]
    for c in candidates:
        rc = c.resolve()
        if rc.exists():
            return rc
    return candidates[0].resolve()


def _strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


def _build_dataset(config: Dict, config_dir: Path, device: torch.device) -> AdaCUTLatentDataset:
    data_cfg = config.get("data", {})
    data_root = _resolve_path(str(data_cfg.get("data_root", "")), config_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Latent data_root not found: {data_root}")
    style_subdirs = data_cfg.get("style_subdirs", [])
    if not style_subdirs:
        raise ValueError("config.data.style_subdirs is empty")
    return AdaCUTLatentDataset(
        data_root=str(data_root),
        style_subdirs=style_subdirs,
        allow_hflip=bool(data_cfg.get("allow_hflip", False)),
        preload_to_gpu=bool(data_cfg.get("preload_to_gpu", False)),
        preload_max_vram_gb=float(data_cfg.get("preload_max_vram_gb", 0.0)),
        preload_reserve_ratio=float(data_cfg.get("preload_reserve_ratio", 0.35)),
        virtual_length_multiplier=int(data_cfg.get("virtual_length_multiplier", 1)),
        device=str(device),
    )


def _build_loader(dataset: AdaCUTLatentDataset, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    use_pin_memory = (device.type == "cuda") and (not bool(getattr(dataset, "preload_to_gpu", False)))
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=use_pin_memory,
    )


def _build_model(ckpt: Dict, config: Dict, device: torch.device) -> LatentAdaCUT:
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing model_state_dict")
    model = build_model_from_config(config.get("model", {}), use_checkpointing=False).to(device)
    clean_sd = _strip_compile_prefix(ckpt["model_state_dict"])
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if unexpected:
        logger.warning("Unexpected model keys from checkpoint: %d", len(unexpected))
    if missing:
        logger.warning("Missing model keys from checkpoint: %d", len(missing))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _distill(
    *,
    model: LatentAdaCUT,
    cartridge: DistilledStyleCartridge,
    objective: AdaCUTObjective,
    loader: DataLoader,
    epochs: int,
    steps_per_epoch: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    channels_last: bool,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(cartridge.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp and device.type == "cuda"))
    cartridge.train()

    max_steps = max(1, int(steps_per_epoch))
    total_epochs = max(1, int(epochs))
    for epoch in range(1, total_epochs + 1):
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        running = 0.0
        count = 0
        for step_idx, batch in enumerate(loader, start=1):
            if step_idx > max_steps:
                break
            content = batch["content"].to(device=device, non_blocking=True)
            target_style = batch["target_style"].to(device=device, non_blocking=True)
            target_style_id = batch["target_style_id"].to(device=device, dtype=torch.long, non_blocking=True)
            source_style_id = batch["source_style_id"].to(device=device, dtype=torch.long, non_blocking=True)
            if channels_last:
                content = content.contiguous(memory_format=torch.channels_last)
                target_style = target_style.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(amp and device.type == "cuda"), dtype=torch.bfloat16):
                style_code, palette = cartridge()
                pred = model(
                    content,
                    style_id=None,
                    step_size=1.0,
                    style_strength=1.0,
                    target_style_latent=None,
                    style_code_override=style_code,
                    override_palette=palette,
                )
                loss_dict = objective.compute(
                    model=model,
                    content=content,
                    target_style=target_style,
                    target_style_id=target_style_id,
                    source_style_id=source_style_id,
                    pred_override=pred,
                )
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach().item())
            count += 1

        mean_loss = running / max(1, count)
        logger.info("Cartridge epoch %d/%d | loss=%.6f | steps=%d", epoch, total_epochs, mean_loss, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill lightweight style cartridge from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Source checkpoint path, e.g. epoch_0030.pt")
    parser.add_argument("--config", type=str, default="", help="Optional config json path; default from checkpoint")
    parser.add_argument("--output_dir", type=str, default="../cartridge_distill", help="Output directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_colors", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--compile", action="store_true", help="torch.compile cartridge module")
    parser.add_argument("--run_full_eval", action="store_true", help="Reserved for future support")
    parser.add_argument("--full_eval_output", type=str, default="", help="Reserved for future support")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = ckpt.get("config", {})
        cfg_path = ckpt_path
        if not config:
            raise ValueError("No config provided in args and checkpoint has no embedded config.")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = _build_model(ckpt, config, device)
    dataset = _build_dataset(config, cfg_path.parent.resolve(), device)
    loader = _build_loader(dataset, int(args.batch_size), int(args.num_workers), device)
    objective = AdaCUTObjective(config)

    style_dim = int(model.style_emb.weight.shape[1])
    cartridge = DistilledStyleCartridge(
        style_dim=style_dim,
        body_channels=int(model.body_channels),
        num_colors=int(args.num_colors),
    ).to(device)
    if args.compile:
        cartridge = torch.compile(cartridge)

    _distill(
        model=model,
        cartridge=cartridge,
        objective=objective,
        loader=loader,
        epochs=int(args.epochs),
        steps_per_epoch=int(args.steps_per_epoch),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        amp=bool(args.amp),
        channels_last=bool(args.channels_last),
        device=device,
    )

    artifact = {
        "cartridge_state_dict": cartridge.state_dict(),
        "style_dim": style_dim,
        "body_channels": int(model.body_channels),
        "num_colors": int(args.num_colors),
        "source_checkpoint": str(ckpt_path),
        "config": config,
    }
    out_file = out_dir / "cartridge.pt"
    torch.save(artifact, out_file)
    logger.info("Saved distilled cartridge: %s", out_file)

    if bool(args.run_full_eval):
        logger.warning("run_full_eval is not implemented for cartridge artifacts yet; skipping.")


if __name__ == "__main__":
    main()
