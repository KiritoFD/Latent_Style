from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .dataset import AdaCUTLatentDataset
except ImportError:  # pragma: no cover
    from dataset import AdaCUTLatentDataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class StyleTokenizer(nn.Module):
    def __init__(self, style_dim: int = 160) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, int(style_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


def _find_style_emb_key(state_dict: Dict[str, torch.Tensor]) -> str:
    candidates = [k for k in state_dict.keys() if k.endswith("style_emb.weight")]
    if not candidates:
        raise KeyError("style_emb.weight not found in checkpoint model_state_dict")
    return candidates[0]


def _build_dataset(config: Dict, config_dir: Path, device: torch.device) -> AdaCUTLatentDataset:
    data_cfg = config.get("data", {})
    data_root = _resolve_path(str(data_cfg.get("data_root", "")), config_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Latent data_root not found: {data_root}")
    style_subdirs = data_cfg.get("style_subdirs", [])
    if not style_subdirs:
        raise ValueError("config.data.style_subdirs is empty")
    ds = AdaCUTLatentDataset(
        data_root=str(data_root),
        style_subdirs=style_subdirs,
        allow_hflip=bool(data_cfg.get("allow_hflip", False)),
        preload_to_gpu=bool(data_cfg.get("preload_to_gpu", False)),
        preload_max_vram_gb=float(data_cfg.get("preload_max_vram_gb", 0.0)),
        preload_reserve_ratio=float(data_cfg.get("preload_reserve_ratio", 0.35)),
        virtual_length_multiplier=int(data_cfg.get("virtual_length_multiplier", 1)),
        device=str(device),
    )
    return ds


def _train_tokenizer(
    tokenizer: StyleTokenizer,
    dataset: AdaCUTLatentDataset,
    style_bank: torch.Tensor,
    *,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    amp: bool,
    channels_last: bool,
    device: torch.device,
) -> None:
    # If tensors are already preloaded on GPU, pin_memory must stay off.
    use_pin_memory = (device.type == "cuda") and (not bool(getattr(dataset, "preload_to_gpu", False)))
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=use_pin_memory,
    )
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp and device.type == "cuda"))
    tokenizer.train()

    max_steps = max(1, int(steps_per_epoch))
    for epoch in range(1, max(1, int(epochs)) + 1):
        dataset.set_epoch(epoch)
        running = 0.0
        count = 0
        for step_idx, batch in enumerate(loader, start=1):
            if step_idx > max_steps:
                break
            x = batch["target_style"].to(device=device, non_blocking=True)
            sid = batch["target_style_id"].to(device=device, dtype=torch.long, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            target = style_bank.index_select(0, sid)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(amp and device.type == "cuda"), dtype=torch.bfloat16):
                pred = tokenizer(x)
                loss = F.mse_loss(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach().item())
            count += 1

        epoch_loss = running / max(1, count)
        logger.info("Tokenizer epoch %d/%d | mse=%.6f | steps=%d", epoch, epochs, epoch_loss, count)


@torch.no_grad()
def _compute_style_prototypes(
    tokenizer: StyleTokenizer,
    dataset: AdaCUTLatentDataset,
    *,
    batch_size: int,
    channels_last: bool,
    device: torch.device,
) -> torch.Tensor:
    num_styles = len(dataset.style_subdirs)
    style_dim = int(tokenizer.encoder[-1].out_features)
    out = torch.zeros((num_styles, style_dim), dtype=torch.float32, device=device)
    tokenizer.eval()

    for sid in range(num_styles):
        pool = dataset.style_tensors[int(sid)]
        n = int(pool.shape[0])
        acc = torch.zeros((style_dim,), dtype=torch.float32, device=device)
        seen = 0
        for start in range(0, n, max(1, int(batch_size))):
            end = min(n, start + max(1, int(batch_size)))
            x = pool[start:end].to(device=device, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            pred = tokenizer(x).float()
            acc += pred.sum(dim=0)
            seen += int(pred.shape[0])
        if seen <= 0:
            raise RuntimeError(f"Empty style pool for style id={sid}")
        out[sid] = acc / float(seen)
    return out


def _build_full_eval_cmd(config: Dict, checkpoint_path: Path, out_dir: Path) -> list[str]:
    cfg_train = config.get("training", {})
    cfg_infer = config.get("inference", {})
    script = Path(__file__).resolve().parent / "utils" / "run_evaluation.py"
    cmd = [
        sys.executable,
        str(script),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(out_dir),
        "--num_steps",
        str(int(cfg_train.get("full_eval_num_steps", cfg_infer.get("num_steps", 1)))),
        "--step_size",
        str(float(cfg_train.get("full_eval_step_size", cfg_infer.get("step_size", 1.0)))),
        "--batch_size",
        str(int(cfg_train.get("full_eval_batch_size", 4))),
        "--max_src_samples",
        str(int(cfg_train.get("full_eval_max_src_samples", 30))),
        "--max_ref_compare",
        str(int(cfg_train.get("full_eval_max_ref_compare", 24))),
        "--max_ref_cache",
        str(int(cfg_train.get("full_eval_max_ref_cache", 80))),
        "--ref_feature_batch_size",
        str(int(cfg_train.get("full_eval_ref_feature_batch_size", 8))),
        "--eval_lpips_chunk_size",
        str(int(cfg_train.get("full_eval_lpips_chunk_size", 8))),
        "--clip_model_name",
        str(cfg_train.get("full_eval_clip_model_name", "openai/clip-vit-base-patch32")),
        "--clip_modelscope_id",
        str(cfg_train.get("full_eval_clip_modelscope_id", "")),
        "--clip_modelscope_cache_dir",
        str(cfg_train.get("full_eval_clip_modelscope_cache_dir", "")),
        "--clip_hf_cache_dir",
        str(cfg_train.get("full_eval_clip_hf_cache_dir", "../eval_cache/hf")),
        "--image_classifier_path",
        str(cfg_train.get("full_eval_image_classifier_path", "../eval_cache/eval_style_image_classifier.pt")),
    ]
    style_strength = cfg_train.get("full_eval_style_strength", cfg_infer.get("style_strength"))
    if style_strength is not None:
        cmd += ["--style_strength", str(float(style_strength))]
    test_image_dir = str(cfg_train.get("test_image_dir", "")).strip()
    if test_image_dir:
        cmd += ["--test_dir", test_image_dir]
    cache_dir = str(cfg_train.get("full_eval_cache_dir", "")).strip()
    if cache_dir:
        cmd += ["--cache_dir", cache_dir]
    if bool(cfg_train.get("full_eval_classifier_only", False)):
        cmd += ["--eval_classifier_only"]
    if bool(cfg_train.get("full_eval_disable_lpips", False)):
        cmd += ["--eval_disable_lpips"]
    if bool(cfg_train.get("full_eval_enable_art_fid", False)):
        cmd += ["--eval_enable_art_fid"]
        cmd += ["--eval_art_fid_max_gen", str(int(cfg_train.get("full_eval_art_fid_max_gen", 120)))]
        cmd += ["--eval_art_fid_max_ref", str(int(cfg_train.get("full_eval_art_fid_max_ref", 120)))]
        cmd += ["--eval_art_fid_batch_size", str(int(cfg_train.get("full_eval_art_fid_batch_size", 8)))]
        if bool(cfg_train.get("full_eval_art_fid_photo_only", False)):
            cmd += ["--eval_art_fid_photo_only"]
    if bool(cfg_train.get("full_eval_reuse_generated", True)):
        cmd += ["--reuse_generated"]
    if bool(cfg_train.get("full_eval_generation_only", False)):
        cmd += ["--generation_only"]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill StyleTokenizer from style_emb and optionally rerun full_eval")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to source checkpoint, e.g. epoch_0120.pt")
    parser.add_argument("--config", type=str, default="", help="Optional config json path; defaults to checkpoint config")
    parser.add_argument("--output_dir", type=str, default="../tokenizer_distill", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA autocast(bf16) for tokenizer training")
    parser.add_argument("--compile", action="store_true", help="torch.compile tokenizer")
    parser.add_argument("--channels_last", action="store_true", help="Use channels_last memory format for tokenizer input")
    parser.add_argument("--run_full_eval", action="store_true", help="Run full_eval using patched checkpoint")
    parser.add_argument("--full_eval_output", type=str, default="", help="Optional full_eval output dir")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing model_state_dict")

    config_from_ckpt = ckpt.get("config", {})
    if args.config:
        config_path = Path(args.config).resolve()
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    elif config_from_ckpt:
        config = config_from_ckpt
        config_path = ckpt_path
    else:
        raise ValueError("No config provided in args and checkpoint has no config")

    config_dir = config_path.parent.resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    state_dict = ckpt["model_state_dict"]
    clean_sd = _strip_compile_prefix(state_dict)
    style_emb_key = _find_style_emb_key(clean_sd)
    style_bank = clean_sd[style_emb_key].float().to(device=device)
    num_styles, style_dim = int(style_bank.shape[0]), int(style_bank.shape[1])
    logger.info("Found style_emb: key=%s shape=(%d, %d)", style_emb_key, num_styles, style_dim)

    dataset = _build_dataset(config, config_dir, device)
    tokenizer = StyleTokenizer(style_dim=style_dim).to(device)
    if args.channels_last:
        tokenizer = tokenizer.to(memory_format=torch.channels_last)
    if args.compile and hasattr(torch, "compile"):
        tokenizer = torch.compile(tokenizer)

    _train_tokenizer(
        tokenizer,
        dataset,
        style_bank,
        epochs=int(args.epochs),
        steps_per_epoch=int(args.steps_per_epoch),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        amp=bool(args.amp),
        channels_last=bool(args.channels_last),
        device=device,
    )

    prototypes = _compute_style_prototypes(
        tokenizer,
        dataset,
        batch_size=int(args.batch_size),
        channels_last=bool(args.channels_last),
        device=device,
    ).detach().cpu()
    logger.info("Computed distilled prototypes: shape=%s", tuple(prototypes.shape))

    tokenizer_path = out_dir / "tokenizer.pt"
    tokenizer_state = tokenizer.state_dict()
    torch.save(
        {
            "tokenizer_state_dict": tokenizer_state,
            "style_dim": style_dim,
            "num_styles": num_styles,
            "style_subdirs": list(dataset.style_subdirs),
            "distilled_prototypes": prototypes,
        },
        tokenizer_path,
    )
    logger.info("Saved tokenizer checkpoint: %s", tokenizer_path)

    patched = copy.deepcopy(ckpt)
    patched_sd = patched["model_state_dict"]
    target_keys = [k for k in patched_sd.keys() if k.endswith("style_emb.weight")]
    if not target_keys:
        raise KeyError("Cannot patch checkpoint: style_emb.weight key not found")
    for k in target_keys:
        patched_sd[k] = prototypes.to(dtype=patched_sd[k].dtype)
    patched_path = out_dir / f"{ckpt_path.stem}_tokenized.pt"
    torch.save(patched, patched_path)
    logger.info("Saved patched checkpoint: %s", patched_path)

    if bool(args.run_full_eval):
        if args.full_eval_output:
            eval_out = Path(args.full_eval_output).expanduser().resolve()
        else:
            eval_out = out_dir / "full_eval_tokenized"
        eval_out.mkdir(parents=True, exist_ok=True)
        cmd = _build_full_eval_cmd(config, patched_path, eval_out)
        logger.info("Running full_eval with patched checkpoint...")
        logger.info("CMD: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parent))
        logger.info("full_eval done. Output: %s", eval_out)


if __name__ == "__main__":
    main()
