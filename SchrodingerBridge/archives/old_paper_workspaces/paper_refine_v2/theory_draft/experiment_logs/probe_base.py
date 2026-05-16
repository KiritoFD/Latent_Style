"""
Base utilities for theory verification experiments.
Shared helper functions for loading model, data, and running inference.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import build_model_from_config
from dataset import AdaCUTLatentDataset


def load_model_and_config(
    checkpoint_path: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[torch.nn.Module, Dict]:
    """Load model from checkpoint and return (model, config)."""
    print(f"[probe] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    model_cfg = config.get("model", {})
    model = build_model_from_config(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    print(f"[probe] Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"[probe] Device: {device}")
    return model, config


def load_dataset(config: Dict) -> AdaCUTLatentDataset:
    """Load dataset from config."""
    data_cfg = config.get("data", {})
    data_root = data_cfg.get("data_root", "../latent-256")
    # Resolve relative paths relative to the project root
    if not Path(data_root).is_absolute():
        data_root = str(SRC_DIR.parent / data_root)
    dataset = AdaCUTLatentDataset(
        data_root=data_root,
        style_subdirs=data_cfg.get("style_subdirs", ["photo", "Hayao", "monet", "vangogh", "cezanne"]),
        allow_hflip=False,
        identity_ratio=None,
        batch_size_hint=32,
        balance_target_styles_per_batch=False,
        preload_to_gpu=False,
        device="cpu",
    )
    print(f"[probe] Dataset: {len(dataset.style_subdirs)} styles, "
          f"{sum(int(t.shape[0]) for t in dataset.style_tensors.values())} total latents")
    return dataset


def get_batch(dataset: AdaCUTLatentDataset, batch_size: int = 32) -> Dict[str, torch.Tensor]:
    """Get a batch of data from dataset."""
    indices = torch.randint(0, len(dataset), (batch_size,))
    batch_list = [dataset[int(idx)] for idx in indices]
    out = {}
    for key in batch_list[0].keys():
        if isinstance(batch_list[0][key], torch.Tensor):
            out[key] = torch.stack([b[key] for b in batch_list], dim=0)
        else:
            # int values -> convert to tensor
            out[key] = torch.tensor([b[key] for b in batch_list], dtype=torch.long)
    return out
