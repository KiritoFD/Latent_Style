import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Sampler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


# -------------------------
# Dataset
# -------------------------
class LatentStyleDataset(Dataset):
    def __init__(self, latents: torch.Tensor, style_ids: torch.Tensor):
        self.latents = latents  # CPU tensor [N,C,H,W]
        self.style_ids = style_ids  # CPU tensor [N]

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.latents[idx], self.style_ids[idx]


def _safe_load_latent(path: Path) -> torch.Tensor:
    try:
        latent = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        latent = torch.load(path, map_location="cpu")
    if isinstance(latent, dict):
        if "latent" in latent:
            latent = latent["latent"]
        else:
            raise ValueError(f"Unsupported latent dict format in {path}")
    if latent.ndim == 4 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    return latent.float().contiguous()


def _read_latent_scale_factor(config: dict) -> float:
    model_cfg = config.get("model", {})
    try:
        scale = float(model_cfg.get("latent_scale_factor", 0.18215))
    except Exception:
        scale = 0.18215
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 0.18215
    return scale


def _default_cache_path(data_root: Path, style_subdirs: list[str], num_styles: int, latent_scale_factor: float) -> Path:
    key = (
        f"{str(data_root.resolve())}|{','.join(style_subdirs)}|{num_styles}|"
        f"scale={latent_scale_factor:.8f}|v3"
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    cache_dir = _ROOT / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"style_classifier_latents_{digest}.pt"


def build_latent_dataset(
    config: dict,
    cache_path: Optional[Path] = None,
    rebuild_cache: bool = False,
    config_dir: Optional[Path] = None,
) -> LatentStyleDataset:
    data_root_raw = Path(config["data"]["data_root"]).expanduser()
    if data_root_raw.is_absolute():
        data_root = data_root_raw.resolve()
    else:
        base_dir = config_dir.resolve() if config_dir is not None else Path.cwd().resolve()
        data_root = (base_dir / data_root_raw).resolve()
    num_styles = int(config["model"]["num_styles"])
    latent_scale_factor = _read_latent_scale_factor(config)
    style_subdirs = config["data"].get("style_subdirs")
    if not style_subdirs:
        style_subdirs = [f"style{i}" for i in range(num_styles)]
    else:
        style_subdirs = list(style_subdirs)

    cache_path = cache_path or _default_cache_path(data_root, style_subdirs, num_styles, latent_scale_factor)
    if cache_path.exists() and not rebuild_cache:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        latents = cached["latents"].cpu()
        style_ids = cached["style_ids"].cpu()
        logger.info(f"Loaded cached latent dataset: {cache_path} (N={latents.shape[0]})")
        return LatentStyleDataset(latents, style_ids)

    latents_list = []
    style_ids_list = []
    total_files = 0
    for style_id, subdir in enumerate(style_subdirs):
        style_dir = data_root / subdir
        files = sorted(style_dir.glob("*.pt")) if style_dir.exists() else []
        if not files:
            logger.warning(f"No latent files found for style={subdir} ({style_dir})")
            continue
        total_files += len(files)
        for p in files:
            latents_list.append(_safe_load_latent(p))
            style_ids_list.append(style_id)

    if not latents_list:
        raise RuntimeError(f"No latent files found under {data_root}")

    latents = torch.stack(latents_list, dim=0).cpu()
    style_ids = torch.tensor(style_ids_list, dtype=torch.long).cpu()

    # Keep same scaling convention as main dataset loader (config-driven).
    if float(latents.std().item()) < 0.5:
        logger.info(
            "Auto-scaling VAE latents by 1/latent_scale_factor=%.6f for classifier dataset",
            latent_scale_factor,
        )
        latents = latents / latent_scale_factor

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": latents,
            "style_ids": style_ids,
            "meta": {
                "latent_scale_factor": float(latent_scale_factor),
                "data_root": str(data_root),
                "style_subdirs": list(style_subdirs),
            },
        },
        cache_path,
    )
    logger.info(f"Built latent cache: {cache_path} (files={total_files}, N={latents.shape[0]})")

    return LatentStyleDataset(latents, style_ids)


# -------------------------
# Augmentations (content suppression)
# -------------------------
def _random_crop_resize(x: torch.Tensor, min_scale: float = 0.7) -> torch.Tensor:
    # x: [B,C,H,W]
    B, C, H, W = x.shape
    if H < 4 or W < 4:
        return x
    scale = float(torch.empty(1).uniform_(min_scale, 1.0).item())
    nh = max(2, int(H * scale))
    nw = max(2, int(W * scale))
    top = int(torch.randint(0, max(1, H - nh + 1), (1,)).item())
    left = int(torch.randint(0, max(1, W - nw + 1), (1,)).item())
    crop = x[:, :, top:top + nh, left:left + nw]
    return F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)


def _random_shift(x: torch.Tensor, max_shift: int = 2) -> torch.Tensor:
    if max_shift <= 0:
        return x
    dy = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    dx = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    if dx == 0 and dy == 0:
        return x
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


def _patch_shuffle(x: torch.Tensor, grid: int = 2) -> torch.Tensor:
    # split into grid x grid patches and permute patches (keeps local texture, breaks layout)
    B, C, H, W = x.shape
    if H % grid != 0 or W % grid != 0 or H < grid or W < grid:
        return x
    ph, pw = H // grid, W // grid
    patches = []
    for i in range(grid):
        for j in range(grid):
            patches.append(x[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw])
    idx = torch.randperm(len(patches), device=x.device)
    patches = [patches[i] for i in idx.tolist()]
    rows = []
    k = 0
    for i in range(grid):
        rows.append(torch.cat(patches[k:k+grid], dim=3))
        k += grid
    return torch.cat(rows, dim=2)

def _lowpass(x: torch.Tensor, lowpass_size: int = 8) -> torch.Tensor:
    # lowpass via down-up sampling (area -> bilinear)
    if lowpass_size <= 0:
        return x
    B, C, H, W = x.shape
    lp = F.interpolate(x, size=(lowpass_size, lowpass_size), mode="area")
    lp = F.interpolate(lp, size=(H, W), mode="bilinear", align_corners=False)
    return lp

def augment_latent(
    x: torch.Tensor,
    p_crop: float = 0.5,
    p_shift: float = 0.5,
    p_patch_shuffle: float = 0.3,
    p_highpass: float = 0.3,
    lowpass_size: int = 8,
    patch_grid: int = 2,
    shift_max: int = 2,
) -> torch.Tensor:
    # x: [B,C,H,W]
    if torch.rand(1).item() < p_crop:
        x = _random_crop_resize(x, min_scale=0.7)
    if torch.rand(1).item() < p_shift:
        x = _random_shift(x, max_shift=shift_max)
    if torch.rand(1).item() < p_patch_shuffle:
        x = _patch_shuffle(x, grid=patch_grid)
    if torch.rand(1).item() < p_highpass:
        lp = _lowpass(x, lowpass_size=lowpass_size)
        x = x - lp
    return x


# -------------------------
# Classifier: GN-CNN + explicit style statistics
# -------------------------
class ResBlock(nn.Module):
    """Latent-space residual block with GroupNorm."""

    def __init__(self, channels: int):
        super().__init__()
        groups = 8
        while groups > 1 and (channels % groups != 0):
            groups //= 2
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x + res


class StyleClassifier(nn.Module):
    """
    Enhanced latent style classifier:
    - ResNet texture stream
    - Explicit statistics stream (gram/mean/std)
    - Spatial-info suppression by global pooling
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        width: int = 64,
        dropout: float = 0.2,
        use_stats: bool = True,
        use_gram: bool = True,
        use_lowpass_stats: bool = True,
        spatial_shuffle: bool = False,
        input_size_train: int = 8,
        input_size_infer: int = 8,
        lowpass_size: int = 8,
        **kwargs,
    ):
        super().__init__()
        w = int(width)

        self.use_stats = bool(use_stats)
        self.use_gram = bool(use_gram)
        self.use_lowpass_stats = bool(use_lowpass_stats)
        self.spatial_shuffle = bool(spatial_shuffle)
        self.input_size_train = int(input_size_train)
        self.input_size_infer = int(input_size_infer)
        self.lowpass_size = int(lowpass_size)

        gn_groups = 8
        while gn_groups > 1 and (w % gn_groups != 0):
            gn_groups //= 2

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w, 3, padding=1),
            nn.GroupNorm(gn_groups, w),
            nn.SiLU(inplace=True),
        )

        self.layer1 = nn.Sequential(ResBlock(w), ResBlock(w))
        self.down1 = nn.Conv2d(w, w * 2, 3, stride=2, padding=1)  # 32->16
        self.layer2 = nn.Sequential(ResBlock(w * 2), ResBlock(w * 2))
        self.down2 = nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1)  # 16->8
        self.layer3 = nn.Sequential(ResBlock(w * 4), ResBlock(w * 4))
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        stats_dim = 0
        if self.use_stats:
            stats_dim += in_channels * 2
        if self.use_lowpass_stats:
            stats_dim += in_channels * 2
        if self.use_gram:
            stats_dim += in_channels * in_channels

        head_in = (w * 4) + stats_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(p=float(dropout)),
            nn.Linear(head_in, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _resize_for_mode(self, x: torch.Tensor) -> torch.Tensor:
        target = self.input_size_train if self.training else self.input_size_infer
        if target > 0 and (x.shape[-1] != target or x.shape[-2] != target):
            x = F.interpolate(x, size=(target, target), mode="area")
        return x

    @staticmethod
    def _calc_gram(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        feat = x.view(b, c, h * w)
        feat_t = feat.transpose(1, 2)
        gram = feat.bmm(feat_t) / max(c * h * w, 1)
        return gram.view(b, -1)

    def _extract_explicit_stats(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        lp_map = _lowpass(x, lowpass_size=self.lowpass_size)
        lp_up = F.interpolate(lp_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        hp_map = x - lp_up

        if self.use_stats:
            parts.append(hp_map.mean(dim=(2, 3)))
            parts.append(hp_map.std(dim=(2, 3), unbiased=False))
        if self.use_lowpass_stats:
            parts.append(lp_map.mean(dim=(2, 3)))
            parts.append(lp_map.std(dim=(2, 3), unbiased=False))
        if self.use_gram:
            parts.append(self._calc_gram(x))

        if not parts:
            return torch.zeros((x.shape[0], 0), device=x.device, dtype=x.dtype)
        return torch.cat(parts, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize_for_mode(x)
        if self.training and self.spatial_shuffle:
            x = _patch_shuffle(x, grid=2)

        feat = self.stem(x)
        feat = self.layer1(feat)
        feat = self.down1(feat)
        feat = self.layer2(feat)
        feat = self.down2(feat)
        feat = self.layer3(feat)
        cnn_feat = self.global_pool(feat).flatten(1)

        stats_feat = self._extract_explicit_stats(x)
        if stats_feat.shape[1] > 0:
            cnn_feat = torch.cat([cnn_feat, stats_feat], dim=1)
        return self.head(cnn_feat)


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=preds.device)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        cm[t, p] += 1
    return cm

@torch.no_grad()
def compute_basic_metrics(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> Dict[str, Any]:
    preds = logits.argmax(dim=1)
    cm = confusion_matrix(preds, y, num_classes=num_classes).to(torch.float32)

    # per-class recall/precision/f1
    tp = torch.diag(cm)
    fn = cm.sum(dim=1) - tp
    fp = cm.sum(dim=0) - tp

    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    acc = float((preds == y).float().mean().item())
    macro_recall = float(recall.mean().item())
    macro_f1 = float(f1.mean().item())
    min_recall = float(recall.min().item())
    recall_std = float(recall.std(unbiased=False).item())

    return {
        "acc": acc,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "recall_per_class": recall.detach().cpu().tolist(),
        "precision_per_class": precision.detach().cpu().tolist(),
        "f1_per_class": f1.detach().cpu().tolist(),
        "min_recall": min_recall,
        "recall_std": recall_std,
        "confusion_matrix": cm.detach().cpu().to(torch.long).tolist(),
    }

@torch.no_grad()
def expected_calibration_error(logits: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> Dict[str, float]:
    probs = F.softmax(logits, dim=1)
    conf, preds = probs.max(dim=1)
    correct = (preds == y).float()

    ece = torch.zeros((), device=logits.device)
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc_bin = correct[m].mean()
            conf_bin = conf[m].mean()
            ece += (m.float().mean()) * torch.abs(acc_bin - conf_bin)

    return {
        "ece": float(ece.item()),
        "mean_confidence": float(conf.mean().item()),
    }


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list = []
    y_list = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits_list.append(model(x))
        y_list.append(y)
    return torch.cat(logits_list, dim=0), torch.cat(y_list, dim=0)


def fit_temperature(
    logits: torch.Tensor,
    y: torch.Tensor,
    iters: int = 200,
    lr: float = 0.05,
) -> float:
    """
    Post-hoc temperature scaling on validation logits.
    Keeps classifier weights frozen and optimizes one scalar T>0.
    """
    if logits.numel() == 0:
        return 1.0
    log_T = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=iters)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        T = log_T.exp()
        loss = F.cross_entropy(logits / T, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(log_T.exp().detach().item())
    if not (T > 0.0) or not torch.isfinite(torch.tensor(T)):
        return 1.0
    return T

@torch.no_grad()
def invariance_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    K: int = 5,
    aug_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Content invariance test:
    For each sample, compare prediction under K stochastic augmentations.
    Measure:
    - top1 agreement with original
    - mean KL(probs_aug || probs_orig)
    """
    model.eval()
    aug_cfg = aug_cfg or {}

    agree = 0
    total = 0
    kl_sum = 0.0
    cnt = 0

    for x, _y in loader:
        x = x.to(device, non_blocking=True)
        logits0 = model(x)
        p0 = F.softmax(logits0, dim=1)
        pred0 = p0.argmax(dim=1)

        for _ in range(K):
            xa = augment_latent(x, **aug_cfg)
            logits1 = model(xa)
            p1 = F.softmax(logits1, dim=1)
            pred1 = p1.argmax(dim=1)

            agree += (pred1 == pred0).sum().item()
            total += pred0.numel()

            # KL(p1 || p0)
            kl = (p1 * (p1.clamp_min(1e-8).log() - p0.clamp_min(1e-8).log())).sum(dim=1).mean()
            kl_sum += float(kl.item())
            cnt += 1

    return {
        "top1_agreement": float(agree / max(total, 1)),
        "mean_kl": float(kl_sum / max(cnt, 1)),
    }

@torch.no_grad()
def run_eval_suite(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    invariance_loader: Optional[DataLoader] = None,
    invariance_K: int = 5,
    aug_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model.eval()
    all_logits = []
    all_y = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits)
        all_y.append(y)
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)

    basic = compute_basic_metrics(logits, y, num_classes=num_classes)
    calib = expected_calibration_error(logits, y, n_bins=15)

    inv = None
    if invariance_loader is not None:
        inv = invariance_metrics(model, invariance_loader, device, num_classes, K=invariance_K, aug_cfg=aug_cfg)

    report = {
        "basic": basic,
        "calibration": calib,
    }
    if inv is not None:
        report["invariance"] = inv

    # automatic usability decision (tweak thresholds as you like)
    usable = True
    reasons = []

    if basic["macro_recall"] < 0.85:
        usable = False; reasons.append(f"macro_recall<{0.85} ({basic['macro_recall']:.3f})")
    if basic["min_recall"] < 0.75:
        usable = False; reasons.append(f"min_recall<{0.75} ({basic['min_recall']:.3f})")
    if calib["ece"] > 0.08:
        usable = False; reasons.append(f"ece>{0.08} ({calib['ece']:.3f})")
    if inv is not None:
        if inv["top1_agreement"] < 0.75:
            usable = False; reasons.append(f"invariance_agreement<{0.75} ({inv['top1_agreement']:.3f})")
        if inv["mean_kl"] > 0.25:
            usable = False; reasons.append(f"invariance_kl>{0.25} ({inv['mean_kl']:.3f})")

    report["classifier_usable"] = usable
    report["fail_reasons"] = reasons
    return report


# -------------------------
# Train
# -------------------------
class FixedRatioBatchSampler(Sampler[list[int]]):
    """
    Build class-balanced mini-batches with fixed per-class quota.
    Works well when style counts are skewed and you want stable recall.
    """

    def __init__(
        self,
        labels: torch.Tensor,
        batch_size: int,
        num_classes: int,
        generator: Optional[torch.Generator] = None,
    ):
        self.labels = labels.long().cpu()
        self.batch_size = int(batch_size)
        self.num_classes = int(num_classes)
        self.generator = generator if generator is not None else torch.Generator()
        if self.batch_size % self.num_classes != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by num_classes ({self.num_classes}) "
                "for fixed_ratio sampling."
            )
        self.per_class = self.batch_size // self.num_classes
        self.class_indices = []
        for c in range(self.num_classes):
            idx = torch.where(self.labels == c)[0]
            if idx.numel() == 0:
                raise ValueError(f"class {c} has zero samples, cannot build fixed ratio sampler")
            self.class_indices.append(idx)
        self._num_batches = max(1, int(self.labels.numel() // self.batch_size))

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        # Per-epoch shuffled pointers for each class bucket.
        shuffled = []
        ptrs = []
        for idx in self.class_indices:
            perm = idx[torch.randperm(idx.numel(), generator=self.generator)]
            shuffled.append(perm)
            ptrs.append(0)

        for _ in range(self._num_batches):
            batch = []
            for c in range(self.num_classes):
                idx_c = shuffled[c]
                p = ptrs[c]
                if p + self.per_class > idx_c.numel():
                    idx_c = self.class_indices[c][torch.randperm(self.class_indices[c].numel(), generator=self.generator)]
                    shuffled[c] = idx_c
                    p = 0
                take = idx_c[p:p + self.per_class]
                ptrs[c] = p + self.per_class
                batch.extend(take.tolist())
            batch_t = torch.tensor(batch, dtype=torch.long)
            order = torch.randperm(batch_t.numel(), generator=self.generator)
            yield batch_t[order].tolist()


def balanced_softmax_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_counts: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Balanced Softmax CE:
    CE(logits + log(class_prior), target)
    """
    priors = class_counts.to(logits.device, logits.dtype).clamp_min(1.0)
    logits_bal = logits + priors.log().unsqueeze(0)
    return F.cross_entropy(logits_bal, targets, label_smoothing=float(label_smoothing))


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }
        self.backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        state = model.state_dict()
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module):
        self.backup = {}
        state = model.state_dict()
        for k, v in state.items():
            if k in self.shadow:
                self.backup[k] = v.detach().clone()
                v.copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self.backup is None:
            return
        state = model.state_dict()
        for k, v in state.items():
            if k in self.backup:
                v.copy_(self.backup[k])
        self.backup = None


def _resolve_num_workers(requested: int, is_windows: bool) -> int:
    if requested >= 0:
        return requested
    if is_windows:
        # Windows worker spawn is expensive; prioritize startup speed.
        return 0
    cpu = os.cpu_count() or 1
    return max(2, min(8, cpu // 2))


def _build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    batch_sampler: Optional[Sampler[list[int]]] = None,
) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if batch_sampler is not None:
        kwargs["batch_sampler"] = batch_sampler
    else:
        kwargs["batch_size"] = batch_size
        kwargs["shuffle"] = shuffle
        kwargs["drop_last"] = drop_last
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def train(
    config_path: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    val_ratio: float,
    width: int,
    dropout: float,
    label_smoothing: float,
    use_amp: bool,
    use_stats_branch: bool,
    use_gram: bool,
    use_lowpass_stats: bool,
    spatial_shuffle: bool,
    # augmentation used during training
    train_aug: bool,
    aug_p_crop: float,
    aug_p_shift: float,
    aug_p_patch_shuffle: float,
    aug_p_highpass: float,
    aug_lowpass_size: int,
    aug_patch_grid: int,
    aug_shift_max: int,
    consistency_weight: float,
    sampling_mode: str,
    use_balanced_softmax: bool,
    warmup_epochs: int,
    ema_decay: float,
    # invariance evaluation
    inv_K: int,
    # startup perf
    num_workers: int,
    val_num_workers: int,
    cache_path: Optional[Path],
    rebuild_cache: bool,
    use_temperature_scaling: bool,
    temp_iters: int,
    temp_lr: float,
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_latent_dataset(
        config,
        cache_path=cache_path,
        rebuild_cache=rebuild_cache,
        config_dir=config_path.parent,
    )

    num_classes = int(config["model"]["num_styles"])
    in_channels = int(config["model"]["latent_channels"])

    # train/val split
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    is_windows = sys.platform.startswith("win")
    nw_train = _resolve_num_workers(num_workers, is_windows=is_windows)
    nw_val = _resolve_num_workers(val_num_workers, is_windows=is_windows)
    pin_memory = bool(device.type == "cuda")
    logger.info(
        f"DataLoader setup | train_workers={nw_train} val_workers={nw_val} "
        f"pin_memory={pin_memory} platform={sys.platform}"
    )

    labels_all = dataset.style_ids
    train_indices = torch.tensor(train_set.indices, dtype=torch.long)
    train_labels = labels_all[train_indices]
    sampling_mode = str(sampling_mode).lower()
    if sampling_mode not in {"shuffle", "fixed_ratio"}:
        raise ValueError(f"Unsupported sampling_mode={sampling_mode}. Use 'shuffle' or 'fixed_ratio'.")

    train_batch_sampler = None
    if sampling_mode == "fixed_ratio":
        train_batch_sampler = FixedRatioBatchSampler(
            labels=train_labels,
            batch_size=batch_size,
            num_classes=num_classes,
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = _build_loader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_batch_sampler is None),
        num_workers=nw_train,
        pin_memory=pin_memory,
        drop_last=True,
        batch_sampler=train_batch_sampler,
    )
    val_loader = _build_loader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw_val,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # smaller subset for invariance test (speed)
    inv_size = min(len(val_set), 512)
    inv_subset, _ = random_split(val_set, [inv_size, len(val_set) - inv_size], generator=torch.Generator().manual_seed(123))
    inv_loader = _build_loader(
        inv_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw_val,
        pin_memory=pin_memory,
        drop_last=False,
    )

    input_size_train = int(config.get("loss", {}).get("style_classifier_input_size_train", 8))
    input_size_infer = int(config.get("loss", {}).get("style_classifier_input_size_infer", input_size_train))
    lowpass_size_cfg = int(config.get("loss", {}).get("style_classifier_lowpass_size", aug_lowpass_size))

    model = StyleClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        width=width,
        dropout=dropout,
        use_stats=use_stats_branch,
        use_gram=use_gram,
        use_lowpass_stats=use_lowpass_stats,
        spatial_shuffle=spatial_shuffle,
        input_size_train=input_size_train,
        input_size_infer=input_size_infer,
        lowpass_size=lowpass_size_cfg,
    ).to(device)

    # class prior for balanced softmax / diagnostics
    class_counts = torch.bincount(train_labels, minlength=num_classes).float()
    class_counts = class_counts.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_epochs = max(int(warmup_epochs), 0)
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EMA(model, decay=float(ema_decay))

    best_val = -1.0
    best_report: Dict[str, Any] = {}
    best_temperature = 1.0

    aug_cfg = dict(
        p_crop=aug_p_crop,
        p_shift=aug_p_shift,
        p_patch_shuffle=aug_p_patch_shuffle,
        p_highpass=aug_p_highpass,
        lowpass_size=aug_lowpass_size,
        patch_grid=aug_patch_grid,
        shift_max=aug_shift_max,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if train_aug:
                    x_aug = augment_latent(x, **aug_cfg)
                    logits_pair = model(torch.cat([x, x_aug], dim=0))
                    logits = logits_pair[: x.shape[0]]
                    logits_aug = logits_pair[x.shape[0] :]

                    if use_balanced_softmax:
                        loss_clean = balanced_softmax_ce(
                            logits, y, class_counts, label_smoothing=float(label_smoothing)
                        )
                        loss_aug = balanced_softmax_ce(
                            logits_aug, y, class_counts, label_smoothing=float(label_smoothing)
                        )
                    else:
                        loss_clean = F.cross_entropy(
                            logits,
                            y,
                            label_smoothing=float(label_smoothing),
                        )
                        loss_aug = F.cross_entropy(
                            logits_aug,
                            y,
                            label_smoothing=float(label_smoothing),
                        )
                    # consistency: augmented prediction should match clean distribution
                    target_probs = F.softmax(logits.detach(), dim=1)
                    loss_consistency = F.kl_div(
                        F.log_softmax(logits_aug, dim=1),
                        target_probs,
                        reduction="batchmean",
                    )
                    loss = 0.5 * (loss_clean + loss_aug) + float(consistency_weight) * loss_consistency
                else:
                    logits = model(x)
                    if use_balanced_softmax:
                        loss = balanced_softmax_ce(
                            logits, y, class_counts, label_smoothing=float(label_smoothing)
                        )
                    else:
                        loss = F.cross_entropy(
                            logits,
                            y,
                            label_smoothing=float(label_smoothing),
                        )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total_loss += float(loss.item())

        scheduler.step()

        train_loss = total_loss / max(len(train_loader), 1)

        # eval suite on val + invariance test
        ema.apply(model)
        report = run_eval_suite(
            model,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            invariance_loader=inv_loader,
            invariance_K=inv_K,
            aug_cfg=aug_cfg,
        )
        ema.restore(model)

        logger.info(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_acc={report['basic']['acc']:.4f} | val_macro_recall={report['basic']['macro_recall']:.4f} | "
            f"val_min_recall={report['basic']['min_recall']:.4f} | ece={report['calibration']['ece']:.4f} | "
            f"inv_agree={report.get('invariance', {}).get('top1_agreement', -1):.4f} | "
            f"sampling={sampling_mode} | bsce={use_balanced_softmax} | "
            f"classifier_usable={report['classifier_usable']} "
        )

        # save best by macro_recall (or gate by classifier_usable)
        if report["basic"]["macro_recall"] > best_val:
            best_val = report["basic"]["macro_recall"]
            best_report = report

            temperature = 1.0
            if use_temperature_scaling:
                try:
                    ema.apply(model)
                    val_logits, val_y = collect_logits(model, val_loader, device)
                    temperature = fit_temperature(val_logits, val_y, iters=temp_iters, lr=temp_lr)
                    calibrated_logits = val_logits / temperature
                    inv = report.get("invariance", {})
                    ece_raw = expected_calibration_error(val_logits, val_y)
                    ece_cal = expected_calibration_error(calibrated_logits, val_y)
                    report["temperature_scaling"] = {
                        "enabled": True,
                        "temperature": float(temperature),
                        "nll_raw": float(F.cross_entropy(val_logits, val_y).item()),
                        "nll_calibrated": float(F.cross_entropy(calibrated_logits, val_y).item()),
                        "ece_raw": float(ece_raw["ece"]),
                        "ece_calibrated": float(ece_cal["ece"]),
                        "mean_confidence_raw": float(ece_raw["mean_confidence"]),
                        "mean_confidence_calibrated": float(ece_cal["mean_confidence"]),
                    }
                    calibrated_usable = True
                    calibrated_reasons = []
                    if report["basic"]["macro_recall"] < 0.85:
                        calibrated_usable = False
                        calibrated_reasons.append(f"macro_recall<0.85 ({report['basic']['macro_recall']:.3f})")
                    if report["basic"]["min_recall"] < 0.75:
                        calibrated_usable = False
                        calibrated_reasons.append(f"min_recall<0.75 ({report['basic']['min_recall']:.3f})")
                    if float(ece_cal["ece"]) > 0.08:
                        calibrated_usable = False
                        calibrated_reasons.append(f"ece_calibrated>0.08 ({ece_cal['ece']:.3f})")
                    if inv:
                        if float(inv.get("top1_agreement", 0.0)) < 0.75:
                            calibrated_usable = False
                            calibrated_reasons.append(
                                f"invariance_agreement<0.75 ({inv.get('top1_agreement', 0.0):.3f})"
                            )
                        if float(inv.get("mean_kl", 0.0)) > 0.25:
                            calibrated_usable = False
                            calibrated_reasons.append(f"invariance_kl>0.25 ({inv.get('mean_kl', 0.0):.3f})")
                    report["classifier_usable_calibrated"] = calibrated_usable
                    report["fail_reasons_calibrated"] = calibrated_reasons
                    logger.info(
                        f"Temperature scaling | T={temperature:.4f} | "
                        f"ECE {report['temperature_scaling']['ece_raw']:.4f}->{report['temperature_scaling']['ece_calibrated']:.4f}"
                    )
                    ema.restore(model)
                except Exception as exc:
                    ema.restore(model)
                    report["temperature_scaling"] = {
                        "enabled": True,
                        "temperature": 1.0,
                        "error": str(exc),
                    }
                    logger.warning(f"Temperature scaling failed, fallback T=1.0: {exc}")
                    temperature = 1.0
            else:
                report["temperature_scaling"] = {"enabled": False, "temperature": 1.0}
            best_temperature = float(temperature)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.shadow,
                "meta": {
                    "in_channels": in_channels,
                    "num_classes": num_classes,
                    "best_val_macro_recall": best_val,
                    "classifier_usable": report["classifier_usable"],
                    "fail_reasons": report["fail_reasons"],
                    "temperature": best_temperature,
                    "sampling_mode": sampling_mode,
                    "use_balanced_softmax": bool(use_balanced_softmax),
                    "warmup_epochs": warmup_epochs,
                    "ema_decay": float(ema_decay),
                },
                "report": report,
            }
            torch.save(payload, output_path)

            # also dump a json report next to ckpt
            json_path = output_path.with_suffix(".report.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload["report"], f, indent=2, ensure_ascii=False)
            logger.info(f"Saved BEST checkpoint to {output_path} (macro_recall={best_val:.4f})")
            logger.info(f"Wrote report to {json_path}")


def main():
    parser = argparse.ArgumentParser("Train robust latent style classifier + auto evaluation report")
    parser.add_argument("--config", type=str, default=str(_ROOT / "config.json"))
    parser.add_argument("--output", type=str, default=str(_ROOT / "style_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--no_amp", action="store_true", default=False)

    parser.add_argument("--no_stats_branch", action="store_true", default=False)
    parser.add_argument("--no_gram", action="store_true", default=False)
    parser.add_argument("--no_lowpass_stats", action="store_true", default=False)
    parser.add_argument("--no_spatial_shuffle", action="store_true", default=True)

    # train augmentation
    parser.add_argument("--train_aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aug_p_crop", type=float, default=0.10)
    parser.add_argument("--aug_p_shift", type=float, default=0.10)
    parser.add_argument("--aug_p_patch_shuffle", type=float, default=0.00)
    parser.add_argument("--aug_p_highpass", type=float, default=0.05)
    parser.add_argument("--aug_lowpass_size", type=int, default=8)
    parser.add_argument("--aug_patch_grid", type=int, default=2)
    parser.add_argument("--aug_shift_max", type=int, default=2)
    parser.add_argument("--consistency_weight", type=float, default=0.02)
    parser.add_argument("--sampling_mode", type=str, default="shuffle", choices=["shuffle", "fixed_ratio"])
    parser.add_argument("--use_balanced_softmax", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # invariance eval
    parser.add_argument("--inv_K", type=int, default=5)
    # startup optimization
    parser.add_argument("--num_workers", type=int, default=-1, help="-1=auto (Windows defaults to 0)")
    parser.add_argument("--val_num_workers", type=int, default=-1, help="-1=auto")
    parser.add_argument("--cache_path", type=str, default="", help="Optional latent cache file")
    parser.add_argument("--rebuild_cache", action="store_true", default=False)
    parser.add_argument("--no_temperature_scaling", action="store_true", default=False)
    parser.add_argument("--temp_iters", type=int, default=200)
    parser.add_argument("--temp_lr", type=float, default=0.05)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    train(
        config_path=Path(args.config),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        width=args.width,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        use_amp=not args.no_amp,
        use_stats_branch=not args.no_stats_branch,
        use_gram=not args.no_gram,
        use_lowpass_stats=not args.no_lowpass_stats,
        spatial_shuffle=not args.no_spatial_shuffle,
        train_aug=bool(args.train_aug),
        aug_p_crop=float(args.aug_p_crop),
        aug_p_shift=float(args.aug_p_shift),
        aug_p_patch_shuffle=float(args.aug_p_patch_shuffle),
        aug_p_highpass=float(args.aug_p_highpass),
        aug_lowpass_size=int(args.aug_lowpass_size),
        aug_patch_grid=int(args.aug_patch_grid),
        aug_shift_max=int(args.aug_shift_max),
        consistency_weight=float(args.consistency_weight),
        sampling_mode=str(args.sampling_mode),
        use_balanced_softmax=bool(args.use_balanced_softmax),
        warmup_epochs=int(args.warmup_epochs),
        ema_decay=float(args.ema_decay),
        inv_K=int(args.inv_K),
        num_workers=int(args.num_workers),
        val_num_workers=int(args.val_num_workers),
        cache_path=Path(args.cache_path).expanduser().resolve() if args.cache_path else None,
        rebuild_cache=bool(args.rebuild_cache),
        use_temperature_scaling=not args.no_temperature_scaling,
        temp_iters=int(args.temp_iters),
        temp_lr=float(args.temp_lr),
    )


if __name__ == "__main__":
    main()
