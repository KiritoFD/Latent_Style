from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from scipy import linalg

try:
    import lpips
except Exception:
    lpips = None


ARTFID_ART_INCEPTION_URL = "https://huggingface.co/matthias-wright/art_inception/resolve/main/art_inception.pth"


def download_art_inception_checkpoint(cache_dir: str | Path | None = None) -> Path:
    root = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir())
    dst_dir = root / "artfid"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "art_inception.pth"
    if dst.exists():
        return dst

    tmp = dst.with_suffix(".tmp")
    with requests.get(ARTFID_ART_INCEPTION_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.replace(dst)
    return dst


class ArtFIDFeatureExtractor(nn.Module):
    """
    Official ArtFID uses a custom art-domain Inception checkpoint.
    The convolutional trunk matches torchvision's inception_v3.
    We load the published checkpoint with strict=False and return the
    pooled 2048-dim feature vector used for FID.
    """

    def __init__(self, ckpt_path: str | Path, device: str | torch.device):
        super().__init__()
        self.model = models.inception_v3(weights=None, aux_logits=False, init_weights=False, transform_input=False)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.model
        x = model.Conv2d_1a_3x3(x)
        x = model.Conv2d_2a_3x3(x)
        x = model.Conv2d_2b_3x3(x)
        x = model.maxpool1(x)
        x = model.Conv2d_3b_1x1(x)
        x = model.Conv2d_4a_3x3(x)
        x = model.maxpool2(x)
        x = model.Mixed_5b(x)
        x = model.Mixed_5c(x)
        x = model.Mixed_5d(x)
        x = model.Mixed_6a(x)
        x = model.Mixed_6b(x)
        x = model.Mixed_6c(x)
        x = model.Mixed_6d(x)
        x = model.Mixed_6e(x)
        x = model.Mixed_7a(x)
        x = model.Mixed_7b(x)
        x = model.Mixed_7c(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)


def load_artfid_feature_extractor(
    *,
    device: str | torch.device,
    cache_dir: str | Path | None = None,
) -> ArtFIDFeatureExtractor:
    ckpt = download_art_inception_checkpoint(cache_dir=cache_dir)
    return ArtFIDFeatureExtractor(ckpt, device=device)


def load_artfid_lpips(
    *,
    device: str | torch.device,
) -> nn.Module:
    if lpips is None:
        raise RuntimeError("lpips is required for academic ArtFID content distance but is not installed.")
    model = lpips.LPIPS(net="alex", verbose=False).to(device)
    model.eval()
    return model


def load_image_tensor(path: str | Path) -> torch.Tensor:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _collect_features(
    paths: list[str],
    *,
    model: nn.Module,
    batch_size: int,
    device: str | torch.device,
) -> np.ndarray:
    feats: list[np.ndarray] = []
    bs = max(1, int(batch_size))
    for start in range(0, len(paths), bs):
        chunk = paths[start : start + bs]
        imgs = torch.stack([load_image_tensor(p) for p in chunk], dim=0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            out = model(imgs)
        feats.append(out.detach().cpu().numpy())
    if not feats:
        return np.empty((0, 2048), dtype=np.float32)
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def compute_artfid_fid_from_paths(
    gen_paths: list[str],
    ref_paths: list[str],
    *,
    model: nn.Module,
    batch_size: int,
    device: str | torch.device,
    ref_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ref_cache_key: str | None = None,
) -> float | None:
    if len(gen_paths) < 2 or len(ref_paths) < 2:
        return None
    gen_feats = _collect_features(gen_paths, model=model, batch_size=batch_size, device=device)
    if gen_feats.shape[0] < 2:
        return None
    if ref_cache is not None and ref_cache_key and ref_cache_key in ref_cache:
        mu_ref, sigma_ref = ref_cache[ref_cache_key]
    else:
        ref_feats = _collect_features(ref_paths, model=model, batch_size=batch_size, device=device)
        if ref_feats.shape[0] < 2:
            return None
        mu_ref = np.mean(ref_feats, axis=0)
        sigma_ref = np.cov(ref_feats, rowvar=False)
        if ref_cache is not None and ref_cache_key:
            ref_cache[ref_cache_key] = (mu_ref, sigma_ref)
    mu_gen = np.mean(gen_feats, axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    return frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)


def compute_artfid_content_distance_from_paths(
    gen_paths: list[str],
    src_paths: list[str],
    *,
    loss_fn: nn.Module,
    batch_size: int,
    device: str | torch.device,
    cpu_fallback: bool = True,
) -> float | None:
    if len(gen_paths) != len(src_paths) or not gen_paths:
        return None

    dists: list[np.ndarray] = []
    bs = max(1, int(batch_size))
    for start in range(0, len(gen_paths), bs):
        gen_chunk = gen_paths[start : start + bs]
        src_chunk = src_paths[start : start + bs]
        x = torch.stack([load_image_tensor(p) for p in gen_chunk], dim=0)
        y = torch.stack([load_image_tensor(p) for p in src_chunk], dim=0)
        try:
            with torch.no_grad():
                cur = loss_fn(
                    (x.to(device=device, dtype=torch.float32) * 2.0) - 1.0,
                    (y.to(device=device, dtype=torch.float32) * 2.0) - 1.0,
                )
        except RuntimeError as exc:
            if (not cpu_fallback) or ("out of memory" not in str(exc).lower()):
                raise
            torch.cuda.empty_cache()
            with torch.no_grad():
                cur = loss_fn(
                    (x.to(device="cpu", dtype=torch.float32) * 2.0) - 1.0,
                    (y.to(device="cpu", dtype=torch.float32) * 2.0) - 1.0,
                )
        dists.append(cur.detach().view(-1).cpu().numpy())
    if not dists:
        return None
    return float(np.mean(np.concatenate(dists, axis=0)))
