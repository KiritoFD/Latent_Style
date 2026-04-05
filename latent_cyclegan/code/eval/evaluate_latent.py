import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm

# Ensure repository root is importable when launching from eval/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_latent_cyclegan import (  # noqa: E402
    ResnetGenerator,
    _load_vae,
    decode_latents_to_images,
    load_yaml_config,
    resolve_vae_source,
)


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass
class EvalConfig:
    shared_cfg: Dict[str, Any]
    model_cfg: Dict[str, Any]
    data_cfg: Dict[str, Any]
    vis_cfg: Dict[str, Any]
    eval_cfg: Dict[str, Any]
    domain_A_name: str
    domain_B_name: str
    base_latent_dir: Path
    base_orig_rgb_dir: Path
    base_vae_recon_dir: Path
    eval_output_dir: Path
    reference_cache_dir: Path
    checkpoint_dir: Path
    generator_checkpoint_path: Path
    max_eval_samples: int
    eval_seed: int
    eval_image_size: int
    eval_batch_size: int
    run_device: torch.device
    use_bf16_autocast: bool


@dataclass
class EvalDirection:
    name: str
    src_domain: str
    tgt_domain: str
    latent_src_dir: Path
    orig_rgb_dir: Path
    vae_recon_dir: Path
    tgt_ref_dir: Path
    out_dir: Path


def _warn(msg: str) -> None:
    warnings.warn(msg, stacklevel=2)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _resolve_repo_path(path_value: Any) -> Path:
    path_str = str(path_value or "").strip()
    if not path_str:
        return Path("")
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _torch_load_trusted(path: Path, map_location: Any = "cpu") -> Any:
    """Load local trusted artifacts across torch versions.

    PyTorch 2.6 changed torch.load default to weights_only=True, which may
    fail for eval caches containing numpy arrays. For locally generated files
    in this repo, we explicitly opt into full loading.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch versions do not support weights_only.
        return torch.load(path, map_location=map_location)


def _safe_stem_name(path: Path) -> str:
    return path.stem


def _pil_rgb(path: Path, image_size: Optional[int] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return img


def _list_images(root: Path) -> List[Path]:
    _require(root.exists(), f"Image directory not found: {root}")
    files: List[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    files = sorted([p for p in files if p.is_file()])
    _require(len(files) > 0, f"No RGB images found in: {root}")
    return files


def _list_latents(root: Path) -> List[Path]:
    _require(root.exists(), f"Latent directory not found: {root}")
    files = sorted([p for p in root.rglob("*.pt") if p.is_file()])
    _require(len(files) > 0, f"No latent .pt files found in: {root}")
    return files


def _build_stem_index(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in _list_images(root):
        index[p.stem] = p
    return index


def _resolve_domain_dir(base_dir: Path, domain_name: str, key_name: str) -> Path:
    path = base_dir / domain_name
    _require(path.exists(), f"{key_name} not found for domain '{domain_name}': {path}")
    return path


def _select_latent_paths(
    latent_paths: Sequence[Path],
    max_eval_samples: int,
    eval_seed: int,
) -> List[Path]:
    # Global deterministic truncation for all downstream alignment by stem.
    selected = list(latent_paths)
    if max_eval_samples > 0 and len(selected) > max_eval_samples:
        rng = random.Random(eval_seed)
        selected = rng.sample(selected, max_eval_samples)
    return sorted(selected, key=lambda p: p.stem)


def _load_latent(path: Path) -> torch.Tensor:
    obj = _torch_load_trusted(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict) and isinstance(obj.get("latent"), torch.Tensor):
        t = obj["latent"]
    else:
        raise ValueError(f"Unsupported latent format: {path}")
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    _require(t.ndim == 3, f"Latent must be [C,H,W] or [1,C,H,W], got {tuple(t.shape)} for {path}")
    return t.detach().float()


def _get_cache_name(target_reference_rgb_dir: Path, eval_image_size: int, clip_backbone_id: str) -> str:
    key = f"{str(target_reference_rgb_dir.resolve())}|{eval_image_size}|inception_v3_pool3|{clip_backbone_id}"
    md5 = hashlib.md5(key.encode("utf-8")).hexdigest()
    clip_tag = clip_backbone_id.replace("/", "_").replace("-", "_")
    return f"eval_ref_cache_{md5}_res{eval_image_size}_inception_v3_pool3_{clip_tag}.pt"


def _batch_iter(seq: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def _mean_cov(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _kid_mean_std_from_features(
    feats_ref: np.ndarray,
    feats_gen: np.ndarray,
    num_subsets: int = 100,
    max_subset_size: int = 1000,
    seed: int = 7,
) -> Tuple[float, float]:
    _require(feats_ref.ndim == 2 and feats_gen.ndim == 2, "KID features must be 2D arrays")
    _require(feats_ref.shape[1] == feats_gen.shape[1], "KID features dim mismatch")

    n_dim = int(feats_ref.shape[1])
    m = min(int(feats_ref.shape[0]), int(feats_gen.shape[0]), int(max_subset_size))
    _require(m >= 2, "KID requires at least 2 samples in each set")

    rng = np.random.default_rng(seed)
    subset_vals: List[float] = []
    for _ in range(max(1, int(num_subsets))):
        x = feats_gen[rng.choice(feats_gen.shape[0], m, replace=False)]
        y = feats_ref[rng.choice(feats_ref.shape[0], m, replace=False)]
        a = (x @ x.T / n_dim + 1.0) ** 3 + (y @ y.T / n_dim + 1.0) ** 3
        b = (x @ y.T / n_dim + 1.0) ** 3
        kid_subset = (a.sum() - np.diag(a).sum()) / (m - 1) - 2.0 * b.sum() / m
        kid_subset = kid_subset / m
        subset_vals.append(float(kid_subset))

    arr = np.asarray(subset_vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    try:
        from scipy import linalg as scipy_linalg
    except Exception as exc:
        raise RuntimeError("scipy is required for FID computation from cached statistics") from exc

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = scipy_linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = scipy_linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(fid)


def _load_clip_model(eval_cfg: Dict[str, Any], device: torch.device):
    try:
        from transformers import CLIPModel, CLIPProcessor
        from transformers.utils import logging as transformers_logging
    except Exception as exc:
        raise RuntimeError("Missing transformers CLIP deps. Install transformers.") from exc

    # Force-disable Hugging Face/Transformers progress bars to avoid stderr spam.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        transformers_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass

    clip_backbone_id = str(eval_cfg.get("clip_backbone_id", "openai/clip-vit-base-patch32"))
    clip_model_cache_dir = str(eval_cfg.get("clip_model_cache_dir", "") or "")
    clip_local_only = bool(eval_cfg.get("clip_local_only", False))
    clip_use_safetensors = bool(eval_cfg.get("clip_use_safetensors", False))

    kwargs: Dict[str, Any] = {
        "local_files_only": clip_local_only,
    }
    if clip_model_cache_dir:
        kwargs["cache_dir"] = clip_model_cache_dir
    if clip_use_safetensors:
        kwargs["use_safetensors"] = True

    model = CLIPModel.from_pretrained(clip_backbone_id, **kwargs).to(device)
    processor = CLIPProcessor.from_pretrained(clip_backbone_id, **kwargs)
    model.eval()
    return clip_backbone_id, model, processor


class _CleanFIDTorchscriptWrapper(nn.Module):
    def __init__(self, scripted_model, resize_inside: bool = False):
        super().__init__()
        self.base = scripted_model.eval()
        self.layers = self.base.layers
        self.resize_inside = resize_inside

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from cleanfid.inception_torchscript import disable_gpu_fuser_on_pt19

        with disable_gpu_fuser_on_pt19():
            bs = x.shape[0]
            if self.resize_inside:
                return self.base(x, return_features=True).view((bs, 2048))
            assert (x.shape[2] == 299) and (x.shape[3] == 299)
            x = (x - 128) / 128
            return self.layers.forward(x).view((bs, 2048))


def _build_cleanfid_feature_extractor(shared_cfg: Dict[str, Any], eval_cfg: Dict[str, Any], device: torch.device):
    from cleanfid import fid

    def _build_default():
        default_path = Path("/tmp/inception-2015-12-05.pt")
        if default_path.exists():
            _log(f"[Phase3][Info] Using cleanfid default Inception weights: {default_path}")
        else:
            _log(
                "[Phase3][Warn] Falling back to cleanfid default Inception weights; "
                f"weights will be downloaded to {default_path}"
            )
        return fid.build_feature_extractor(mode="clean", device=device, use_dataparallel=False)

    raw_path = str(
        eval_cfg.get("cleanfid_inception_path", "") or shared_cfg.get("cleanfid_inception_path", "") or ""
    ).strip()
    if not raw_path:
        return _build_default()

    path = _resolve_repo_path(raw_path)
    if not path.exists():
        _log(
            "[Phase3][Warn] Configured cleanfid inception path does not exist: "
            f"{path}. Falling back to default cleanfid weights."
        )
        return _build_default()
    _log(f"[Phase3][Info] Using configured cleanfid inception weights: {path}")

    if path.is_dir():
        expected = path / "inception-2015-12-05.pt"
        if not expected.exists():
            _log(
                "[Phase3][Warn] Configured cleanfid inception directory does not contain "
                f"inception-2015-12-05.pt: {path}. Falling back to default cleanfid weights."
            )
            return _build_default()
        from cleanfid.inception_torchscript import InceptionV3W

        model = InceptionV3W(str(path), download=False, resize_inside=False).to(device)
    else:
        try:
            scripted_model = torch.jit.load(str(path)).eval()
        except Exception as exc:
            state_dict_hint = ""
            try:
                maybe_obj = torch.load(str(path), map_location="cpu")
                if isinstance(maybe_obj, dict):
                    state_dict_hint = (
                        " The file looks like a regular PyTorch state_dict (.pth), not the "
                        "TorchScript cleanfid Inception weights."
                    )
            except Exception:
                pass
            _log(
                "[Phase3][Warn] Invalid cleanfid inception weights file: "
                f"{path}. Expected the TorchScript file 'inception-2015-12-05.pt' or a directory containing it."
                f"{state_dict_hint} Original error: {exc}"
            )
            _log("[Phase3][Warn] Falling back to default cleanfid weights.")
            return _build_default()
        model = _CleanFIDTorchscriptWrapper(scripted_model, resize_inside=False).to(device)

    model.eval()
    return lambda x: model(x)


def _to_feature_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if hasattr(output, "pooler_output") and torch.is_tensor(output.pooler_output):
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state[:, 0]
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported CLIP output type: {type(output)}")


@torch.no_grad()
def _clip_image_features(
    image_paths: Sequence[Path],
    clip_model,
    clip_processor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    for batch_paths in tqdm(list(_batch_iter(list(image_paths), batch_size)), desc="CLIP image feats", ascii=True):
        imgs = [_pil_rgb(p) for p in batch_paths]
        inputs = clip_processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = _to_feature_tensor(clip_model.get_image_features(**inputs))
        outputs = F.normalize(outputs.float(), dim=-1)
        feats.append(outputs.cpu())
        del imgs, inputs, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(feats, dim=0)


def _phase0_parse_config(args: argparse.Namespace) -> EvalConfig:
    config_path = _resolve_repo_path(args.config)
    cfg = load_yaml_config(str(config_path))

    initial_eval_cfg = cfg.get("cyclegan_eval", {})
    _require(isinstance(initial_eval_cfg, dict), "config.cyclegan_eval must be a dict")

    experiment_dir_raw = str(initial_eval_cfg.get("experiment_dir", "") or "").strip()
    experiment_dir = Path("")
    if experiment_dir_raw:
        experiment_dir_candidate = _resolve_repo_path(experiment_dir_raw)
        experiment_dir = experiment_dir_candidate
        experiment_config_path = experiment_dir / "config.yaml"
        _require(experiment_dir.exists(), f"cyclegan_eval.experiment_dir not found: {experiment_dir}")
        _require(experiment_config_path.exists(), f"experiment config not found: {experiment_config_path}")
        base_cfg = cfg
        cfg = load_yaml_config(str(experiment_config_path))
        merged_shared_cfg = cfg.get("shared", {})
        _require(isinstance(merged_shared_cfg, dict), "experiment config.shared must be a dict")
        merged_shared_cfg = dict(merged_shared_cfg)
        initial_shared_cfg = base_cfg.get("shared", {})
        _require(isinstance(initial_shared_cfg, dict), "config.shared must be a dict")
        merged_shared_cfg.update(initial_shared_cfg)
        cfg["shared"] = merged_shared_cfg

        merged_eval_cfg = cfg.get("cyclegan_eval", {})
        _require(isinstance(merged_eval_cfg, dict), "experiment config.cyclegan_eval must be a dict")
        merged_eval_cfg = dict(merged_eval_cfg)
        merged_eval_cfg.update(initial_eval_cfg)
        merged_eval_cfg["experiment_dir"] = str(experiment_dir)
        cfg["cyclegan_eval"] = merged_eval_cfg

    model_cfg = cfg.get("model", {})
    shared_cfg = cfg.get("shared", {})
    vis_cfg = cfg.get("visualization", {})
    eval_cfg = cfg.get("cyclegan_eval", {})
    data_cfg = cfg.get("data", {})

    _require(isinstance(shared_cfg, dict), "config.shared must be a dict")
    _require(isinstance(model_cfg, dict), "config.model must be a dict")
    _require(isinstance(vis_cfg, dict), "config.visualization must be a dict")
    _require(isinstance(eval_cfg, dict), "config.cyclegan_eval must be a dict")
    _require(isinstance(data_cfg, dict), "config.data must be a dict")

    domain_A_name = str(eval_cfg.get("domain_A_name", "") or "").strip()
    domain_B_name = str(eval_cfg.get("domain_B_name", "") or "").strip()

    base_latent_dir = _resolve_repo_path(args.base_latent_dir or eval_cfg.get("base_latent_dir", "") or "")
    base_orig_rgb_dir = _resolve_repo_path(args.base_orig_rgb_dir or eval_cfg.get("base_orig_rgb_dir", "") or "")
    base_vae_recon_dir = _resolve_repo_path(args.base_vae_recon_dir or eval_cfg.get("base_vae_recon_dir", "") or "")

    if experiment_dir:
        default_checkpoint_dir = experiment_dir / "model"
        default_generator_checkpoint_path = default_checkpoint_dir / "last.pt"
        default_eval_output_dir = experiment_dir / "eval"
        eval_output_dir = _resolve_repo_path(args.eval_output_dir or default_eval_output_dir)
        checkpoint_dir = _resolve_repo_path(args.checkpoint_dir or default_checkpoint_dir)
        generator_checkpoint_path = _resolve_repo_path(
            args.generator_checkpoint_path or default_generator_checkpoint_path
        )
    else:
        default_checkpoint_dir = _resolve_repo_path(cfg.get("train", {}).get("checkpoint_dir", "outputs/model"))
        default_generator_checkpoint_path = default_checkpoint_dir / "last.pt"
        default_eval_output_dir = _resolve_repo_path(eval_cfg.get("eval_output_dir", "outputs/eval_latent"))
        eval_output_dir = _resolve_repo_path(
            args.eval_output_dir or eval_cfg.get("eval_output_dir", "") or default_eval_output_dir
        )
        checkpoint_dir = _resolve_repo_path(
            args.checkpoint_dir or cfg.get("train", {}).get("checkpoint_dir", "") or default_checkpoint_dir
        )
        generator_checkpoint_path = _resolve_repo_path(
            args.generator_checkpoint_path or eval_cfg.get("generator_checkpoint_path", "") or default_generator_checkpoint_path
        )

    reference_cache_dir = _resolve_repo_path(
        args.reference_cache_dir or eval_cfg.get("reference_cache_dir", "outputs/eval_cache_latent")
    )

    eval_image_size = int(args.eval_image_size or eval_cfg.get("eval_image_size", 256))
    eval_batch_size = int(args.eval_batch_size or eval_cfg.get("eval_batch_size", 8))

    device_str = str(args.run_device or eval_cfg.get("run_device", "cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    run_device = torch.device(device_str)
    use_bf16_autocast = bool(
        args.use_bf16_autocast if args.use_bf16_autocast is not None else eval_cfg.get("use_bf16_autocast", True)
    )
    max_eval_samples = int(eval_cfg.get("max_eval_samples", -1))
    eval_seed = int(eval_cfg.get("eval_seed", 42))

    _require(domain_A_name, "cyclegan_eval.domain_A_name is required")
    _require(domain_B_name, "cyclegan_eval.domain_B_name is required")
    _require(str(base_latent_dir), "cyclegan_eval.base_latent_dir is required")
    _require(str(base_orig_rgb_dir), "cyclegan_eval.base_orig_rgb_dir is required")
    _require(str(base_vae_recon_dir), "cyclegan_eval.base_vae_recon_dir is required")

    _require(base_latent_dir.exists(), f"base_latent_dir not found: {base_latent_dir}")
    _require(base_orig_rgb_dir.exists(), f"base_orig_rgb_dir not found: {base_orig_rgb_dir}")
    _require(base_vae_recon_dir.exists(), f"base_vae_recon_dir not found: {base_vae_recon_dir}")

    # Validate domain subfolders for both directions in Phase 0.
    _resolve_domain_dir(base_latent_dir, domain_A_name, "base_latent_dir")
    _resolve_domain_dir(base_latent_dir, domain_B_name, "base_latent_dir")
    _resolve_domain_dir(base_orig_rgb_dir, domain_A_name, "base_orig_rgb_dir")
    _resolve_domain_dir(base_orig_rgb_dir, domain_B_name, "base_orig_rgb_dir")
    _resolve_domain_dir(base_vae_recon_dir, domain_A_name, "base_vae_recon_dir")
    _resolve_domain_dir(base_vae_recon_dir, domain_B_name, "base_vae_recon_dir")

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    reference_cache_dir.mkdir(parents=True, exist_ok=True)

    return EvalConfig(
        shared_cfg=shared_cfg,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        vis_cfg=vis_cfg,
        eval_cfg=eval_cfg,
        domain_A_name=domain_A_name,
        domain_B_name=domain_B_name,
        base_latent_dir=base_latent_dir,
        base_orig_rgb_dir=base_orig_rgb_dir,
        base_vae_recon_dir=base_vae_recon_dir,
        eval_output_dir=eval_output_dir,
        reference_cache_dir=reference_cache_dir,
        checkpoint_dir=checkpoint_dir,
        generator_checkpoint_path=generator_checkpoint_path,
        max_eval_samples=max_eval_samples,
        eval_seed=eval_seed,
        eval_image_size=eval_image_size,
        eval_batch_size=eval_batch_size,
        run_device=run_device,
        use_bf16_autocast=use_bf16_autocast,
    )


def _phase1_build_or_load_ref_cache(
    ecfg: EvalConfig,
    direction: EvalDirection,
    clip_runtime: Optional[Dict[str, Any]] = None,
    force_regen_cache: bool = False,
) -> Dict[str, Any]:
    cleanfid_model = None

    if clip_runtime is None:
        clip_model_id, clip_model, clip_processor = _load_clip_model(ecfg.eval_cfg, ecfg.run_device)
    else:
        clip_model_id = str(clip_runtime["clip_model_id"])
        clip_model = clip_runtime["clip_model"]
        clip_processor = clip_runtime["clip_processor"]

    def _build_or_load_domain_cache(reference_rgb_dir: Path, cache_tag: str) -> Dict[str, Any]:
        nonlocal cleanfid_model

        cache_name = f"{cache_tag}_" + _get_cache_name(reference_rgb_dir, ecfg.eval_image_size, clip_model_id)
        cache_path = ecfg.reference_cache_dir / cache_name

        if cache_path.exists() and not force_regen_cache:
            _log(f"[Phase1] Use existing {cache_tag} cache: {cache_path}")
            payload = _torch_load_trusted(cache_path, map_location="cpu")
            payload["cache_path"] = str(cache_path)
            return payload

        _log(f"[Phase1] Building {cache_tag} cache: {reference_rgb_dir}")
        ref_paths = _list_images(reference_rgb_dir)
        ref_cache_max_images = int(ecfg.eval_cfg.get("ref_cache_max_images", 0))

        num_ref_images = None
        shuffle_ref = False
        if ref_cache_max_images > 0:
            num_ref_images = ref_cache_max_images
            shuffle_ref = True
            if len(ref_paths) > ref_cache_max_images:
                random.seed(42)
                ref_paths = random.sample(ref_paths, ref_cache_max_images)

        if cleanfid_model is None:
            cleanfid_model = _build_cleanfid_feature_extractor(ecfg.shared_cfg, ecfg.eval_cfg, ecfg.run_device)

        cleanfid_features = fid.get_folder_features(
            fdir=str(reference_rgb_dir),
            model=cleanfid_model,
            mode="clean",
            num_workers=4,
            num=num_ref_images,
            shuffle=shuffle_ref,
            seed=42,
            batch_size=max(8, ecfg.eval_batch_size),
            device=ecfg.run_device,
            verbose=True,
        )
        cleanfid_features = np.asarray(cleanfid_features)
        mu, sigma = _mean_cov(cleanfid_features)

        clip_feats = _clip_image_features(
            image_paths=ref_paths,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=ecfg.run_device,
            batch_size=ecfg.eval_batch_size,
        )
        clip_proto_img = F.normalize(clip_feats.mean(dim=0, keepdim=False), dim=0).cpu()

        payload: Dict[str, Any] = {
            "meta": {
                "cache_tag": cache_tag,
                "reference_rgb_dir": str(reference_rgb_dir.resolve()),
                "eval_image_size": ecfg.eval_image_size,
                "extractors": ["inception_v3_pool3", clip_model_id],
                "num_ref_images_cleanfid": int(cleanfid_features.shape[0]),
                "num_ref_images_clip": int(clip_feats.shape[0]),
            },
            "fid_kid": {
                "mu": mu.astype(np.float64),
                "sigma": sigma.astype(np.float64),
            },
            "clip": {
                "image_prototype": clip_proto_img,
            },
            "cache_path": str(cache_path),
        }
        torch.save(payload, cache_path)
        _log(f"[Phase1] {cache_tag} cache saved: {cache_path}")
        return payload

    source_cache = _build_or_load_domain_cache(direction.orig_rgb_dir, "source")
    target_cache = _build_or_load_domain_cache(direction.tgt_ref_dir, "target")

    merged_cache: Dict[str, Any] = {
        "meta": {
            "direction": direction.name,
            "source_domain": direction.src_domain,
            "target_domain": direction.tgt_domain,
        },
        # Distribution metrics (FID/KID) are always against target domain.
        "fid_kid": target_cache["fid_kid"],
        "clip": {
            "source_image_prototype": source_cache["clip"]["image_prototype"],
            "target_image_prototype": target_cache["clip"]["image_prototype"],
        },
        "cache_path": {
            "source": source_cache.get("cache_path", ""),
            "target": target_cache.get("cache_path", ""),
        },
        "_clip_runtime": {
            "clip_model_id": clip_model_id,
            "clip_model": clip_model,
            "clip_processor": clip_processor,
        },
    }
    return merged_cache


def _load_generators(ecfg: EvalConfig) -> Dict[str, torch.nn.Module]:
    _require(ecfg.generator_checkpoint_path.exists(), f"checkpoint not found: {ecfg.generator_checkpoint_path}")

    payload = _torch_load_trusted(ecfg.generator_checkpoint_path, map_location="cpu")
    _require(isinstance(payload, dict), f"Unsupported checkpoint payload type: {type(payload)}")

    def _make_generator() -> ResnetGenerator:
        return ResnetGenerator(
            in_ch=int(ecfg.model_cfg.get("in_channels", 4)),
            out_ch=int(ecfg.model_cfg.get("out_channels", 4)),
            ngf=int(ecfg.model_cfg.get("ngf", 256)),
            n_res_blocks=int(ecfg.model_cfg.get("n_res_blocks", 9)),
            out_activation=str(ecfg.model_cfg.get("out_activation", "none")),
            use_global_residual=bool(ecfg.model_cfg.get("use_global_residual", False)),
            use_pointwise_only=bool(ecfg.model_cfg.get("use_pointwise_only", False)),
        )

    def _pick_state_dict(keys: Sequence[str]) -> Optional[Dict[str, torch.Tensor]]:
        for k in keys:
            sd_obj = payload.get(k)
            if isinstance(sd_obj, dict) and all(isinstance(v, torch.Tensor) for v in sd_obj.values()):
                return sd_obj
        return None

    sd_g = _pick_state_dict(("G", "netG_A", "generator", "state_dict"))
    sd_f = _pick_state_dict(("F", "netG_B"))

    if sd_g is None and all(isinstance(v, torch.Tensor) for v in payload.values()):
        sd_g = payload
    _require(sd_g is not None, f"Cannot locate A2B generator weights in: {ecfg.generator_checkpoint_path}")
    _require(sd_f is not None, f"Cannot locate B2A generator weights (F/netG_B) in: {ecfg.generator_checkpoint_path}")

    G = _make_generator()
    F_gen = _make_generator()

    missing_g, unexpected_g = G.load_state_dict(sd_g, strict=False)
    missing_f, unexpected_f = F_gen.load_state_dict(sd_f, strict=False)
    if missing_g:
        _warn(f"G state_dict missing keys: {len(missing_g)}")
    if unexpected_g:
        _warn(f"G state_dict unexpected keys: {len(unexpected_g)}")
    if missing_f:
        _warn(f"F state_dict missing keys: {len(missing_f)}")
    if unexpected_f:
        _warn(f"F state_dict unexpected keys: {len(unexpected_f)}")

    G.to(ecfg.run_device).eval()
    F_gen.to(ecfg.run_device).eval()
    return {"A2B": G, "B2A": F_gen}


def _save_tensor_image(img: torch.Tensor, out_path: Path) -> None:
    # ToPILImage does not support bfloat16, so force float32 on CPU.
    img = img.detach().to(dtype=torch.float32).cpu().clamp(0, 1)
    pil = transforms.ToPILImage()(img)
    pil.save(out_path)


def _make_3panel(orig: Image.Image, recon: Image.Image, transfer: Image.Image, stem: str) -> Image.Image:
    w = orig.width
    h = orig.height
    label_h = 28
    canvas = Image.new("RGB", (w * 3, h + label_h), (245, 245, 245))
    canvas.paste(orig, (0, label_h))
    canvas.paste(recon, (w, label_h))
    canvas.paste(transfer, (w * 2, label_h))

    draw = ImageDraw.Draw(canvas)
    labels = ["Orig_Img", "Recon_Img", "Transferred_Img"]
    for i, txt in enumerate(labels):
        draw.text((10 + i * w, 6), txt, fill=(20, 20, 20))
    draw.text((10, h + 8), stem, fill=(20, 20, 20))
    return canvas


@torch.no_grad()
def _phase2_infer_and_visualize(
    ecfg: EvalConfig,
    direction: EvalDirection,
    generator: torch.nn.Module,
    vae,
) -> Dict[str, Any]:
    image_out_dir = direction.out_dir / "eval_images"
    vis3_out_dir = direction.out_dir / "eval_vis_3panel"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    vis3_out_dir.mkdir(parents=True, exist_ok=True)

    latent_paths_all = _list_latents(direction.latent_src_dir)
    selected_latent_paths = _select_latent_paths(
        latent_paths=latent_paths_all,
        max_eval_samples=ecfg.max_eval_samples,
        eval_seed=ecfg.eval_seed,
    )
    selected_stems = [_safe_stem_name(p) for p in selected_latent_paths]

    orig_index = _build_stem_index(direction.orig_rgb_dir)
    recon_index = _build_stem_index(direction.vae_recon_dir)

    latents_scaled = bool(ecfg.data_cfg.get("latents_scaled", False))
    latent_divisor = float(ecfg.data_cfg.get("latent_divisor", 1.0))
    vae_scaling_factor = float(ecfg.vis_cfg.get("vae_scaling_factor", 0.18215))

    transferred_paths: List[Path] = []
    paired_stems: List[str] = []

    pbar = tqdm(list(_batch_iter(selected_latent_paths, ecfg.eval_batch_size)), desc=f"[{direction.name}] Phase2 inference", ascii=True)
    for latent_batch_paths in pbar:
        stems = [_safe_stem_name(p) for p in latent_batch_paths]
        latent_tensors = [_load_latent(p) for p in latent_batch_paths]

        try:
            latent_batch = torch.stack(latent_tensors, dim=0).to(ecfg.run_device, non_blocking=True)
            latent_in = latent_batch / latent_divisor if not math.isclose(latent_divisor, 1.0) else latent_batch
            use_amp = ecfg.use_bf16_autocast and ecfg.run_device.type == "cuda"
            amp_dtype = torch.bfloat16 if use_amp else torch.float32
            with torch.autocast(device_type=ecfg.run_device.type, enabled=use_amp, dtype=amp_dtype):
                fake_latent = generator(latent_in)
                fake_img = decode_latents_to_images(
                    vae=vae,
                    latents=fake_latent,
                    latents_scaled=latents_scaled,
                    scaling_factor=vae_scaling_factor,
                    latent_divisor=latent_divisor,
                )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            _warn(f"OOM on batch size={len(latent_batch_paths)}; fallback to per-sample for this batch")
            if ecfg.run_device.type == "cuda":
                torch.cuda.empty_cache()
            fake_list: List[torch.Tensor] = []
            for latent_tensor in latent_tensors:
                one = latent_tensor.unsqueeze(0).to(ecfg.run_device, non_blocking=True)
                one_in = one / latent_divisor if not math.isclose(latent_divisor, 1.0) else one
                one_fake_latent = generator(one_in)
                one_fake_img = decode_latents_to_images(
                    vae=vae,
                    latents=one_fake_latent,
                    latents_scaled=latents_scaled,
                    scaling_factor=vae_scaling_factor,
                    latent_divisor=latent_divisor,
                )
                fake_list.append(one_fake_img[0].detach().cpu())
                del one, one_in, one_fake_latent, one_fake_img
            fake_img = torch.stack(fake_list, dim=0)

        for i, stem in enumerate(stems):
            out_path = image_out_dir / f"{stem}.png"
            _save_tensor_image(fake_img[i], out_path)
            transferred_paths.append(out_path)

            orig_path = orig_index.get(stem)
            recon_path = recon_index.get(stem)
            if orig_path is None or recon_path is None:
                _warn(f"[{direction.name}] Missing stem={stem} in orig/recon dir, skip 3-panel visualization")
                continue

            orig_img = _pil_rgb(orig_path, image_size=ecfg.eval_image_size)
            recon_img = _pil_rgb(recon_path, image_size=ecfg.eval_image_size)
            transfer_img = _pil_rgb(out_path, image_size=ecfg.eval_image_size)
            panel = _make_3panel(orig_img, recon_img, transfer_img, stem)
            panel.save(vis3_out_dir / f"{stem}.png")
            paired_stems.append(stem)

        del latent_tensors
        if "latent_batch" in locals():
            del latent_batch
        if "latent_in" in locals():
            del latent_in
        if "fake_latent" in locals():
            del fake_latent
        if "fake_img" in locals():
            del fake_img
        if ecfg.run_device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "transferred_paths": transferred_paths,
        "transferred_dir": image_out_dir,
        "selected_stems": selected_stems,
        "paired_stems": paired_stems,
    }

def _phase3_metrics(
    ecfg: EvalConfig,
    direction: EvalDirection,
    ref_cache: Dict[str, Any],
    phase2_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    from cleanfid import fid

    cleanfid_model = _build_cleanfid_feature_extractor(ecfg.shared_cfg, ecfg.eval_cfg, ecfg.run_device)

    disable_lpips = bool(ecfg.eval_cfg.get("skip_lpips", False))
    disable_clip = bool(ecfg.eval_cfg.get("skip_clip", False))

    transferred_dir = Path(phase2_outputs["transferred_dir"])
    transferred_paths = _list_images(transferred_dir)

    # Distribution metrics.
    gen_feats = fid.get_folder_features(
        fdir=str(transferred_dir),
        model=cleanfid_model,
        mode="clean",
        num_workers=4,
        batch_size=max(8, ecfg.eval_batch_size),
        device=ecfg.run_device,
        verbose=True,
    )
    gen_feats = np.asarray(gen_feats)
    mu_g, sigma_g = _mean_cov(gen_feats)
    mu_r = np.asarray(ref_cache["fid_kid"]["mu"])
    sigma_r = np.asarray(ref_cache["fid_kid"]["sigma"])
    fid_value = _frechet_distance(mu_g, sigma_g, mu_r, sigma_r)

    max_ref_compare = int(ecfg.eval_cfg.get("kid_ref_max_images", 0))
    ref_feats_kid = fid.get_folder_features(
        fdir=str(direction.tgt_ref_dir),
        model=cleanfid_model,
        mode="clean",
        num_workers=4,
        num=(max_ref_compare if max_ref_compare > 0 else None),
        shuffle=(max_ref_compare > 0),
        seed=7,
        batch_size=max(8, ecfg.eval_batch_size),
        device=ecfg.run_device,
        verbose=True,
    )
    ref_feats_kid = np.asarray(ref_feats_kid)

    kid_subsets = int(ecfg.eval_cfg.get("kid_num_subsets", 100))
    kid_subset_size = int(ecfg.eval_cfg.get("kid_subset_size", 1000))
    kid_value, kid_std_value = _kid_mean_std_from_features(
        feats_ref=ref_feats_kid,
        feats_gen=gen_feats,
        num_subsets=kid_subsets,
        max_subset_size=kid_subset_size,
        seed=7,
    )

    # Pairwise metrics.
    orig_index = _build_stem_index(direction.orig_rgb_dir)
    gen_index = _build_stem_index(transferred_dir)
    selected_stems = [str(s) for s in phase2_outputs.get("selected_stems", [])]
    pair_stems = [s for s in selected_stems if s in orig_index and s in gen_index]

    lpips_vals: List[float] = []
    clip_dir_vals: List[float] = []
    clip_style_vals: List[float] = []
    rows: List[Dict[str, Any]] = []

    lpips_model = None
    if not disable_lpips:
        try:
            import lpips

            lpips_model = lpips.LPIPS(net="vgg").to(ecfg.run_device)
            lpips_model.eval()
        except Exception as exc:
            _warn(f"LPIPS disabled due to import/init error: {exc}")

    clip_model = None
    clip_processor = None
    clip_source_proto = None
    clip_target_proto = None
    if not disable_clip:
        runtime = ref_cache.get("_clip_runtime", {})
        clip_model = runtime.get("clip_model")
        clip_processor = runtime.get("clip_processor")
        clip_source_proto = ref_cache.get("clip", {}).get("source_image_prototype")
        clip_target_proto = ref_cache.get("clip", {}).get("target_image_prototype")
        if (
            clip_model is None
            or clip_processor is None
            or clip_source_proto is None
            or clip_target_proto is None
        ):
            _warn("CLIP runtime/prototype missing in cache, CLIP metrics skipped")
            disable_clip = True

    style_vec = None
    if not disable_clip and clip_source_proto is not None and clip_target_proto is not None:
        src_proto = F.normalize(clip_source_proto.float().to(ecfg.run_device), dim=0)
        tgt_proto = F.normalize(clip_target_proto.float().to(ecfg.run_device), dim=0)
        style_vec = F.normalize(tgt_proto - src_proto, dim=0)

    for batch_stems in tqdm(list(_batch_iter(pair_stems, ecfg.eval_batch_size)), desc=f"[{direction.name}] Phase3 pairwise", ascii=True):
        orig_tensors: List[torch.Tensor] = []
        gen_tensors: List[torch.Tensor] = []
        orig_imgs: List[Image.Image] = []
        gen_imgs: List[Image.Image] = []

        for stem in batch_stems:
            o = _pil_rgb(orig_index[stem], ecfg.eval_image_size)
            g = _pil_rgb(gen_index[stem], ecfg.eval_image_size)
            orig_imgs.append(o)
            gen_imgs.append(g)
            orig_tensors.append(transforms.ToTensor()(o))
            gen_tensors.append(transforms.ToTensor()(g))

        orig_batch = torch.stack(orig_tensors, dim=0).to(ecfg.run_device)
        gen_batch = torch.stack(gen_tensors, dim=0).to(ecfg.run_device)

        if lpips_model is not None:
            # LPIPS expects [-1, 1]
            o_lp = orig_batch * 2 - 1
            g_lp = gen_batch * 2 - 1
            lp = lpips_model(o_lp, g_lp).view(-1)
            lp_cpu = lp.detach().cpu().tolist()
        else:
            lp_cpu = [float("nan")] * len(batch_stems)

        if not disable_clip and clip_model is not None and clip_processor is not None:
            inp_src = clip_processor(images=orig_imgs, return_tensors="pt")
            inp_gen = clip_processor(images=gen_imgs, return_tensors="pt")
            inp_src = {k: v.to(ecfg.run_device) for k, v in inp_src.items()}
            inp_gen = {k: v.to(ecfg.run_device) for k, v in inp_gen.items()}
            feat_src = _to_feature_tensor(clip_model.get_image_features(**inp_src))
            feat_gen = _to_feature_tensor(clip_model.get_image_features(**inp_gen))
            feat_src = F.normalize(feat_src.float(), dim=-1)
            feat_gen = F.normalize(feat_gen.float(), dim=-1)

            tgt = F.normalize(clip_target_proto.to(feat_gen.device).unsqueeze(0), dim=-1)
            style_vals = F.cosine_similarity(feat_gen, tgt.expand_as(feat_gen), dim=-1)
            style_cpu = style_vals.detach().cpu().tolist()

            dir_cpu: List[float]
            if style_vec is not None:
                edit_vec = F.normalize(feat_gen - feat_src, dim=-1)
                dir_vals = F.cosine_similarity(edit_vec, style_vec.unsqueeze(0).expand_as(edit_vec), dim=-1)
                dir_cpu = dir_vals.detach().cpu().tolist()
            else:
                dir_cpu = [float("nan")] * len(batch_stems)
        else:
            style_cpu = [float("nan")] * len(batch_stems)
            dir_cpu = [float("nan")] * len(batch_stems)

        for i, stem in enumerate(batch_stems):
            row = {
                "stem": stem,
                "orig_path": str(orig_index[stem]),
                "transferred_path": str(gen_index[stem]),
                "lpips": float(lp_cpu[i]) if not math.isnan(lp_cpu[i]) else None,
                "clip_dir": float(dir_cpu[i]) if not math.isnan(dir_cpu[i]) else None,
                "clip_style": float(style_cpu[i]) if not math.isnan(style_cpu[i]) else None,
            }
            rows.append(row)
            if not math.isnan(lp_cpu[i]):
                lpips_vals.append(float(lp_cpu[i]))
            if not math.isnan(dir_cpu[i]):
                clip_dir_vals.append(float(dir_cpu[i]))
            if not math.isnan(style_cpu[i]):
                clip_style_vals.append(float(style_cpu[i]))

        del orig_batch, gen_batch
        if ecfg.run_device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "direction": direction.name,
        "src_domain": direction.src_domain,
        "tgt_domain": direction.tgt_domain,
        "fid": float(fid_value),
        "kid_mean": kid_value,
        "kid_std": kid_std_value,
        "lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else None,
        "clip_dir_mean": float(np.mean(clip_dir_vals)) if clip_dir_vals else None,
        "clip_style_mean": float(np.mean(clip_style_vals)) if clip_style_vals else None,
        "num_generated": len(transferred_paths),
        "num_selected_stems": len(selected_stems),
        "num_pairs": len(rows),
        "reference_cache": ref_cache.get("cache_path", ""),
    }
    return {
        "summary": report,
        "rows": rows,
    }


def _write_reports(direction: EvalDirection, metric_payload: Dict[str, Any]) -> None:
    report_json = direction.out_dir / f"eval_metrics_{direction.name}.json"
    report_csv = direction.out_dir / f"eval_metrics_{direction.name}.csv"

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(metric_payload["summary"], f, indent=2, ensure_ascii=False)

    rows = metric_payload["rows"]
    fields = ["stem", "orig_path", "transferred_path", "lpips", "clip_dir", "clip_style"]
    with open(report_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})

    _log(f"[{direction.name}][Phase3] Metrics summary:")
    for k, v in metric_payload["summary"].items():
        _log(f"  - {k}: {v}")
    _log(f"[{direction.name}][Phase3] Saved JSON report to: {report_json}")
    _log(f"[{direction.name}][Phase3] Saved CSV report to: {report_csv}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Latent CycleGAN evaluation in RGB space")
    p.add_argument("--config", type=str, default="configs/example.yaml")

    # Phase 0 override with highest priority. Keep names consistent with YAML keys.
    p.add_argument("--base_latent_dir", type=str, default="")
    p.add_argument("--base_orig_rgb_dir", type=str, default="")
    p.add_argument("--base_vae_recon_dir", type=str, default="")

    p.add_argument("--checkpoint_dir", type=str, default="")
    p.add_argument("--generator_checkpoint_path", type=str, default="")
    p.add_argument("--eval_output_dir", type=str, default="")
    p.add_argument("--reference_cache_dir", type=str, default="")
    p.add_argument("--eval_image_size", type=int, default=0)
    p.add_argument("--eval_batch_size", type=int, default=0)
    p.add_argument("--run_device", type=str, default="")
    p.add_argument("--use_bf16_autocast", action="store_true", default=None)
    p.add_argument("--no_use_bf16_autocast", dest="use_bf16_autocast", action="store_false")
    p.add_argument("--force_regen_cache", action="store_true")
    return p


def _build_directions(ecfg: EvalConfig) -> List[EvalDirection]:
    def _mk(name: str, src_domain: str, tgt_domain: str) -> EvalDirection:
        latent_src_dir = _resolve_domain_dir(ecfg.base_latent_dir, src_domain, "base_latent_dir")
        orig_rgb_dir = _resolve_domain_dir(ecfg.base_orig_rgb_dir, src_domain, "base_orig_rgb_dir")
        vae_recon_dir = _resolve_domain_dir(ecfg.base_vae_recon_dir, src_domain, "base_vae_recon_dir")
        tgt_ref_dir = _resolve_domain_dir(ecfg.base_orig_rgb_dir, tgt_domain, "base_orig_rgb_dir")
        out_dir = ecfg.eval_output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        return EvalDirection(
            name=name,
            src_domain=src_domain,
            tgt_domain=tgt_domain,
            latent_src_dir=latent_src_dir,
            orig_rgb_dir=orig_rgb_dir,
            vae_recon_dir=vae_recon_dir,
            tgt_ref_dir=tgt_ref_dir,
            out_dir=out_dir,
        )

    return [
        _mk("A2B", ecfg.domain_A_name, ecfg.domain_B_name),
        _mk("B2A", ecfg.domain_B_name, ecfg.domain_A_name),
    ]


def main() -> None:
    args = build_argparser().parse_args()

    _log("[Phase0] Parsing YAML and applying CLI overrides...")
    ecfg = _phase0_parse_config(args)

    _log("[Setup] Loading dual generators from one checkpoint...")
    generators = _load_generators(ecfg)

    _log("[Setup] Loading VAE once for all directions...")
    vae_name, vae_subfolder = resolve_vae_source({"shared": ecfg.shared_cfg}, ecfg.vis_cfg)
    vae = _load_vae(
        model_name_or_path=vae_name or "runwayml/stable-diffusion-v1-5",
        subfolder=vae_subfolder,
        device=ecfg.run_device,
    )

    _log("[Setup] Loading CLIP once for all directions...")
    clip_model_id, clip_model, clip_processor = _load_clip_model(ecfg.eval_cfg, ecfg.run_device)
    clip_runtime = {
        "clip_model_id": clip_model_id,
        "clip_model": clip_model,
        "clip_processor": clip_processor,
    }

    directions = _build_directions(ecfg)
    for direction in directions:
        _log(
            f"[{direction.name}] src={direction.src_domain} tgt={direction.tgt_domain} "
            f"max_eval_samples={ecfg.max_eval_samples} eval_seed={ecfg.eval_seed}"
        )

        _log(f"[{direction.name}][Phase1] Strict cache pre-computation...")
        ref_cache = _phase1_build_or_load_ref_cache(
            ecfg=ecfg,
            direction=direction,
            clip_runtime=clip_runtime,
            force_regen_cache=args.force_regen_cache,
        )

        _log(f"[{direction.name}][Phase2] Latent inference and 3-panel visualization...")
        phase2_outputs = _phase2_infer_and_visualize(
            ecfg=ecfg,
            direction=direction,
            generator=generators[direction.name],
            vae=vae,
        )

        _log(f"[{direction.name}][Phase3] Metric computation and aggregation...")
        metric_payload = _phase3_metrics(
            ecfg=ecfg,
            direction=direction,
            ref_cache=ref_cache,
            phase2_outputs=phase2_outputs,
        )
        _write_reports(direction=direction, metric_payload=metric_payload)


if __name__ == "__main__":
    main()
