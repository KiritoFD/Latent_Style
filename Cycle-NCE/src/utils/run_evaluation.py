"""
LGT Evaluation Pro: Optimized with Pipeline Offloading, Async I/O & Vectorization
Target Hardware: RTX 4070 Laptop (8GB VRAM) | CPU: 7940HX
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from contextlib import nullcontext
import torch

# 妫ｅ啯鏆?Enable Tensor Cores for float32 matrix multiplication (Fixes UserWarning)
torch.set_float32_matmul_precision('high')

import numpy as np
import csv
import random
import gc
import time
import hashlib
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Metrics
try:
    import lpips
except ImportError:
    lpips = None

try:
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
from torchvision.utils import save_image
# Project imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.inference import LGTInference, load_vae, encode_image, decode_latent

# ==========================================
# Optimized Feature Extractors
# ==========================================

class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        # Load VGG only once, freeze immediately
        # Use weights parameter instead of deprecated pretrained=True
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval().to(device)
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.layer_ids = [8, 15] 
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1))

    def get_features(self, x):
        # Normalize and extract
        x = (x.to(self.mean.device) - self.mean) / self.std
        feats = []
        h = x
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.layer_ids:
                feats.append(h.detach().cpu()) # Store on CPU to save VRAM
        return feats

def to_lpips_input(img_tensor):
    return img_tensor * 2.0 - 1.0


def compute_swd(feat_x: torch.Tensor, feat_y: torch.Tensor, num_projections: int = 256) -> torch.Tensor:
    """
    Compute per-sample SWD between two feature maps.
    feat_x, feat_y: [B, C, H, W], same shape.
    Returns: [B] SWD values.
    """
    if feat_x.shape != feat_y.shape:
        raise ValueError(f"SWD shape mismatch: {feat_x.shape} vs {feat_y.shape}")
    if feat_x.ndim != 4:
        raise ValueError(f"SWD expects 4D tensors [B,C,H,W], got {feat_x.ndim}D")

    b, c, h, w = feat_x.shape
    x = feat_x.permute(0, 2, 3, 1).reshape(b, h * w, c).to(torch.float32)
    y = feat_y.permute(0, 2, 3, 1).reshape(b, h * w, c).to(torch.float32)

    proj = torch.randn(c, int(num_projections), device=x.device, dtype=x.dtype)
    proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)

    proj_x = torch.matmul(x, proj)
    proj_y = torch.matmul(y, proj)

    proj_x, _ = torch.sort(proj_x, dim=1)
    proj_y, _ = torch.sort(proj_y, dim=1)
    return ((proj_x - proj_y) ** 2).mean(dim=(1, 2))


def compute_lf_structure_similarity_batch(
    img_x: torch.Tensor,
    img_y: torch.Tensor,
    blur_kernel: int = 21,
    sigma: float = 5.0,
) -> torch.Tensor:
    """
    Compute low-frequency structure similarity per sample for batches in [0,1].
    This suppresses high-frequency style strokes and compares only coarse layout.
    img_x, img_y: [B, 3, H, W]
    returns: [B]
    """
    if img_x.shape != img_y.shape:
        raise ValueError(f"LF-SSIM shape mismatch: {img_x.shape} vs {img_y.shape}")

    device, dtype = img_x.device, img_x.dtype
    weights = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=dtype).view(1, 3, 1, 1)
    x = (img_x * weights).sum(dim=1, keepdim=True).to(torch.float32)
    y = (img_y * weights).sum(dim=1, keepdim=True).to(torch.float32)

    k = int(blur_kernel)
    if k <= 0:
        raise ValueError(f"blur_kernel must be > 0, got {k}")
    if (k % 2) == 0:
        raise ValueError(f"blur_kernel must be odd, got {k}")

    coord = torch.arange(k, device=device, dtype=torch.float32) - (k // 2)
    g = torch.exp(-(coord**2) / (2 * sigma**2))
    g = g / g.sum()
    window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    pad = k // 2
    lf_x = F.conv2d(x, window, padding=pad)
    lf_y = F.conv2d(y, window, padding=pad)
    lf_x_flat = lf_x.view(lf_x.shape[0], -1)
    lf_y_flat = lf_y.view(lf_y.shape[0], -1)
    return F.cosine_similarity(lf_x_flat, lf_y_flat, dim=1).to(torch.float32)


def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda oom" in msg)


def _lpips_forward_safe(
    loss_fn,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device: str,
    chunk_size: int,
    cpu_fallback: bool,
    tag: str = "lpips",
) -> torch.Tensor:
    """
    Robust LPIPS forward:
    - Runs in chunks to reduce peak memory.
    - On CUDA OOM, halves chunk size and retries.
    - If chunk size reaches 1 and still OOM, optionally falls back to CPU.
    Returns a CPU tensor shaped [N].
    """
    n = int(x.shape[0])
    if n <= 0:
        return torch.empty((0,), dtype=torch.float32)

    cur_chunk = max(1, min(int(chunk_size), n))

    while True:
        try:
            outs = []
            with torch.no_grad():
                for s in range(0, n, cur_chunk):
                    e = min(s + cur_chunk, n)
                    d = loss_fn(to_lpips_input(x[s:e]), to_lpips_input(y[s:e]))
                    outs.append(d.detach().cpu().view(-1))
            return torch.cat(outs, dim=0)
        except RuntimeError as exc:
            if not (str(device).startswith("cuda") and _is_cuda_oom(exc)):
                raise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if cur_chunk > 1:
                nxt = max(1, cur_chunk // 2)
                if nxt == cur_chunk:
                    nxt = cur_chunk - 1
                print(f"  WARNING: CUDA OOM in {tag}, reduce LPIPS chunk {cur_chunk} -> {nxt}")
                cur_chunk = nxt
                continue
            if not cpu_fallback:
                raise

            print(f"  WARNING: CUDA OOM in {tag} at chunk=1, fallback to CPU LPIPS")
            prev_dev = torch.device(device)
            try:
                loss_fn = loss_fn.to("cpu")
                x_cpu = x.detach().cpu()
                y_cpu = y.detach().cpu()
                outs = []
                with torch.no_grad():
                    for s in range(0, n, cur_chunk):
                        e = min(s + cur_chunk, n)
                        d = loss_fn(to_lpips_input(x_cpu[s:e]), to_lpips_input(y_cpu[s:e]))
                        outs.append(d.detach().cpu().view(-1))
                return torch.cat(outs, dim=0)
            finally:
                try:
                    loss_fn.to(prev_dev)
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

def save_image_task(tensor_cpu, path):
    """Async save task to avoid blocking GPU loop"""
    try:
        save_image(tensor_cpu, path)
    except Exception as e:
        print(f"Error saving {path}: {e}")

def _extract_clip_embeddings(output):
    """
    Robust extraction logic for CLIP. 
    Handles Tensor, Tuple, ModelOutput, and Dict objects.
    """
    # Case 1: Direct Tensor
    if isinstance(output, torch.Tensor):
        return output
    
    # Case 2: HuggingFace ModelOutput object (dot access)
    if hasattr(output, 'image_embeds') and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, 'text_embeds') and output.text_embeds is not None:
        return output.text_embeds
    # 妫ｅ啯鏆?Fix: Support pooler_output (BaseModelOutputWithPooling)
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output
        
    # Case 3: Dict-like
    if isinstance(output, dict):
        if 'image_embeds' in output: return output['image_embeds']
        if 'text_embeds' in output: return output['text_embeds']
        if 'pooler_output' in output: return output['pooler_output']
            
    # Case 4: Tuple/List (Fallback)
    if isinstance(output, (tuple, list)):
        if len(output) > 0 and isinstance(output[0], torch.Tensor):
            return output[0]

    # Debug info if all fails
    type_str = str(type(output))
    msg = f"Could not find embeddings in CLIP output. Output Type: {type_str}"
    if isinstance(output, dict) or hasattr(output, 'keys'):
        msg += f", Keys: {list(output.keys())}"
    raise RuntimeError(msg)


def _try_load_clip_backbone(model_name: str, device: str, allow_network: bool):
    model_name = str(model_name or "").strip()
    if not model_name:
        return None, None, "disabled"
    try:
        from transformers import AutoProcessor, CLIPModel

        kwargs = {"local_files_only": (not bool(allow_network))}
        processor = AutoProcessor.from_pretrained(model_name, **kwargs)
        model = CLIPModel.from_pretrained(model_name, **kwargs).to(device).eval()
        if not hasattr(model, "get_image_features"):
            raise RuntimeError(f"Model '{model_name}' does not support get_image_features for image-only eval.")
        for p in model.parameters():
            p.requires_grad_(False)
        return model, processor, model_name
    except Exception as e:
        print(f"  ERROR: failed to load CLIP model '{model_name}': {e}")
        return None, None, "disabled"


def _encode_clip_images(clip_model, clip_processor, batch_01: torch.Tensor, device: str) -> torch.Tensor:
    to_pil = T.ToPILImage()
    pil_images = [to_pil(img.detach().cpu()) for img in batch_01]
    inputs = clip_processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        raise RuntimeError("CLIP processor output missing pixel_values.")
    with torch.no_grad():
        emb = clip_model.get_image_features(pixel_values=pixel_values)
        if not isinstance(emb, torch.Tensor):
            emb = _extract_clip_embeddings(emb)
        if not isinstance(emb, torch.Tensor):
            raise RuntimeError(f"Unexpected CLIP embedding output type: {type(emb)}")
    return F.normalize(emb.float(), dim=1).detach().cpu()


def _load_eval_image_tensor(path: Path, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size))
    return T.ToTensor()(img)


def _parse_generated_name(filename: str, style_names: list[str]) -> tuple[str, str, str] | None:
    """
    Parse generated image name:
    {src_style}_{src_stem}_to_{tgt_style}.jpg
    """
    stem = Path(filename).stem
    if "_to_" not in stem:
        return None
    left, tgt_style = stem.rsplit("_to_", 1)
    # prefer longest style name first to avoid prefix ambiguity
    for src_style in sorted(style_names, key=lambda x: len(x), reverse=True):
        prefix = f"{src_style}_"
        if left.startswith(prefix):
            src_stem = left[len(prefix):]
            if src_stem:
                return src_style, src_stem, tgt_style
    return None


def _is_ref_cache_valid(ref_features: dict, need_clip: bool) -> bool:
    if not isinstance(ref_features, dict) or not ref_features:
        return False

    valid_style_count = 0
    for feats in ref_features.values():
        if not isinstance(feats, list) or not feats:
            continue

        has_vgg = False
        has_clip = False
        for sample in feats:
            if not isinstance(sample, dict):
                continue
            vgg = sample.get("vgg")
            clip = sample.get("clip")
            if isinstance(vgg, list) and len(vgg) > 0 and all(isinstance(t, torch.Tensor) for t in vgg):
                has_vgg = True
            if isinstance(clip, torch.Tensor):
                has_clip = True
            if has_vgg and (has_clip or not need_clip):
                break

        if not has_vgg:
            continue
        if need_clip and not has_clip:
            continue
        valid_style_count += 1

    return valid_style_count > 0


def _build_style_ref_prototypes(
    test_images: dict,
    vae,
    device: str,
    ref_count: int = 8,
) -> dict:
    """
    Build deterministic style reference latents per style by averaging
    the first `ref_count` images (sorted order) in each target style.
    """
    style_ref_prototypes = {}
    for style_id, (_, img_list) in test_images.items():
        if not img_list:
            continue
        count = max(1, min(len(img_list), int(ref_count)))
        selected = img_list[:count]
        batch = torch.stack([_load_eval_image_tensor(p) for p in selected], dim=0).to(device)
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
        with torch.no_grad(), amp_ctx:
            latents = encode_image(vae, batch, device)
        style_ref_prototypes[style_id] = latents.mean(dim=0, keepdim=True).detach()
    return style_ref_prototypes

# ==========================================
# Main Logic
# ==========================================

def _parse_epoch_from_ckpt_name(path: Path):
    stem = path.stem
    if not stem.startswith("epoch_"):
        return None
    try:
        return int(stem.split("_", 1)[1])
    except Exception:
        return None


def _resolve_existing_path(raw_path: str | None, base_dirs: list[Path]) -> Path | None:
    candidates = _build_path_candidates(raw_path, base_dirs)
    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand
    return None


def _build_path_candidates(raw_path: str | None, base_dirs: list[Path]) -> list[Path]:
    if raw_path is None:
        return []
    text = str(raw_path).strip()
    if not text:
        return []

    p = Path(text).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        for base in base_dirs:
            candidates.append((base / p).resolve())
        candidates.append(p.resolve())
    return candidates


def _resolve_dir_path(raw_path: str | None, base_dirs: list[Path]) -> Path:
    """
    Resolve directory path predictably across different launch cwd.
    Preference:
    1) absolute path
    2) first existing candidate from base_dirs + raw_path
    3) first base_dir + raw_path
    """
    text = str(raw_path or "").strip()
    if not text:
        raise ValueError("Directory path is empty.")

    p = Path(text).expanduser()
    if p.is_absolute():
        return p.resolve()

    for base in base_dirs:
        cand = (base / p).resolve()
        if cand.exists():
            return cand
    return (base_dirs[0] / p).resolve()


def _load_json_config(raw_path: str | None, base_dirs: list[Path]) -> dict:
    resolved = _resolve_existing_path(raw_path, base_dirs)
    if resolved is None:
        return {}
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            print(f"Loaded fallback config: {resolved}")
            return data
    except Exception as exc:
        print(f"WARNING: failed to read fallback config {resolved}: {exc}")
    return {}


def _find_latest_ckpt_under_dir(scan_dir: Path) -> Path | None:
    candidates: list[Path] = []
    # Common layouts: run/epoch_*.pt or run/checkpoints/epoch_*.pt
    candidates.extend(sorted(scan_dir.glob("epoch_*.pt")))
    candidates.extend(sorted((scan_dir / "checkpoints").glob("epoch_*.pt")))
    # Fallback: recursive search for unusual layouts.
    if not candidates:
        for p in scan_dir.rglob("epoch_*.pt"):
            parts_lower = {x.lower() for x in p.parts}
            if "full_eval" in parts_lower:
                continue
            candidates.append(p)
    if not candidates:
        return None

    def _score(path: Path):
        ep = _parse_epoch_from_ckpt_name(path)
        if ep is None:
            ep = -1
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        return (ep, mtime, str(path))

    return max(candidates, key=_score)


def _infer_full_eval_out_dir_for_ckpt(ckpt_path: Path) -> Path:
    ep = _parse_epoch_from_ckpt_name(ckpt_path)
    if ckpt_path.parent.name.lower() == "checkpoints":
        run_dir = ckpt_path.parent.parent
    else:
        run_dir = ckpt_path.parent
    if ep is None:
        return run_dir / "full_eval" / ckpt_path.stem
    return run_dir / "full_eval" / f"epoch_{ep:04d}"


def _default_image_classifier_path() -> Path:
    # Keep legacy save/load location stable across runs.
    return Path(__file__).resolve().parents[2] / "artifacts" / "eval_classifier" / "eval_style_image_classifier.pt"


def _auto_run_missing_full_eval(args) -> None:
    src_dir = Path(__file__).resolve().parents[1]
    scan_root = src_dir.parent
    sibling_dirs = sorted([d for d in scan_root.iterdir() if d.is_dir()], key=lambda x: x.name.lower())

    print(f"Auto full-eval | scan root: {scan_root}")
    to_run = []
    skipped = []
    for d in sibling_dirs:
        ckpt_path = _find_latest_ckpt_under_dir(d)
        if ckpt_path is None:
            continue
        out_dir = _infer_full_eval_out_dir_for_ckpt(ckpt_path)
        summary_path = out_dir / "summary.json"
        if summary_path.exists() and not args.force_regen:
            skipped.append((d.name, ckpt_path))
            continue
        to_run.append((d.name, ckpt_path, out_dir))

    if skipped:
        print("Auto full-eval | already done:")
        for name, ckpt in skipped:
            print(f"  - {name}: {ckpt}")
    if not to_run:
        print("Auto full-eval | nothing to run.")
        return

    print("Auto full-eval | pending:")
    for name, ckpt, out_dir in to_run:
        print(f"  - {name}: {ckpt} -> {out_dir}")
    this_file = Path(__file__).resolve()
    for _, ckpt_path, out_dir in to_run:
        cmd = [
            sys.executable,
            str(this_file),
            "--checkpoint", str(ckpt_path),
            "--output", str(out_dir),
            "--mode", str(args.mode),
        ]
        if args.cache_dir:
            cmd += ["--cache_dir", str(args.cache_dir)]
        if args.clip_allow_network:
            cmd += ["--clip_allow_network"]
        if args.test_dir:
            cmd += ["--test_dir", str(args.test_dir)]
        if args.style_strength is not None:
            cmd += ["--style_strength", str(args.style_strength)]
        if args.force_regen:
            cmd += ["--force_regen"]
        if args.eval_classifier_only:
            cmd += ["--eval_classifier_only"]
        if args.eval_disable_lpips:
            cmd += ["--eval_disable_lpips"]
        if args.reuse_generated:
            cmd += ["--reuse_generated"]
        if args.generation_only:
            cmd += ["--generation_only"]

        print(f"\n[Auto] Running: {ckpt_path}")
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help="Single-checkpoint mode: path to checkpoint")
    parser.add_argument('--output', '--out', dest='output', type=str, default=None, help="Single-checkpoint mode: output directory")
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'gen', 'ana'], help="Run mode: full/gen/ana")
    parser.add_argument('--config', type=str, default="../config.json", help="Auto mode config path")
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None, help="Directory to store shared feature caches")
    parser.add_argument('--num_steps', type=int, default=None)
    parser.add_argument('--step_size', type=float, default=None)
    parser.add_argument('--style_strength', type=float, default=None, help="Global style strength in [0,1]; default uses checkpoint config")
    parser.add_argument('--max_src_samples', type=int, default=None, help="Max source images per style; <=0 means all")
    parser.add_argument('--max_ref_compare', type=int, default=None, help="Max refs for SWD style compare; <=0 means all cached refs")
    parser.add_argument('--max_ref_cache', type=int, default=None, help="Max reference images per style used for cache/features; <=0 means all")
    parser.add_argument('--ref_feature_batch_size', type=int, default=None, help="Batch size for reference feature extraction")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size increased due to offloading")
    parser.add_argument('--force_regen', action='store_true', help="Force regenerate evaluation outputs/metrics (does not rebuild global ref cache)")
    parser.add_argument('--force_regen_ref_cache', action='store_true', help="Force rebuild global reference-feature cache only")
    parser.add_argument('--image_classifier_path', type=str, default="", help="Path to robust image classifier checkpoint for evaluation (optional)")
    parser.add_argument('--clip_model_name', type=str, default="", help="HF/local CLIP model name or local directory")
    parser.add_argument('--clip_modelscope_id', type=str, default="", help="Optional ModelScope model id for CLIP fallback")
    parser.add_argument('--clip_modelscope_cache_dir', type=str, default="", help="Optional ModelScope cache directory")
    parser.add_argument('--clip_allow_network', action='store_true', help="Allow online model fetch if local cache is missing (default off)")
    parser.add_argument('--vae_model_id', type=str, default="", help="VAE model id/preset (default: sdxl official)")
    parser.add_argument('--vae_dtype', type=str, default="", help="VAE dtype: fp32/fp16/bf16 (default: fp32)")
    parser.add_argument('--eval_classifier_only', action='store_true', help="Run only classifier evaluation (skip LF-SSIM/SWD)")
    parser.add_argument('--eval_disable_lpips', action='store_true', help="Deprecated. Kept for compatibility; ignored.")
    parser.add_argument('--eval_swd_projections', type=int, default=128, help="Number of random projections used by style SWD")
    parser.add_argument('--eval_lf_blur_kernel', type=int, default=None, help="LF-SSIM Gaussian blur kernel size (odd)")
    parser.add_argument('--eval_lf_sigma', type=float, default=None, help="LF-SSIM Gaussian sigma")
    parser.add_argument('--eval_ssim_window_size', type=int, default=None, help="Deprecated alias of --eval_lf_blur_kernel")
    parser.add_argument('--eval_ssim_sigma', type=float, default=None, help="Deprecated alias of --eval_lf_sigma")
    parser.add_argument('--reuse_generated', action='store_true', help="Reuse existing generated *_to_*.jpg in output dir and skip generation")
    parser.add_argument('--generation_only', action='store_true', help="Only generate translated images, skip all evaluation metrics")
    parser.add_argument(
        '--style_ref_mode',
        type=str,
        default='none',
        choices=['none', 'prototype', 'random', 'self'],
        help="Reference strategy. Deployment path uses style_id only; non-none modes are ignored."
    )
    parser.add_argument('--style_ref_count', type=int, default=8, help="Number of images to build per-style prototype")
    parser.add_argument('--style_ref_seed', type=int, default=2026, help="Random seed when style_ref_mode=random")
    args = parser.parse_args()

    # Mode dispatch (explicit and backward-compatible with old flags).
    if args.mode == "gen":
        args.generation_only = True
        args.reuse_generated = False
    elif args.mode == "ana":
        args.generation_only = False
        args.reuse_generated = True
        args.force_regen = False

    if (args.checkpoint is None) ^ (args.output is None):
        raise ValueError("Both --checkpoint and --output must be provided together.")
    if args.checkpoint is None and args.output is None:
        _auto_run_missing_full_eval(args)
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup Paths & Config
    path_bases = [
        Path.cwd(),
        Path(__file__).resolve().parent,      # src/utils
        Path(__file__).resolve().parents[1],  # src
        Path(__file__).resolve().parents[2],  # Cycle-NCE
    ]

    out_dir = _resolve_dir_path(args.output, path_bases)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Thread Pool for Async I/O
    io_pool = ThreadPoolExecutor(max_workers=4)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists(): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg_ckpt = ckpt.get('config', {})
    cfg_file = _load_json_config(
        args.config,
        [
            Path.cwd(),
            Path(__file__).resolve().parent,      # src/utils
            Path(__file__).resolve().parents[1],  # src
            Path(__file__).resolve().parents[2],  # Cycle-NCE
        ],
    )

    # Merge with checkpoint config taking precedence.
    cfg = dict(cfg_file)
    cfg.update(cfg_ckpt if isinstance(cfg_ckpt, dict) else {})
    cfg_train = dict(cfg_file.get('training', {}))
    cfg_train.update(cfg_ckpt.get('training', {}) if isinstance(cfg_ckpt, dict) else {})
    cfg_infer = dict(cfg_file.get('inference', {}))
    cfg_infer.update(cfg_ckpt.get('inference', {}) if isinstance(cfg_ckpt, dict) else {})
    cfg_data = dict(cfg_file.get('data', {}))
    cfg_data.update(cfg_ckpt.get('data', {}) if isinstance(cfg_ckpt, dict) else {})

    # Fill missing CLI args from checkpoint config for simpler trainer-side invocation.
    if args.cache_dir is None:
        args.cache_dir = cfg_train.get("full_eval_cache_dir", "../eval_cache")
    if args.num_steps is None:
        args.num_steps = int(cfg_train.get("full_eval_num_steps", cfg_infer.get("num_steps", 1)))
    if args.step_size is None:
        args.step_size = float(cfg_train.get("full_eval_step_size", cfg_infer.get("step_size", 1.0)))
    if args.style_strength is None:
        cfg_style = cfg_train.get("full_eval_style_strength", cfg_infer.get("style_strength", None))
        args.style_strength = None if cfg_style is None else float(cfg_style)
    if args.max_src_samples is None:
        args.max_src_samples = int(cfg_train.get("full_eval_max_src_samples", 30))
    if args.max_ref_compare is None:
        args.max_ref_compare = int(cfg_train.get("full_eval_max_ref_compare", 50))
    if args.max_ref_cache is None:
        args.max_ref_cache = int(cfg_train.get("full_eval_max_ref_cache", 256))
    if args.ref_feature_batch_size is None:
        args.ref_feature_batch_size = int(cfg_train.get("full_eval_ref_feature_batch_size", 64))
    if args.batch_size is None:
        args.batch_size = int(cfg_train.get("full_eval_batch_size", 20))
    if not args.clip_model_name:
        args.clip_model_name = str(cfg_train.get("full_eval_clip_model_name", "openai/clip-vit-base-patch32"))
    if not args.clip_modelscope_id:
        args.clip_modelscope_id = str(cfg_train.get("full_eval_clip_modelscope_id", ""))
    if not args.clip_modelscope_cache_dir:
        args.clip_modelscope_cache_dir = str(cfg_train.get("full_eval_clip_modelscope_cache_dir", ""))
    if not args.vae_model_id:
        args.vae_model_id = str(cfg_train.get("full_eval_vae_model_id", cfg_infer.get("vae_model_id", "sdxl")))
    if not args.vae_dtype:
        args.vae_dtype = str(cfg_train.get("full_eval_vae_dtype", cfg_infer.get("vae_dtype", "fp32")))
    if (not args.image_classifier_path) and str(cfg_train.get("full_eval_image_classifier_path", "")).strip():
        args.image_classifier_path = str(cfg_train.get("full_eval_image_classifier_path"))
    if not args.image_classifier_path:
        args.image_classifier_path = str(_default_image_classifier_path())
    if (not args.eval_classifier_only) and bool(cfg_train.get("full_eval_classifier_only", False)):
        args.eval_classifier_only = True
    # Deprecated flag kept for backward compatibility.
    if (not args.eval_disable_lpips) and bool(cfg_train.get("full_eval_disable_lpips", False)):
        args.eval_disable_lpips = True
    if args.eval_swd_projections is None:
        args.eval_swd_projections = int(cfg_train.get("full_eval_swd_projections", 128))
    if args.eval_lf_blur_kernel is None:
        if args.eval_ssim_window_size is not None:
            args.eval_lf_blur_kernel = int(args.eval_ssim_window_size)
        else:
            args.eval_lf_blur_kernel = int(
                cfg_train.get("full_eval_lf_blur_kernel", cfg_train.get("full_eval_ssim_window_size", 21))
            )
    if args.eval_lf_sigma is None:
        if args.eval_ssim_sigma is not None:
            args.eval_lf_sigma = float(args.eval_ssim_sigma)
        else:
            args.eval_lf_sigma = float(cfg_train.get("full_eval_lf_sigma", cfg_train.get("full_eval_ssim_sigma", 5.0)))
    if (not args.reuse_generated) and bool(cfg_train.get("full_eval_reuse_generated", False)):
        args.reuse_generated = True
    if (not args.generation_only) and bool(cfg_train.get("full_eval_generation_only", False)):
        args.generation_only = True
    cache_dir = _resolve_dir_path(args.cache_dir, path_bases)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve Test Data Path
    test_dir_raw = args.test_dir if args.test_dir else cfg_train.get('test_image_dir', '')
    test_dir_bases = [
        Path.cwd(),
        checkpoint_path.parent.resolve(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[2],
    ]
    print(f"Test dir raw input: {repr(test_dir_raw)}")
    print("Test dir candidate paths:")
    for cand in _build_path_candidates(test_dir_raw, test_dir_bases):
        print(f"  - {cand}")
    resolved_test_dir = _resolve_existing_path(
        test_dir_raw,
        test_dir_bases,
    )
    if resolved_test_dir is None:
        raise ValueError(f"Test directory not found: {test_dir_raw}")
    test_dir = resolved_test_dir
    print(f"Resolved test dir: {test_dir}")

    style_subdirs = cfg_data.get('style_subdirs', [])
    if not style_subdirs:
        # Fallback: auto-detect subdirs
        style_subdirs = [d.name for d in test_dir.iterdir() if d.is_dir()]

    style_ref_mode = str(cfg_train.get('full_eval_style_ref_mode', args.style_ref_mode)).lower()
    style_ref_count = int(cfg_train.get('full_eval_style_ref_count', args.style_ref_count))
    style_ref_seed = int(cfg_train.get('full_eval_style_ref_seed', args.style_ref_seed))
    print(f"Style reference mode: {style_ref_mode} (count={style_ref_count}, seed={style_ref_seed})")
    if style_ref_mode != "none":
        print("WARNING: inference is style_id-only; style_ref_mode is ignored.")
    
    test_images = {}
    for style_id, style_name in enumerate(style_subdirs):
        s_dir = test_dir / style_name
        if not s_dir.exists(): continue
        # Only take valid images
        images = sorted([p for p in s_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.webp']])
        test_images[style_id] = (style_name, images)

    # Prepare Source List
    all_src_info = []
    max_src_samples = int(args.max_src_samples)
    for s_id, (s_name, s_list) in test_images.items():
        rng = random.Random(42)
        sampled = s_list[:]
        rng.shuffle(sampled)
        if max_src_samples > 0:
            sampled = sampled[:max_src_samples]
        for p in sampled:
            all_src_info.append({'path': p, 'style_id': s_id, 'style_name': s_name})

    # Buffer to pass data from Phase 1 to Phase 2
    generated_buffer = []
    style_name_to_id = {name: idx for idx, name in enumerate(style_subdirs)}
    src_lookup = {(x["style_name"], x["path"].stem): x["path"] for x in all_src_info}
    num_src_total = len(all_src_info)
    num_styles = len(style_subdirs)

    if args.reuse_generated:
        print(f"\nPhase 1: Reuse generated images from {out_dir}")
        reuse_files = sorted(out_dir.glob("*_to_*.jpg"))
        for p in reuse_files:
            parsed = _parse_generated_name(p.name, style_subdirs)
            if parsed is None:
                continue
            src_style, src_stem, tgt_style = parsed
            src_path = src_lookup.get((src_style, src_stem))
            tgt_id = style_name_to_id.get(tgt_style)
            if src_path is None or tgt_id is None:
                continue
            try:
                gen_img = _load_eval_image_tensor(p)
            except Exception as e:
                print(f"  WARNING: failed loading generated image {p}: {e}")
                continue
            generated_buffer.append(
                {
                    "src_path": src_path,
                    "src_style": src_style,
                    "tgt_style_name": tgt_style,
                    "tgt_style_id": int(tgt_id),
                    "gen_latent": None,
                    "gen_img": gen_img,
                    "gen_name": p.name,
                }
            )
        print(f"  Reused {len(generated_buffer)} generated images")

    if not generated_buffer and args.mode == "ana":
        raise RuntimeError(
            f"Mode 'ana' requires existing generated images in {out_dir}. "
            f"Run '--mode gen' first, then '--mode ana'."
        )

    if not generated_buffer:
        print(f"\nPhase 1: Generation (Batch Size {args.batch_size})")

        lgt = LGTInference(
            str(checkpoint_path),
            device=device,
            num_steps=args.num_steps,
            step_size=args.step_size,
            style_strength=args.style_strength,
        )
        vae = load_vae(device=device, model_id=args.vae_model_id, torch_dtype=args.vae_dtype)
        model_scale = float(getattr(lgt.model, "latent_scale_factor", 0.18215))
        vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
        scale_in = model_scale / max(vae_scale, 1e-8)
        scale_out = vae_scale / max(model_scale, 1e-8)
        if abs(scale_in - 1.0) > 1e-4:
            print(f"WARNING: latent scale mismatch (model={model_scale:.6f}, vae={vae_scale:.6f}). Applying rescale.")

        # Process in batches
        for b_start in range(0, num_src_total, args.batch_size):
            b_end = min(b_start + args.batch_size, num_src_total)
            batch_info = all_src_info[b_start:b_end]
            print(f"  Generating Batch {b_start//args.batch_size + 1}/{(num_src_total-1)//args.batch_size + 1}")

            # Load Source Images
            src_tensors = []
            for item in batch_info:
                src_tensors.append(_load_eval_image_tensor(item['path']))

            src_batch = torch.stack(src_tensors).to(device)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    # Inversion
                    latents_src = encode_image(vae, src_batch, device)
                    if abs(scale_in - 1.0) > 1e-4:
                        latents_src = latents_src * scale_in
                    latents_x0 = lgt.inversion(latents_src)

                    # Generation for each target style
                    for tgt_id in range(num_styles):
                        tgt_name = style_subdirs[tgt_id]
                        tgt_ids = torch.full((len(batch_info),), tgt_id, device=device, dtype=torch.long)
                        latents_gen = lgt.generation(latents_x0, tgt_ids)
                        if abs(scale_out - 1.0) > 1e-4:
                            latents_gen = latents_gen * scale_out
                        imgs_gen = decode_latent(vae, latents_gen, device) # [B, 3, H, W]

                        # Offload to CPU & Save Async
                        latents_gen_cpu = latents_gen.detach().float().cpu()
                        imgs_gen_cpu = imgs_gen.cpu()

                        for i in range(len(batch_info)):
                            src_item = batch_info[i]
                            out_name = f"{src_item['style_name']}_{src_item['path'].stem}_to_{tgt_name}.jpg"
                            out_path = out_dir / out_name

                            # Async Save
                            io_pool.submit(save_image_task, imgs_gen_cpu[i], out_path)

                            # Store for Phase 2
                            generated_buffer.append({
                                'src_path': src_item['path'],
                                'src_style': src_item['style_name'],
                                'tgt_style_name': tgt_name,
                                'tgt_style_id': tgt_id,
                                'gen_latent': latents_gen_cpu[i], # Keep latent for classifier evaluation
                                'gen_img': imgs_gen_cpu[i], # Keep in RAM
                                'gen_name': out_name
                            })

        # Unload Generation Models
        del lgt, vae
        torch.cuda.empty_cache()
        gc.collect()
        print("  Generation models unloaded")
    if not generated_buffer:
        raise RuntimeError(f"No generated samples to evaluate in {out_dir}")

    if args.generation_only:
        print("\nGeneration-only mode enabled: skip Phase 2 metrics/classifier/SSIM/SWD.")
        io_pool.shutdown(wait=True)
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        summary = {
            "checkpoint": str(checkpoint_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "gen",
            "generated_count": int(len(generated_buffer)),
            "output_dir": str(out_dir),
            "note": "Metrics are intentionally skipped. Run evaluation later with --mode ana.",
        }
        sum_path = out_dir / "summary.json"
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved: {sum_path}")
        return

    # ==========================================
    # PHASE 2: EVALUATION (VGG + SWD)
    # ==========================================
    print(f"\n妫ｅ啯鐣?Phase 2: Evaluation")
    
    # Load Classifier if requested
    image_classifier = None
    classifier_label_names = list(style_subdirs)

    if args.image_classifier_path:
        try:
            from utils.image_classify import load_eval_image_classifier
            image_ckpt = Path(args.image_classifier_path)
            if not image_ckpt.is_absolute():
                image_ckpt = (Path(__file__).resolve().parent / image_ckpt).resolve()
            if not image_ckpt.exists():
                print(f"  WARNING: image style classifier checkpoint not found: {image_ckpt}")
            else:
                image_classifier = load_eval_image_classifier(image_ckpt, device=device)
                classifier_label_names = list(image_classifier.classes)
                print(f"  Loaded image style classifier: {image_ckpt} (classes={len(classifier_label_names)})")
        except Exception as e:
            print(f"  WARNING: failed to load image style classifier: {e}")
            image_classifier = None

    # Skip other metrics if classifier-only mode
    run_full_metrics = not args.eval_classifier_only

    # Load Evaluators
    vgg_extractor = None
    clip_model = None
    clip_processor = None
    has_clip = False
    clip_model_tag = "disabled"
    if run_full_metrics:
        vgg_extractor = VGGFeatureExtractor(device=device)
        clip_model, clip_processor, clip_model_tag = _try_load_clip_backbone(
            args.clip_model_name,
            device=device,
            allow_network=bool(args.clip_allow_network),
        )
        has_clip = clip_model is not None and clip_processor is not None
        if not has_clip:
            raise RuntimeError(
                f"CLIP model unavailable for full_eval (model={args.clip_model_name}). "
                "Please ensure local weights exist or run with --clip_allow_network."
            )
        print(f"  CLIP enabled: model={clip_model_tag}")

    # Prepare Reference Features (Cache)
    style_sig = ",".join(style_subdirs)
    dataset_sig = f"{str(test_dir.resolve())}|{style_sig}|{clip_model_tag}|v2"
    dataset_hash = hashlib.md5(dataset_sig.encode()).hexdigest()[:10]
    max_ref_cache = int(args.max_ref_cache)
    max_ref_cache_tag = "all" if max_ref_cache <= 0 else str(max_ref_cache)
    cache_file = cache_dir / f"ref_feats_{dataset_hash}_m{max_ref_cache_tag}.pt"
    ref_features = {}
    # Keep reference cache independent from output regeneration.
    must_rebuild_ref_cache = bool(args.force_regen_ref_cache)

    if run_full_metrics and cache_file.exists() and not must_rebuild_ref_cache:
        print(f"Found global reference cache: {cache_file}")
        try:
            ref_features = torch.load(cache_file, map_location='cpu')
            if _is_ref_cache_valid(ref_features, need_clip=has_clip):
                print("  Reference cache loaded successfully")
            else:
                print("  Reference cache invalid/incomplete for current metrics, rebuilding...")
                ref_features = {}
        except Exception as e:
            print(f"  Reference cache load failed ({e}), rebuilding...")
            ref_features = {}

    if run_full_metrics and not ref_features:
        print(f"\nComputing Reference Features (global cache miss): {cache_file}")
        for style_id, (style_name, img_list) in test_images.items():
            ref_features[style_id] = []

            sampled_refs = img_list[:]
            if max_ref_cache > 0:
                sampled_refs = sampled_refs[:min(len(sampled_refs), max_ref_cache)]
            ref_bs = max(1, int(args.ref_feature_batch_size))

            pbar = tqdm(range(0, len(sampled_refs), ref_bs), desc=f"Featurizing {style_name}")
            for b_start in pbar:
                batch_paths = sampled_refs[b_start:b_start + ref_bs]
                try:
                    batch_t = torch.stack(
                        [T.ToTensor()(Image.open(img_path).convert('RGB').resize((256, 256))) for img_path in batch_paths],
                        dim=0,
                    ).to(device)

                    amp_ctx = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
                    with torch.no_grad(), amp_ctx:
                        v_feats = vgg_extractor.get_features(batch_t)
                    clip_feats = None
                    if has_clip:
                        clip_feats = _encode_clip_images(clip_model, clip_processor, batch_t, device)

                    for i, img_path in enumerate(batch_paths):
                        row = {
                            'path': str(img_path),
                            'vgg': [vf[i:i+1] for vf in v_feats],
                        }
                        if clip_feats is not None:
                            row['clip'] = clip_feats[i:i+1]
                        ref_features[style_id].append(row)
                except Exception as e:
                    print(f"Skipping batch {b_start}-{b_start + len(batch_paths)} in {style_name}: {e}")

        # No lock by design: best-effort write only.
        try:
            tmp_cache = cache_file.with_suffix(cache_file.suffix + f".tmp.{os.getpid()}")
            torch.save(ref_features, tmp_cache)
            os.replace(tmp_cache, cache_file)
            print(f"Global reference cache saved: {cache_file}")
        except Exception as save_exc:
            print(f"  WARNING: failed to write shared reference cache (ignored): {save_exc}")

    # Prepare reference VGG tensors for faster SWD computation.
    ref_vgg_matrices = {}  # style_id -> list[layer][N_ref, C, H, W] on GPU
    ref_clip_matrices = {}  # style_id -> [N_ref, D] on GPU
    if run_full_metrics and vgg_extractor is not None:
        for sid, feats in ref_features.items():
            vgg_lists = [f.get('vgg') for f in feats if isinstance(f, dict) and isinstance(f.get('vgg'), list)]
            if vgg_lists:
                try:
                    num_layers = len(vgg_lists[0])
                    packed = []
                    for li in range(num_layers):
                        layer_feats = [v[li] for v in vgg_lists if li < len(v) and isinstance(v[li], torch.Tensor)]
                        if not layer_feats:
                            continue
                        packed.append(torch.cat(layer_feats, dim=0).to(device, dtype=torch.float32))
                    if packed:
                        ref_vgg_matrices[sid] = packed
                except Exception as e:
                    print(f"  WARNING: failed to prepare VGG tensors for style {sid}: {e}")
            if has_clip:
                try:
                    clip_rows = [f.get('clip') for f in feats if isinstance(f, dict) and isinstance(f.get('clip'), torch.Tensor)]
                    if clip_rows:
                        ref_clip_matrices[sid] = torch.cat(clip_rows, dim=0).to(device, dtype=torch.float32)
                except Exception as e:
                    print(f"  WARNING: failed to prepare CLIP tensors for style {sid}: {e}")
    if run_full_metrics:
        if len(ref_vgg_matrices) == 0:
            raise RuntimeError("Reference feature cache has no valid VGG features; cannot compute style metrics.")
        if has_clip and len(ref_clip_matrices) == 0:
            raise RuntimeError("Reference feature cache has no valid CLIP features; cannot compute clip_style_sim.")

    csv_path = out_dir / 'metrics.csv'
    # Re-evaluation on reused images should overwrite metrics to avoid mixing old/new classifier outputs.
    csv_mode = 'w' if args.force_regen or args.reuse_generated or not csv_path.exists() else 'a'
    csv_file = open(csv_path, csv_mode, newline='')
    columns = [
        'src_style',
        'tgt_style',
        'src_image',
        'gen_image',
        'content_lf_ssim',
        'style',
        'style_swd',
        'clip_style_sim',
        'pred_style',
        'class_correct',
    ]
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    if csv_mode == 'w': writer.writeheader()

    # Process Generated Buffer
    total_gen = len(generated_buffer)
    print(f"  Processing {total_gen} generated images...")
    
    for b_start in range(0, total_gen, args.batch_size):
        b_end = min(b_start + args.batch_size, total_gen)
        batch_items = generated_buffer[b_start:b_end]
        
        gen_imgs_cpu = torch.stack([item['gen_img'] for item in batch_items])
        gen_imgs = gen_imgs_cpu.to(device)
        
        src_tensors = []
        for item in batch_items:
            img = Image.open(item['src_path']).convert('RGB').resize((256, 256))
            src_tensors.append(T.ToTensor()(img))
        src_imgs = torch.stack(src_tensors).to(device)
        
        with torch.no_grad():
            # 1. Low-frequency content structure similarity (higher is better)
            c_lf_vals = [0.0] * len(batch_items)
            if run_full_metrics:
                lf_vals = compute_lf_structure_similarity_batch(
                    gen_imgs.float(),
                    src_imgs.float(),
                    blur_kernel=int(args.eval_lf_blur_kernel),
                    sigma=float(args.eval_lf_sigma),
                )
                c_lf_vals = lf_vals.cpu().numpy()

            # 2. VGG feature extraction for style SWD
            gen_vgg_feats = None
            gen_clip_feats = None
            if run_full_metrics and vgg_extractor is not None:
                amp_ctx = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
                with torch.no_grad(), amp_ctx:
                    gen_vgg_feats = [vf.to(device, dtype=torch.float32) for vf in vgg_extractor.get_features(gen_imgs)]
                if has_clip:
                    gen_clip_feats = _encode_clip_images(clip_model, clip_processor, gen_imgs, device).to(device, dtype=torch.float32)

            # 3. Classifier Predictions
            pred_indices = [-1] * len(batch_items)
            if image_classifier is not None:
                preds = image_classifier.predict_indices(gen_imgs).cpu().numpy().tolist()
                pred_indices = preds

            # 4. Style Metrics & Row Writing
            for i, item in enumerate(batch_items):
                tgt_id = item['tgt_style_id']
                tgt_name = item['tgt_style_name']
                
                # --- Classifier Logic ---
                pred_idx = pred_indices[i]
                
                pred_style_name = "N/A"
                class_correct = "N/A"
                
                if image_classifier is not None and pred_idx != -1:
                    if pred_idx < len(classifier_label_names):
                        pred_style_name = classifier_label_names[pred_idx]
                        is_correct = (str(pred_style_name).lower() == str(tgt_name).lower())
                        class_correct = 1 if is_correct else 0
                    else:
                        pred_style_name = f"Unknown({pred_idx})"
                        class_correct = 0

                # --- Style SWD ---
                s_swd_score = 0.0
                if gen_vgg_feats is not None and tgt_id in ref_vgg_matrices:
                    ref_layers = ref_vgg_matrices[tgt_id]
                    layer_scores = []
                    max_ref_compare = int(args.max_ref_compare)
                    for li, gen_layer in enumerate(gen_vgg_feats):
                        if li >= len(ref_layers):
                            continue
                        refs_layer = ref_layers[li]
                        if refs_layer.ndim != 4 or refs_layer.shape[0] <= 0:
                            continue
                        use_n = refs_layer.shape[0] if max_ref_compare <= 0 else min(refs_layer.shape[0], max_ref_compare)
                        refs = refs_layer[:use_n]
                        gen_rep = gen_layer[i:i+1].expand(use_n, -1, -1, -1)
                        swd_vals = compute_swd(gen_rep, refs, num_projections=int(args.eval_swd_projections))
                        layer_scores.append(float(swd_vals.mean().item()))
                    if layer_scores:
                        s_swd_score = float(np.mean(layer_scores))
                clip_style_sim = 0.0
                if gen_clip_feats is not None and tgt_id in ref_clip_matrices:
                    refs = ref_clip_matrices[tgt_id]
                    if refs.ndim == 2 and refs.shape[0] > 0:
                        max_ref_compare = int(args.max_ref_compare)
                        use_n = refs.shape[0] if max_ref_compare <= 0 else min(refs.shape[0], max_ref_compare)
                        ref_clip = refs[:use_n]
                        gen_clip = gen_clip_feats[i:i+1].expand(use_n, -1)
                        clip_style_sim = float(F.cosine_similarity(gen_clip, ref_clip, dim=1).mean().item())
                style_score = clip_style_sim if has_clip else s_swd_score
                
                writer.writerow({
                    'src_style': item['src_style'],
                    'tgt_style': item['tgt_style_name'],
                    'src_image': item['src_path'].name,
                    'gen_image': item['gen_name'],
                    'content_lf_ssim': c_lf_vals[i],
                    'style': style_score,
                    'style_swd': s_swd_score,
                    'clip_style_sim': clip_style_sim,
                    'pred_style': pred_style_name,
                    'class_correct': class_correct
                })
            
            csv_file.flush()

    csv_file.close()
    io_pool.shutdown(wait=True)
    
    generate_summary_json(csv_path, out_dir, checkpoint_path)

def generate_summary_json(csv_path, out_dir, ckpt_path):
    print("\n妫ｅ啯鎯?Generating Summary...")
    rows = []
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader: rows.append(r)
            
    if not rows: return

    def to_f(x): return float(x) if x else 0.0

    matrix = defaultdict(lambda: defaultdict(list))
    for r in rows:
        matrix[r['src_style']][r['tgt_style']].append(r)

    matrix_json = {}
    transfer_pool = []
    identity_pool = []
    photo_transfer_pool = []
    
    # Classification Stats
    y_true = []
    y_pred = []

    for src, targets in matrix.items():
        matrix_json[src] = {}
        for tgt, items in targets.items():
            stats = {
                'count': len(items),
                'style': np.mean([to_f(x.get('style', x.get('clip_style_sim', x.get('style_swd', 0.0)))) for x in items]),
                'style_swd': np.mean([to_f(x.get('style_swd', 0.0)) for x in items]),
                'clip_style_sim': np.mean([to_f(x.get('clip_style_sim', 0.0)) for x in items]),
                'content_lf_ssim': np.mean([to_f(x.get('content_lf_ssim', x.get('content_ssim', 0.0))) for x in items]),
            }
            stats['content_ssim'] = stats['content_lf_ssim']  # backward-compatible alias
            
            # Classification Accuracy for this pair
            cls_results = [x['class_correct'] for x in items if x['class_correct'] != 'N/A']
            if cls_results:
                acc = np.mean([int(c) for c in cls_results])
                stats['classifier_acc'] = acc
                
                # Collect for global report
                for x in items:
                    if x['class_correct'] != 'N/A':
                        y_true.append(tgt)
                        y_pred.append(x['pred_style'])
            else:
                stats['classifier_acc'] = None

            matrix_json[src][tgt] = stats
            
            if src == tgt:
                identity_pool.append(stats)
            else:
                transfer_pool.append(stats)
                if src == 'photo':
                    photo_transfer_pool.append(stats)

    def pool_avg(pool, key):
        if not pool: return 0.0
        return float(np.mean([x[key] for x in pool]))

    # Generate Classification Report
    cls_report = None
    detailed_metrics = {}
    if SKLEARN_AVAILABLE and y_true:
        try:
            cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # 闁圭粯鍔曡ぐ鍥即鐎靛憡绾悷娆忓€诲▓鎴犳嫚閿斿墽鐭庡ǎ鍥ｅ墲娴?
            precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(cls_report.keys())[:-3], zero_division=0)
            unique_labels = list(cls_report.keys())[:-3]
            for i, label in enumerate(unique_labels):
                detailed_metrics[label] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1_score": float(f1[i]),
                    "support": int(support[i])
                }
            print("\n闁?Classification Report:")
            print(classification_report(y_true, y_pred, zero_division=0))
        except Exception as e:
            print(f"闁宠法濯寸粭?Failed to generate classification report: {e}")
    elif y_true:
        # Manual simple accuracy if sklearn missing
        correct = sum(1 for t, p in zip(y_true, y_pred) if t.lower() == p.lower())
        cls_report = {"accuracy": correct / len(y_true)}

    summary = {
        'checkpoint': str(ckpt_path),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'matrix_breakdown': matrix_json,
        'analysis': {
            'style_transfer_ability': {
                'style': pool_avg(transfer_pool, 'style'),
                'style_swd': pool_avg(transfer_pool, 'style_swd'),
                'clip_style_sim': pool_avg(transfer_pool, 'clip_style_sim'),
                'content_lf_ssim': pool_avg(transfer_pool, 'content_lf_ssim'),
                'content_ssim': pool_avg(transfer_pool, 'content_lf_ssim'),  # backward-compatible alias
                'classifier_acc': pool_avg([t for t in transfer_pool if t['classifier_acc'] is not None], 'classifier_acc')
            },
            'photo_to_art_performance': {
                'style': pool_avg(photo_transfer_pool, 'style'),
                'style_swd': pool_avg(photo_transfer_pool, 'style_swd'),
                'clip_style_sim': pool_avg(photo_transfer_pool, 'clip_style_sim'),
                'valid': len(photo_transfer_pool) > 0,
                'classifier_acc': pool_avg([t for t in photo_transfer_pool if t['classifier_acc'] is not None], 'classifier_acc')
            }
        },
        'classification_report': cls_report,
        'detailed_style_metrics': detailed_metrics
    }
    
    sum_path = out_dir / 'summary.json'
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"闁?Summary saved: {sum_path}")

if __name__ == '__main__':
    main() 
   
