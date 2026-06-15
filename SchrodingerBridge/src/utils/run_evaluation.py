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
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor

# Metrics
try:
    import lpips
except ImportError:
    lpips = None

import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
from torchvision.utils import save_image
from scipy import linalg

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.inference import LGTInference, load_vae, encode_image, decode_latent
from utils.artfid_metric import (
    compute_artfid_content_distance_from_paths,
    compute_artfid_fid_from_paths,
    load_artfid_feature_extractor,
    load_artfid_lpips,
)
from utils.targetwise_artfid_summary import write_targetwise_artfid_summary
from utils.introstyle_eval import (
    IntroStyleFeatureExtractor,
    introstyle_style_vector,
    mean_pool_scores,
    resolve_introstyle_model_path,
    style_bank_paths,
)
from config_schema import load_config, load_inference_defaults, merge_config_dicts, resolve_full_eval_section

# KID (official implementation via torchmetrics)
try:
    from torchmetrics.image.kid import KernelInceptionDistance
except Exception:
    KernelInceptionDistance = None

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _PROJECT_ROOT
_DEFAULT_HF_CLIP_REPO = "openai/clip-vit-base-patch32"
_DEFAULT_CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
_DEFAULT_CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def _resolve_default_local_clip_dir() -> Path:
    """
    Prefer the workspace's shared CLIP snapshot, but keep compatibility with
    older layouts that stored it under Cycle-NCE/eval_cache.
    """
    candidates = [
        _WORKSPACE_ROOT / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32",
        _WORKSPACE_ROOT / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32",
        _PROJECT_ROOT / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


_DEFAULT_LOCAL_CLIP_DIR = _resolve_default_local_clip_dir()


def _manual_clip_candidates(cache_dir: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if cache_dir is not None:
        candidates.append(cache_dir / "manual_clip" / "openai-clip-vit-base-patch32")
    candidates.extend([
        _DEFAULT_LOCAL_CLIP_DIR,
        _WORKSPACE_ROOT / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32",
        _WORKSPACE_ROOT / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32",
        _PROJECT_ROOT / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32",
    ])
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique

def _safe_to_eval_device(batch, device: str):
    """
    Move processor outputs to device when possible (BatchEncoding supports .to()).
    """
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    return batch


def _clip_image_size_from_runtime(clip_model, clip_processor) -> int:
    image_size = None
    image_processor = getattr(clip_processor, "image_processor", None)
    if image_processor is not None:
        size_cfg = getattr(image_processor, "size", None)
        if isinstance(size_cfg, dict):
            image_size = size_cfg.get("shortest_edge") or size_cfg.get("height") or size_cfg.get("width")
        elif isinstance(size_cfg, int):
            image_size = size_cfg
    if image_size is None:
        vision_cfg = getattr(getattr(clip_model, "config", None), "vision_config", None)
        image_size = getattr(vision_cfg, "image_size", None)
    if image_size is None:
        visual = getattr(clip_model, "visual", None)
        image_size = getattr(visual, "input_resolution", None)
    return int(image_size or 224)


def _clip_mean_std_from_runtime(clip_processor) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    image_processor = getattr(clip_processor, "image_processor", None)
    mean = getattr(image_processor, "image_mean", None)
    std = getattr(image_processor, "image_std", None)
    if mean is None:
        mean = _DEFAULT_CLIP_IMAGE_MEAN
    if std is None:
        std = _DEFAULT_CLIP_IMAGE_STD
    return tuple(float(x) for x in mean), tuple(float(x) for x in std)


def _prepare_clip_pixels(images_01: torch.Tensor, *, image_size: int, mean, std) -> torch.Tensor:
    if images_01.ndim == 3:
        images_01 = images_01.unsqueeze(0)
    imgs = images_01.to(dtype=torch.float32)
    h, w = imgs.shape[-2:]
    crop = min(h, w)
    if h != crop or w != crop:
        top = max(0, (h - crop) // 2)
        left = max(0, (w - crop) // 2)
        imgs = imgs[:, :, top:top + crop, left:left + crop]
    if imgs.shape[-2:] != (image_size, image_size):
        imgs = F.interpolate(
            imgs,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    mean_t = torch.tensor(mean, device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
    imgs = (imgs - mean_t) / std_t
    return imgs.contiguous(memory_format=torch.channels_last)


def _load_clip_from_source(CLIPModel, CLIPProcessor, src: str, device: str, *, local_only: bool, cache_dir: str):
    model_kwargs = {}
    proc_kwargs = {}
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
        proc_kwargs["cache_dir"] = cache_dir
    if local_only:
        model_kwargs["local_files_only"] = True
        proc_kwargs["local_files_only"] = True
    try:
        model = CLIPModel.from_pretrained(src, **model_kwargs).to(device)
        processor = CLIPProcessor.from_pretrained(src, **proc_kwargs)
        return model, processor
    except TypeError:
        # Compatibility with older transformers signatures.
        # In strict local mode, old transformers may ignore local-only semantics.
        # Guard against accidental online fetch by allowing fallback only for real local paths.
        if local_only:
            src_path = Path(str(src)).expanduser()
            if not src_path.exists():
                raise RuntimeError(
                    "Current transformers build does not support local_files_only in from_pretrained, "
                    f"and source is not a local path: {src}. Please provide a local snapshot directory."
                )
        model_kwargs.pop("local_files_only", None)
        proc_kwargs.pop("local_files_only", None)
        model = CLIPModel.from_pretrained(src, **model_kwargs).to(device)
        processor = CLIPProcessor.from_pretrained(src, **proc_kwargs)
        return model, processor


def _find_local_hf_snapshot(cache_root: Path, repo_id: str) -> str | None:
    """
    Resolve a local HF snapshot path for offline loading.
    Supports both cache layouts:
    - <cache>/models--org--repo/snapshots/<rev>
    - <cache>/hub/models--org--repo/snapshots/<rev>
    """
    repo_key = str(repo_id).strip().replace("/", "--")
    if not repo_key:
        return None
    model_dir_name = f"models--{repo_key}"
    roots = [cache_root, cache_root / "hub"]
    snapshots: list[Path] = []
    for root in roots:
        model_dir = root / model_dir_name
        snap_dir = model_dir / "snapshots"
        if snap_dir.exists():
            snapshots.extend([p for p in snap_dir.iterdir() if p.is_dir()])
    if not snapshots:
        return None
    snapshots = sorted(snapshots, key=lambda p: p.name, reverse=True)

    def _path_usable(p: Path) -> bool:
        if not p.exists():
            return False
        if p.is_symlink():
            try:
                return p.resolve(strict=False).exists()
            except Exception:
                return False
        return True

    def _clip_snapshot_missing_files(p: Path) -> list[str]:
        missing = []
        if not _path_usable(p / "config.json"):
            missing.append("config.json")
        if not (_path_usable(p / "pytorch_model.bin") or _path_usable(p / "model.safetensors")):
            missing.append("pytorch_model.bin|model.safetensors")
        if not (_path_usable(p / "preprocessor_config.json") or _path_usable(p / "processor_config.json")):
            missing.append("preprocessor_config.json|processor_config.json")
        if not (_path_usable(p / "tokenizer.json") or (_path_usable(p / "vocab.json") and _path_usable(p / "merges.txt"))):
            missing.append("tokenizer.json|(vocab.json+merges.txt)")
        if not _path_usable(p / "tokenizer_config.json"):
            missing.append("tokenizer_config.json")
        return missing

    def _is_complete_clip_snapshot(p: Path) -> bool:
        return len(_clip_snapshot_missing_files(p)) == 0

    for s in snapshots:
        if _is_complete_clip_snapshot(s):
            return str(s.resolve())
    # Fallback to the latest snapshot if none is complete.
    return str(snapshots[0].resolve())


def _debug_clip_cache_state(cache_root: Path, repo_id: str) -> str:
    repo_key = str(repo_id).strip().replace("/", "--")
    if not repo_key:
        return "empty clip_model_name"
    model_dir_name = f"models--{repo_key}"
    roots = [cache_root, cache_root / "hub"]
    snapshots: list[Path] = []
    for root in roots:
        snap_dir = root / model_dir_name / "snapshots"
        if snap_dir.exists():
            snapshots.extend([p for p in snap_dir.iterdir() if p.is_dir()])
    if not snapshots:
        return f"no snapshots under {cache_root}"

    snapshots = sorted(snapshots, key=lambda p: p.name, reverse=True)

    def _path_usable(p: Path) -> bool:
        if not p.exists():
            return False
        if p.is_symlink():
            try:
                return p.resolve(strict=False).exists()
            except Exception:
                return False
        return True

    def _missing_list(p: Path) -> list[str]:
        missing = []
        if not _path_usable(p / "config.json"):
            missing.append("config.json")
        if not (_path_usable(p / "pytorch_model.bin") or _path_usable(p / "model.safetensors")):
            missing.append("pytorch_model.bin|model.safetensors")
        if not (_path_usable(p / "preprocessor_config.json") or _path_usable(p / "processor_config.json")):
            missing.append("preprocessor_config.json|processor_config.json")
        if not (_path_usable(p / "tokenizer.json") or (_path_usable(p / "vocab.json") and _path_usable(p / "merges.txt"))):
            missing.append("tokenizer.json|(vocab.json+merges.txt)")
        if not _path_usable(p / "tokenizer_config.json"):
            missing.append("tokenizer_config.json")
        return missing

    lines = []
    for s in snapshots[:3]:
        miss = _missing_list(s)
        if miss:
            lines.append(f"{s.name}: missing/broken -> {', '.join(miss)}")
        else:
            lines.append(f"{s.name}: OK")
    return "; ".join(lines)

def to_lpips_input(img_tensor):
    return img_tensor * 2.0 - 1.0


def _runtime_debug_scalars(raw: object) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                out[str(key)] = float(torch.nan_to_num(value.detach().float()).item())
        elif isinstance(value, (int, float, bool)):
            out[str(key)] = float(value)
    return out


def _attention_entropy_scalar(attn: torch.Tensor | None) -> float:
    if attn is None:
        return 0.0
    probs = attn.detach().float().clamp_min(1e-8)
    return float((-(probs * probs.log()).sum(dim=-1).mean()).item())


def _runtime_observability_from_model(model: torch.nn.Module | None) -> dict[str, float]:
    if model is None:
        return {}
    stats: dict[str, float] = {}
    structured = getattr(model, "structured_style_tokenizer", None)
    for key, value in _runtime_debug_scalars(getattr(structured, "last_debug", {})).items():
        stats[f"structured_style_tokenizer_{key}"] = value
    for key, value in _runtime_debug_scalars(getattr(model, "last_output_appearance_debug", {})).items():
        stats[str(key)] = value
    for key, value in _runtime_debug_scalars(getattr(model, "last_i2sb_transport_debug", {})).items():
        stats[f"i2sb_{key}"] = value
    for key, value in _runtime_debug_scalars(getattr(model, "last_solver_noise_debug", {})).items():
        stats[f"solver_{key}"] = value
    for key, value in _runtime_debug_scalars(getattr(model, "last_style_delta_debug", {})).items():
        stats[str(key)] = value
    for key, value in _runtime_debug_scalars(getattr(model, "last_style_strength_debug", {})).items():
        stats[str(key)] = value
    topology_attn = getattr(model, "last_semantic_topology_attn", None)
    stats["semantic_topology_attn_entropy"] = _attention_entropy_scalar(topology_attn)
    stats["semantic_topology_attn_active"] = 1.0 if topology_attn is not None else 0.0
    return stats


def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda oom" in msg)


def _batched_paths(items: list[Path], n: int) -> list[list[Path]]:
    step = max(1, int(n))
    return [items[i:i + step] for i in range(0, len(items), step)]


def _run_introstyle_sidecar(
    *,
    metrics_csv: Path,
    images_dir: Path,
    test_dir: Path,
    cache_dir: Path,
    device: str,
    model_id: str,
    modelscope_id: str,
    modelscope_cache_dir: str,
    allow_network: bool,
    style_bank_root: str,
    bank_limit_per_style: int,
    batch_size: int,
    topk: int,
    t: int,
    up_ft_index: int,
    ensemble_size: int,
) -> dict[str, object] | None:
    bank_root = Path(str(style_bank_root).strip()) if str(style_bank_root).strip() else test_dir
    if not bank_root.exists():
        print(f"  WARNING: IntroStyle skipped; style bank root missing: {bank_root}")
        return None
    if not metrics_csv.exists():
        print(f"  WARNING: IntroStyle skipped; metrics.csv missing: {metrics_csv}")
        return None

    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        metric_rows = list(csv.DictReader(f))
    if not metric_rows:
        print("  WARNING: IntroStyle skipped; metrics.csv has no rows.")
        return None

    resolved_model_id = resolve_introstyle_model_path(
        model_id=str(model_id),
        modelscope_id=str(modelscope_id),
        modelscope_cache_dir=str(modelscope_cache_dir or (cache_dir / "modelscope")),
        allow_network=bool(allow_network),
    )
    extractor = IntroStyleFeatureExtractor(
        model_id=resolved_model_id,
        device=str(device),
        t=int(t),
        up_ft_index=int(up_ft_index),
        ensemble_size=int(ensemble_size),
    )

    bank_paths = style_bank_paths(bank_root, per_style_limit=max(0, int(bank_limit_per_style)))
    bank_vectors: dict[str, torch.Tensor] = {}
    for style_name, paths in bank_paths.items():
        feats = extractor.encode_paths(paths, batch_size=max(1, int(batch_size)))
        bank_vectors[style_name] = introstyle_style_vector(feats)
    if not bank_vectors:
        print(f"  WARNING: IntroStyle skipped; no bank images found under {bank_root}")
        return None

    all_paths: list[Path] = []
    filtered_rows: list[dict[str, str]] = []
    for row in metric_rows:
        name = str(row.get("gen_image", "")).strip()
        if not name:
            continue
        path = images_dir / Path(name).name
        if not path.exists():
            continue
        all_paths.append(path)
        filtered_rows.append(row)
    if not filtered_rows:
        print("  WARNING: IntroStyle skipped; generated images not found on disk.")
        return None

    style_names = sorted(bank_vectors.keys())
    sidecar_rows: list[dict[str, object]] = []
    for chunk_idx, chunk_paths in enumerate(_batched_paths(all_paths, max(1, int(batch_size)))):
        metas = filtered_rows[chunk_idx * max(1, int(batch_size)):(chunk_idx + 1) * max(1, int(batch_size))]
        feats = extractor.encode_paths(chunk_paths, batch_size=len(chunk_paths))
        vecs = introstyle_style_vector(feats)
        scores = mean_pool_scores(vecs, bank_vectors, topk=max(1, int(topk)))
        for i, row in enumerate(metas):
            target = str(row["tgt_style"])
            source = str(row["src_style"])
            if target not in scores or source not in scores:
                continue
            target_score = float(scores[target][i].item())
            source_score = float(scores[source][i].item())
            non_target_scores = [(name, float(scores[name][i].item())) for name in style_names if name != target]
            best_non_target_style, best_non_target_score = max(non_target_scores, key=lambda x: x[1])
            sidecar_rows.append(
                {
                    "src_style": source,
                    "tgt_style": target,
                    "src_image": str(row.get("src_image", "")),
                    "gen_image": str(row.get("gen_image", "")),
                    "introstyle_target_style_score": target_score,
                    "introstyle_source_style_score": source_score,
                    "introstyle_best_non_target_style": best_non_target_style,
                    "introstyle_best_non_target_score": best_non_target_score,
                    "introstyle_style_margin": target_score - best_non_target_score,
                }
            )

    if not sidecar_rows:
        print("  WARNING: IntroStyle skipped; no scored rows were produced.")
        return None

    intro_csv = metrics_csv.parent / "introstyle_metrics.csv"
    intro_json = metrics_csv.parent / "introstyle_summary.json"
    with intro_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "src_style",
                "tgt_style",
                "src_image",
                "gen_image",
                "introstyle_target_style_score",
                "introstyle_source_style_score",
                "introstyle_best_non_target_style",
                "introstyle_best_non_target_score",
                "introstyle_style_margin",
            ],
        )
        writer.writeheader()
        writer.writerows(sidecar_rows)

    matrix: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    transfer_target_scores: list[float] = []
    transfer_margins: list[float] = []
    identity_target_scores: list[float] = []
    photo_transfer_target_scores: list[float] = []
    photo_transfer_margins: list[float] = []
    for src in sorted({str(r["src_style"]) for r in sidecar_rows}):
        for tgt in sorted({str(r["tgt_style"]) for r in sidecar_rows if str(r["src_style"]) == src}):
            pair = [r for r in sidecar_rows if str(r["src_style"]) == src and str(r["tgt_style"]) == tgt]
            if not pair:
                continue
            pair_stats = {
                "count": float(len(pair)),
                "introstyle_target_style_score": float(np.mean([float(r["introstyle_target_style_score"]) for r in pair])),
                "introstyle_source_style_score": float(np.mean([float(r["introstyle_source_style_score"]) for r in pair])),
                "introstyle_best_non_target_score": float(np.mean([float(r["introstyle_best_non_target_score"]) for r in pair])),
                "introstyle_style_margin": float(np.mean([float(r["introstyle_style_margin"]) for r in pair])),
            }
            matrix[src][tgt] = pair_stats
            if src == tgt:
                identity_target_scores.append(pair_stats["introstyle_target_style_score"])
            else:
                transfer_target_scores.append(pair_stats["introstyle_target_style_score"])
                transfer_margins.append(pair_stats["introstyle_style_margin"])
                if src.lower() == "photo":
                    photo_transfer_target_scores.append(pair_stats["introstyle_target_style_score"])
                    photo_transfer_margins.append(pair_stats["introstyle_style_margin"])

    identity_mean = float(np.mean(identity_target_scores)) if identity_target_scores else None
    transfer_mean = float(np.mean(transfer_target_scores)) if transfer_target_scores else None
    photo_mean = float(np.mean(photo_transfer_target_scores)) if photo_transfer_target_scores else None
    payload: dict[str, object] = {
        "metrics_csv": str(intro_csv),
        "style_bank_root": str(bank_root),
        "model_id": resolved_model_id,
        "matrix_breakdown": matrix,
        "analysis": {
            "style_transfer_ability": {
                "introstyle_target_style_score": transfer_mean,
                "introstyle_style_margin": float(np.mean(transfer_margins)) if transfer_margins else None,
                "introstyle_delta_idt": (transfer_mean - identity_mean) if transfer_mean is not None and identity_mean is not None else None,
            },
            "identity_reconstruction": {
                "introstyle_target_style_score": identity_mean,
            },
            "photo_to_art_performance": {
                "introstyle_target_style_score": photo_mean,
                "introstyle_style_margin": float(np.mean(photo_transfer_margins)) if photo_transfer_margins else None,
                "introstyle_delta_idt": (photo_mean - identity_mean) if photo_mean is not None and identity_mean is not None else None,
            },
        },
    }
    intro_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "metrics_csv": str(intro_csv),
        "summary_json": str(intro_json),
        "payload": payload,
    }


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

def _images_01_to_uint8_hwc_cpu(images: torch.Tensor) -> torch.Tensor:
    """
    Quantize decoded [0,1] images on GPU before the CPU copy.
    Copying uint8 NHWC is much cheaper than copying float CHW and converting
    again inside PIL/torchvision.
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)
    packed = images.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    return packed.permute(0, 2, 3, 1).contiguous().cpu()


def _uint8_hwc_to_float_chw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected uint8 HWC image, got shape={tuple(image.shape)}")
    return image.permute(2, 0, 1).contiguous().to(torch.float32).div_(255.0)


def _compute_style_rgb_stats(
    test_images: dict[int, tuple[str, list[Path]]],
    *,
    image_size: int,
    ref_limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_styles = max(test_images.keys(), default=-1) + 1
    means = torch.full((num_styles, 3, 1, 1), 0.5, dtype=torch.float32)
    stds = torch.full((num_styles, 3, 1, 1), 0.25, dtype=torch.float32)
    limit = int(ref_limit)
    for style_id, (_, paths) in test_images.items():
        selected = list(paths)
        if limit > 0:
            selected = selected[:limit]
        if not selected:
            continue
        sum_rgb = torch.zeros(3, dtype=torch.float64)
        sumsq_rgb = torch.zeros(3, dtype=torch.float64)
        pixel_count = 0
        for path in selected:
            try:
                img = _load_eval_image_tensor(path, size=image_size).to(dtype=torch.float64)
            except Exception:
                continue
            flat = img.view(3, -1)
            sum_rgb += flat.sum(dim=1)
            sumsq_rgb += flat.square().sum(dim=1)
            pixel_count += int(flat.shape[1])
        if pixel_count <= 0:
            continue
        mean = sum_rgb / float(pixel_count)
        var = (sumsq_rgb / float(pixel_count) - mean.square()).clamp_min(1e-6)
        means[int(style_id), :, 0, 0] = mean.to(dtype=torch.float32)
        stds[int(style_id), :, 0, 0] = var.sqrt().to(dtype=torch.float32)
    return means, stds


def _apply_postdecode_style_rgb_affine(
    images: torch.Tensor,
    target_ids: torch.Tensor,
    target_means: torch.Tensor | None,
    target_stds: torch.Tensor | None,
    *,
    strength: float,
    mean_strength: float,
    std_strength: float,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, dict[str, float]]:
    if target_means is None or target_stds is None or images.numel() == 0:
        return images, {"postdecode_rgbcal_active": 0.0}
    strength = max(0.0, min(1.0, float(strength)))
    mean_strength = max(0.0, min(1.0, float(mean_strength)))
    std_strength = max(0.0, min(1.0, float(std_strength)))
    if strength <= 0.0 or (mean_strength <= 0.0 and std_strength <= 0.0):
        return images, {"postdecode_rgbcal_active": 0.0}

    target_ids = target_ids.to(device=images.device, dtype=torch.long).clamp(0, target_means.shape[0] - 1)
    tgt_mean = target_means.to(device=images.device, dtype=images.dtype)[target_ids]
    tgt_std = target_stds.to(device=images.device, dtype=images.dtype)[target_ids].clamp_min(eps)
    img_mean = images.mean(dim=(-2, -1), keepdim=True)
    img_std = images.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(eps)
    out_mean = img_mean.lerp(tgt_mean, mean_strength)
    out_std = img_std.lerp(tgt_std, std_strength).clamp_min(eps)
    affine = (images - img_mean) / img_std * out_std + out_mean
    adjusted = images.lerp(affine, strength).clamp(0.0, 1.0)
    return adjusted, {
        "postdecode_rgbcal_active": 1.0,
        "postdecode_rgbcal_strength": float(strength),
        "postdecode_rgbcal_mean_strength": float(mean_strength),
        "postdecode_rgbcal_std_strength": float(std_strength),
        "postdecode_rgbcal_mean_delta": float((tgt_mean - img_mean).detach().abs().mean().cpu().item()),
        "postdecode_rgbcal_std_delta": float((tgt_std - img_std).detach().abs().mean().cpu().item()),
    }


@torch.no_grad()
def _compute_style_latent_stats(
    test_images: dict[int, tuple[str, list[Path]]],
    *,
    vae,
    device: str,
    ref_limit: int,
    scale_in: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_styles = max(test_images.keys(), default=-1) + 1
    means = torch.zeros((num_styles, 4, 1, 1), dtype=torch.float32)
    stds = torch.ones((num_styles, 4, 1, 1), dtype=torch.float32)
    limit = int(ref_limit)
    for style_id, (_, paths) in test_images.items():
        selected = list(paths)
        if limit > 0:
            selected = selected[:limit]
        if not selected:
            continue
        sum_lat = None
        sumsq_lat = None
        value_count = 0
        for path in selected:
            try:
                img = _load_eval_image_tensor(path).unsqueeze(0).to(device)
                latent = encode_image(vae, img, device)
                if abs(scale_in - 1.0) > 1e-4:
                    latent = latent * scale_in
                latent = latent.detach().float()
            except Exception:
                continue
            flat = latent[0].view(latent.shape[1], -1).to(dtype=torch.float64, device="cpu")
            if sum_lat is None:
                sum_lat = torch.zeros(flat.shape[0], dtype=torch.float64)
                sumsq_lat = torch.zeros(flat.shape[0], dtype=torch.float64)
            sum_lat += flat.sum(dim=1)
            sumsq_lat += flat.square().sum(dim=1)
            value_count += int(flat.shape[1])
        if value_count <= 0 or sum_lat is None or sumsq_lat is None:
            continue
        mean = sum_lat / float(value_count)
        var = (sumsq_lat / float(value_count) - mean.square()).clamp_min(1e-6)
        means[int(style_id), :, 0, 0] = mean.to(dtype=torch.float32)
        stds[int(style_id), :, 0, 0] = var.sqrt().to(dtype=torch.float32)
    return means, stds


def _apply_latent_style_affine(
    latents: torch.Tensor,
    target_ids: torch.Tensor,
    target_means: torch.Tensor | None,
    target_stds: torch.Tensor | None,
    *,
    strength: float,
    mean_strength: float,
    std_strength: float,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, dict[str, float]]:
    if target_means is None or target_stds is None or latents.numel() == 0:
        return latents, {"latent_style_affine_active": 0.0}
    strength = max(0.0, min(1.0, float(strength)))
    mean_strength = max(0.0, min(1.0, float(mean_strength)))
    std_strength = max(0.0, min(1.0, float(std_strength)))
    if strength <= 0.0 or (mean_strength <= 0.0 and std_strength <= 0.0):
        return latents, {"latent_style_affine_active": 0.0}

    target_ids = target_ids.to(device=latents.device, dtype=torch.long).clamp(0, target_means.shape[0] - 1)
    tgt_mean = target_means.to(device=latents.device, dtype=latents.dtype)[target_ids]
    tgt_std = target_stds.to(device=latents.device, dtype=latents.dtype)[target_ids].clamp_min(eps)
    lat_mean = latents.mean(dim=(-2, -1), keepdim=True)
    lat_std = latents.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(eps)
    out_mean = lat_mean.lerp(tgt_mean, mean_strength)
    out_std = lat_std.lerp(tgt_std, std_strength).clamp_min(eps)
    affine = (latents - lat_mean) / lat_std * out_std + out_mean
    adjusted = latents.lerp(affine, strength)
    return adjusted, {
        "latent_style_affine_active": 1.0,
        "latent_style_affine_strength": float(strength),
        "latent_style_affine_mean_strength": float(mean_strength),
        "latent_style_affine_std_strength": float(std_strength),
        "latent_style_affine_mean_delta": float((tgt_mean - lat_mean).detach().abs().mean().cpu().item()),
        "latent_style_affine_std_delta": float((tgt_std - lat_std).detach().abs().mean().cpu().item()),
    }


def save_image_task(image_cpu, path, backend: str = "pil_png"):
    """Async save task to avoid blocking GPU loop."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if torch.is_tensor(image_cpu) and image_cpu.dtype == torch.uint8:
            if backend == "torchvision_png":
                from torchvision.io import write_png

                chw = image_cpu.permute(2, 0, 1).contiguous() if image_cpu.ndim == 3 else image_cpu.contiguous()
                write_png(chw.cpu(), str(path))
                return
            arr = image_cpu.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                image = Image.fromarray(arr.squeeze(-1) if arr.shape[-1] == 1 else arr)
                image.save(path, format="PNG")
                return
        save_image(image_cpu, path)
    except Exception as e:
        print(f"Error saving {path}: {e}")


def _sync_cuda_if(device: str, enabled: bool) -> None:
    if enabled and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _add_timing(timings: dict[str, float], key: str, start: float) -> None:
    timings[key] = float(timings.get(key, 0.0) + (time.perf_counter() - start))


def _list_reuse_generated_files(out_dir: Path) -> list[Path]:
    # Prefer new layout: out_dir/images/*.jpg, keep backward compatibility.
    candidates = []
    candidates.extend(sorted((out_dir / "images").glob("*_to_*.jpg")))
    candidates.extend(sorted(out_dir.glob("*_to_*.jpg")))
    candidates.extend(sorted((out_dir / "images").glob("*_to_*.png")))
    candidates.extend(sorted(out_dir.glob("*_to_*.png")))
    dedup = {}
    for p in candidates:
        dedup[str(p.resolve())] = p
    return sorted(dedup.values(), key=lambda x: str(x))


def _should_auto_reuse_generated(
    *,
    out_dir: Path,
    expected_count: int,
    force_regen: bool,
    reuse_generated: bool,
) -> tuple[bool, int]:
    if force_regen:
        return False, 0
    files = _list_reuse_generated_files(out_dir)
    found = len(files)
    if reuse_generated:
        return True, found
    if found <= 0:
        return False, 0
    if expected_count > 0 and found >= expected_count:
        return True, found
    return False, found


def _resolve_gen_image_path(out_dir: Path, gen_image_value: str) -> Path | None:
    raw = str(gen_image_value or "").strip()
    if not raw:
        return None
    p = Path(raw)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((out_dir / p).resolve())
        candidates.append((out_dir / "images" / p.name).resolve())
        candidates.append((out_dir / p.name).resolve())
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _save_summary_grid_png(rows, out_dir: Path, style_order: list[str] | None = None) -> Path | None:
    if not rows:
        return None
    if not style_order:
        style_order = sorted({str(r.get("src_style", "")) for r in rows if str(r.get("src_style", ""))})
    if not style_order:
        return None

    def _to_f(v, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    # src_style -> src_image -> tgt_style -> {path, clip_style, content_lpips}
    by_src = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        src_style = str(r.get("src_style", ""))
        src_image = str(r.get("src_image", ""))
        tgt_style = str(r.get("tgt_style", ""))
        p = _resolve_gen_image_path(out_dir, str(r.get("gen_image", "")))
        if (not src_style) or (not src_image) or (not tgt_style) or (p is None):
            continue
        by_src[src_style][src_image][tgt_style] = {
            "path": p,
            "clip_style": _to_f(r.get("clip_style", 0.0), 0.0),
            "content_lpips": _to_f(r.get("content_lpips", 0.0), 0.0),
        }

    # Pick one representative source image per row style:
    # maximize mean clip_style across transfers to OTHER styles.
    chosen = {}
    for src_style in style_order:
        candidates = by_src.get(src_style, {})
        if not candidates:
            chosen[src_style] = {}
            continue
        best_key = None
        best_map = None
        best_src_img = None
        for src_img, tgt_map in candidates.items():
            transfer_scores = []
            for tgt_style in style_order:
                if tgt_style == src_style:
                    continue
                item = tgt_map.get(tgt_style)
                if item is None:
                    continue
                transfer_scores.append(float(item.get("clip_style", 0.0)))
            coverage = len(transfer_scores)
            if coverage <= 0:
                continue
            mean_clip = float(np.mean(transfer_scores))
            min_clip = float(np.min(transfer_scores))
            # Higher mean clip first, then min clip, then coverage.
            rank_key = (mean_clip, min_clip, coverage, src_img)
            if best_key is None or rank_key > best_key:
                best_key = rank_key
                best_map = tgt_map
                best_src_img = src_img
        if best_map is None:
            ranked = sorted(candidates.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            best_src_img, best_map = ranked[0]
        chosen[src_style] = {
            "src_image": str(best_src_img),
            "tgt_map": best_map,
        }

    existing_paths = []
    for src_style in style_order:
        tgt_map = chosen.get(src_style, {}).get("tgt_map", {})
        for tgt_style in style_order:
            item = tgt_map.get(tgt_style)
            p = item.get("path") if isinstance(item, dict) else None
            if p is not None and p.exists():
                existing_paths.append(p)
    if not existing_paths:
        return None

    # Keep original resolution; no downscaling.
    sizes = []
    for p in existing_paths:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception:
            pass
    if not sizes:
        return None
    cell_w = max(w for w, _ in sizes)
    cell_h = max(h for _, h in sizes)
    n = len(style_order)

    try:
        font = ImageFont.truetype("arial.ttf", size=28)
        font_small = ImageFont.truetype("arial.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    bg = (0, 0, 0)
    fg = (255, 255, 255)
    pad = 18
    header_h = 56
    metric_h = 24
    left_w = 220
    canvas_w = left_w + n * cell_w + (n + 1) * pad
    canvas_h = header_h + n * (cell_h + metric_h) + (n + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg)
    draw = ImageDraw.Draw(canvas)

    for ci, tgt_style in enumerate(style_order):
        x = left_w + pad + ci * (cell_w + pad)
        y = 8
        draw.text((x, y), tgt_style, fill=fg, font=font)

    for ri, src_style in enumerate(style_order):
        x = 6
        y = header_h + pad + ri * (cell_h + metric_h + pad) + max(0, (cell_h - 28) // 2)
        draw.text((x, y), src_style, fill=fg, font=font)
        src_img = chosen.get(src_style, {}).get("src_image", "")
        if src_img:
            draw.text((x, y + 30), Path(src_img).stem, fill=(200, 200, 200), font=font_small)
        tgt_map = chosen.get(src_style, {}).get("tgt_map", {})
        for ci, tgt_style in enumerate(style_order):
            px = left_w + pad + ci * (cell_w + pad)
            py = header_h + pad + ri * (cell_h + metric_h + pad)
            item = tgt_map.get(tgt_style)
            p = item.get("path") if isinstance(item, dict) else None
            if p is None or not p.exists():
                continue
            try:
                with Image.open(p).convert("RGB") as im:
                    canvas.paste(im, (px, py))
            except Exception:
                continue
            clip_style = float(item.get("clip_style", 0.0))
            c_lpips = float(item.get("content_lpips", 0.0))
            stat_text = f"clip={clip_style:.3f} lpips={c_lpips:.3f}"
            draw.text((px + 4, py + cell_h + 3), stat_text, fill=(230, 230, 230), font=font_small)

    out_path = out_dir / "summary_grid.png"
    canvas.save(out_path, format="PNG")
    print(f"Summary grid saved: {out_path}")
    print("Summary grid source selection (max transfer clip_style mean):")
    for src_style in style_order:
        src_img = chosen.get(src_style, {}).get("src_image", "")
        print(f"  {src_style}: {Path(src_img).stem if src_img else '(none)'}")
    return out_path

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


@torch.no_grad()
def _extract_inception_feats(paths, runner, max_images: int = 200):
    return runner.extract(paths, max_images=max_images)


def _collect_metric_image_paths(paths, max_images: int) -> list[str]:
    out = []
    seen = set()
    for raw in list(paths or []):
        try:
            p = Path(str(raw))
        except Exception:
            continue
        if not p.exists() or not p.is_file():
            continue
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
        if len(out) >= max(1, int(max_images)):
            break
    return out


class _InceptionFeatRunner:
    def __init__(self, device: str, batch_size: int = 16):
        self.device = str(device)
        self.batch_size = max(1, int(batch_size))
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)
        self.tfm = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, paths, max_images: int = 200):
        if not paths:
            return np.empty((0, 2048), dtype=np.float64)
        sel = list(paths)[: max(1, int(max_images))]
        feats = []
        for s in range(0, len(sel), self.batch_size):
            e = min(s + self.batch_size, len(sel))
            imgs = []
            for p in sel[s:e]:
                try:
                    imgs.append(self.tfm(Image.open(p).convert("RGB")))
                except Exception:
                    continue
            if not imgs:
                continue
            x = torch.stack(imgs, dim=0).to(self.device)
            y = self.model(x)
            if y.ndim > 2:
                y = torch.flatten(y, 1)
            feats.append(y.detach().cpu().double().numpy())
        if not feats:
            return np.empty((0, 2048), dtype=np.float64)
        return np.concatenate(feats, axis=0)

    def close(self):
        del self.model
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()


def _frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def _compute_art_fid_for_pair(
    gen_paths,
    ref_paths,
    src_paths,
    *,
    feature_model,
    lpips_loss_fn,
    device: str,
    batch_size: int,
    max_gen: int,
    max_ref: int,
    ref_cache: dict | None = None,
    ref_cache_key: str | None = None,
):
    gen = _collect_metric_image_paths(gen_paths, max_gen)
    ref = _collect_metric_image_paths(ref_paths, max_ref)
    src = _collect_metric_image_paths(src_paths, max_gen)
    if len(gen) < 2 or len(ref) < 2:
        return None, None, None
    if not src:
        return None, None, None
    n = min(len(gen), len(src))
    gen = gen[:n]
    src = src[:n]
    if len(gen) < 1:
        return None, None, None

    artfid_fid = compute_artfid_fid_from_paths(
        gen,
        ref,
        model=feature_model,
        batch_size=max(1, int(batch_size)),
        device=device,
        ref_cache=ref_cache,
        ref_cache_key=ref_cache_key,
    )
    artfid_content = compute_artfid_content_distance_from_paths(
        gen,
        src,
        loss_fn=lpips_loss_fn,
        batch_size=max(1, int(batch_size)),
        device=device,
    )
    if artfid_fid is None or artfid_content is None:
        return artfid_fid, artfid_content, None
    art_fid = (1.0 + float(artfid_fid)) * (1.0 + float(artfid_content))
    return float(artfid_fid), float(artfid_content), float(art_fid)


def _compute_fid_for_pair(
    src_paths,
    ref_paths,
    *,
    runner,
    device: str,
    max_gen: int,
    max_ref: int,
    ref_cache: dict | None = None,
    ref_cache_key: str | None = None,
):
    gen = _collect_metric_image_paths(src_paths, max_gen)
    ref = _collect_metric_image_paths(ref_paths, max_ref)
    if len(gen) < 2 or len(ref) < 2:
        return None
    # Prefer the lightweight runner path when available so `ref_cache`
    # can actually amortize repeated target-style computations.
    # torchmetrics FID is kept as a fallback only when runner is unavailable.
    if runner is None and FrechetInceptionDistance is not None:
        fid = FrechetInceptionDistance(normalize=False).to(device)
        fid.eval()

        def _update(paths: list[str], *, real: bool):
            bs = max(1, int(getattr(runner, "batch_size", 16)))
            for i in range(0, len(paths), bs):
                chunk = paths[i : i + bs]
                imgs = torch.stack([_load_uint8_rgb_tensor_299(p) for p in chunk], dim=0).to(device)
                fid.update(imgs, real=bool(real))

        _update(ref, real=True)
        _update(gen, real=False)
        score = fid.compute()
        return float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)

    s_feats = _extract_inception_feats(gen, runner=runner, max_images=max_gen)
    if ref_cache is not None and ref_cache_key is not None and ref_cache_key in ref_cache:
        r_feats = ref_cache[ref_cache_key]
    else:
        r_feats = _extract_inception_feats(ref, runner=runner, max_images=max_ref)
        if ref_cache is not None and ref_cache_key is not None:
            ref_cache[ref_cache_key] = r_feats
    if s_feats.shape[0] < 2 or r_feats.shape[0] < 2:
        return None
    mu_s, cov_s = s_feats.mean(axis=0), np.cov(s_feats, rowvar=False)
    mu_r, cov_r = r_feats.mean(axis=0), np.cov(r_feats, rowvar=False)
    return float(_frechet_distance(mu_s, cov_s, mu_r, cov_r))


def _load_uint8_rgb_tensor_299(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((299, 299), Image.Resampling.BICUBIC)
    # np.asarray(PIL.Image) can produce a non-writable view, which triggers a PyTorch warning.
    # Copy to ensure a writable, contiguous buffer.
    arr = np.asarray(img, dtype=np.uint8).copy()
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unexpected image shape for KID: {arr.shape}")
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _compute_kid_for_pair(
    gen_paths: list[str],
    ref_paths: list[str],
    *,
    device: str,
    subset_size: int,
    max_gen: int,
    max_ref: int,
    batch_size: int,
) -> tuple[float | None, float | None]:
    if KernelInceptionDistance is None:
        raise RuntimeError("torchmetrics is required for KID (KernelInceptionDistance) but is not available.")
    g = _collect_metric_image_paths(gen_paths, max_gen)
    r = _collect_metric_image_paths(ref_paths, max_ref)
    if not g or not r:
        return None, None

    # torchmetrics enforces subset_size <= number of samples for both sets.
    subset = max(2, int(subset_size))
    subset = min(subset, len(g), len(r))
    if subset < 2:
        return None, None

    kid = KernelInceptionDistance(subset_size=int(subset)).to(device)
    kid.eval()

    def _update(paths: list[str], *, real: bool):
        bs = max(1, int(batch_size))
        for i in range(0, len(paths), bs):
            chunk = paths[i : i + bs]
            imgs = torch.stack([_load_uint8_rgb_tensor_299(p) for p in chunk], dim=0).to(device)
            kid.update(imgs, real=bool(real))

    _update(r, real=True)
    _update(g, real=False)
    mean, std = kid.compute()
    mean_f = float(mean.detach().cpu().item()) if hasattr(mean, "detach") else float(mean)
    std_f = float(std.detach().cpu().item()) if hasattr(std, "detach") else float(std)
    return mean_f, std_f


def _load_eval_image_tensor(path: Path, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size))
    return T.ToTensor()(img)


def _parse_generated_name(filename: str, style_names: list[str]) -> tuple[str, str, str] | None:
    """
    Parse generated image names from both the native evaluator layout and the
    baseline bridge scripts:
    - {src_style}_{src_stem}_to_{tgt_style}.png
    - {src_style}__{src_stem}__to__{tgt_style}.png
    """
    stem = Path(filename).stem
    if "__to__" in stem:
        left, tgt_style = stem.rsplit("__to__", 1)
        if "__" in left:
            src_style, src_stem = left.split("__", 1)
            if src_style in style_names and tgt_style in style_names and src_stem:
                return src_style, src_stem, tgt_style
        return None
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


def _infer_style_names_from_generated_files(files: list[Path]) -> list[str]:
    styles = set()
    for p in files:
        stem = p.stem
        if "__to__" in stem:
            left, tgt = stem.rsplit("__to__", 1)
            if tgt:
                styles.add(str(tgt))
            if "__" in left:
                src_style = left.split("__", 1)[0]
                if src_style:
                    styles.add(str(src_style))
            continue
        if "_to_" in stem:
            left, tgt = stem.rsplit("_to_", 1)
            if tgt:
                styles.add(str(tgt))
            if "_" in left:
                src_style = left.split("_", 1)[0]
                if src_style:
                    styles.add(str(src_style))
    return sorted(styles)


def _is_ref_cache_valid(ref_features: dict, need_clip: bool) -> bool:
    if not isinstance(ref_features, dict) or not ref_features:
        return False
    if not need_clip:
        return True
    for feats in ref_features.values():
        if not isinstance(feats, list):
            return False
        if not feats:
            continue
        sample = feats[0]
        clip = sample.get("clip") if isinstance(sample, dict) else None
        if clip is None or not isinstance(clip, torch.Tensor):
            return False
    return True


def _source_path_key(path: str | Path) -> str:
    return str(Path(path).resolve())


def _is_source_cache_valid(source_cache: dict, *, image_size: int, need_clip: bool) -> bool:
    if not isinstance(source_cache, dict) or not source_cache:
        return False
    meta = source_cache.get("meta")
    items = source_cache.get("items")
    if not isinstance(meta, dict) or not isinstance(items, dict) or not items:
        return False
    if int(meta.get("image_size", -1)) != int(image_size):
        return False
    for payload in items.values():
        if not isinstance(payload, dict):
            return False
        img = payload.get("img")
        if not isinstance(img, torch.Tensor) or img.ndim != 3:
            return False
        if tuple(img.shape[-2:]) != (int(image_size), int(image_size)):
            return False
        clip = payload.get("clip")
        if need_clip and (clip is None or not isinstance(clip, torch.Tensor)):
            return False
    return True


def _acquire_lock(lock_path: Path, timeout_sec: int = 600, poll_sec: float = 1.0) -> bool:
    deadline = time.time() + max(1, int(timeout_sec))
    while time.time() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()}\n")
            return True
        except FileExistsError:
            time.sleep(max(0.1, float(poll_sec)))
    return False


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
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None

    p = Path(text).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        for base in base_dirs:
            candidates.append((base / p).resolve())
        candidates.append(p.resolve())

    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand
    return None


def _candidate_eval_image_roots(root: Path) -> list[Path]:
    root_str = str(root)
    candidates = [root]
    if os.name == "nt":
        preferred: list[Path] = []
        if "_samam_512_classview_real" in root_str:
            preferred.append(Path(root_str.replace("_samam_512_classview_real", "_512_images")))
        elif "_samam_512_classview" in root_str:
            preferred.append(Path(root_str.replace("_samam_512_classview", "_512_images")))
        if "_classview" in root_str and "_classview_real" not in root_str:
            preferred.append(Path(root_str.replace("_classview", "_classview_real")))
        candidates = preferred + candidates

    ordered: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def _prefer_readable_eval_image_root(root: Path) -> Path:
    for candidate in _candidate_eval_image_roots(root):
        if candidate.exists() and candidate != root:
            print(f"[fallback] readable eval image root: {root} -> {candidate}")
            return candidate
    return root


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
            "--cache_dir", str(args.cache_dir),
            "--num_steps", str(args.num_steps),
            "--step_size", str(args.step_size),
            "--vae_model", str(args.vae_model),
            "--max_src_samples", str(args.max_src_samples),
            "--max_ref_compare", str(args.max_ref_compare),
            "--max_ref_cache", str(args.max_ref_cache),
            "--ref_feature_batch_size", str(args.ref_feature_batch_size),
            "--batch_size", str(args.batch_size),
            "--target_chunk_size", str(args.target_chunk_size),
            "--vae_decode_batch_size", str(args.vae_decode_batch_size),
            "--image_save_workers", str(args.image_save_workers),
            "--image_save_backend", str(args.image_save_backend),
            "--postprocess_mode", str(args.postprocess_mode),
            "--postprocess_strength", str(args.postprocess_strength),
            "--postprocess_mean_strength", str(args.postprocess_mean_strength),
            "--postprocess_std_strength", str(args.postprocess_std_strength),
            "--postprocess_ref_limit", str(args.postprocess_ref_limit),
            "--latent_postprocess_mode", str(args.latent_postprocess_mode),
            "--latent_postprocess_strength", str(args.latent_postprocess_strength),
            "--latent_postprocess_mean_strength", str(args.latent_postprocess_mean_strength),
            "--latent_postprocess_std_strength", str(args.latent_postprocess_std_strength),
            "--latent_postprocess_ref_limit", str(args.latent_postprocess_ref_limit),
            "--clip_model_name", str(args.clip_model_name),
            "--clip_modelscope_id", str(args.clip_modelscope_id),
            "--clip_modelscope_cache_dir", str(args.clip_modelscope_cache_dir),
            "--clip_hf_cache_dir", str(args.clip_hf_cache_dir),
            "--introstyle_model_id", str(args.introstyle_model_id),
            "--introstyle_modelscope_id", str(args.introstyle_modelscope_id),
            "--introstyle_modelscope_cache_dir", str(args.introstyle_modelscope_cache_dir),
            "--introstyle_bank_limit_per_style", str(args.introstyle_bank_limit_per_style),
            "--introstyle_batch_size", str(args.introstyle_batch_size),
            "--introstyle_topk", str(args.introstyle_topk),
            "--introstyle_t", str(args.introstyle_t),
            "--introstyle_up_ft_index", str(args.introstyle_up_ft_index),
            "--introstyle_ensemble_size", str(args.introstyle_ensemble_size),
        ]
        if bool(args.allow_metric_postprocess):
            cmd.append("--allow_metric_postprocess")
        if args.clip_allow_network:
            cmd += ["--clip_allow_network"]
        if args.introstyle_allow_network:
            cmd += ["--introstyle_allow_network"]
        if args.test_dir:
            cmd += ["--test_dir", str(args.test_dir)]
        if args.introstyle_style_bank_root:
            cmd += ["--introstyle_style_bank_root", str(args.introstyle_style_bank_root)]
        if args.style_strength is not None:
            cmd += ["--style_strength", str(args.style_strength)]
        if args.force_regen:
            cmd += ["--force_regen"]
        if args.eval_disable_lpips:
            cmd += ["--eval_disable_lpips"]
        if args.eval_enable_art_fid:
            cmd += ["--eval_enable_art_fid"]
            cmd += ["--eval_art_fid_max_gen", str(args.eval_art_fid_max_gen)]
            cmd += ["--eval_art_fid_max_ref", str(args.eval_art_fid_max_ref)]
            cmd += ["--eval_art_fid_batch_size", str(args.eval_art_fid_batch_size)]
            if args.eval_art_fid_photo_only:
                cmd += ["--eval_art_fid_photo_only"]
        else:
            cmd += ["--no-eval_enable_art_fid"]
        if args.eval_enable_kid:
            cmd += ["--eval_enable_kid"]
            cmd += ["--eval_kid_max_gen", str(args.eval_kid_max_gen)]
            cmd += ["--eval_kid_max_ref", str(args.eval_kid_max_ref)]
            cmd += ["--eval_kid_subset_size", str(args.eval_kid_subset_size)]
            cmd += ["--eval_kid_batch_size", str(args.eval_kid_batch_size)]
        else:
            cmd += ["--no-eval_enable_kid"]
        if args.reuse_generated:
            cmd += ["--reuse_generated"]
        if args.generation_only:
            cmd += ["--generation_only"]
        if not bool(args.save_generated_images):
            cmd += ["--no-save_generated_images"]
        if not bool(args.save_summary_grid):
            cmd += ["--no-save_summary_grid"]
        if not bool(args.eval_only_lpips_clip_style):
            cmd += ["--no-eval_only_lpips_clip_style"]
        if bool(getattr(args, "transfer_only", False)):
            cmd += ["--transfer_only"]
        if bool(args.eval_enable_introstyle):
            cmd += ["--eval_enable_introstyle"]
        else:
            cmd += ["--no-eval_enable_introstyle"]

        print(f"\n[Auto] Running: {ckpt_path}")
        subprocess.run(cmd, check=True)


def main():
    runtime_defaults = load_inference_defaults()
    full_eval_defaults = runtime_defaults.get("full_eval", {}) or {}
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_dir', nargs='?', default=None, help="One-shot mode: target full_eval directory (reuse existing images).")
    parser.add_argument('--checkpoint', type=str, default=None, help="Single-checkpoint mode: path to checkpoint")
    parser.add_argument('--output', type=str, default=None, help="Single-checkpoint mode: output directory")
    parser.add_argument('--config_override', type=str, default="", help="Optional config json merged on top of the checkpoint config for eval-only inference overrides.")
    parser.add_argument('--style_subdirs', type=str, default="", help="Optional comma-separated style names for reuse-only eval without checkpoint")
    parser.add_argument('--config', type=str, default="../config.json", help="Auto mode config path")
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default="../eval_cache", help="Directory to store shared feature caches")
    parser.add_argument('--num_steps', type=int, default=int(full_eval_defaults.get("num_steps", 12)))
    parser.add_argument('--step_size', type=float, default=float(full_eval_defaults.get("step_size", 1.0)))
    parser.add_argument('--style_strength', type=float, default=full_eval_defaults.get("style_strength", None), help="Global style strength. Values above 1 require model.style_strength_max > 1.")
    parser.add_argument('--residual_scale', type=float, default=1.0, help="Post-endpoint latent residual scale for inference strengthening. 1.0 keeps default behavior.")
    parser.add_argument(
        '--vae_model',
        type=str,
        default=str(full_eval_defaults.get("vae_model", "ema")),
        help="VAE preset or HF id for encode/decode. Supports ema, mse, sd15, sdxl, sdxl-fp32, sdxl-fp16-fix, or a HF repo id.",
    )
    parser.add_argument('--vae_decode_scale', type=float, default=None, help="Override VAE scaling factor for decode only; encode/model latent scale stay unchanged.")
    parser.add_argument('--vae_compile_decoder', action='store_true', help="Compile the SD VAE decoder wrapper for eval/generation.")
    parser.add_argument('--vae_compile_method', type=str, default="pt2", choices=["pt2", "jit"])
    parser.add_argument('--vae_compile_mode', type=str, default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument('--vae_compile_fullgraph', action='store_true', help="Use fullgraph=True for compiled VAE decoder.")
    parser.add_argument('--vae_compile_cache_dir', type=str, default="", help="Persistent torch.compile cache directory for the VAE decoder.")
    parser.add_argument('--style_adapter', type=str, default="", help="Optional external style adapter (.pt) to override tokenizer state and, on legacy families only, style_spatial_id_16")
    parser.add_argument('--max_src_samples', type=int, default=int(full_eval_defaults.get("max_src_samples", 30)), help="Max source images per style; <=0 means all")
    parser.add_argument('--max_ref_compare', type=int, default=int(full_eval_defaults.get("max_ref_compare", 50)), help="Max refs for LPIPS style compare; <=0 means all cached refs")
    parser.add_argument('--max_ref_cache', type=int, default=int(full_eval_defaults.get("max_ref_cache", 256)), help="Max reference images per style used for cache/features; <=0 means all")
    parser.add_argument('--ref_feature_batch_size', type=int, default=int(full_eval_defaults.get("ref_feature_batch_size", 64)), help="Batch size for reference feature extraction")
    parser.add_argument('--batch_size', type=int, default=int(full_eval_defaults.get("batch_size", 8)), help="Generation batch size for evaluation. Lower this if VRAM is tight.")
    parser.add_argument(
        '--target_chunk_size',
        type=int,
        default=int(full_eval_defaults.get("target_chunk_size", 1)),
        help="Number of target styles to generate per source batch. 1 preserves legacy behavior; >1 improves GPU/VAE utilization.",
    )
    parser.add_argument(
        '--vae_decode_batch_size',
        type=int,
        default=int(full_eval_defaults.get("vae_decode_batch_size", 0)),
        help="Decode generated latents in chunks. <=0 uses batch_size*target_chunk_size.",
    )
    parser.add_argument('--image_save_workers', type=int, default=int(full_eval_defaults.get("image_save_workers", 4)), help="Async image writer worker count.")
    parser.add_argument(
        '--image_save_backend',
        type=str,
        default=str(full_eval_defaults.get("image_save_backend", "pil_png")),
        choices=["pil_png", "torchvision_png"],
        help="Generated image save backend.",
    )
    parser.add_argument(
        '--save_generated_images',
        action=argparse.BooleanOptionalAction,
        default=bool(full_eval_defaults.get("save_generated_images", True)),
        help="Persist generated PNGs under output/images. Disable for fast metric-only sweeps.",
    )
    parser.add_argument('--profile_timing', action='store_true', help="Synchronize CUDA around generation stages for accurate timing breakdown.")
    parser.add_argument('--save_summary_grid', action=argparse.BooleanOptionalAction, default=bool(full_eval_defaults.get("save_summary_grid", True)), help="Save visual summary_grid.png. Disable for pure throughput timing.")
    parser.add_argument(
        '--keep_generated_on_device',
        action=argparse.BooleanOptionalAction,
        default=bool(full_eval_defaults.get("keep_generated_on_device", True)),
        help=(
            "Keep decoded images on GPU for metric-only eval to avoid GPU->CPU->GPU copies. "
            "Automatically disabled when generated images or sidecar metrics require host/image files."
        ),
    )
    parser.add_argument('--force_regen', action='store_true', help="Force regenerate evaluation outputs/metrics (does not rebuild global ref cache)")
    parser.add_argument('--force_regen_ref_cache', action='store_true', help="Force rebuild global reference-feature cache only")
    parser.add_argument('--ref_cache_lock_timeout', type=int, default=900, help="Seconds to wait for another process building reference cache")
    parser.add_argument('--clip_model_name', type=str, default=_DEFAULT_HF_CLIP_REPO, help="HF repo id or local CLIP directory")
    parser.add_argument('--clip_modelscope_id', type=str, default="", help="Optional ModelScope model id for CLIP fallback")
    parser.add_argument('--clip_modelscope_cache_dir', type=str, default="", help="Optional ModelScope cache directory")
    parser.add_argument('--clip_hf_cache_dir', type=str, default="", help="HuggingFace cache dir for CLIP; default uses <cache_dir>/hf")
    parser.add_argument('--clip_allow_network', action='store_true', help="Allow online model fetch if local cache is missing (default off)")
    parser.add_argument(
        '--clip_backend',
        type=str,
        default="hf",
        choices=["openai", "hf", "none"],
        help="CLIP backend for clip_* metrics: openai (official), hf (transformers), none (disable).",
    )
    parser.add_argument(
        '--clip_openai_model',
        type=str,
        default="ViT-B/32",
        help="OpenAI CLIP model name for --clip_backend openai (e.g. ViT-B/32).",
    )
    parser.add_argument(
        '--clip_optional',
        action='store_true',
        help="If CLIP cannot be loaded, continue with clip_* = 0 (default: fail to avoid silent zeros).",
    )
    parser.add_argument('--eval_disable_lpips', action='store_true', help="Skip LPIPS metrics (keep CLIP)")
    parser.add_argument(
        '--eval_only_lpips_clip_style',
        action=argparse.BooleanOptionalAction,
        default=bool(full_eval_defaults.get("only_lpips_clip_style", True)),
        help=(
            "Compute only content LPIPS and CLIP style similarity by default. "
            "Use --no-eval_only_lpips_clip_style to also compute clip_dir and clip_content."
        ),
    )
    parser.add_argument(
        '--transfer_only',
        action=argparse.BooleanOptionalAction,
        default=bool(full_eval_defaults.get("transfer_only", False)),
        help="Skip identity src_style==tgt_style pairs for fast convergence eval. Default off for full board comparability.",
    )
    parser.add_argument(
        '--eval_enable_introstyle',
        action=argparse.BooleanOptionalAction,
        default=bool(full_eval_defaults.get("enable_introstyle", False)),
        help="Enable IntroStyle sidecar evaluation and write introstyle_metrics.csv / introstyle_summary.json.",
    )
    parser.add_argument(
        '--introstyle_style_bank_root',
        type=str,
        default=str(full_eval_defaults.get("introstyle_style_bank_root", "")),
        help="Held-out style-bank root for IntroStyle. Defaults to --test_dir when empty.",
    )
    parser.add_argument(
        '--introstyle_model_id',
        type=str,
        default=str(full_eval_defaults.get("introstyle_model_id", "")),
        help="Local path or repo id for IntroStyle backbone. Prefer a local ModelScope snapshot on remote.",
    )
    parser.add_argument(
        '--introstyle_modelscope_id',
        type=str,
        default=str(full_eval_defaults.get("introstyle_modelscope_id", "stabilityai/stable-diffusion-2-1-base")),
        help="ModelScope repo id used when --introstyle_model_id is empty.",
    )
    parser.add_argument(
        '--introstyle_modelscope_cache_dir',
        type=str,
        default=str(full_eval_defaults.get("introstyle_modelscope_cache_dir", "")),
        help="ModelScope cache dir for IntroStyle backbone.",
    )
    parser.add_argument(
        '--introstyle_allow_network',
        action='store_true',
        default=bool(full_eval_defaults.get("introstyle_allow_network", False)),
        help="Allow IntroStyle backbone download if local ModelScope cache is missing.",
    )
    parser.add_argument('--introstyle_bank_limit_per_style', type=int, default=int(full_eval_defaults.get("introstyle_bank_limit_per_style", 64)))
    parser.add_argument('--introstyle_batch_size', type=int, default=int(full_eval_defaults.get("introstyle_batch_size", 4)))
    parser.add_argument('--introstyle_topk', type=int, default=int(full_eval_defaults.get("introstyle_topk", 8)))
    parser.add_argument('--introstyle_t', type=int, default=int(full_eval_defaults.get("introstyle_t", 25)))
    parser.add_argument('--introstyle_up_ft_index', type=int, default=int(full_eval_defaults.get("introstyle_up_ft_index", 1)))
    parser.add_argument('--introstyle_ensemble_size', type=int, default=int(full_eval_defaults.get("introstyle_ensemble_size", 1)))
    parser.add_argument(
        '--eval_enable_art_fid',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable ArtFID/FID metric (default: disabled). Use --eval_enable_art_fid to enable.",
    )
    parser.add_argument('--eval_art_fid_max_gen', type=int, default=200, help="Max generated images per pair for FID_style")
    parser.add_argument('--eval_art_fid_max_ref', type=int, default=200, help="Max target-style reference images per pair for FID_style")
    parser.add_argument('--eval_art_fid_batch_size', type=int, default=16, help="Batch size for inception feature extraction in ArtFID")
    parser.add_argument('--eval_art_fid_photo_only', action='store_true', help="Compute ArtFID/FID only for photo->art directions")
    parser.add_argument(
        '--eval_enable_kid',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable KID metric (default: disabled). Use --eval_enable_kid to enable.",
    )
    parser.add_argument('--eval_kid_max_gen', type=int, default=200, help="Max generated images per pair for KID")
    parser.add_argument('--eval_kid_max_ref', type=int, default=200, help="Max target-style reference images per pair for KID")
    parser.add_argument('--eval_kid_subset_size', type=int, default=50, help="Subset size for KID (torchmetrics)")
    parser.add_argument('--eval_kid_batch_size', type=int, default=8, help="Batch size for KID image loading/inception")
    parser.add_argument(
        '--eval_lpips_chunk_size',
        type=int,
        default=int(full_eval_defaults.get("lpips_chunk_size", 4)),
        help="LPIPS chunk size. CUDA OOM automatically retries with smaller chunks unless CPU fallback is disabled.",
    )
    parser.add_argument('--eval_lpips_no_cpu_fallback', action='store_true', help="Disable CPU fallback when LPIPS CUDA OOM occurs")
    parser.add_argument(
        '--postprocess_mode',
        type=str,
        default=str(full_eval_defaults.get("postprocess_mode", "none")),
        choices=["none", "style_rgb_affine"],
        help="Optional decoded-RGB postprocess before image save and metrics.",
    )
    parser.add_argument('--postprocess_strength', type=float, default=float(full_eval_defaults.get("postprocess_strength", 0.0)))
    parser.add_argument('--postprocess_mean_strength', type=float, default=float(full_eval_defaults.get("postprocess_mean_strength", 1.0)))
    parser.add_argument('--postprocess_std_strength', type=float, default=float(full_eval_defaults.get("postprocess_std_strength", 1.0)))
    parser.add_argument('--postprocess_ref_limit', type=int, default=int(full_eval_defaults.get("postprocess_ref_limit", 64)))
    parser.add_argument(
        '--latent_postprocess_mode',
        type=str,
        default=str(full_eval_defaults.get("latent_postprocess_mode", "none")),
        choices=["none", "style_latent_affine"],
        help="Optional latent-space postprocess before VAE decode and metrics.",
    )
    parser.add_argument('--latent_postprocess_strength', type=float, default=float(full_eval_defaults.get("latent_postprocess_strength", 0.0)))
    parser.add_argument('--latent_postprocess_mean_strength', type=float, default=float(full_eval_defaults.get("latent_postprocess_mean_strength", 1.0)))
    parser.add_argument('--latent_postprocess_std_strength', type=float, default=float(full_eval_defaults.get("latent_postprocess_std_strength", 1.0)))
    parser.add_argument('--latent_postprocess_ref_limit', type=int, default=int(full_eval_defaults.get("latent_postprocess_ref_limit", 64)))
    parser.add_argument(
        '--allow_metric_postprocess',
        action='store_true',
        default=bool(full_eval_defaults.get("allow_metric_postprocess", False)),
        help="Allow RGB/latent style affine postprocess in metric-producing runs. Off by default because it is not model capacity.",
    )
    parser.add_argument('--reuse_generated', action='store_true', help="Reuse existing generated images in output dir/images (or legacy output dir) and skip generation")
    parser.add_argument('--generation_only', action='store_true', help="Only generate translated images, skip all evaluation metrics")
    parser.add_argument('--seed', type=int, default=-1, help="Seed RNGs for reproducible VAE latent sampling/generation; <0 leaves RNG state untouched.")
    args = parser.parse_args()
    raw_cli_flags = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}

    def _cli_provided(name: str) -> bool:
        underscore = f"--{name}"
        hyphen = f"--{name.replace('_', '-')}"
        no_underscore = f"--no-{name}"
        no_hyphen = f"--no-{name.replace('_', '-')}"
        return bool({underscore, hyphen, no_underscore, no_hyphen} & raw_cli_flags)

    if int(args.seed) >= 0:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    # One-shot mode: `run_evaluation.py <full_eval_dir>`
    if args.eval_dir and not args.output:
        args.output = str(args.eval_dir)
        args.reuse_generated = True
        args.force_regen = True

    if args.output is None:
        if args.checkpoint is None:
            _auto_run_missing_full_eval(args)
            return
        raise ValueError("--output is required when --checkpoint is provided.")
    if args.checkpoint is None and (not args.reuse_generated):
        raise ValueError("--checkpoint is required unless --reuse_generated is set.")
    if args.checkpoint is None and args.generation_only:
        raise ValueError("--generation_only requires --checkpoint (cannot generate without model checkpoint).")
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
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _resolve_dir_path(args.cache_dir, path_bases)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir = (
        _resolve_dir_path(args.clip_hf_cache_dir, path_bases)
        if str(args.clip_hf_cache_dir).strip()
        else (cache_dir / "hf").resolve()
    )
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    # Handle both HF cache layouts:
    # - modern: <hf_cache>/hub/models--*
    # - legacy/manual: <hf_cache>/models--*
    hub_cache_dir = (hf_cache_dir / "hub").resolve()
    if not hub_cache_dir.exists() and any(hf_cache_dir.glob("models--*")):
        hub_cache_dir = hf_cache_dir
    # Pin all HuggingFace caches to one stable directory for offline reuse.
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str((hf_cache_dir / "transformers").resolve())
    if str(getattr(args, "clip_backend", "hf")).strip().lower() == "hf" and not bool(args.clip_allow_network):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["MODELSCOPE_OFFLINE"] = "1"
    print(f"HF cache dir: {hf_cache_dir}")
    print(f"HF hub cache dir: {hub_cache_dir}")
    
    io_pool = None
    
    checkpoint_path: Path | None = None
    cfg = {}
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {})
        if str(args.config_override).strip():
            cfg = merge_config_dicts(cfg, load_config(str(args.config_override).strip()))
        resolved_full_eval = resolve_full_eval_section(cfg)
        if resolved_full_eval:
            if "num_steps" in resolved_full_eval and not _cli_provided("num_steps"):
                args.num_steps = int(resolved_full_eval["num_steps"])
            if "step_size" in resolved_full_eval and not _cli_provided("step_size"):
                args.step_size = float(resolved_full_eval["step_size"])
            if "style_strength" in resolved_full_eval and not _cli_provided("style_strength"):
                args.style_strength = resolved_full_eval["style_strength"]
            if "vae_model" in resolved_full_eval and not _cli_provided("vae_model"):
                args.vae_model = str(resolved_full_eval["vae_model"])
            if "batch_size" in resolved_full_eval and not _cli_provided("batch_size"):
                args.batch_size = int(resolved_full_eval["batch_size"])
            if "max_src_samples" in resolved_full_eval and not _cli_provided("max_src_samples"):
                args.max_src_samples = int(resolved_full_eval["max_src_samples"])
            if "max_ref_compare" in resolved_full_eval and not _cli_provided("max_ref_compare"):
                args.max_ref_compare = int(resolved_full_eval["max_ref_compare"])
            if "max_ref_cache" in resolved_full_eval and not _cli_provided("max_ref_cache"):
                args.max_ref_cache = int(resolved_full_eval["max_ref_cache"])
            if "ref_feature_batch_size" in resolved_full_eval and not _cli_provided("ref_feature_batch_size"):
                args.ref_feature_batch_size = int(resolved_full_eval["ref_feature_batch_size"])
            if "target_chunk_size" in resolved_full_eval and not _cli_provided("target_chunk_size"):
                args.target_chunk_size = int(resolved_full_eval["target_chunk_size"])
            if "vae_decode_batch_size" in resolved_full_eval and not _cli_provided("vae_decode_batch_size"):
                args.vae_decode_batch_size = int(resolved_full_eval["vae_decode_batch_size"])
            if "transfer_only" in resolved_full_eval and not _cli_provided("transfer_only"):
                args.transfer_only = bool(resolved_full_eval["transfer_only"])
            if "image_save_workers" in resolved_full_eval and not _cli_provided("image_save_workers"):
                args.image_save_workers = int(resolved_full_eval["image_save_workers"])
            if "image_save_backend" in resolved_full_eval and not _cli_provided("image_save_backend"):
                args.image_save_backend = str(resolved_full_eval["image_save_backend"])
            if "save_generated_images" in resolved_full_eval and not _cli_provided("save_generated_images"):
                args.save_generated_images = bool(resolved_full_eval["save_generated_images"])
            if "save_summary_grid" in resolved_full_eval and not _cli_provided("save_summary_grid"):
                args.save_summary_grid = bool(resolved_full_eval["save_summary_grid"])
            if "keep_generated_on_device" in resolved_full_eval and not _cli_provided("keep_generated_on_device"):
                args.keep_generated_on_device = bool(resolved_full_eval["keep_generated_on_device"])
            if "lpips_chunk_size" in resolved_full_eval and not _cli_provided("eval_lpips_chunk_size"):
                args.eval_lpips_chunk_size = int(resolved_full_eval["lpips_chunk_size"])
            if "enable_introstyle" in resolved_full_eval and not _cli_provided("eval_enable_introstyle"):
                args.eval_enable_introstyle = bool(resolved_full_eval["enable_introstyle"])
            if "introstyle_style_bank_root" in resolved_full_eval and not _cli_provided("introstyle_style_bank_root"):
                args.introstyle_style_bank_root = str(resolved_full_eval["introstyle_style_bank_root"])
            if "introstyle_model_id" in resolved_full_eval and not _cli_provided("introstyle_model_id"):
                args.introstyle_model_id = str(resolved_full_eval["introstyle_model_id"])
            if "introstyle_modelscope_id" in resolved_full_eval and not _cli_provided("introstyle_modelscope_id"):
                args.introstyle_modelscope_id = str(resolved_full_eval["introstyle_modelscope_id"])
            if "introstyle_modelscope_cache_dir" in resolved_full_eval and not _cli_provided("introstyle_modelscope_cache_dir"):
                args.introstyle_modelscope_cache_dir = str(resolved_full_eval["introstyle_modelscope_cache_dir"])
            if "introstyle_allow_network" in resolved_full_eval and not _cli_provided("introstyle_allow_network"):
                args.introstyle_allow_network = bool(resolved_full_eval["introstyle_allow_network"])
            if "introstyle_bank_limit_per_style" in resolved_full_eval and not _cli_provided("introstyle_bank_limit_per_style"):
                args.introstyle_bank_limit_per_style = int(resolved_full_eval["introstyle_bank_limit_per_style"])
            if "introstyle_batch_size" in resolved_full_eval and not _cli_provided("introstyle_batch_size"):
                args.introstyle_batch_size = int(resolved_full_eval["introstyle_batch_size"])
            if "introstyle_topk" in resolved_full_eval and not _cli_provided("introstyle_topk"):
                args.introstyle_topk = int(resolved_full_eval["introstyle_topk"])
            if "introstyle_t" in resolved_full_eval and not _cli_provided("introstyle_t"):
                args.introstyle_t = int(resolved_full_eval["introstyle_t"])
            if "introstyle_up_ft_index" in resolved_full_eval and not _cli_provided("introstyle_up_ft_index"):
                args.introstyle_up_ft_index = int(resolved_full_eval["introstyle_up_ft_index"])
            if "introstyle_ensemble_size" in resolved_full_eval and not _cli_provided("introstyle_ensemble_size"):
                args.introstyle_ensemble_size = int(resolved_full_eval["introstyle_ensemble_size"])
            if "postprocess_mode" in resolved_full_eval and not _cli_provided("postprocess_mode"):
                args.postprocess_mode = str(resolved_full_eval["postprocess_mode"])
            if "postprocess_strength" in resolved_full_eval and not _cli_provided("postprocess_strength"):
                args.postprocess_strength = float(resolved_full_eval["postprocess_strength"])
            if "postprocess_mean_strength" in resolved_full_eval and not _cli_provided("postprocess_mean_strength"):
                args.postprocess_mean_strength = float(resolved_full_eval["postprocess_mean_strength"])
            if "postprocess_std_strength" in resolved_full_eval and not _cli_provided("postprocess_std_strength"):
                args.postprocess_std_strength = float(resolved_full_eval["postprocess_std_strength"])
            if "postprocess_ref_limit" in resolved_full_eval and not _cli_provided("postprocess_ref_limit"):
                args.postprocess_ref_limit = int(resolved_full_eval["postprocess_ref_limit"])
            if "latent_postprocess_mode" in resolved_full_eval and not _cli_provided("latent_postprocess_mode"):
                args.latent_postprocess_mode = str(resolved_full_eval["latent_postprocess_mode"])
            if "latent_postprocess_strength" in resolved_full_eval and not _cli_provided("latent_postprocess_strength"):
                args.latent_postprocess_strength = float(resolved_full_eval["latent_postprocess_strength"])
            if "latent_postprocess_mean_strength" in resolved_full_eval and not _cli_provided("latent_postprocess_mean_strength"):
                args.latent_postprocess_mean_strength = float(resolved_full_eval["latent_postprocess_mean_strength"])
            if "latent_postprocess_std_strength" in resolved_full_eval and not _cli_provided("latent_postprocess_std_strength"):
                args.latent_postprocess_std_strength = float(resolved_full_eval["latent_postprocess_std_strength"])
            if "latent_postprocess_ref_limit" in resolved_full_eval and not _cli_provided("latent_postprocess_ref_limit"):
                args.latent_postprocess_ref_limit = int(resolved_full_eval["latent_postprocess_ref_limit"])
            if "allow_metric_postprocess" in resolved_full_eval and not _cli_provided("allow_metric_postprocess"):
                args.allow_metric_postprocess = bool(resolved_full_eval["allow_metric_postprocess"])
    else:
        print("Single-run eval in reuse-only mode (no checkpoint).")

    # Generation-only probe runs are usually used for timing. Unless the caller
    # explicitly requests a collage, keep summary-grid export off so the timing
    # reflects generation rather than post-processing overhead.
    if args.generation_only and not _cli_provided("save_summary_grid"):
        args.save_summary_grid = False
    if (bool(args.save_summary_grid) or bool(args.eval_enable_art_fid) or bool(args.eval_enable_kid) or bool(args.eval_enable_introstyle) or bool(args.reuse_generated)) and not bool(args.save_generated_images):
        print("Enabling --save_generated_images because summary-grid, ArtFID/KID, IntroStyle, or reuse_generated needs image files.")
        args.save_generated_images = True

    image_save_workers = max(1, int(args.image_save_workers))
    io_pool = ThreadPoolExecutor(max_workers=image_save_workers) if bool(args.save_generated_images) else None
    
    # Resolve Test Data Path
    test_dir_raw = args.test_dir if args.test_dir else cfg.get('training', {}).get('test_image_dir', '')
    if not str(test_dir_raw).strip():
        # Reuse-only fallback for convenience.
        test_dir_raw = "../../style_data/overfit50"
    resolved_test_dir = _resolve_existing_path(
        test_dir_raw,
        [
            Path.cwd(),
            *( [checkpoint_path.parent.resolve()] if checkpoint_path is not None else [] ),
            Path(__file__).resolve().parent,
            Path(__file__).resolve().parents[1],
            Path(__file__).resolve().parents[2],
        ],
    )
    if resolved_test_dir is None:
        raise ValueError(f"Test directory not found: {test_dir_raw}")
    test_dir = _prefer_readable_eval_image_root(resolved_test_dir)

    style_subdirs = [x.strip() for x in str(args.style_subdirs).split(",") if x.strip()]
    if not style_subdirs:
        style_subdirs = list(cfg.get('data', {}).get('style_subdirs', []))
    if not style_subdirs:
        style_subdirs = [d.name for d in test_dir.iterdir() if d.is_dir()]
    if (not style_subdirs) and args.reuse_generated:
        style_subdirs = _infer_style_names_from_generated_files(_list_reuse_generated_files(out_dir))
    if not style_subdirs:
        raise ValueError("Failed to infer style names. Provide --style_subdirs or valid --test_dir folders.")
    
    test_images = {}
    for style_id, style_name in enumerate(style_subdirs):
        s_dir = test_dir / style_name
        if not s_dir.exists(): continue
        # Only take valid images
        images = sorted([p for p in s_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.webp']])
        test_images[style_id] = (style_name, images)

    postprocess_mode = str(args.postprocess_mode).strip().lower()
    if postprocess_mode not in {"none", "style_rgb_affine"}:
        raise ValueError(f"Unsupported postprocess_mode: {args.postprocess_mode}")
    metric_postprocess_requested = (
        (postprocess_mode != "none" and float(args.postprocess_strength) > 0.0)
        or (
            str(args.latent_postprocess_mode).strip().lower() != "none"
            and float(args.latent_postprocess_strength) > 0.0
        )
    )
    if metric_postprocess_requested and not bool(args.allow_metric_postprocess):
        raise ValueError(
            "Metric-affecting RGB/latent affine postprocess is disabled by default. "
            "Pass --allow_metric_postprocess, or set full_eval.allow_metric_postprocess=true / "
            "training.full_eval_allow_metric_postprocess=true, only for diagnostic calibration runs."
        )
    post_rgb_means = None
    post_rgb_stds = None
    if postprocess_mode == "style_rgb_affine" and float(args.postprocess_strength) > 0.0:
        post_rgb_means, post_rgb_stds = _compute_style_rgb_stats(
            test_images,
            image_size=256,
            ref_limit=int(args.postprocess_ref_limit),
        )
        print(
            "Post-decode RGB calibration enabled: "
            f"strength={float(args.postprocess_strength):.3f}, "
            f"mean={float(args.postprocess_mean_strength):.3f}, "
            f"std={float(args.postprocess_std_strength):.3f}, "
            f"ref_limit={int(args.postprocess_ref_limit)}"
        )
    latent_postprocess_mode = str(args.latent_postprocess_mode).strip().lower()
    if latent_postprocess_mode not in {"none", "style_latent_affine"}:
        raise ValueError(f"Unsupported latent_postprocess_mode: {args.latent_postprocess_mode}")
    latent_post_means = None
    latent_post_stds = None

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
    runtime_observability_rows = []
    style_name_to_id = {name: idx for idx, name in enumerate(style_subdirs)}
    src_lookup = {(x["style_name"], x["path"].stem): x["path"] for x in all_src_info}
    num_src_total = len(all_src_info)
    num_styles = len(style_subdirs)
    transfer_only = bool(getattr(args, "transfer_only", False))
    if transfer_only:
        print("Transfer-only eval enabled: skipping identity src_style==tgt_style pairs.")
    expected_generated = (
        sum(max(0, num_styles - 1) for _ in all_src_info)
        if transfer_only
        else num_src_total * num_styles
    )
    timings: dict[str, float] = {}
    wall_start = time.perf_counter()

    auto_reuse, found_generated = _should_auto_reuse_generated(
        out_dir=out_dir,
        expected_count=expected_generated,
        force_regen=bool(args.force_regen),
        reuse_generated=bool(args.reuse_generated),
    )
    if auto_reuse and not args.reuse_generated:
        args.reuse_generated = True
        print(
            f"\nPhase 1: Auto-reuse enabled from {images_dir} "
            f"(found {found_generated} generated images, expected {expected_generated})"
        )
    elif (not args.reuse_generated) and found_generated > 0 and expected_generated > 0:
        print(
            f"\nPhase 1: Found {found_generated}/{expected_generated} generated images under {images_dir}; "
            "treating cache as incomplete, regenerate."
        )

    if args.reuse_generated:
        print(f"\nPhase 1: Reuse generated images from {images_dir}")
        reuse_files = _list_reuse_generated_files(out_dir)
        fast_metric_half_cpu = (
            bool(args.eval_only_lpips_clip_style)
            and (not bool(args.save_generated_images))
            and (not bool(args.eval_enable_introstyle))
            and (not bool(args.eval_enable_art_fid))
            and (not bool(args.eval_enable_kid))
        )
        for p in reuse_files:
            parsed = _parse_generated_name(p.name, style_subdirs)
            if parsed is None:
                continue
            src_style, src_stem, tgt_style = parsed
            if transfer_only and src_style == tgt_style:
                continue
            src_path = src_lookup.get((src_style, src_stem))
            tgt_id = style_name_to_id.get(tgt_style)
            if src_path is None or tgt_id is None:
                continue
            try:
                gen_img = _load_eval_image_tensor(p)
                if fast_metric_half_cpu:
                    gen_img = gen_img.to(dtype=torch.float16)
            except Exception as e:
                print(f"  WARNING: failed loading generated image {p}: {e}")
                continue
            generated_buffer.append(
                {
                    "src_path": src_path,
                    "src_style": src_style,
                    "tgt_style_name": tgt_style,
                    "tgt_style_id": int(tgt_id),
                    "gen_img": gen_img,
                    "gen_name": p.name,
                    "runtime_observability": None,
                }
            )
        print(f"  Reused {len(generated_buffer)} generated images")

    if not generated_buffer:
        if checkpoint_path is None:
            raise RuntimeError("No reusable images found and no checkpoint provided. Cannot run generation phase.")
        print(f"\nPhase 1: Generation (Batch Size {args.batch_size})")
        fast_metric_half_cpu = (
            bool(args.eval_only_lpips_clip_style)
            and (not bool(args.save_generated_images))
            and (not bool(args.eval_enable_introstyle))
            and (not bool(args.eval_enable_art_fid))
            and (not bool(args.eval_enable_kid))
        )
        keep_generated_on_device = (
            bool(getattr(args, "keep_generated_on_device", True))
            and fast_metric_half_cpu
            and (not bool(args.generation_only))
        )
        if keep_generated_on_device:
            print("  Fast metric path: keep decoded generated tensors on GPU (no PNG/sidecar host roundtrip).")

        t0 = time.perf_counter()
        lgt = LGTInference(
            str(checkpoint_path),
            device=device,
            num_steps=args.num_steps,
            step_size=args.step_size,
            style_strength=args.style_strength,
            residual_scale=args.residual_scale,
            style_adapter_path=(args.style_adapter or None),
            config_override_path=(args.config_override or None),
        )
        _sync_cuda_if(device, bool(args.profile_timing))
        _add_timing(timings, "load_lancet", t0)
        t0 = time.perf_counter()
        vae = load_vae(
            device,
            model_id=str(args.vae_model),
            cache_dir=str(hf_cache_dir),
            compile_decoder=bool(args.vae_compile_decoder),
            compile_method=str(args.vae_compile_method),
            compile_mode=str(args.vae_compile_mode),
            compile_fullgraph=bool(args.vae_compile_fullgraph),
            compile_cache_dir=str(args.vae_compile_cache_dir),
        )
        _sync_cuda_if(device, bool(args.profile_timing))
        _add_timing(timings, "load_vae", t0)
        model_scale = float(getattr(lgt.model, "latent_scale_factor", 0.18215))
        vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
        scale_in = model_scale / max(vae_scale, 1e-8)
        scale_out = vae_scale / max(model_scale, 1e-8)
        if abs(scale_in - 1.0) > 1e-4:
            print(f"WARNING: latent scale mismatch (model={model_scale:.6f}, vae={vae_scale:.6f}). Applying rescale.")
        if latent_postprocess_mode == "style_latent_affine" and float(args.latent_postprocess_strength) > 0.0:
            latent_post_means, latent_post_stds = _compute_style_latent_stats(
                test_images,
                vae=vae,
                device=device,
                ref_limit=int(args.latent_postprocess_ref_limit),
                scale_in=scale_in,
            )
            print(
                "Latent style-affine calibration enabled: "
                f"strength={float(args.latent_postprocess_strength):.3f}, "
                f"mean={float(args.latent_postprocess_mean_strength):.3f}, "
                f"std={float(args.latent_postprocess_std_strength):.3f}, "
                f"ref_limit={int(args.latent_postprocess_ref_limit)}"
            )

        # Process in batches
        for b_start in range(0, num_src_total, args.batch_size):
            b_end = min(b_start + args.batch_size, num_src_total)
            batch_info = all_src_info[b_start:b_end]
            print(f"  Generating Batch {b_start//args.batch_size + 1}/{(num_src_total-1)//args.batch_size + 1}")

            # Load Source Images
            t0 = time.perf_counter()
            src_tensors = []
            for item in batch_info:
                src_tensors.append(_load_eval_image_tensor(item['path']))

            src_batch = torch.stack(src_tensors).to(device)
            _sync_cuda_if(device, bool(args.profile_timing))
            _add_timing(timings, "source_load_to_device", t0)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    # Inversion
                    t0 = time.perf_counter()
                    latents_src = encode_image(vae, src_batch, device)
                    if abs(scale_in - 1.0) > 1e-4:
                        latents_src = latents_src * scale_in
                    latents_x0 = lgt.inversion(latents_src)
                    _sync_cuda_if(device, bool(args.profile_timing))
                    _add_timing(timings, "encode_inversion", t0)

                    target_chunk = max(1, min(num_styles, int(args.target_chunk_size)))
                    default_decode_bs = max(1, len(batch_info) * target_chunk)
                    vae_decode_bs = max(1, int(args.vae_decode_batch_size) if int(args.vae_decode_batch_size) > 0 else default_decode_bs)

                    # Generation for target-style chunks. This keeps the legacy source
                    # batch loop but can batch several style IDs through LANCET and VAE,
                    # which is where full_eval previously under-used the GPU.
                    for tgt_start in range(0, num_styles, target_chunk):
                        tgt_end = min(num_styles, tgt_start + target_chunk)
                        chunk_style_ids = list(range(tgt_start, tgt_end))
                        pair_latents = []
                        pair_tgt_ids = []
                        meta = []
                        for tgt_id in chunk_style_ids:
                            tgt_name = style_subdirs[tgt_id]
                            for src_idx, src_item in enumerate(batch_info):
                                if transfer_only and int(src_item["style_id"]) == int(tgt_id):
                                    continue
                                out_name = f"{src_item['style_name']}_{src_item['path'].stem}_to_{tgt_name}.png"
                                pair_latents.append(latents_x0[src_idx:src_idx + 1])
                                pair_tgt_ids.append(int(tgt_id))
                                meta.append((src_item, tgt_name, tgt_id, out_name))
                        if not pair_latents:
                            continue
                        repeated_latents = torch.cat(pair_latents, dim=0)
                        tgt_ids = torch.tensor(pair_tgt_ids, device=device, dtype=torch.long)
                        t0 = time.perf_counter()
                        latents_gen = lgt.generation(repeated_latents, tgt_ids)
                        chunk_runtime_observability = _runtime_observability_from_model(getattr(lgt, "model", None))
                        latent_post_debug = None
                        if latent_postprocess_mode == "style_latent_affine":
                            latents_gen, latent_post_debug = _apply_latent_style_affine(
                                latents_gen,
                                tgt_ids,
                                latent_post_means,
                                latent_post_stds,
                                strength=float(args.latent_postprocess_strength),
                                mean_strength=float(args.latent_postprocess_mean_strength),
                                std_strength=float(args.latent_postprocess_std_strength),
                            )
                        if abs(scale_out - 1.0) > 1e-4:
                            latents_gen = latents_gen * scale_out
                        _sync_cuda_if(device, bool(args.profile_timing))
                        _add_timing(timings, "lancet_generation", t0)

                        for dec_start in range(0, latents_gen.shape[0], vae_decode_bs):
                            dec_end = min(latents_gen.shape[0], dec_start + vae_decode_bs)
                            t0 = time.perf_counter()
                            imgs_gen = decode_latent(
                                vae,
                                latents_gen[dec_start:dec_end],
                                device,
                                scaling_factor=args.vae_decode_scale,
                            )
                            post_debug = None
                            if postprocess_mode == "style_rgb_affine":
                                dec_meta = meta[dec_start:dec_end]
                                dec_tgt_ids = torch.tensor(
                                    [int(x[2]) for x in dec_meta],
                                    device=imgs_gen.device,
                                    dtype=torch.long,
                                )
                                imgs_gen, post_debug = _apply_postdecode_style_rgb_affine(
                                    imgs_gen,
                                    dec_tgt_ids,
                                    post_rgb_means,
                                    post_rgb_stds,
                                    strength=float(args.postprocess_strength),
                                    mean_strength=float(args.postprocess_mean_strength),
                                    std_strength=float(args.postprocess_std_strength),
                                )
                            _sync_cuda_if(device, bool(args.profile_timing))
                            _add_timing(timings, "vae_decode", t0)
                            save_generated_images = bool(args.save_generated_images)
                            imgs_gen_cpu = None
                            imgs_gen_u8_cpu = None
                            if save_generated_images:
                                t0 = time.perf_counter()
                                imgs_gen_u8_cpu = _images_01_to_uint8_hwc_cpu(imgs_gen)
                                _sync_cuda_if(device, bool(args.profile_timing))
                                _add_timing(timings, "uint8_cpu_copy", t0)
                            elif keep_generated_on_device:
                                t0 = time.perf_counter()
                                imgs_gen_cpu = imgs_gen.detach().to(dtype=torch.float16).contiguous()
                                _sync_cuda_if(device, bool(args.profile_timing))
                                _add_timing(timings, "generated_gpu_keep", t0)
                            else:
                                t0 = time.perf_counter()
                                cpu_dtype = torch.float16 if fast_metric_half_cpu else torch.float32
                                imgs_gen_cpu = imgs_gen.detach().to(device="cpu", dtype=cpu_dtype).contiguous()
                                _sync_cuda_if(device, bool(args.profile_timing))
                                _add_timing(timings, "generated_cpu_copy", t0)
                            t0 = time.perf_counter()
                            for local_i, (src_item, tgt_name, tgt_id, out_name) in enumerate(meta[dec_start:dec_end]):
                                runtime_observability = dict(chunk_runtime_observability) if chunk_runtime_observability else {}
                                if isinstance(latent_post_debug, dict):
                                    runtime_observability.update(latent_post_debug)
                                if isinstance(post_debug, dict):
                                    runtime_observability.update(post_debug)
                                gen_name = out_name
                                gen_img_payload = imgs_gen_cpu[local_i] if imgs_gen_cpu is not None else imgs_gen_u8_cpu[local_i]
                                if save_generated_images:
                                    out_path = images_dir / out_name
                                    out_rel = Path("images") / out_name
                                    io_pool.submit(
                                        save_image_task,
                                        gen_img_payload,
                                        out_path,
                                        str(args.image_save_backend),
                                    )
                                    gen_name = out_rel.as_posix()
                                generated_buffer.append({
                                    'src_path': src_item['path'],
                                    'src_style': src_item['style_name'],
                                    'tgt_style_name': tgt_name,
                                    'tgt_style_id': tgt_id,
                                    'gen_img': gen_img_payload,
                                    'gen_name': gen_name,
                                    'runtime_observability': runtime_observability or None,
                                })
                            if save_generated_images:
                                _add_timing(timings, "image_save_submit", t0)
                            del imgs_gen, imgs_gen_cpu, imgs_gen_u8_cpu
                        del repeated_latents, tgt_ids, latents_gen, pair_latents, pair_tgt_ids

        # Unload Generation Models
        del lgt, vae
        torch.cuda.empty_cache()
        gc.collect()
        print("  Generation models unloaded")
    if not generated_buffer:
        raise RuntimeError(f"No generated samples to evaluate in {out_dir}")

    if args.generation_only:
        print("\nGeneration-only mode enabled: skip Phase 2 metrics/LPIPS/CLIP.")
        t0 = time.perf_counter()
        if io_pool is not None:
            io_pool.shutdown(wait=True)
            _add_timing(timings, "image_save_join", t0)
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        timings["wall_total"] = float(time.perf_counter() - wall_start)
        summary = {
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else "(reuse-only:no-checkpoint)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "generation_only",
            "generated_count": int(len(generated_buffer)),
            "output_dir": str(out_dir),
            "settings": {
                "vae_model": str(args.vae_model),
                "batch_size": int(args.batch_size),
                "target_chunk_size": int(args.target_chunk_size),
                "vae_decode_batch_size": int(args.vae_decode_batch_size),
                "transfer_only": bool(getattr(args, "transfer_only", False)),
                "image_save_workers": int(args.image_save_workers),
                "image_save_backend": str(args.image_save_backend),
                "save_generated_images": bool(args.save_generated_images),
                "keep_generated_on_device": bool(getattr(args, "keep_generated_on_device", True)),
                "lpips_chunk_size": int(args.eval_lpips_chunk_size),
                "profile_timing": bool(args.profile_timing),
                "save_summary_grid": bool(args.save_summary_grid),
                "postprocess_mode": str(args.postprocess_mode),
                "postprocess_strength": float(args.postprocess_strength),
                "postprocess_mean_strength": float(args.postprocess_mean_strength),
                "postprocess_std_strength": float(args.postprocess_std_strength),
                "postprocess_ref_limit": int(args.postprocess_ref_limit),
                "latent_postprocess_mode": str(args.latent_postprocess_mode),
                "latent_postprocess_strength": float(args.latent_postprocess_strength),
                "latent_postprocess_mean_strength": float(args.latent_postprocess_mean_strength),
                "latent_postprocess_std_strength": float(args.latent_postprocess_std_strength),
                "latent_postprocess_ref_limit": int(args.latent_postprocess_ref_limit),
            },
            "timings_sec": {k: float(v) for k, v in sorted(timings.items())},
            "note": "Metrics are intentionally skipped. Run evaluation later with --reuse_generated.",
        }
        sum_path = out_dir / "summary.json"
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved: {sum_path}")
        if bool(args.save_summary_grid):
            print("  Saving summary grid... (disable with --no-save_summary_grid for pure throughput timing)")
            grid_rows = []
            for it in generated_buffer:
                grid_rows.append(
                    {
                        "src_style": it["src_style"],
                        "tgt_style": it["tgt_style_name"],
                        "src_image": Path(it["src_path"]).name,
                        "gen_image": it["gen_name"],
                    }
                )
            t0 = time.perf_counter()
            summary_grid_path = _save_summary_grid_png(grid_rows, out_dir, style_order=list(style_subdirs))
            _add_timing(timings, "summary_grid", t0)
            timings["wall_total"] = float(time.perf_counter() - wall_start)
            summary["timings_sec"] = {k: float(v) for k, v in sorted(timings.items())}
            if summary_grid_path is not None:
                summary["summary_grid_path"] = str(summary_grid_path)
            with open(sum_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Summary updated: {sum_path}")
        return

    # ==========================================
    # PHASE 2: EVALUATION (LPIPS + CLIP)
    # ==========================================
    print(f"\n妫ｅ啯鐣?Phase 2: Evaluation")
    eval_phase_start = time.perf_counter()

    run_full_metrics = True
    only_lpips_clip_style = bool(args.eval_only_lpips_clip_style)

    # Load Evaluators
    loss_fn = None
    clip_model = None
    clip_processor = None
    has_clip = False
    clip_backend = str(getattr(args, "clip_backend", "hf")).strip().lower()
    clip_preprocess = None  # OpenAI CLIP preprocess
    clip_encode_images_01 = None  # Callable[[Tensor[B,3,H,W]], Tensor[B,D]] on device
    if clip_backend == "openai":
        clip_model_tag = f"openai:{str(getattr(args, 'clip_openai_model', 'ViT-B/32')).strip() or 'ViT-B/32'}"
    elif clip_backend == "hf":
        clip_model_tag = str(args.clip_model_name).strip() or "openai/clip-vit-base-patch32"
    else:
        clip_model_tag = "none"

    if run_full_metrics:
        # Initialize LPIPS
        if args.eval_disable_lpips:
            loss_fn = None
        elif lpips is None:
            print("  WARNING: lpips module not available. Install with: pip install lpips")
        else:
            try:
                loss_fn = lpips.LPIPS(net='vgg', verbose=False).to(device)
                print("  LPIPS Loaded")
            except Exception as e:
                print(f"  WARNING: Failed to load LPIPS: {e}")

        if clip_backend == "none":
            has_clip = False
        elif clip_backend == "openai":
            try:
                import clip as openai_clip

                clip_cache_root = (cache_dir / "clip_openai").resolve()
                clip_cache_root.mkdir(parents=True, exist_ok=True)
                model_name = str(getattr(args, "clip_openai_model", "ViT-B/32")).strip() or "ViT-B/32"

                if not bool(args.clip_allow_network):
                    # Fail fast (avoid hanging downloads) if weights are missing.
                    url = getattr(openai_clip, "_MODELS", {}).get(model_name)
                    if url:
                        expected = clip_cache_root / Path(str(url)).name
                        if not expected.exists():
                            raise FileNotFoundError(
                                f"OpenAI CLIP weights not found in cache: {expected}. "
                                f"Run once with --clip_allow_network to download, or pre-download into {clip_cache_root}."
                            )

                clip_model, clip_preprocess = openai_clip.load(
                    model_name,
                    device=device,
                    download_root=str(clip_cache_root),
                )
                clip_model.eval()
                has_clip = True
                clip_model_tag = f"openai:{model_name}"
                print(f"  CLIP Loaded (OpenAI): {model_name} (cache={clip_cache_root})")
            except Exception as e:
                if bool(getattr(args, "clip_optional", False)):
                    print(f"  WARNING: OpenAI CLIP unavailable, continue without CLIP metrics: {e}")
                    has_clip = False
                    clip_model = None
                    clip_preprocess = None
                else:
                    raise
        elif clip_backend == "hf":
            clip_model_name = str(args.clip_model_name).strip() or _DEFAULT_HF_CLIP_REPO
            try:
                from transformers import CLIPModel, CLIPProcessor

                clip_sources = []
                local_only = (not bool(args.clip_allow_network))
                model_name_raw = str(args.clip_model_name).strip()
                for candidate in _manual_clip_candidates(cache_dir):
                    if candidate.exists():
                        clip_sources.append(str(candidate.resolve()))

                if model_name_raw:
                    model_name_path = Path(model_name_raw).expanduser()
                    if model_name_path.exists():
                        clip_sources.append(str(model_name_path.resolve()))

                local_snapshot = _find_local_hf_snapshot(hf_cache_dir, clip_model_name)
                if local_snapshot:
                    clip_sources.append(local_snapshot)

                if not local_only:
                    clip_sources.append(clip_model_name)

                # Preserve insertion order while removing duplicates.
                clip_sources = list(dict.fromkeys(clip_sources))

                if local_only and not clip_sources:
                    dbg = _debug_clip_cache_state(hf_cache_dir, clip_model_name)
                    raise FileNotFoundError(
                        "Offline CLIP load requires local cache, but no local source was found. "
                        f"clip_model_name={clip_model_name}, hf_cache_dir={hf_cache_dir}. "
                        f"Cache diagnosis: {dbg}"
                    )

                ms_id = str(args.clip_modelscope_id).strip()
                if ms_id:
                    try:
                        from modelscope.hub.snapshot_download import snapshot_download

                        ms_kwargs = {}
                        ms_cache_dir = str(args.clip_modelscope_cache_dir).strip()
                        if not ms_cache_dir:
                            ms_cache_dir = str((hf_cache_dir / "modelscope").resolve())
                        ms_kwargs["cache_dir"] = ms_cache_dir
                        try:
                            ms_local = snapshot_download(
                                ms_id, local_files_only=(not bool(args.clip_allow_network)), **ms_kwargs
                            )
                        except TypeError:
                            if bool(args.clip_allow_network):
                                ms_local = snapshot_download(ms_id, **ms_kwargs)
                            else:
                                raise
                        clip_sources.append(ms_local)
                        print(f"  ModelScope CLIP cache: {ms_local}")
                    except Exception as ms_exc:
                        print(f"  WARNING: ModelScope CLIP fallback unavailable: {ms_exc}")

                last_err = None
                for src in clip_sources:
                    try:
                        clip_model, clip_processor = _load_clip_from_source(
                            CLIPModel,
                            CLIPProcessor,
                            src,
                            device,
                            local_only=(not bool(args.clip_allow_network)),
                            cache_dir=str(hf_cache_dir),
                        )
                        clip_model.eval()
                        has_clip = True
                        clip_model_tag = str(src)
                        print(f"  CLIP Loaded (HF) from: {src}")
                        break
                    except Exception as load_exc:
                        last_err = load_exc
                        continue
                if not has_clip and last_err is not None:
                    raise last_err
            except Exception as e:
                if bool(getattr(args, "clip_optional", False)):
                    print(f"  WARNING: HF CLIP unavailable, continue without CLIP metrics: {e}")
                    try:
                        dbg = _debug_clip_cache_state(hf_cache_dir, clip_model_name)
                        print(f"  CLIP cache diagnosis: {dbg}")
                    except Exception:
                        pass
                    has_clip = False
                    clip_model = None
                    clip_processor = None
                else:
                    raise
        else:
            raise ValueError(f"Invalid --clip_backend: {clip_backend}")

        if has_clip and clip_model is not None:
            clip_image_size = _clip_image_size_from_runtime(clip_model, clip_processor)
            clip_mean, clip_std = _clip_mean_std_from_runtime(clip_processor)
            if clip_backend == "openai":
                def clip_encode_images_01(images_01):  # noqa: ANN001
                    imgs = _prepare_clip_pixels(images_01, image_size=clip_image_size, mean=clip_mean, std=clip_std)
                    feats = clip_model.encode_image(imgs)
                    feats = feats.to(dtype=torch.float32)
                    if feats.ndim == 1:
                        feats = feats.unsqueeze(0)
                    return feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            else:

                def clip_encode_images_01(images_01):  # noqa: ANN001
                    pixels = _prepare_clip_pixels(images_01, image_size=clip_image_size, mean=clip_mean, std=clip_std)
                    out = clip_model.get_image_features(pixel_values=pixels)
                    feats = _extract_clip_embeddings(out).to(device, dtype=torch.float32)
                    if feats.ndim == 1:
                        feats = feats.unsqueeze(0)
                    return feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)

    # Prepare Reference Features (Cache)
    style_sig = ",".join(style_subdirs)
    dataset_sig = f"{str(test_dir.resolve())}|{style_sig}|{clip_model_tag}|tensorclip-v3"
    dataset_hash = hashlib.md5(dataset_sig.encode()).hexdigest()[:10]
    max_ref_cache = int(args.max_ref_cache)
    max_ref_cache_tag = "all" if max_ref_cache <= 0 else str(max_ref_cache)
    cache_file = cache_dir / f"ref_feats_{dataset_hash}_m{max_ref_cache_tag}.pt"
    lock_file = cache_file.with_suffix(cache_file.suffix + ".lock")

    ref_features = {}
    ref_cache_status = "not_used"
    # Keep reference cache independent from output regeneration.
    must_rebuild_ref_cache = bool(args.force_regen_ref_cache)

    if run_full_metrics and cache_file.exists() and not must_rebuild_ref_cache:
        print(f"Found global reference cache: {cache_file}")
        try:
            ref_features = torch.load(cache_file, map_location='cpu')
            if _is_ref_cache_valid(ref_features, need_clip=has_clip):
                print("  Reference cache loaded successfully")
                ref_cache_status = "loaded"
            else:
                print("  Reference cache invalid for current metrics, rebuilding...")
                ref_features = {}
        except Exception as e:
            print(f"  Reference cache load failed ({e}), rebuilding...")
            ref_features = {}

    if run_full_metrics and not ref_features:
        got_lock = _acquire_lock(lock_file, timeout_sec=int(args.ref_cache_lock_timeout), poll_sec=1.0)
        if not got_lock:
            raise TimeoutError(f"Timed out waiting for reference-cache lock: {lock_file}")
        try:
            # Double-check after lock: another process may have completed cache.
            if cache_file.exists() and not must_rebuild_ref_cache:
                try:
                    ref_features = torch.load(cache_file, map_location='cpu')
                    if _is_ref_cache_valid(ref_features, need_clip=has_clip):
                        print(f"Loaded global reference cache after waiting: {cache_file}")
                        ref_cache_status = "loaded_after_wait"
                    else:
                        ref_features = {}
                except Exception:
                    ref_features = {}

            if not ref_features:
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
                            batch_tensors = torch.stack([_load_eval_image_tensor(img_path) for img_path in batch_paths], dim=0).to(device)
                            with torch.no_grad():
                                c_emb = None
                                if has_clip and clip_model is not None and clip_encode_images_01 is not None:
                                    c_emb = clip_encode_images_01(batch_tensors).detach().cpu()

                            for i, img_path in enumerate(batch_paths):
                                ref_features[style_id].append({
                                    'path': str(img_path),
                                    'clip': c_emb[i:i+1] if c_emb is not None else None
                                })
                            del batch_tensors
                        except Exception as e:
                            print(f"Skipping batch {b_start}-{b_start + len(batch_paths)} in {style_name}: {e}")

                tmp_cache = cache_file.with_suffix(cache_file.suffix + f".tmp.{os.getpid()}")
                torch.save(ref_features, tmp_cache)
                os.replace(tmp_cache, cache_file)
                print(f"Global reference cache saved: {cache_file}")
                ref_cache_status = "rebuilt"
        finally:
            try:
                lock_file.unlink(missing_ok=True)
            except Exception:
                pass

    # Optimize Reference CLIP Features for Vectorization
    ref_clip_matrices = {} # style_id -> Tensor[N_ref, D] (GPU)
    ref_clip_prototypes = {}  # style_id -> Tensor[1, D] (GPU)
    
    if run_full_metrics and has_clip and clip_model is not None:
        for sid, feats in ref_features.items():
            clips = [f['clip'] for f in feats if f['clip'] is not None]
            if clips:
                try:
                    # Detect dimension dynamically from the first clip
                    current_dim = clips[0].shape[-1]
                    
                    valid_clips = []
                    for c in clips:
                        if c.ndim == 1: c = c.unsqueeze(0)
                        if c.shape[-1] == current_dim: valid_clips.append(c)
                    
                    if valid_clips:
                        # Stack: [N, D]
                        stacked = torch.cat(valid_clips, dim=0)
                        # Double check norm
                        stacked = stacked / (stacked.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                        ref_clip_matrices[sid] = stacked.to(device, dtype=torch.float32)
                        proto = stacked.mean(dim=0, keepdim=True)
                        proto = proto / (proto.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                        ref_clip_prototypes[sid] = proto.to(device, dtype=torch.float32)
                except Exception as e:
                    print(f"  闁宠法濯寸粭?Failed to prepare CLIP matrix for style {sid}: {e}")

    fast_metric_half_cpu = (
        bool(args.eval_only_lpips_clip_style)
        and (not bool(args.save_generated_images))
        and (not bool(args.eval_enable_introstyle))
        and (not bool(args.eval_enable_art_fid))
        and (not bool(args.eval_enable_kid))
    )

    csv_path = out_dir / 'metrics.csv'
    # Re-evaluation on reused images should overwrite metrics to avoid mixing old/new outputs.
    csv_mode = 'w' if args.force_regen or args.reuse_generated or not csv_path.exists() else 'a'
    csv_file = open(csv_path, csv_mode, newline='', encoding='utf-8')
    columns = [
        'src_style',
        'tgt_style',
        'src_image',
        'gen_image',
        'content_lpips',
        
        'clip_dir',
        'clip_style',
        'clip_content',
    ]
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    if csv_mode == 'w': writer.writeheader()

    # Process Generated Buffer
    total_gen = len(generated_buffer)
    print(f"  Processing {total_gen} generated images...")
    src_eval_size = 256
    need_src_clip_cache = bool(
        has_clip and clip_model is not None and clip_encode_images_01 is not None and (not only_lpips_clip_style)
    )
    unique_src_keys = sorted({_source_path_key(item['src_path']) for item in generated_buffer})
    src_cache_file = cache_dir / f"src_feats_{dataset_hash}_img{src_eval_size}_clip{1 if need_src_clip_cache else 0}.pt"
    src_cache_lock = src_cache_file.with_suffix(src_cache_file.suffix + ".lock")
    src_cache_payload = {}
    src_cache_status = "not_used"
    must_rebuild_src_cache = bool(args.force_regen_ref_cache)
    if src_cache_file.exists() and not must_rebuild_src_cache:
        print(f"Found global source cache: {src_cache_file}")
        try:
            src_cache_payload = torch.load(src_cache_file, map_location='cpu')
            if _is_source_cache_valid(src_cache_payload, image_size=src_eval_size, need_clip=need_src_clip_cache):
                print("  Source cache loaded successfully")
                src_cache_status = "loaded"
            else:
                print("  Source cache invalid for current eval settings, rebuilding...")
                src_cache_payload = {}
        except Exception as e:
            print(f"  Source cache load failed ({e}), rebuilding...")
            src_cache_payload = {}
    if not src_cache_payload:
        got_lock = _acquire_lock(src_cache_lock, timeout_sec=int(args.ref_cache_lock_timeout), poll_sec=1.0)
        if not got_lock:
            raise TimeoutError(f"Timed out waiting for source-cache lock: {src_cache_lock}")
        try:
            if src_cache_file.exists() and not must_rebuild_src_cache:
                try:
                    src_cache_payload = torch.load(src_cache_file, map_location='cpu')
                    if _is_source_cache_valid(src_cache_payload, image_size=src_eval_size, need_clip=need_src_clip_cache):
                        print(f"Loaded global source cache after waiting: {src_cache_file}")
                        src_cache_status = "loaded_after_wait"
                    else:
                        src_cache_payload = {}
                except Exception:
                    src_cache_payload = {}
            if not src_cache_payload:
                print(f"\nComputing Source Cache (global miss): {src_cache_file}")
                src_items: dict[str, dict[str, torch.Tensor | None]] = {}
                src_bs = max(1, int(args.ref_feature_batch_size))
                pbar = tqdm(range(0, len(unique_src_keys), src_bs), desc="Caching source images")
                for b_start in pbar:
                    batch_keys = unique_src_keys[b_start:b_start + src_bs]
                    batch_paths = [Path(key) for key in batch_keys]
                    batch_tensors_cpu = torch.stack(
                        [_load_eval_image_tensor(src_path, size=src_eval_size) for src_path in batch_paths],
                        dim=0,
                    ).contiguous()
                    clip_cpu = None
                    if need_src_clip_cache:
                        batch_tensors_gpu = batch_tensors_cpu.to(device)
                        clip_cpu = clip_encode_images_01(batch_tensors_gpu).detach().cpu()
                        del batch_tensors_gpu
                    for idx, key in enumerate(batch_keys):
                        src_items[key] = {
                            "img": batch_tensors_cpu[idx].clone(),
                            "clip": None if clip_cpu is None else clip_cpu[idx].clone(),
                        }
                src_cache_payload = {
                    "meta": {
                        "image_size": int(src_eval_size),
                        "clip_enabled": bool(need_src_clip_cache),
                        "count": int(len(src_items)),
                    },
                    "items": src_items,
                }
                tmp_src_cache = src_cache_file.with_suffix(src_cache_file.suffix + f".tmp.{os.getpid()}")
                torch.save(src_cache_payload, tmp_src_cache)
                os.replace(tmp_src_cache, src_cache_file)
                print(f"Global source cache saved: {src_cache_file}")
                src_cache_status = "rebuilt"
        finally:
            try:
                src_cache_lock.unlink(missing_ok=True)
            except Exception:
                pass

    src_cache_items = src_cache_payload.get("items", {}) if isinstance(src_cache_payload, dict) else {}
    src_img_cache = {
        str(key): value["img"]
        for key, value in src_cache_items.items()
        if isinstance(value, dict) and isinstance(value.get("img"), torch.Tensor)
    }
    src_clip_cache = {
        str(key): value["clip"]
        for key, value in src_cache_items.items()
        if isinstance(value, dict) and isinstance(value.get("clip"), torch.Tensor)
    }
    metric_loop_start = time.perf_counter()
    
    for b_start in range(0, total_gen, args.batch_size):
        b_end = min(b_start + args.batch_size, total_gen)
        batch_items = generated_buffer[b_start:b_end]
        
        gen_imgs_cpu = torch.stack(
            [
                _uint8_hwc_to_float_chw(item['gen_img'])
                if torch.is_tensor(item['gen_img']) and item['gen_img'].dtype == torch.uint8
                else item['gen_img']
                for item in batch_items
            ],
            dim=0,
        ).contiguous()
        gen_imgs = gen_imgs_cpu.to(device, non_blocking=True)

        src_tensors = []
        src_keys = []
        for item in batch_items:
            src_key = _source_path_key(item['src_path'])
            src_keys.append(src_key)
            cached = src_img_cache.get(src_key)
            if cached is None:
                cached = _load_eval_image_tensor(Path(item['src_path']), size=src_eval_size)
                if fast_metric_half_cpu:
                    cached = cached.to(dtype=torch.float16)
                src_img_cache[src_key] = cached
            src_tensors.append(cached)
        src_imgs_cpu = torch.stack(src_tensors, dim=0).contiguous()
        src_imgs = src_imgs_cpu.to(device, non_blocking=True)
        
        with torch.no_grad():
            # 1. Content LPIPS
            c_lpips_vals = []
            if loss_fn:
                gen_f32 = gen_imgs.float()
                src_f32 = src_imgs.float()
                lpips_chunk = max(1, int(args.eval_lpips_chunk_size))
                lpips_cpu_fallback = not bool(args.eval_lpips_no_cpu_fallback)
                dists = _lpips_forward_safe(
                    loss_fn,
                    gen_f32,
                    src_f32,
                    device=device,
                    chunk_size=lpips_chunk,
                    cpu_fallback=lpips_cpu_fallback,
                    tag="content_lpips",
                )
                c_lpips_vals = dists.numpy()
            else:
                c_lpips_vals = [0.0] * len(batch_items)

            # 2. CLIP Features
            gen_clips = None
            src_clips = None
            c_clip_scores = [0.0] * len(batch_items)
            batch_clip_style_scores = None
            batch_clip_dir_scores = None
            
            if has_clip and clip_model is not None and clip_encode_images_01 is not None:
                gen_clips = clip_encode_images_01(gen_imgs)
                if not only_lpips_clip_style:
                    # Src CLIP (cache by source path; source repeats across many target styles)
                    miss_indices = [i for i, k in enumerate(src_keys) if k not in src_clip_cache]
                    if miss_indices:
                        src_miss = clip_encode_images_01(src_imgs[miss_indices])
                        src_miss_cpu = src_miss.detach().cpu()
                        for j, idx in enumerate(miss_indices):
                            src_clip_cache[src_keys[idx]] = src_miss_cpu[j].clone()
                    src_clips = torch.stack([src_clip_cache[k] for k in src_keys], dim=0).to(device, dtype=torch.float32)
                    c_clip_scores = F.cosine_similarity(gen_clips, src_clips).cpu().float().numpy()
                proto_items = []
                proto_valid = []
                for item in batch_items:
                    proto = ref_clip_prototypes.get(item["tgt_style_id"])
                    if proto is not None and proto.shape[-1] == gen_clips.shape[-1]:
                        proto_items.append(proto.view(-1))
                        proto_valid.append(True)
                    else:
                        proto_items.append(torch.zeros((gen_clips.shape[-1],), device=device, dtype=torch.float32))
                        proto_valid.append(False)
                if proto_items:
                    proto_batch = torch.stack(proto_items, dim=0).to(device=device, dtype=torch.float32)
                    batch_clip_style_scores = F.cosine_similarity(gen_clips.float(), proto_batch, dim=-1).detach().cpu().float().numpy()
                    for j, ok in enumerate(proto_valid):
                        if not ok:
                            batch_clip_style_scores[j] = 0.0
                    if (not only_lpips_clip_style) and src_clips is not None:
                        dir_gen = gen_clips.float() - src_clips.float()
                        dir_tgt = proto_batch - src_clips.float()
                        dir_gen = dir_gen / (dir_gen.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                        dir_tgt = dir_tgt / (dir_tgt.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                        batch_clip_dir_scores = F.cosine_similarity(dir_gen, dir_tgt, dim=-1).detach().cpu().float().numpy()
                        for j, ok in enumerate(proto_valid):
                            if not ok:
                                batch_clip_dir_scores[j] = 0.0

            # 3. Style Metrics & Row Writing
            for i, item in enumerate(batch_items):
                # --- CLIP metrics ---
                # clip_dir: directional similarity in edit space.
                # clip_style: absolute similarity to target style prototype.
                s_clip_dir = 0.0
                s_clip_style = 0.0
                if batch_clip_style_scores is not None:
                    s_clip_style = float(batch_clip_style_scores[i])
                if batch_clip_dir_scores is not None:
                    s_clip_dir = float(batch_clip_dir_scores[i])

                writer.writerow({
                    'src_style': item['src_style'],
                    'tgt_style': item['tgt_style_name'],
                    'src_image': item['src_path'].name,
                    'gen_image': item['gen_name'],
                    'content_lpips': c_lpips_vals[i],
                    
                    'clip_dir': s_clip_dir,
                    'clip_style': s_clip_style,
                    'clip_content': c_clip_scores[i],
                })
                observability = item.get("runtime_observability")
                if isinstance(observability, dict) and observability:
                    runtime_observability_rows.append(
                        {
                            "src_style": str(item["src_style"]),
                            "tgt_style": str(item["tgt_style_name"]),
                            **{str(k): float(v) for k, v in observability.items()},
                        }
                    )
            
            csv_file.flush()

    _sync_cuda_if(device, bool(args.profile_timing))
    _add_timing(timings, "eval_metrics_loop", metric_loop_start)
    csv_file.close()
    introstyle_result = None
    if bool(args.eval_enable_introstyle) and not bool(args.generation_only):
        t0 = time.perf_counter()
        try:
            introstyle_result = _run_introstyle_sidecar(
                metrics_csv=csv_path,
                images_dir=images_dir,
                test_dir=test_dir,
                cache_dir=cache_dir,
                device=device,
                model_id=str(args.introstyle_model_id),
                modelscope_id=str(args.introstyle_modelscope_id),
                modelscope_cache_dir=str(args.introstyle_modelscope_cache_dir),
                allow_network=bool(args.introstyle_allow_network),
                style_bank_root=str(args.introstyle_style_bank_root),
                bank_limit_per_style=int(args.introstyle_bank_limit_per_style),
                batch_size=int(args.introstyle_batch_size),
                topk=int(args.introstyle_topk),
                t=int(args.introstyle_t),
                up_ft_index=int(args.introstyle_up_ft_index),
                ensemble_size=int(args.introstyle_ensemble_size),
            )
        except Exception as exc:
            print(f"WARNING: IntroStyle sidecar failed: {exc}")
            introstyle_result = None
        _add_timing(timings, "eval_introstyle", t0)
    t0 = time.perf_counter()
    if io_pool is not None:
        io_pool.shutdown(wait=True)
        _add_timing(timings, "image_save_join", t0)
    timings["eval_total"] = float(time.perf_counter() - eval_phase_start)
    timings["wall_total"] = float(time.perf_counter() - wall_start)
    
    style_real_paths = {}
    for _, (style_name, img_list) in test_images.items():
        style_real_paths[style_name] = [str(p) for p in img_list]
    ckpt_for_summary = checkpoint_path if checkpoint_path is not None else Path("(reuse-only:no-checkpoint)")
    generate_summary_json(
        csv_path,
        out_dir,
        ckpt_for_summary,
        style_order=list(style_subdirs),
        style_real_paths=style_real_paths,
        source_style_paths=style_real_paths,
        device=device,
        enable_art_fid=bool(args.eval_enable_art_fid),
        art_fid_max_gen=int(args.eval_art_fid_max_gen),
        art_fid_max_ref=int(args.eval_art_fid_max_ref),
        art_fid_batch_size=int(args.eval_art_fid_batch_size),
        art_fid_photo_only=bool(args.eval_art_fid_photo_only),
        cache_dir=cache_dir,
        enable_kid=bool(args.eval_enable_kid),
        kid_max_gen=int(args.eval_kid_max_gen),
        kid_max_ref=int(args.eval_kid_max_ref),
        kid_subset_size=int(args.eval_kid_subset_size),
        kid_batch_size=int(args.eval_kid_batch_size),
        settings={
            "vae_model": str(args.vae_model),
            "batch_size": int(args.batch_size),
            "target_chunk_size": int(args.target_chunk_size),
            "vae_decode_batch_size": int(args.vae_decode_batch_size),
            "transfer_only": bool(getattr(args, "transfer_only", False)),
            "image_save_workers": int(args.image_save_workers),
            "image_save_backend": str(args.image_save_backend),
            "save_generated_images": bool(args.save_generated_images),
            "keep_generated_on_device": bool(getattr(args, "keep_generated_on_device", True)),
            "lpips_chunk_size": int(args.eval_lpips_chunk_size),
            "profile_timing": bool(args.profile_timing),
            "save_summary_grid": bool(args.save_summary_grid),
            "only_lpips_clip_style": bool(args.eval_only_lpips_clip_style),
            "postprocess_mode": str(args.postprocess_mode),
            "postprocess_strength": float(args.postprocess_strength),
            "postprocess_mean_strength": float(args.postprocess_mean_strength),
            "postprocess_std_strength": float(args.postprocess_std_strength),
            "postprocess_ref_limit": int(args.postprocess_ref_limit),
            "allow_metric_postprocess": bool(args.allow_metric_postprocess),
            "latent_postprocess_mode": str(args.latent_postprocess_mode),
            "latent_postprocess_strength": float(args.latent_postprocess_strength),
            "latent_postprocess_mean_strength": float(args.latent_postprocess_mean_strength),
            "latent_postprocess_std_strength": float(args.latent_postprocess_std_strength),
            "latent_postprocess_ref_limit": int(args.latent_postprocess_ref_limit),
            "enable_introstyle": bool(args.eval_enable_introstyle),
            "introstyle_style_bank_root": str(args.introstyle_style_bank_root) if str(args.introstyle_style_bank_root).strip() else str(test_dir),
            "introstyle_model_id": str(args.introstyle_model_id),
            "introstyle_modelscope_id": str(args.introstyle_modelscope_id),
            "cache_artifacts": {
                "reference_cache": {
                    "path": str(cache_file),
                    "status": str(ref_cache_status),
                    "entry_count": int(sum(len(v) for v in ref_features.values())) if isinstance(ref_features, dict) else 0,
                },
                "source_cache": {
                    "path": str(src_cache_file),
                    "status": str(src_cache_status),
                    "entry_count": int(len(src_cache_items)),
                },
            },
        },
        timings=timings,
        introstyle_result=introstyle_result,
        runtime_observability_rows=runtime_observability_rows,
    )

def generate_summary_json(
    csv_path,
    out_dir,
    ckpt_path,
    *,
    style_order=None,
    style_real_paths=None,
    source_style_paths=None,
    device: str = "cpu",
    enable_art_fid: bool = False,
    art_fid_max_gen: int = 200,
    art_fid_max_ref: int = 200,
    art_fid_batch_size: int = 16,
    art_fid_photo_only: bool = False,
    cache_dir: str | Path | None = None,
    enable_kid: bool = False,
    kid_max_gen: int = 200,
    kid_max_ref: int = 200,
    kid_subset_size: int = 50,
    kid_batch_size: int = 8,
    settings: dict | None = None,
    timings: dict | None = None,
    introstyle_result: dict[str, object] | None = None,
    runtime_observability_rows: list[dict[str, object]] | None = None,
):
    print("\n妫ｅ啯鎯?Generating Summary...")
    rows = []
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader: rows.append(r)
            
    if not rows: return
    introstyle_payload = introstyle_result.get("payload") if isinstance(introstyle_result, dict) else None
    introstyle_matrix = introstyle_payload.get("matrix_breakdown", {}) if isinstance(introstyle_payload, dict) else {}
    introstyle_analysis = introstyle_payload.get("analysis", {}) if isinstance(introstyle_payload, dict) else {}

    def to_f(x): return float(x) if x else 0.0

    fid_runner = None
    ref_fid_cache = {}
    artfid_feature_model = None
    artfid_lpips_loss = None
    artfid_ref_cache = {}
    if enable_art_fid:
        if style_real_paths is None:
            raise RuntimeError("ArtFID/FID requested but style_real_paths is missing.")
        try:
            fid_runner = _InceptionFeatRunner(
                device=device,
                batch_size=max(1, int(art_fid_batch_size)),
            )
        except Exception as e:
            fid_runner = None
            if FrechetInceptionDistance is None:
                raise RuntimeError(
                    "ArtFID/FID requested but no available backend. "
                    "Install torchmetrics[image] and torch-fidelity, or ensure Inception weights are available offline."
                ) from e
            print(f"  WARNING: Inception runner unavailable, fallback to torchmetrics FID: {e}")
        artfid_feature_model = load_artfid_feature_extractor(
            device=device,
            cache_dir=cache_dir,
        )
        artfid_lpips_loss = load_artfid_lpips(device=device)
    if enable_kid and KernelInceptionDistance is None:
        raise RuntimeError("KID requested (--eval_enable_kid) but torchmetrics is not available.")
    if enable_kid:
        if style_real_paths is None:
            raise RuntimeError("KID requested (--eval_enable_kid) but style_real_paths is missing.")
        # torchmetrics KID depends on torch-fidelity for Inception weights/features.
        try:
            import torch_fidelity  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "KID requested (--eval_enable_kid) but torch-fidelity is not available. "
                "Install it via `pip install torch-fidelity` (or `pip install torchmetrics[image]`)."
            ) from e

    matrix = defaultdict(lambda: defaultdict(list))
    for r in rows:
        matrix[r['src_style']][r['tgt_style']].append(r)

    src_name_to_path = {}
    if isinstance(source_style_paths, dict):
        for s_name, paths in source_style_paths.items():
            d = {}
            for p in (paths or []):
                try:
                    pp = Path(str(p))
                    d[pp.name] = str(pp)
                except Exception:
                    continue
            src_name_to_path[str(s_name)] = d

    # Build reusable path lists once per (src_style, tgt_style) pair to avoid
    # repeated path resolution / dict lookups across multiple metric families.
    pair_metric_paths = {}
    for src, targets in matrix.items():
        src_map = src_name_to_path.get(str(src), {})
        for tgt, items in targets.items():
            gen_paths = []
            src_paths = []
            for x in items:
                gp = _resolve_gen_image_path(out_dir, x.get('gen_image', ''))
                if gp is not None:
                    gen_paths.append(str(gp.resolve()))
                sp = src_map.get(str(x.get('src_image', '')))
                if sp:
                    src_paths.append(sp)
            pair_metric_paths[(str(src), str(tgt))] = {
                "gen_paths": gen_paths,
                "src_paths": src_paths,
                "ref_paths": list(style_real_paths.get(tgt, [])) if isinstance(style_real_paths, dict) else [],
            }

    matrix_json = {}
    all_pool = []
    transfer_pool = []
    identity_pool = []
    photo_transfer_pool = []

    for src, targets in matrix.items():
        matrix_json[src] = {}
        for tgt, items in targets.items():
            mean_content_lpips = np.mean([to_f(x['content_lpips']) for x in items])
            stats = {
                'count': len(items),
                'clip_dir': np.mean([to_f(x.get('clip_dir', x.get('clip_style', 0.0))) for x in items]),
                'clip_style': np.mean([to_f(x.get('clip_style', x.get('clip_dir', 0.0))) for x in items]),
                
                'content_lpips': mean_content_lpips,
                'clip_content': np.mean([to_f(x.get('clip_content', 0)) for x in items]),
            }
            intro_pair = None
            if isinstance(introstyle_matrix, dict):
                intro_pair = (introstyle_matrix.get(str(src), {}) or {}).get(str(tgt))
            if isinstance(intro_pair, dict):
                stats['introstyle_target_style_score'] = intro_pair.get('introstyle_target_style_score')
                stats['introstyle_source_style_score'] = intro_pair.get('introstyle_source_style_score')
                stats['introstyle_best_non_target_score'] = intro_pair.get('introstyle_best_non_target_score')
                stats['introstyle_style_margin'] = intro_pair.get('introstyle_style_margin')
            else:
                stats['introstyle_target_style_score'] = None
                stats['introstyle_source_style_score'] = None
                stats['introstyle_best_non_target_score'] = None
                stats['introstyle_style_margin'] = None
            should_compute_art_fid = bool(enable_art_fid and style_real_paths is not None)
            if should_compute_art_fid and art_fid_photo_only:
                should_compute_art_fid = (src.lower() == "photo" and src.lower() != tgt.lower())
            if should_compute_art_fid:
                try:
                    pair_paths = pair_metric_paths.get((str(src), str(tgt)), {})
                    gen_paths = list(pair_paths.get("gen_paths", []))
                    src_paths = list(pair_paths.get("src_paths", []))
                    ref_paths = list(pair_paths.get("ref_paths", []))
                    artfid_style_fid, artfid_content_lpips, art_fid = _compute_art_fid_for_pair(
                        gen_paths,
                        ref_paths,
                        src_paths,
                        feature_model=artfid_feature_model,
                        lpips_loss_fn=artfid_lpips_loss,
                        device=device,
                        batch_size=max(1, int(art_fid_batch_size)),
                        max_gen=max(1, int(art_fid_max_gen)),
                        max_ref=max(1, int(art_fid_max_ref)),
                        ref_cache=artfid_ref_cache,
                        ref_cache_key=str(tgt),
                    )
                    stats['art_fid_fid'] = artfid_style_fid
                    stats['art_fid_content_lpips'] = artfid_content_lpips
                    stats['art_fid'] = art_fid
                    fid_style = _compute_fid_for_pair(
                        gen_paths,
                        ref_paths,
                        runner=fid_runner,
                        device=device,
                        max_gen=max(1, int(art_fid_max_gen)),
                        max_ref=max(1, int(art_fid_max_ref)),
                        ref_cache=ref_fid_cache,
                        ref_cache_key=str(tgt),
                    )
                    stats['fid_style'] = fid_style
                    fid_baseline = _compute_fid_for_pair(
                        src_paths,
                        ref_paths,
                        runner=fid_runner,
                        device=device,
                        max_gen=max(1, int(art_fid_max_gen)),
                        max_ref=max(1, int(art_fid_max_ref)),
                        ref_cache=ref_fid_cache,
                        ref_cache_key=str(tgt),
                    )
                    stats['fid_baseline'] = fid_baseline
                    if fid_style is not None and fid_baseline is not None:
                        delta_fid = float(fid_baseline) - float(fid_style)
                        stats['delta_fid'] = delta_fid
                        stats['delta_fid_ratio'] = float(delta_fid / max(float(fid_baseline), 1e-8))
                    else:
                        stats['delta_fid'] = None
                        stats['delta_fid_ratio'] = None
                except Exception as e:
                    print(f"WARNING: ArtFID failed for {src}->{tgt}: {e}")
                    stats['fid_style'] = None
                    stats['art_fid_fid'] = None
                    stats['art_fid_content_lpips'] = None
                    stats['art_fid'] = None
                    stats['fid_baseline'] = None
                    stats['delta_fid'] = None
                    stats['delta_fid_ratio'] = None
            else:
                stats['fid_style'] = None
                stats['art_fid_fid'] = None
                stats['art_fid_content_lpips'] = None
                stats['art_fid'] = None
                stats['fid_baseline'] = None
                stats['delta_fid'] = None
                stats['delta_fid_ratio'] = None

            if enable_kid and style_real_paths is not None:
                try:
                    pair_paths = pair_metric_paths.get((str(src), str(tgt)), {})
                    gen_paths = list(pair_paths.get("gen_paths", []))
                    src_paths = list(pair_paths.get("src_paths", []))
                    ref_paths = list(pair_paths.get("ref_paths", []))
                    kid_style, kid_style_std = _compute_kid_for_pair(
                        gen_paths,
                        ref_paths,
                        device=device,
                        subset_size=max(2, int(kid_subset_size)),
                        max_gen=max(1, int(kid_max_gen)),
                        max_ref=max(1, int(kid_max_ref)),
                        batch_size=max(1, int(kid_batch_size)),
                    )
                    stats['kid_style'] = kid_style
                    stats['kid_style_std'] = kid_style_std
                    kid_baseline, kid_baseline_std = _compute_kid_for_pair(
                        src_paths,
                        ref_paths,
                        device=device,
                        subset_size=max(2, int(kid_subset_size)),
                        max_gen=max(1, int(kid_max_gen)),
                        max_ref=max(1, int(kid_max_ref)),
                        batch_size=max(1, int(kid_batch_size)),
                    )
                    stats['kid_baseline'] = kid_baseline
                    stats['kid_baseline_std'] = kid_baseline_std
                    if kid_style is not None and kid_baseline is not None:
                        delta_kid = float(kid_baseline) - float(kid_style)
                        stats['delta_kid'] = delta_kid
                        stats['delta_kid_ratio'] = float(delta_kid / max(float(kid_baseline), 1e-8))
                    else:
                        stats['delta_kid'] = None
                        stats['delta_kid_ratio'] = None
                except Exception as e:
                    print(f"WARNING: KID failed for {src}->{tgt}: {e}")
                    stats['kid_style'] = None
                    stats['kid_style_std'] = None
                    stats['kid_baseline'] = None
                    stats['kid_baseline_std'] = None
                    stats['delta_kid'] = None
                    stats['delta_kid_ratio'] = None
            else:
                stats['kid_style'] = None
                stats['kid_style_std'] = None
                stats['kid_baseline'] = None
                stats['kid_baseline_std'] = None
                stats['delta_kid'] = None
                stats['delta_kid_ratio'] = None
            
            matrix_json[src][tgt] = stats
            all_pool.append(stats)
            
            if src == tgt:
                identity_pool.append(stats)
            else:
                transfer_pool.append(stats)
                if src == 'photo':
                    photo_transfer_pool.append(stats)

    def pool_avg(pool, key, default=0.0):
        if not pool:
            return default
        return float(np.mean([x[key] for x in pool]))

    def build_pool_summary(pool, *, valid: bool | None = None):
        return {
            'clip_dir': pool_avg(pool, 'clip_dir'),
            'clip_style': pool_avg(pool, 'clip_style'),
            'clip_content': pool_avg(pool, 'clip_content'),
            'content_lpips': pool_avg(pool, 'content_lpips'),
            'introstyle_target_style_score': pool_avg([t for t in pool if t.get('introstyle_target_style_score') is not None], 'introstyle_target_style_score', default=None),
            'introstyle_source_style_score': pool_avg([t for t in pool if t.get('introstyle_source_style_score') is not None], 'introstyle_source_style_score', default=None),
            'introstyle_best_non_target_score': pool_avg([t for t in pool if t.get('introstyle_best_non_target_score') is not None], 'introstyle_best_non_target_score', default=None),
            'introstyle_style_margin': pool_avg([t for t in pool if t.get('introstyle_style_margin') is not None], 'introstyle_style_margin', default=None),
            'art_fid_content_lpips': pool_avg([t for t in pool if t.get('art_fid_content_lpips') is not None], 'art_fid_content_lpips', default=None),
            'fid_baseline': pool_avg([t for t in pool if t.get('fid_baseline') is not None], 'fid_baseline', default=None),
            'fid': pool_avg([t for t in pool if t.get('fid_style') is not None], 'fid_style', default=None),
            'delta_fid': pool_avg([t for t in pool if t.get('delta_fid') is not None], 'delta_fid', default=None),
            'delta_fid_ratio': pool_avg([t for t in pool if t.get('delta_fid_ratio') is not None], 'delta_fid_ratio', default=None),
            'art_fid_fid': pool_avg([t for t in pool if t.get('art_fid_fid') is not None], 'art_fid_fid', default=None),
            'art_fid': pool_avg([t for t in pool if t.get('art_fid') is not None], 'art_fid', default=None),
            'kid_baseline': pool_avg([t for t in pool if t.get('kid_baseline') is not None], 'kid_baseline', default=None),
            'kid': pool_avg([t for t in pool if t.get('kid_style') is not None], 'kid_style', default=None),
            'delta_kid': pool_avg([t for t in pool if t.get('delta_kid') is not None], 'delta_kid', default=None),
            'delta_kid_ratio': pool_avg([t for t in pool if t.get('delta_kid_ratio') is not None], 'delta_kid_ratio', default=None),
            **({'valid': bool(valid)} if valid is not None else {}),
        }

    def build_runtime_observability_summary(pool):
        if not pool:
            return None
        metric_keys = sorted(
            {
                str(key)
                for row in pool
                for key, value in row.items()
                if key not in {"src_style", "tgt_style"} and isinstance(value, (int, float))
            }
        )
        summary = {}
        for key in metric_keys:
            values = [float(row[key]) for row in pool if isinstance(row.get(key), (int, float))]
            if values:
                summary[key] = float(np.mean(values))
        return summary or None

    all_pairs_summary = build_pool_summary(all_pool)
    transfer_summary = build_pool_summary(transfer_pool)
    identity_summary = build_pool_summary(identity_pool)
    photo_summary = build_pool_summary(photo_transfer_pool, valid=len(photo_transfer_pool) > 0)
    runtime_all_pairs_summary = build_runtime_observability_summary(runtime_observability_rows or [])
    runtime_transfer_summary = build_runtime_observability_summary(
        [row for row in (runtime_observability_rows or []) if str(row.get("src_style", "")) != str(row.get("tgt_style", ""))]
    )
    runtime_identity_summary = build_runtime_observability_summary(
        [row for row in (runtime_observability_rows or []) if str(row.get("src_style", "")) == str(row.get("tgt_style", ""))]
    )
    runtime_photo_summary = build_runtime_observability_summary(
        [
            row
            for row in (runtime_observability_rows or [])
            if str(row.get("src_style", "")).lower() == "photo"
            and str(row.get("src_style", "")) != str(row.get("tgt_style", ""))
        ]
    )
    identity_intro = identity_summary.get('introstyle_target_style_score')
    if identity_intro is not None:
        for bucket in (all_pairs_summary, transfer_summary, photo_summary):
            target_intro = bucket.get('introstyle_target_style_score')
            if target_intro is not None:
                bucket['introstyle_delta_idt'] = float(target_intro) - float(identity_intro)

    summary = {
        'checkpoint': str(ckpt_path),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'metrics_note': {
            'clip_dir': "cos( CLIP(gen)-CLIP(src), CLIP(target_style_proto)-CLIP(src) ) - Measures edit direction.",
            'clip_style': "cos( CLIP(gen), CLIP(target_style_proto) ) - Measures absolute style similarity.",
            'clip_content': "cos( CLIP(gen), CLIP(src) ) - Measures semantic/content preservation.",
            'introstyle_target_style_score': "IntroStyle similarity to the held-out target-style bank.",
            'introstyle_source_style_score': "IntroStyle similarity to the held-out source-style bank.",
            'introstyle_best_non_target_score': "Strongest IntroStyle similarity among non-target banks.",
            'introstyle_style_margin': "introstyle_target_style_score - introstyle_best_non_target_score.",
            'introstyle_delta_idt': "Pool-level IntroStyle target score minus the identity pool target score.",
            'fid_baseline': "FID between source-domain images and target-style references.",
            'fid': "FID between generated images and target-style real references (Inception features).",
            'delta_fid': "fid_baseline - fid (higher is better).",
            'delta_fid_ratio': "delta_fid / fid_baseline (relative improvement ratio).",
            'art_fid_fid': "Academic ArtFID style term: FID computed with the official art-domain Inception checkpoint.",
            'art_fid_content_lpips': "Academic ArtFID content term: mean LPIPS-Alex between generated and source content images.",
            'art_fid': "Academic ArtFID: (1 + art_fid_fid) * (1 + art_fid_content_lpips).",
            'kid_baseline': "KID between source-domain images and target-style references (torchmetrics).",
            'kid': "KID between generated images and target-style references (torchmetrics).",
            'delta_kid': "kid_baseline - kid (higher is better).",
            'delta_kid_ratio': "delta_kid / kid_baseline (relative improvement ratio).",
        },
        'settings': dict(settings or {}),
        'timings_sec': {str(k): float(v) for k, v in sorted((timings or {}).items())},
        'matrix_breakdown': matrix_json,
        'analysis': {
            'all_pairs_overview': all_pairs_summary,
            'style_transfer_ability': transfer_summary,
            'identity_reconstruction': identity_summary,
            'photo_to_art_performance': photo_summary,
        },
        'runtime_observability': {
            'available': bool(runtime_observability_rows),
            'all_pairs_overview': runtime_all_pairs_summary,
            'style_transfer_ability': runtime_transfer_summary,
            'identity_reconstruction': runtime_identity_summary,
            'photo_to_art_performance': runtime_photo_summary,
        },
    }
    if isinstance(introstyle_result, dict):
        summary['introstyle_sidecar'] = {
            'metrics_csv': introstyle_result.get('metrics_csv'),
            'summary_json': introstyle_result.get('summary_json'),
        }
    if isinstance(introstyle_analysis, dict):
        summary['introstyle_analysis_sidecar'] = introstyle_analysis

    if bool((settings or {}).get("save_summary_grid", True)):
        summary['summary_grid_pending'] = True

    sum_path = out_dir / 'summary.json'
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {sum_path}")
    aggregate_artfid_path = write_targetwise_artfid_summary(sum_path)
    if aggregate_artfid_path is not None:
        print(f"Targetwise ArtFID summary saved: {aggregate_artfid_path}")

    summary_grid_path = None
    if bool((settings or {}).get("save_summary_grid", True)):
        print("  Saving summary grid... (disable with --no-save_summary_grid for pure throughput timing)")
        t0 = time.perf_counter()
        summary_grid_path = _save_summary_grid_png(rows, out_dir, style_order=style_order)
        _add_timing(timings, "summary_grid", t0)
        summary.pop('summary_grid_pending', None)
        summary['timings_sec'] = {str(k): float(v) for k, v in sorted((timings or {}).items())}
        if summary_grid_path is not None:
            summary['summary_grid_path'] = str(summary_grid_path)
        with open(sum_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary updated: {sum_path}")
    if fid_runner is not None:
        fid_runner.close()

if __name__ == '__main__':
    main() 
   
