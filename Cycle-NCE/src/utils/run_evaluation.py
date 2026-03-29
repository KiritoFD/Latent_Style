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

try:
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from scipy import linalg
# Project imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.inference import LGTInference, load_vae, encode_image, decode_latent
from utils.classify import DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT, load_eval_image_classifier
from utils.artfid_metric import (
    compute_artfid_content_distance_from_paths,
    compute_artfid_fid_from_paths,
    load_artfid_feature_extractor,
    load_artfid_lpips,
)

# KID (official implementation via torchmetrics)
try:
    from torchmetrics.image.kid import KernelInceptionDistance
except Exception:
    KernelInceptionDistance = None

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None

def _safe_to_eval_device(batch, device: str):
    """
    Move processor outputs to device when possible (BatchEncoding supports .to()).
    """
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    return batch


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


def _infer_style_names_from_generated_files(files: list[Path]) -> list[str]:
    styles = set()
    for p in files:
        stem = p.stem
        if "_to_" not in stem:
            continue
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
            "--max_src_samples", str(args.max_src_samples),
            "--max_ref_compare", str(args.max_ref_compare),
            "--max_ref_cache", str(args.max_ref_cache),
            "--ref_feature_batch_size", str(args.ref_feature_batch_size),
            "--batch_size", str(args.batch_size),
            "--image_classifier_path", str(args.image_classifier_path),
            "--clip_model_name", str(args.clip_model_name),
            "--clip_modelscope_id", str(args.clip_modelscope_id),
            "--clip_modelscope_cache_dir", str(args.clip_modelscope_cache_dir),
            "--clip_hf_cache_dir", str(args.clip_hf_cache_dir),
        ]
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

        print(f"\n[Auto] Running: {ckpt_path}")
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_dir', nargs='?', default=None, help="One-shot mode: target full_eval directory (reuse existing images).")
    parser.add_argument('--checkpoint', type=str, default=None, help="Single-checkpoint mode: path to checkpoint")
    parser.add_argument('--output', type=str, default=None, help="Single-checkpoint mode: output directory")
    parser.add_argument('--style_subdirs', type=str, default="", help="Optional comma-separated style names for reuse-only eval without checkpoint")
    parser.add_argument('--config', type=str, default="../config.json", help="Auto mode config path")
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default="../eval_cache", help="Directory to store shared feature caches")
    parser.add_argument('--num_steps', type=int, default=15)
    parser.add_argument('--step_size', type=float, default=1.0)
    parser.add_argument('--style_strength', type=float, default=None, help="Global style strength in [0,1]; default uses checkpoint config")
    parser.add_argument('--max_src_samples', type=int, default=30, help="Max source images per style; <=0 means all")
    parser.add_argument('--max_ref_compare', type=int, default=50, help="Max refs for LPIPS style compare; <=0 means all cached refs")
    parser.add_argument('--max_ref_cache', type=int, default=256, help="Max reference images per style used for cache/features; <=0 means all")
    parser.add_argument('--ref_feature_batch_size', type=int, default=64, help="Batch size for reference feature extraction")
    parser.add_argument('--batch_size', type=int, default=20, help="Batch size increased due to offloading")
    parser.add_argument('--force_regen', action='store_true', help="Force regenerate evaluation outputs/metrics (does not rebuild global ref cache)")
    parser.add_argument('--force_regen_ref_cache', action='store_true', help="Force rebuild global reference-feature cache only")
    parser.add_argument('--ref_cache_lock_timeout', type=int, default=900, help="Seconds to wait for another process building reference cache")
    parser.add_argument('--image_classifier_path', type=str, default=str(DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT), help="Path to robust image classifier checkpoint for evaluation")
    parser.add_argument('--clip_model_name', type=str, default="openai/clip-vit-base-patch32", help="HF/local CLIP model name or local directory")
    parser.add_argument('--clip_modelscope_id', type=str, default="", help="Optional ModelScope model id for CLIP fallback")
    parser.add_argument('--clip_modelscope_cache_dir', type=str, default="", help="Optional ModelScope cache directory")
    parser.add_argument('--clip_hf_cache_dir', type=str, default="", help="HuggingFace cache dir for CLIP; default uses <cache_dir>/hf")
    parser.add_argument('--clip_allow_network', action='store_true', help="Allow online model fetch if local cache is missing (default off)")
    parser.add_argument(
        '--clip_backend',
        type=str,
        default="openai",
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
    parser.add_argument('--eval_classifier_only', action='store_true', help="Run only classifier evaluation (skip LPIPS/CLIP)")
    parser.add_argument('--eval_disable_lpips', action='store_true', help="Skip LPIPS metrics (keep CLIP)")
    parser.add_argument(
        '--eval_only_lpips_clip_style',
        action='store_true',
        help="Compute only content LPIPS and CLIP style similarity (skip style LPIPS, clip_dir, clip_content, classifier).",
    )
    parser.add_argument(
        '--eval_enable_art_fid',
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ArtFID/FID metric (default: enabled). Use --no-eval_enable_art_fid to disable.",
    )
    parser.add_argument('--eval_art_fid_max_gen', type=int, default=200, help="Max generated images per pair for FID_style")
    parser.add_argument('--eval_art_fid_max_ref', type=int, default=200, help="Max target-style reference images per pair for FID_style")
    parser.add_argument('--eval_art_fid_batch_size', type=int, default=16, help="Batch size for inception feature extraction in ArtFID")
    parser.add_argument('--eval_art_fid_photo_only', action='store_true', help="Compute ArtFID/FID only for photo->art directions")
    parser.add_argument(
        '--eval_enable_kid',
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable KID metric (default: enabled). Use --no-eval_enable_kid to disable.",
    )
    parser.add_argument('--eval_kid_max_gen', type=int, default=200, help="Max generated images per pair for KID")
    parser.add_argument('--eval_kid_max_ref', type=int, default=200, help="Max target-style reference images per pair for KID")
    parser.add_argument('--eval_kid_subset_size', type=int, default=50, help="Subset size for KID (torchmetrics)")
    parser.add_argument('--eval_kid_batch_size', type=int, default=8, help="Batch size for KID image loading/inception")
    parser.add_argument('--eval_lpips_chunk_size', type=int, default=2, help="LPIPS chunk size for conservative VRAM usage")
    parser.add_argument('--eval_lpips_no_cpu_fallback', action='store_true', help="Disable CPU fallback when LPIPS CUDA OOM occurs")
    parser.add_argument('--reuse_generated', action='store_true', help="Reuse existing generated images in output dir/images (or legacy output dir) and skip generation")
    parser.add_argument('--generation_only', action='store_true', help="Only generate translated images, skip all evaluation metrics")
    args = parser.parse_args()
    if args.eval_classifier_only and args.eval_only_lpips_clip_style:
        raise ValueError("--eval_classifier_only conflicts with --eval_only_lpips_clip_style")

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
    
    # Thread Pool for Async I/O
    io_pool = ThreadPoolExecutor(max_workers=4)
    
    checkpoint_path: Path | None = None
    cfg = {}
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {})
    else:
        print("Single-run eval in reuse-only mode (no checkpoint).")
    
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
    test_dir = resolved_test_dir

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
        print(f"\nPhase 1: Reuse generated images from {images_dir}")
        reuse_files = _list_reuse_generated_files(out_dir)
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
                    "gen_img": gen_img,
                    "gen_name": p.name,
                }
            )
        print(f"  Reused {len(generated_buffer)} generated images")

    if not generated_buffer:
        if checkpoint_path is None:
            raise RuntimeError("No reusable images found and no checkpoint provided. Cannot run generation phase.")
        print(f"\nPhase 1: Generation (Batch Size {args.batch_size})")

        lgt = LGTInference(
            str(checkpoint_path),
            device=device,
            num_steps=args.num_steps,
            step_size=args.step_size,
            style_strength=args.style_strength,
        )
        vae = load_vae(device)
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
                        imgs_gen_cpu = imgs_gen.cpu()

                        for i in range(len(batch_info)):
                            src_item = batch_info[i]
                            out_name = f"{src_item['style_name']}_{src_item['path'].stem}_to_{tgt_name}.jpg"
                            out_path = images_dir / out_name
                            out_rel = Path("images") / out_name

                            # Async Save
                            io_pool.submit(save_image_task, imgs_gen_cpu[i], out_path)

                            # Store for Phase 2
                            generated_buffer.append({
                                'src_path': src_item['path'],
                                'src_style': src_item['style_name'],
                                'tgt_style_name': tgt_name,
                                'tgt_style_id': tgt_id,
                                'gen_img': imgs_gen_cpu[i], # Keep in RAM
                                'gen_name': out_rel.as_posix()
                            })

        # Unload Generation Models
        del lgt, vae
        torch.cuda.empty_cache()
        gc.collect()
        print("  Generation models unloaded")
    if not generated_buffer:
        raise RuntimeError(f"No generated samples to evaluate in {out_dir}")

    if args.generation_only:
        print("\nGeneration-only mode enabled: skip Phase 2 metrics/classifier/LPIPS/CLIP.")
        io_pool.shutdown(wait=True)
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
        _save_summary_grid_png(grid_rows, out_dir, style_order=list(style_subdirs))
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        summary = {
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else "(reuse-only:no-checkpoint)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "generation_only",
            "generated_count": int(len(generated_buffer)),
            "output_dir": str(out_dir),
            "note": "Metrics are intentionally skipped. Run evaluation later with --reuse_generated.",
        }
        sum_path = out_dir / "summary.json"
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved: {sum_path}")
        return

    # ==========================================
    # PHASE 2: EVALUATION (VGG + CLIP)
    # ==========================================
    print(f"\n妫ｅ啯鐣?Phase 2: Evaluation")
    
    # Load image classifier (single evaluation path)
    image_classifier = None
    classifier_label_names = list(style_subdirs)

    if args.image_classifier_path:
        try:
            classifier_path_candidates = [str(args.image_classifier_path)]
            cfg_classifier = str(cfg.get("training", {}).get("full_eval_image_classifier_path", "")).strip()
            if cfg_classifier:
                classifier_path_candidates.append(cfg_classifier)
            classifier_path_candidates.extend(
                [
                    str(cache_dir / "eval_style_image_classifier.pt"),
                    str(Path(__file__).resolve().parents[2] / "eval_cache" / "eval_style_image_classifier.pt"),
                ]
            )

            base_dirs = [
                Path.cwd(),
                out_dir,
                cache_dir,
                Path(__file__).resolve().parent,
                Path(__file__).resolve().parents[1],
                Path(__file__).resolve().parents[2],
            ]
            if checkpoint_path is not None:
                base_dirs.insert(1, checkpoint_path.parent.resolve())

            resolved_ckpt = None
            for raw in classifier_path_candidates:
                resolved_ckpt = _resolve_existing_path(raw, base_dirs)
                if resolved_ckpt is not None:
                    break

            if resolved_ckpt is None:
                print("  WARNING: image style classifier checkpoint not found. Tried:")
                for raw in classifier_path_candidates:
                    print(f"    - {raw}")
            else:
                image_classifier = load_eval_image_classifier(resolved_ckpt, device=device)
                classifier_label_names = list(image_classifier.classes)
                print(f"  Loaded image style classifier: {resolved_ckpt} (classes={len(classifier_label_names)})")
        except Exception as e:
            print(f"  WARNING: failed to load image style classifier: {e}")
            image_classifier = None

    # Skip other metrics if classifier-only mode
    run_full_metrics = not args.eval_classifier_only
    only_lpips_clip_style = bool(args.eval_only_lpips_clip_style)

    # Load Evaluators
    loss_fn = None
    clip_model = None
    clip_processor = None
    has_clip = False
    clip_backend = str(getattr(args, "clip_backend", "hf")).strip().lower()
    clip_preprocess = None  # OpenAI CLIP preprocess
    clip_encode_pils = None  # Callable[[list[PIL.Image]], Tensor[B,D]] on device
    if clip_backend == "openai":
        clip_model_tag = f"openai:{str(getattr(args, 'clip_openai_model', 'ViT-B/32')).strip() or 'ViT-B/32'}"
    elif clip_backend == "hf":
        clip_model_tag = str(args.clip_model_name).strip() or "openai/clip-vit-base-patch32"
    else:
        clip_model_tag = "none"
    to_pil = ToPILImage()

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
            clip_model_name = str(args.clip_model_name).strip() or "openai/clip-vit-base-patch32"
            try:
                from transformers import CLIPModel, CLIPProcessor

                clip_sources = []
                local_only = (not bool(args.clip_allow_network))
                model_name_raw = str(args.clip_model_name).strip()
                if model_name_raw:
                    # 1) Direct local path.
                    model_name_path = Path(model_name_raw).expanduser()
                    if model_name_path.exists():
                        clip_sources.append(str(model_name_path.resolve()))

                    # 2) HF cache snapshot path.
                    local_snapshot = _find_local_hf_snapshot(hf_cache_dir, clip_model_name)
                    if local_snapshot:
                        clip_sources.append(local_snapshot)

                    # 3) Remote repo-id fallback only when network is explicitly allowed.
                    if not local_only:
                        clip_sources.append(clip_model_name)

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
            if clip_backend == "openai":
                if clip_preprocess is None:
                    raise RuntimeError("OpenAI CLIP preprocess missing")

                def clip_encode_pils(pils):  # noqa: ANN001
                    imgs = torch.stack([clip_preprocess(im) for im in pils], dim=0).to(device)
                    feats = clip_model.encode_image(imgs)
                    feats = feats.to(dtype=torch.float32)
                    if feats.ndim == 1:
                        feats = feats.unsqueeze(0)
                    return feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            else:

                def clip_encode_pils(pils):  # noqa: ANN001
                    inputs = _safe_to_eval_device(clip_processor(images=pils, return_tensors='pt'), device)
                    out = clip_model.get_image_features(**inputs)
                    feats = _extract_clip_embeddings(out).to(device, dtype=torch.float32)
                    if feats.ndim == 1:
                        feats = feats.unsqueeze(0)
                    return feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)

    # Prepare Reference Features (Cache)
    style_sig = ",".join(style_subdirs)
    dataset_sig = f"{str(test_dir.resolve())}|{style_sig}|{clip_model_tag}|v2"
    dataset_hash = hashlib.md5(dataset_sig.encode()).hexdigest()[:10]
    max_ref_cache = int(args.max_ref_cache)
    max_ref_cache_tag = "all" if max_ref_cache <= 0 else str(max_ref_cache)
    cache_file = cache_dir / f"ref_feats_{dataset_hash}_m{max_ref_cache_tag}.pt"
    lock_file = cache_file.with_suffix(cache_file.suffix + ".lock")

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
                            # Keep raw PILs for CLIP so CLIPProcessor applies its own canonical resize/crop.
                            batch_pils = [Image.open(img_path).convert('RGB') for img_path in batch_paths]
                            with torch.no_grad():
                                c_emb = None
                                if has_clip and clip_model is not None:
                                    c_emb = clip_encode_pils(batch_pils).detach().cpu()

                            for i, img_path in enumerate(batch_paths):
                                ref_features[style_id].append({
                                    'path': str(img_path),
                                    'clip': c_emb[i:i+1] if c_emb is not None else None
                                })
                        except Exception as e:
                            print(f"Skipping batch {b_start}-{b_start + len(batch_paths)} in {style_name}: {e}")

                tmp_cache = cache_file.with_suffix(cache_file.suffix + f".tmp.{os.getpid()}")
                torch.save(ref_features, tmp_cache)
                os.replace(tmp_cache, cache_file)
                print(f"Global reference cache saved: {cache_file}")
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

    # Cache source images/CLIP embeddings to avoid repeated work across many target styles.
    src_img_cache = {}   # abs src path -> Tensor[3,256,256] on CPU (LPIPS path)
    src_pil_cache = {}   # abs src path -> PIL.Image (CLIP path)
    src_clip_cache = {}  # abs src path -> Tensor[D] on CPU

    csv_path = out_dir / 'metrics.csv'
    # Re-evaluation on reused images should overwrite metrics to avoid mixing old/new classifier outputs.
    csv_mode = 'w' if args.force_regen or args.reuse_generated or not csv_path.exists() else 'a'
    csv_file = open(csv_path, csv_mode, newline='')
    columns = [
        'src_style',
        'tgt_style',
        'src_image',
        'gen_image',
        'content_lpips',
        
        'clip_dir',
        'clip_style',
        'clip_content',
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
        
        gen_imgs_cpu = torch.stack([item['gen_img'] for item in batch_items]).contiguous()
        gen_imgs = gen_imgs_cpu.to(device, non_blocking=True)

        src_tensors = []
        src_keys = []
        for item in batch_items:
            src_key = str(Path(item['src_path']).resolve())
            src_keys.append(src_key)
            cached = src_img_cache.get(src_key)
            if cached is None:
                cached = _load_eval_image_tensor(Path(item['src_path']))
                src_img_cache[src_key] = cached
            src_tensors.append(cached)
        src_imgs_cpu = torch.stack(src_tensors, dim=0).contiguous()
        src_imgs = src_imgs_cpu.to(device, non_blocking=True)
        
        with torch.no_grad():
            # 1. Content LPIPS (Skip if classifier only)
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

            # 2. CLIP Features (Skip if classifier only)
            gen_clips = None
            src_clips = None
            c_clip_scores = [0.0] * len(batch_items)
            
            if has_clip and clip_model is not None:
                # Gen CLIP
                pil_gens = [to_pil(img.float()) for img in gen_imgs_cpu]
                gen_clips = clip_encode_pils(pil_gens)
                if not only_lpips_clip_style:
                    # Src CLIP (cache by source path; source repeats across many target styles)
                    miss_indices = [i for i, k in enumerate(src_keys) if k not in src_clip_cache]
                    if miss_indices:
                        pil_srcs_miss = []
                        for i in miss_indices:
                            src_path = str(Path(batch_items[i]['src_path']).resolve())
                            pil_img = src_pil_cache.get(src_path)
                            if pil_img is None:
                                pil_img = Image.open(batch_items[i]['src_path']).convert('RGB')
                                src_pil_cache[src_path] = pil_img
                            pil_srcs_miss.append(pil_img)
                        src_miss = clip_encode_pils(pil_srcs_miss)
                        src_miss_cpu = src_miss.detach().cpu()
                        for j, idx in enumerate(miss_indices):
                            src_clip_cache[src_keys[idx]] = src_miss_cpu[j].clone()
                    src_clips = torch.stack([src_clip_cache[k] for k in src_keys], dim=0).to(device, dtype=torch.float32)
                    c_clip_scores = F.cosine_similarity(gen_clips, src_clips).cpu().float().numpy()

            # 3. Classifier Predictions
            pred_indices = [-1] * len(batch_items)
            if image_classifier is not None and (not only_lpips_clip_style):
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

                # --- CLIP metrics ---
                # clip_dir: directional similarity in edit space.
                # clip_style: absolute similarity to target style prototype.
                s_clip_dir = 0.0
                s_clip_style = 0.0
                if only_lpips_clip_style:
                    if has_clip and gen_clips is not None and tgt_id in ref_clip_prototypes:
                        tgt_proto = ref_clip_prototypes[tgt_id]  # [1, D]
                        gen_emb = gen_clips[i:i+1]              # [1, D]
                        if gen_emb.shape[-1] == tgt_proto.shape[-1]:
                            s_clip_style = F.cosine_similarity(gen_emb, tgt_proto).item()
                elif has_clip and gen_clips is not None and src_clips is not None and tgt_id in ref_clip_prototypes:
                    tgt_proto = ref_clip_prototypes[tgt_id]  # [1, D]
                    gen_emb = gen_clips[i:i+1]              # [1, D]
                    src_emb = src_clips[i:i+1]              # [1, D]
                    if gen_emb.shape[-1] == tgt_proto.shape[-1] == src_emb.shape[-1]:
                        dir_gen = gen_emb - src_emb
                        dir_tgt = tgt_proto - src_emb
                        dir_gen = dir_gen / (dir_gen.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                        dir_tgt = dir_tgt / (dir_tgt.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                        s_clip_dir = F.cosine_similarity(dir_gen, dir_tgt).item()
                        s_clip_style = F.cosine_similarity(gen_emb, tgt_proto).item()

                writer.writerow({
                    'src_style': item['src_style'],
                    'tgt_style': item['tgt_style_name'],
                    'src_image': item['src_path'].name,
                    'gen_image': item['gen_name'],
                    'content_lpips': c_lpips_vals[i],
                    
                    'clip_dir': s_clip_dir,
                    'clip_style': s_clip_style,
                    'clip_content': c_clip_scores[i],
                    'pred_style': pred_style_name,
                    'class_correct': class_correct
                })
            
            csv_file.flush()

    csv_file.close()
    io_pool.shutdown(wait=True)
    
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
):
    print("\n妫ｅ啯鎯?Generating Summary...")
    rows = []
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader: rows.append(r)
            
    if not rows: return

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
    
    # Classification Stats
    y_true = []
    y_pred = []

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

    def build_pool_summary(pool, *, valid: bool | None = None):
        return {
            'clip_dir': pool_avg(pool, 'clip_dir'),
            'clip_style': pool_avg(pool, 'clip_style'),
            'clip_content': pool_avg(pool, 'clip_content'),
            'content_lpips': pool_avg(pool, 'content_lpips'),
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
            'classifier_acc': pool_avg([t for t in pool if t['classifier_acc'] is not None], 'classifier_acc'),
            **({'valid': bool(valid)} if valid is not None else {}),
        }

    summary = {
        'checkpoint': str(ckpt_path),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'metrics_note': {
            'clip_dir': "cos( CLIP(gen)-CLIP(src), CLIP(target_style_proto)-CLIP(src) ) - Measures edit direction.",
            'clip_style': "cos( CLIP(gen), CLIP(target_style_proto) ) - Measures absolute style similarity.",
            'clip_content': "cos( CLIP(gen), CLIP(src) ) - Measures semantic/content preservation.",
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
        'matrix_breakdown': matrix_json,
        'analysis': {
            'all_pairs_overview': build_pool_summary(all_pool),
            'style_transfer_ability': build_pool_summary(transfer_pool),
            'identity_reconstruction': build_pool_summary(identity_pool),
            'photo_to_art_performance': build_pool_summary(photo_transfer_pool, valid=len(photo_transfer_pool) > 0),
        },
        'classification_report': cls_report,
        'detailed_style_metrics': detailed_metrics
    }
    
    sum_path = out_dir / 'summary.json'
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"闁?Summary saved: {sum_path}")
    _save_summary_grid_png(rows, out_dir, style_order=style_order)
    if fid_runner is not None:
        fid_runner.close()

if __name__ == '__main__':
    main() 
   
