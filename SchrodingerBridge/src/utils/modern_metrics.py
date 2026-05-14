from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _mean(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _safe_rel_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def list_style_images(style_dir: Path) -> list[Path]:
    return sorted([p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name.lower())


def load_metrics_rows(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@dataclass
class ModernMetricConfig:
    test_dir: Path
    device: str = "cuda"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    dino_model_name: str = "facebook/dinov2-small"
    cmmd_sigma: float = 10.0
    batch_size: int = 16


class ClipEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def encode_paths(self, paths: list[Path], batch_size: int) -> torch.Tensor:
        feats: list[torch.Tensor] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if hasattr(self.model, "get_image_features"):
                result = self.model.get_image_features(**inputs)
            else:
                result = self.model(**inputs)
            if isinstance(result, torch.Tensor):
                emb = result
            elif hasattr(result, "image_embeds") and result.image_embeds is not None:
                emb = result.image_embeds
            elif hasattr(result, "pooler_output") and result.pooler_output is not None:
                emb = result.pooler_output
            elif hasattr(result, "last_hidden_state") and result.last_hidden_state is not None:
                emb = result.last_hidden_state[:, 0, :]
            else:
                raise TypeError(f"Unsupported CLIP output type: {type(result)!r}")
            emb = F.normalize(emb.float(), p=2, dim=-1)
            feats.append(emb.cpu())
            for img in images:
                img.close()
        return torch.cat(feats, dim=0) if feats else torch.empty((0, 512), dtype=torch.float32)


class DinoStructureEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def structure_distance(self, src_paths: list[Path], gen_paths: list[Path], batch_size: int) -> float | None:
        values: list[float] = []
        for start in range(0, len(src_paths), batch_size):
            src_batch = src_paths[start:start + batch_size]
            gen_batch = gen_paths[start:start + batch_size]
            src_images = [Image.open(p).convert("RGB") for p in src_batch]
            gen_images = [Image.open(p).convert("RGB") for p in gen_batch]
            src_inputs = self.processor(images=src_images, return_tensors="pt")
            gen_inputs = self.processor(images=gen_images, return_tensors="pt")
            src_inputs = {k: v.to(self.device) for k, v in src_inputs.items()}
            gen_inputs = {k: v.to(self.device) for k, v in gen_inputs.items()}
            src_out = self.model(**src_inputs, output_hidden_states=True)
            gen_out = self.model(**gen_inputs, output_hidden_states=True)
            # Penultimate patch tokens; this is a stable spatial descriptor without fragile attention hooks.
            src_tokens = src_out.hidden_states[-2][:, 1:, :]
            gen_tokens = gen_out.hidden_states[-2][:, 1:, :]
            src_tokens = F.normalize(src_tokens.float(), p=2, dim=-1)
            gen_tokens = F.normalize(gen_tokens.float(), p=2, dim=-1)
            ssm_src = torch.bmm(src_tokens, src_tokens.transpose(1, 2))
            ssm_gen = torch.bmm(gen_tokens, gen_tokens.transpose(1, 2))
            batch_vals = F.mse_loss(ssm_gen, ssm_src, reduction="none").mean(dim=(1, 2))
            values.extend(batch_vals.detach().cpu().tolist())
            for img in src_images + gen_images:
                img.close()
        return _mean(values)


class VggGramEmbedder:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        base = self._load_vgg19().features.eval().to(self.device)
        for p in base.parameters():
            p.requires_grad_(False)
        self.model = base
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=VGG19_Weights.IMAGENET1K_V1.transforms().mean, std=VGG19_Weights.IMAGENET1K_V1.transforms().std),
            ]
        )
        # feature indices immediately after relu1_1, relu2_1, relu4_1, relu5_1
        self.capture_indices = {1: "relu1_1", 6: "relu2_1", 20: "relu4_1", 29: "relu5_1"}

    @staticmethod
    def _load_vgg19():
        try:
            return vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except RuntimeError as exc:
            if "invalid hash value" not in str(exc):
                raise
            cache_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "vgg19-dcbb9e9d.pth"
            if cache_path.exists():
                cache_path.unlink()
            return vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

    def _load_batch(self, paths: list[Path]) -> torch.Tensor:
        images = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            images.append(self.preprocess(img))
            img.close()
        return torch.stack(images, dim=0).to(self.device)

    def _extract_features(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        x = batch
        out: dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx in self.capture_indices:
                out[self.capture_indices[idx]] = x
        return out

    @staticmethod
    def _gram(feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        flat = feat.view(b, c, h * w)
        return torch.bmm(flat, flat.transpose(1, 2)) / float(h * w)

    @torch.inference_mode()
    def style_prototype(self, paths: list[Path], batch_size: int) -> dict[str, torch.Tensor]:
        grams_by_layer: dict[str, list[torch.Tensor]] = defaultdict(list)
        for start in range(0, len(paths), batch_size):
            batch = self._load_batch(paths[start:start + batch_size])
            feats = self._extract_features(batch)
            for layer, feat in feats.items():
                grams_by_layer[layer].append(self._gram(feat).cpu())
        return {layer: torch.cat(chunks, dim=0).mean(dim=0) for layer, chunks in grams_by_layer.items()}

    @torch.inference_mode()
    def gram_distances(self, gen_paths: list[Path], target_proto: dict[str, torch.Tensor], batch_size: int) -> tuple[float | None, float | None]:
        micro_vals: list[float] = []
        macro_vals: list[float] = []
        for start in range(0, len(gen_paths), batch_size):
            batch = self._load_batch(gen_paths[start:start + batch_size])
            feats = self._extract_features(batch)
            grams = {layer: self._gram(feat).cpu() for layer, feat in feats.items()}
            for idx in range(batch.shape[0]):
                micro = []
                macro = []
                for layer in ("relu1_1", "relu2_1"):
                    micro.append(F.mse_loss(grams[layer][idx], target_proto[layer]).item())
                for layer in ("relu4_1", "relu5_1"):
                    macro.append(F.mse_loss(grams[layer][idx], target_proto[layer]).item())
                micro_vals.append(float(sum(micro) / len(micro)))
                macro_vals.append(float(sum(macro) / len(macro)))
        return _mean(micro_vals), _mean(macro_vals)


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    dist = torch.cdist(x, y, p=2.0) ** 2
    return torch.exp(-dist / (2.0 * sigma * sigma))


def compute_cmmd(real_feats: torch.Tensor, gen_feats: torch.Tensor, sigma: float) -> float | None:
    n = real_feats.shape[0]
    m = gen_feats.shape[0]
    if n < 2 or m < 2:
        return None
    k_xx = rbf_kernel(real_feats, real_feats, sigma=sigma)
    k_yy = rbf_kernel(gen_feats, gen_feats, sigma=sigma)
    k_xy = rbf_kernel(real_feats, gen_feats, sigma=sigma)
    k_xx_sum = (k_xx.sum() - torch.trace(k_xx)) / float(n * (n - 1))
    k_yy_sum = (k_yy.sum() - torch.trace(k_yy)) / float(m * (m - 1))
    k_xy_sum = k_xy.sum() * (2.0 / float(n * m))
    cmmd_sq = k_xx_sum + k_yy_sum - k_xy_sum
    return float(torch.sqrt(torch.relu(cmmd_sq)).item())


def _pool_summary(pool: list[dict[str, Any]], *, valid: bool | None = None) -> dict[str, Any]:
    base = {
        "clip_dir": _mean([x.get("clip_dir") for x in pool]),
        "clip_style": _mean([x.get("clip_style") for x in pool]),
        "clip_content": _mean([x.get("clip_content") for x in pool]),
        "content_lpips": _mean([x.get("content_lpips") for x in pool]),
        "classifier_acc": _mean([x.get("classifier_acc") for x in pool]),
        "cmmd": _mean([x.get("cmmd") for x in pool]),
        "dino_structure": _mean([x.get("dino_structure") for x in pool]),
        "gram_micro": _mean([x.get("gram_micro") for x in pool]),
        "gram_macro": _mean([x.get("gram_macro") for x in pool]),
    }
    if valid is not None:
        base["valid"] = bool(valid)
    return base


def _update_metrics_note(metrics_note: dict[str, Any]) -> dict[str, Any]:
    note = dict(metrics_note or {})
    note.update(
        {
            "cmmd": "CLIP-MMD between generated images and target-style real references. Lower is better.",
            "dino_structure": "DINOv2 penultimate-patch self-similarity distance between generated and source content images. Lower is better.",
            "gram_micro": "Mean shallow VGG Gram MSE (relu1_1/relu2_1) against target-style prototype. Lower is better.",
            "gram_macro": "Mean deep VGG Gram MSE (relu4_1/relu5_1) against target-style prototype. Lower is better.",
        }
    )
    return note


def append_modern_metrics_to_summary(eval_dir: Path, cfg: ModernMetricConfig) -> dict[str, Any]:
    eval_dir = eval_dir.resolve()
    summary_path = eval_dir / "summary.json"
    metrics_path = eval_dir / "metrics.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary.json: {summary_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing metrics.csv: {metrics_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = load_metrics_rows(metrics_path)
    images_root = eval_dir / "images" if (eval_dir / "images").is_dir() else eval_dir

    pair_buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pair_buckets[(row["src_style"], row["tgt_style"])].append(row)

    style_dirs = {name: cfg.test_dir / name for name in {row["tgt_style"] for row in rows} | {row["src_style"] for row in rows}}
    ref_paths_by_style = {style: list_style_images(path) for style, path in style_dirs.items() if path.is_dir()}

    clip = ClipEmbedder(cfg.clip_model_name, cfg.device)
    dino = DinoStructureEmbedder(cfg.dino_model_name, cfg.device)
    vgg = VggGramEmbedder(cfg.device)

    clip_ref_cache: dict[str, torch.Tensor] = {}
    gram_ref_cache: dict[str, dict[str, torch.Tensor]] = {}
    for style, ref_paths in ref_paths_by_style.items():
        if ref_paths:
            clip_ref_cache[style] = clip.encode_paths(ref_paths, batch_size=cfg.batch_size)
            gram_ref_cache[style] = vgg.style_prototype(ref_paths, batch_size=cfg.batch_size)

    matrix = payload.get("matrix_breakdown", {}) or {}
    all_pool: list[dict[str, Any]] = []
    identity_pool: list[dict[str, Any]] = []
    transfer_pool: list[dict[str, Any]] = []
    photo_pool: list[dict[str, Any]] = []

    for (src_style, tgt_style), bucket in pair_buckets.items():
        gen_paths = [images_root / row["gen_image"] for row in bucket]
        src_paths = [cfg.test_dir / src_style / row["src_image"] for row in bucket]
        clip_style_val = _mean([float(row["clip_style"]) for row in bucket if row.get("clip_style")])
        clip_content_val = _mean([float(row["clip_content"]) for row in bucket if row.get("clip_content")])
        clip_dir_val = _mean([float(row["clip_dir"]) for row in bucket if row.get("clip_dir")])
        lpips_val = _mean([float(row["content_lpips"]) for row in bucket if row.get("content_lpips")])
        cls_vals = [row.get("class_correct") for row in bucket if row.get("class_correct") not in {None, "", "N/A"}]
        cls_acc = None
        if cls_vals:
            cls_acc = _mean([float(int(v)) for v in cls_vals])

        cmmd_val = None
        if tgt_style in clip_ref_cache and gen_paths:
            gen_clip = clip.encode_paths(gen_paths, batch_size=cfg.batch_size)
            cmmd_val = compute_cmmd(clip_ref_cache[tgt_style], gen_clip, sigma=cfg.cmmd_sigma)

        dino_val = dino.structure_distance(src_paths, gen_paths, batch_size=cfg.batch_size) if gen_paths else None

        gram_micro = None
        gram_macro = None
        if tgt_style in gram_ref_cache and gen_paths:
            gram_micro, gram_macro = vgg.gram_distances(gen_paths, gram_ref_cache[tgt_style], batch_size=cfg.batch_size)

        pair_stats = matrix.setdefault(src_style, {}).setdefault(tgt_style, {})
        pair_stats.update(
            {
                "count": len(bucket),
                "clip_dir": clip_dir_val,
                "clip_style": clip_style_val,
                "clip_content": clip_content_val,
                "content_lpips": lpips_val,
                "classifier_acc": cls_acc,
                "cmmd": cmmd_val,
                "dino_structure": dino_val,
                "gram_micro": gram_micro,
                "gram_macro": gram_macro,
            }
        )
        all_pool.append(pair_stats)
        if src_style == tgt_style:
            identity_pool.append(pair_stats)
        else:
            transfer_pool.append(pair_stats)
            if src_style == "photo":
                photo_pool.append(pair_stats)

    payload["metrics_note"] = _update_metrics_note(payload.get("metrics_note", {}))
    payload["analysis"] = dict(payload.get("analysis", {}) or {})
    payload["analysis"]["all_pairs_overview"] = _pool_summary(all_pool)
    payload["analysis"]["style_transfer_ability"] = _pool_summary(transfer_pool)
    payload["analysis"]["identity_reconstruction"] = _pool_summary(identity_pool)
    payload["analysis"]["photo_to_art_performance"] = _pool_summary(photo_pool, valid=bool(photo_pool))
    payload["matrix_breakdown"] = matrix
    payload["modern_metric_runtime"] = {
        "clip_model_name": cfg.clip_model_name,
        "dino_model_name": cfg.dino_model_name,
        "cmmd_sigma": cfg.cmmd_sigma,
        "test_dir": str(cfg.test_dir),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
