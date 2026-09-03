from __future__ import annotations

"""Build id-only multi-prototype style-memory adapters.

The reference-memory generation probe showed that explicit target-style latents
can lift style, but using a reference at test time is not the final protocol.
This script compresses internal training-set style features into an adapter-side
prototype bank that is selected only by style id at inference.
"""

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_style_embedding_distill import (  # noqa: E402
    _load_checkpoint_model,
    _load_latent,
    _memory_tier_eval_batch_size,
    _run_full_eval,
    _style_latent_index,
)
from run_style_embedding_mainline_calibration import _apply_style_adapter  # noqa: E402


@dataclass(frozen=True)
class BankAdapterRecipe:
    name: str
    mode: str
    num_prototypes: int
    blend: float
    route_strength: float = 0.0
    route_temperature: float = 8.0
    residual_strength: float = 0.0
    residual_tanh_scale: float = 0.55
    residual_highpass_kernel: int = 1
    residual_center_base: bool = True
    residual_center_content: bool = False
    residual_gate_gamma: float = 0.0
    residual_gate_floor: float = 0.20
    residual_gate_kernel: int = 5
    typed_prototypes_per_type: int = 0
    typed_gate_gamma: float = 2.5
    typed_gate_temperature: float = 1.0
    typed_prior_scale: float = 0.65
    contrastive_logit_scale: float = 2.0
    contrastive_highpass_weight: float = 0.15
    contrastive_edge_weight: float = 0.10
    max_samples_per_style: int = 128
    batch_size: int = 16
    highpass_kernel: int = 5
    highpass_boost: float = 1.0
    temperature: float = 1.0


RECIPES = [
    BankAdapterRecipe(
        name="bm00_hightex_k4_blend65",
        mode="high_texture",
        num_prototypes=4,
        blend=0.65,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="bm01_diverse_k4_blend65",
        mode="diverse_low",
        num_prototypes=4,
        blend=0.65,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="bm02_hightex_k4_boost_blend75",
        mode="high_texture",
        num_prototypes=4,
        blend=0.75,
        highpass_boost=1.12,
        temperature=0.75,
    ),
    BankAdapterRecipe(
        name="br00_route_hightex_k4_s45",
        mode="high_texture",
        num_prototypes=4,
        blend=0.0,
        route_strength=0.45,
        route_temperature=8.0,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="br01_route_hightex_k4_s65",
        mode="high_texture",
        num_prototypes=4,
        blend=0.0,
        route_strength=0.65,
        route_temperature=8.0,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="br02_route_diverse_k4_s65",
        mode="diverse_low",
        num_prototypes=4,
        blend=0.0,
        route_strength=0.65,
        route_temperature=8.0,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="mr00_residual_hightex_k4_s22",
        mode="high_texture",
        num_prototypes=4,
        blend=0.0,
        route_strength=0.0,
        route_temperature=8.0,
        residual_strength=0.22,
        residual_tanh_scale=0.55,
        residual_highpass_kernel=1,
        residual_center_base=True,
        residual_gate_gamma=0.0,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="mr01_residual_hightex_k4_hp_s32",
        mode="high_texture",
        num_prototypes=4,
        blend=0.0,
        route_strength=0.0,
        route_temperature=8.0,
        residual_strength=0.32,
        residual_tanh_scale=0.45,
        residual_highpass_kernel=5,
        residual_center_base=True,
        residual_gate_gamma=4.0,
        residual_gate_floor=0.25,
        residual_gate_kernel=5,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="mr02_residual_contentdir_k4_s18",
        mode="high_texture",
        num_prototypes=4,
        blend=0.0,
        route_strength=0.0,
        route_temperature=8.0,
        residual_strength=0.18,
        residual_tanh_scale=0.45,
        residual_highpass_kernel=1,
        residual_center_base=False,
        residual_center_content=True,
        residual_gate_gamma=0.0,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="mr03_residual_contentdir_k4_hp_s24",
        mode="high_texture",
        num_prototypes=4,
        blend=0.0,
        route_strength=0.0,
        route_temperature=8.0,
        residual_strength=0.24,
        residual_tanh_scale=0.40,
        residual_highpass_kernel=5,
        residual_center_base=False,
        residual_center_content=True,
        residual_gate_gamma=4.0,
        residual_gate_floor=0.25,
        residual_gate_kernel=5,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="mt00_typed_fet_k6_s18",
        mode="typed_flat_edge_texton",
        num_prototypes=6,
        typed_prototypes_per_type=2,
        blend=0.0,
        residual_strength=0.18,
        residual_tanh_scale=0.45,
        residual_highpass_kernel=1,
        residual_center_base=False,
        residual_center_content=True,
        residual_gate_gamma=0.0,
        typed_gate_gamma=2.4,
        typed_gate_temperature=0.9,
        typed_prior_scale=0.65,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="mt01_typed_fet_k6_hp_s24",
        mode="typed_flat_edge_texton",
        num_prototypes=6,
        typed_prototypes_per_type=2,
        blend=0.0,
        residual_strength=0.24,
        residual_tanh_scale=0.40,
        residual_highpass_kernel=5,
        residual_center_base=False,
        residual_center_content=True,
        residual_gate_gamma=4.0,
        residual_gate_floor=0.25,
        residual_gate_kernel=5,
        typed_gate_gamma=2.8,
        typed_gate_temperature=0.8,
        typed_prior_scale=0.75,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="mt02_typed_uniform_fet_k6_s20",
        mode="typed_flat_edge_texton",
        num_prototypes=6,
        typed_prototypes_per_type=2,
        blend=0.0,
        route_temperature=0.10,
        residual_strength=0.20,
        residual_tanh_scale=0.45,
        residual_highpass_kernel=1,
        residual_center_base=False,
        residual_center_content=True,
        residual_gate_gamma=0.0,
        typed_gate_gamma=2.4,
        typed_gate_temperature=0.9,
        typed_prior_scale=0.65,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="mt03_typed_uniform_fet_k6_hp_s24",
        mode="typed_flat_edge_texton",
        num_prototypes=6,
        typed_prototypes_per_type=2,
        blend=0.0,
        route_temperature=0.10,
        residual_strength=0.24,
        residual_tanh_scale=0.40,
        residual_highpass_kernel=5,
        residual_center_base=False,
        residual_center_content=True,
        residual_gate_gamma=4.0,
        residual_gate_floor=0.25,
        residual_gate_kernel=5,
        typed_gate_gamma=2.8,
        typed_gate_temperature=0.8,
        typed_prior_scale=0.75,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="ma00_stylepure_k6_s20",
        mode="style_contrastive",
        num_prototypes=6,
        blend=0.0,
        route_temperature=8.0,
        residual_strength=0.20,
        residual_tanh_scale=0.45,
        residual_highpass_kernel=1,
        residual_center_base=False,
        residual_center_content=True,
        contrastive_logit_scale=2.0,
        contrastive_highpass_weight=0.12,
        contrastive_edge_weight=0.08,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="ma01_typed_stylepure_k9_s22",
        mode="typed_style_contrastive",
        num_prototypes=9,
        typed_prototypes_per_type=3,
        blend=0.0,
        route_temperature=8.0,
        residual_strength=0.22,
        residual_tanh_scale=0.42,
        residual_highpass_kernel=1,
        residual_center_base=False,
        residual_center_content=True,
        typed_gate_gamma=2.6,
        typed_gate_temperature=0.85,
        typed_prior_scale=0.80,
        contrastive_logit_scale=2.2,
        contrastive_highpass_weight=0.12,
        contrastive_edge_weight=0.08,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="ma02_typed_stylepure_uniform_hp_k9_s24",
        mode="typed_style_contrastive",
        num_prototypes=9,
        typed_prototypes_per_type=3,
        blend=0.0,
        route_temperature=0.10,
        residual_strength=0.24,
        residual_tanh_scale=0.40,
        residual_highpass_kernel=5,
        residual_center_base=False,
        residual_center_content=True,
        residual_gate_gamma=4.0,
        residual_gate_floor=0.25,
        residual_gate_kernel=5,
        typed_gate_gamma=2.8,
        typed_gate_temperature=0.80,
        typed_prior_scale=0.85,
        contrastive_logit_scale=2.4,
        contrastive_highpass_weight=0.10,
        contrastive_edge_weight=0.06,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="mf00_fisher_k6_s20",
        mode="fisher_style_contrastive",
        num_prototypes=6,
        blend=0.0,
        route_temperature=8.0,
        residual_strength=0.20,
        residual_tanh_scale=0.45,
        residual_highpass_kernel=1,
        residual_center_base=False,
        residual_center_content=True,
        contrastive_logit_scale=2.2,
        contrastive_highpass_weight=0.12,
        contrastive_edge_weight=0.08,
        highpass_boost=1.0,
    ),
    BankAdapterRecipe(
        name="mf01_typed_fisher_k9_s22",
        mode="typed_fisher_style_contrastive",
        num_prototypes=9,
        typed_prototypes_per_type=3,
        blend=0.0,
        route_temperature=8.0,
        residual_strength=0.22,
        residual_tanh_scale=0.43,
        residual_highpass_kernel=1,
        residual_center_base=False,
        residual_center_content=True,
        typed_gate_gamma=2.6,
        typed_gate_temperature=0.85,
        typed_prior_scale=0.70,
        contrastive_logit_scale=2.2,
        contrastive_highpass_weight=0.12,
        contrastive_edge_weight=0.08,
        highpass_boost=1.0,
    ),
]

STYLE_CONTRASTIVE_MODES = {"style_contrastive", "typed_style_contrastive"}
FISHER_CONTRASTIVE_MODES = {"fisher_style_contrastive", "typed_fisher_style_contrastive"}
TYPED_MODES = {"typed_flat_edge_texton", "typed_style_contrastive", "typed_fisher_style_contrastive"}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _parse_recipes(spec: str) -> list[BankAdapterRecipe]:
    if not spec.strip():
        return RECIPES
    keep = {item.strip() for item in spec.split(",") if item.strip()}
    selected = [recipe for recipe in RECIPES if recipe.name in keep]
    if not selected:
        raise ValueError(f"No matching memory-bank adapter recipes for {spec!r}")
    return selected


def _resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (ROOT / path).resolve()


def _resolve_latent_root(config: dict, requested: Path | None) -> Path:
    requested = _resolve_path(requested)
    if requested is not None:
        return requested
    data_root = str((config.get("data", {}) or {}).get("data_root", "")).strip()
    if data_root:
        p = Path(data_root)
        return p if p.is_absolute() else (ROOT / p).resolve()
    return ROOT.parent / "latent-256-sd15-ema"


def _read_summary_metrics(summary: dict) -> dict:
    overview = dict(summary.get("analysis", {}).get("all_pairs_overview", {}) or {})
    cross_by_target = dict(summary.get("analysis", {}).get("cross_by_target_style", {}) or {})
    hayao_cross = dict(cross_by_target.get("Hayao", {}) or cross_by_target.get("hayao", {}) or {})
    valid_targets = [
        (str(name), dict(payload))
        for name, payload in cross_by_target.items()
        if isinstance(payload, dict) and payload.get("clip_style") is not None
    ]
    valid_targets.sort(key=lambda item: float(item[1].get("clip_style", float("inf"))))
    weakest = valid_targets[0] if valid_targets else ("", {})
    return {
        "clip_style": overview.get("clip_style", float("nan")),
        "clip_content": overview.get("clip_content", float("nan")),
        "content_lpips": overview.get("content_lpips", float("nan")),
        "ec": overview.get("edge_consistency", overview.get("ec", float("nan"))),
        "hayao_cross_clip_style": hayao_cross.get("clip_style", float("nan")),
        "hayao_cross_content_lpips": hayao_cross.get("content_lpips", float("nan")),
        "weakest_cross_target": weakest[0],
        "weakest_cross_clip_style": weakest[1].get("clip_style", float("nan")),
        "weakest_cross_content_lpips": weakest[1].get("content_lpips", float("nan")),
    }


def _highpass(x: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel <= 1:
        return x.float()
    if kernel % 2 == 0:
        kernel += 1
    return x.float() - F.avg_pool2d(x.float(), kernel_size=kernel, stride=1, padding=kernel // 2)


def _lowpass(x: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel <= 1:
        return x.float()
    if kernel % 2 == 0:
        kernel += 1
    return F.avg_pool2d(x.float(), kernel_size=kernel, stride=1, padding=kernel // 2)


def _flatten_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.flatten(1).float(), dim=1, eps=1e-6)


def _edge_energy(x: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    low = _lowpass(x, kernel)
    gx = low[..., :, 1:] - low[..., :, :-1]
    gy = low[..., 1:, :] - low[..., :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.sqrt(gx.square() + gy.square() + 1e-12).mean(dim=(1, 2, 3))


def _style_measure_descriptor(x: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    low = _lowpass(x, kernel)
    high = _highpass(x, kernel)
    # Channel statistics give the atom a distributional style coordinate rather
    # than a raw local texture coordinate.
    parts = [
        low.mean(dim=(2, 3)),
        low.std(dim=(2, 3), unbiased=False),
        high.abs().mean(dim=(2, 3)),
        high.std(dim=(2, 3), unbiased=False),
    ]
    return F.normalize(torch.cat(parts, dim=1).float(), dim=1, eps=1e-6)


def _contrastive_style_scores(
    descriptors_by_style: list[torch.Tensor],
) -> list[torch.Tensor]:
    centroids = torch.stack(
        [F.normalize(desc.mean(dim=0), dim=0, eps=1e-6) for desc in descriptors_by_style],
        dim=0,
    )
    out: list[torch.Tensor] = []
    for style_id, desc in enumerate(descriptors_by_style):
        sim = desc @ centroids.T
        own = sim[:, style_id]
        other = sim.masked_fill(
            torch.eye(sim.shape[1], dtype=torch.bool)[style_id].view(1, -1),
            -1.0e4,
        ).max(dim=1).values
        out.append(own - other)
    return out


def _fisher_project_descriptors(
    descriptors_by_style: list[torch.Tensor],
    *,
    max_dim: int = 4,
    reg: float = 0.05,
) -> list[torch.Tensor]:
    """Project internal style descriptors into a Fisher-discriminative space."""

    xs = [desc.float() for desc in descriptors_by_style]
    x = torch.cat(xs, dim=0)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    xs = [(desc - mean) / std for desc in xs]
    x = torch.cat(xs, dim=0)
    global_mean = x.mean(dim=0, keepdim=True)
    dim = x.shape[1]
    sw = torch.zeros((dim, dim), dtype=x.dtype, device=x.device)
    sb = torch.zeros((dim, dim), dtype=x.dtype, device=x.device)
    for desc in xs:
        cls_mean = desc.mean(dim=0, keepdim=True)
        centered = desc - cls_mean
        sw = sw + centered.T @ centered / max(1, desc.shape[0] - 1)
        diff = cls_mean - global_mean
        sb = sb + desc.shape[0] * (diff.T @ diff)
    sw = sw / max(1, len(xs))
    sb = sb / max(1, x.shape[0])
    eye = torch.eye(dim, dtype=x.dtype, device=x.device)
    scale = torch.trace(sw).clamp_min(1e-6) / max(1, dim)
    sw = sw + eye * (float(reg) * scale)
    evals, evecs = torch.linalg.eigh(sw)
    inv_sqrt = evecs @ torch.diag(evals.clamp_min(1e-6).rsqrt()) @ evecs.T
    fisher = inv_sqrt @ sb @ inv_sqrt
    eigvals, eigvecs = torch.linalg.eigh((fisher + fisher.T) * 0.5)
    keep = min(max(1, int(max_dim)), len(xs) - 1, dim)
    proj = inv_sqrt @ eigvecs[:, torch.argsort(eigvals, descending=True)[:keep]]
    return [F.normalize(desc @ proj, dim=1, eps=1e-6) for desc in xs]


def _fisher_style_scores(descriptors_by_style: list[torch.Tensor]) -> list[torch.Tensor]:
    return _contrastive_style_scores(_fisher_project_descriptors(descriptors_by_style))


def _top_unique(score: torch.Tensor, count: int, used: set[int], *, largest: bool) -> list[int]:
    order = torch.argsort(score.float(), descending=largest).tolist()
    out: list[int] = []
    for idx in order:
        idx = int(idx)
        if idx in used:
            continue
        used.add(idx)
        out.append(idx)
        if len(out) >= count:
            break
    if not out and order:
        out.append(int(order[0]))
    while out and len(out) < count:
        out.append(out[-1])
    return out


@torch.inference_mode()
def _encode_body_features(model, latents: torch.Tensor, style_id: int, device: str) -> torch.Tensor:
    latents = latents.to(device=device, dtype=next(model.parameters()).dtype)
    sid = torch.full((latents.shape[0],), int(style_id), dtype=torch.long, device=latents.device)
    style_code = model.encode_style_id(sid)
    feat = latents / max(float(getattr(model, "latent_scale_factor", 0.18215)), 1e-8)
    h = model.enc_in_act(model.enc_in(feat))
    h = model._run_style_blocks(
        h,
        blocks=model.hires_body,
        style_code=style_code,
        base_idx=0,
        gate_scale=0.0,
    )
    body = model.down(h)
    return model._normalize_style_map(body)


def _boost_texture(model, feat: torch.Tensor, *, recipe: BankAdapterRecipe) -> torch.Tensor:
    if abs(recipe.highpass_boost - 1.0) < 1e-6:
        return model._normalize_style_map(feat)
    high = _highpass(feat, recipe.highpass_kernel)
    low = feat.float() - high
    return model._normalize_style_map(low + high * float(recipe.highpass_boost))


def _select_diverse_indices(features: torch.Tensor, k: int, *, kernel: int) -> list[int]:
    desc = _flatten_norm(_lowpass(features, kernel))
    chosen = [int(torch.argmax(_highpass(features, kernel).abs().mean(dim=(1, 2, 3))).item())]
    while len(chosen) < min(k, features.shape[0]):
        chosen_desc = desc.index_select(0, torch.tensor(chosen, device=desc.device))
        sim = desc @ chosen_desc.T
        min_dist = 1.0 - sim.max(dim=1).values
        min_dist[torch.tensor(chosen, device=desc.device)] = -1.0
        chosen.append(int(torch.argmax(min_dist).item()))
    return chosen


@torch.inference_mode()
def _style_bank_for_recipe(
    model,
    latent_index: dict[str, list[Path]],
    *,
    style_names: list[str],
    recipe: BankAdapterRecipe,
    device: str,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    banks: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    type_ids_all: list[torch.Tensor] = []
    style_stats: list[tuple[float, float]] = []
    rows: list[dict] = []
    k = max(1, int(recipe.num_prototypes))

    pools: list[dict[str, object]] = []
    for style_id, style_name in enumerate(style_names):
        paths = list(latent_index[style_name])
        rng.shuffle(paths)
        paths = paths[: max(k, min(int(recipe.max_samples_per_style), len(paths)))]
        feats_chunks = []
        for start in range(0, len(paths), recipe.batch_size):
            batch_paths = paths[start : start + recipe.batch_size]
            latents = torch.cat([_load_latent(path) for path in batch_paths], dim=0)
            feats_chunks.append(_encode_body_features(model, latents, style_id, device).cpu())
        feats = torch.cat(feats_chunks, dim=0)
        pools.append(
            {
                "style_id": style_id,
                "style_name": style_name,
                "paths": paths,
                "feats": feats,
                "hp": _highpass(feats, recipe.highpass_kernel).abs().mean(dim=(1, 2, 3)),
                "edge": _edge_energy(feats, recipe.highpass_kernel),
                "descriptor": _style_measure_descriptor(feats, recipe.highpass_kernel),
            }
        )

    descriptors = [pool["descriptor"] for pool in pools]  # type: ignore[list-item]
    if recipe.mode in FISHER_CONTRASTIVE_MODES:
        style_purity = _fisher_style_scores(descriptors)
    else:
        style_purity = _contrastive_style_scores(descriptors)

    for pool in pools:
        style_id = int(pool["style_id"])
        style_name = str(pool["style_name"])
        paths = list(pool["paths"])  # type: ignore[arg-type]
        feats = pool["feats"]  # type: ignore[assignment]
        hp = pool["hp"]  # type: ignore[assignment]
        edge = pool["edge"]  # type: ignore[assignment]
        purity = style_purity[style_id].float()
        hp_z = (hp.float() - hp.float().mean()) / hp.float().std(unbiased=False).clamp_min(1e-6)
        edge_z = (edge.float() - edge.float().mean()) / edge.float().std(unbiased=False).clamp_min(1e-6)
        style_score = (
            purity
            + float(recipe.contrastive_highpass_weight) * hp_z
            + float(recipe.contrastive_edge_weight) * edge_z
        )
        if recipe.mode == "high_texture":
            chosen = torch.topk(hp, k=min(k, feats.shape[0])).indices.tolist()
            type_ids = torch.full((len(chosen),), 2, dtype=torch.long)
        elif recipe.mode == "diverse_low":
            chosen = _select_diverse_indices(feats, k, kernel=recipe.highpass_kernel)
            type_ids = torch.full((len(chosen),), 1, dtype=torch.long)
        elif recipe.mode in {"style_contrastive", "fisher_style_contrastive"}:
            chosen = torch.topk(style_score, k=min(k, feats.shape[0])).indices.tolist()
            type_ids = torch.full((len(chosen),), 2, dtype=torch.long)
        elif recipe.mode == "typed_flat_edge_texton":
            per_type = max(1, int(recipe.typed_prototypes_per_type or max(1, k // 3)))
            used: set[int] = set()
            flat_idx = _top_unique(hp, per_type, used, largest=False)
            edge_idx = _top_unique(edge, per_type, used, largest=True)
            texton_idx = _top_unique(hp, per_type, used, largest=True)
            chosen = (flat_idx + edge_idx + texton_idx)[:k]
            type_ids = torch.tensor(
                ([0] * len(flat_idx) + [1] * len(edge_idx) + [2] * len(texton_idx))[: len(chosen)],
                dtype=torch.long,
            )
        elif recipe.mode in {"typed_style_contrastive", "typed_fisher_style_contrastive"}:
            per_type = max(1, int(recipe.typed_prototypes_per_type or max(1, k // 3)))
            used = set()
            flat_mask_score = style_score - 0.55 * hp_z - 0.20 * edge_z
            edge_mask_score = style_score + 0.75 * edge_z - 0.20 * hp_z
            texton_mask_score = style_score + 0.65 * hp_z
            flat_idx = _top_unique(flat_mask_score, per_type, used, largest=True)
            edge_idx = _top_unique(edge_mask_score, per_type, used, largest=True)
            texton_idx = _top_unique(texton_mask_score, per_type, used, largest=True)
            chosen = (flat_idx + edge_idx + texton_idx)[:k]
            type_ids = torch.tensor(
                ([0] * len(flat_idx) + [1] * len(edge_idx) + [2] * len(texton_idx))[: len(chosen)],
                dtype=torch.long,
            )
        else:
            raise ValueError(f"Unsupported prototype mode: {recipe.mode}")
        proto = feats.index_select(0, torch.tensor(chosen, dtype=torch.long))
        proto = _boost_texture(model, proto, recipe=recipe).cpu()
        if proto.shape[0] < k:
            pad = proto[-1:].expand(k - proto.shape[0], -1, -1, -1)
            proto = torch.cat([proto, pad], dim=0)
        if type_ids.shape[0] < k:
            type_ids = F.pad(type_ids, (0, k - type_ids.shape[0]), value=int(type_ids[-1].item()))
        score = hp.index_select(0, torch.tensor(chosen, dtype=torch.long))
        purity_chosen = purity.index_select(0, torch.tensor(chosen, dtype=torch.long))
        logit = (
            (purity_chosen * float(recipe.contrastive_logit_scale)).float()
            if recipe.mode in STYLE_CONTRASTIVE_MODES | FISHER_CONTRASTIVE_MODES
            else torch.zeros_like(score).float()
            if recipe.mode == "typed_flat_edge_texton"
            else (score / max(float(recipe.temperature), 1e-4)).float()
        )
        if logit.shape[0] < k:
            logit = F.pad(logit, (0, k - logit.shape[0]), value=float(logit[-1].item()))
        banks.append(proto[:k])
        logits.append(logit[:k])
        type_ids_all.append(type_ids[:k])
        style_stats.append((float(hp.mean().item()), float(edge.mean().item())))
        for rank, idx in enumerate(chosen[:k]):
            rows.append(
                {
                    "style_id": style_id,
                    "style_name": style_name,
                    "rank": rank,
                    "prototype_type": ["flat", "edge", "texton"][int(type_ids[rank].item())] if recipe.mode in TYPED_MODES else recipe.mode,
                    "sample_index": int(idx),
                    "sample_path": str(paths[int(idx)]),
                    "highpass_score": float(hp[int(idx)].item()),
                    "edge_score": float(edge[int(idx)].item()),
                    "style_purity": float(purity[int(idx)].item()),
                    "logit": float(logit[rank].item()),
                }
            )
    type_ids_tensor = torch.stack(type_ids_all, dim=0)
    if recipe.mode in TYPED_MODES:
        stats = torch.tensor(style_stats, dtype=torch.float32)
        hp_z = (stats[:, 0] - stats[:, 0].mean()) / stats[:, 0].std(unbiased=False).clamp_min(1e-6)
        edge_z = (stats[:, 1] - stats[:, 1].mean()) / stats[:, 1].std(unbiased=False).clamp_min(1e-6)
        type_logits = torch.stack([-hp_z - 0.25 * edge_z, edge_z - 0.25 * hp_z, hp_z], dim=1)
        type_logits = type_logits * float(recipe.typed_prior_scale)
    else:
        type_logits = torch.empty(0)
    return torch.stack(banks, dim=0), torch.stack(logits, dim=0), type_ids_tensor, type_logits, rows


def _save_memory_adapter(
    path: Path,
    model,
    *,
    bank: torch.Tensor,
    logits: torch.Tensor,
    type_ids: torch.Tensor,
    type_logits: torch.Tensor,
    recipe: BankAdapterRecipe,
) -> None:
    payload = {
        "style_emb.weight": model.style_emb.weight.detach().cpu(),
        "style_spatial_id_16": model.style_spatial_id_16.detach().cpu(),
        "style_memory_bank_16": bank.detach().cpu(),
        "style_memory_bank_logits": logits.detach().cpu(),
        "style_memory_bank_blend": torch.tensor(float(recipe.blend), dtype=torch.float32),
        "style_memory_bank_route_strength": torch.tensor(float(recipe.route_strength), dtype=torch.float32),
        "style_memory_bank_route_temperature": torch.tensor(float(recipe.route_temperature), dtype=torch.float32),
        "style_memory_bank_type_gate_gamma": torch.tensor(float(recipe.typed_gate_gamma), dtype=torch.float32),
        "style_memory_bank_type_gate_temperature": torch.tensor(float(recipe.typed_gate_temperature), dtype=torch.float32),
        "style_memory_bank_residual_strength": torch.tensor(float(recipe.residual_strength), dtype=torch.float32),
        "style_memory_bank_residual_tanh_scale": torch.tensor(float(recipe.residual_tanh_scale), dtype=torch.float32),
        "style_memory_bank_residual_highpass_kernel": torch.tensor(float(recipe.residual_highpass_kernel), dtype=torch.float32),
        "style_memory_bank_residual_center_base": torch.tensor(1.0 if recipe.residual_center_base else 0.0, dtype=torch.float32),
        "style_memory_bank_residual_center_content": torch.tensor(1.0 if recipe.residual_center_content else 0.0, dtype=torch.float32),
        "style_memory_bank_residual_gate_gamma": torch.tensor(float(recipe.residual_gate_gamma), dtype=torch.float32),
        "style_memory_bank_residual_gate_floor": torch.tensor(float(recipe.residual_gate_floor), dtype=torch.float32),
        "style_memory_bank_residual_gate_kernel": torch.tensor(float(recipe.residual_gate_kernel), dtype=torch.float32),
    }
    if recipe.mode in {"typed_flat_edge_texton", "typed_style_contrastive"}:
        payload["style_memory_bank_type_ids"] = type_ids.detach().cpu().long()
        payload["style_memory_bank_type_logits"] = type_logits.detach().cpu()
    tokenizer = getattr(model, "style_tokenizer", None)
    if tokenizer is not None:
        payload["style_tokenizer.grammar_vocab.weight"] = tokenizer.grammar_vocab.weight.detach().cpu()
        payload["style_tokenizer.band_vocab.weight"] = tokenizer.band_vocab.weight.detach().cpu()
        identity = getattr(tokenizer, "identity_vocab", None)
        if torch.is_tensor(identity):
            payload["style_tokenizer.identity_vocab"] = identity.detach().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _run_recipe(
    recipe: BankAdapterRecipe,
    *,
    checkpoint: Path,
    init_style_adapter: Path,
    latent_root: Path,
    out_root: Path,
    style_names: list[str],
    eval_batch_size: int,
    vae_model: str,
    seed: int,
    device: str,
    skip_eval: bool,
) -> dict:
    rng = random.Random(seed)
    model, config = _load_checkpoint_model(checkpoint, device)
    _apply_style_adapter(model, init_style_adapter, device)
    model.eval()
    latent_index = _style_latent_index(latent_root, style_names)
    recipe_dir = out_root / recipe.name
    recipe_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    bank, logits, type_ids, type_logits, prototype_rows = _style_bank_for_recipe(
        model,
        latent_index,
        style_names=style_names,
        recipe=recipe,
        device=device,
        rng=rng,
    )
    adapter_path = recipe_dir / "style_adapter.pt"
    _save_memory_adapter(
        adapter_path,
        model,
        bank=bank,
        logits=logits,
        type_ids=type_ids,
        type_logits=type_logits,
        recipe=recipe,
    )
    with (recipe_dir / "prototype_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "style_id",
                "style_name",
                "rank",
                "prototype_type",
                "sample_index",
                "sample_path",
                "highpass_score",
                "edge_score",
                "style_purity",
                "logit",
            ],
        )
        writer.writeheader()
        writer.writerows(prototype_rows)
    _write_json(
        recipe_dir / "memory_bank_config.json",
        {
            "recipe": recipe.__dict__,
            "checkpoint": str(checkpoint),
            "init_style_adapter": str(init_style_adapter),
            "latent_root": str(latent_root),
            "one_line_hypothesis": (
                "If explicit reference latents work because they supply real style-source features, "
                "an id-only adapter-side prototype bank should recover part of that style lift without a test-time reference."
            ),
        },
    )
    if skip_eval:
        return {"recipe": recipe.name, "mode": "skip_eval", "seconds": time.time() - start}
    summary = _run_full_eval(
        checkpoint=checkpoint,
        style_adapter=adapter_path,
        output_dir=recipe_dir / "full_eval",
        batch_size=eval_batch_size,
        vae_model=vae_model,
    )
    row = _read_summary_metrics(summary)
    row.update({"recipe": recipe.name, "seconds": time.time() - start, "style_adapter": str(adapter_path)})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/evaluate id-only multi-prototype style-memory adapters.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "exp/vae_backend/ema_transport_moment/ema_transport_adain_w34_guard/epoch_0006.pt")
    parser.add_argument("--init-style-adapter", type=Path, default=ROOT / "exp/style_embedding_mainline_calibration/ema_transport_adain_w34_e6_fulltrain/m02_embspatial_highpass_style/style_adapter.pt")
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/style_memory_bank_adapter_probe")
    parser.add_argument("--recipes", default="")
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--vae-model", default="auto")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    checkpoint = _resolve_path(args.checkpoint)
    init_style_adapter = _resolve_path(args.init_style_adapter)
    out_root = _resolve_path(args.out_root)
    if checkpoint is None or init_style_adapter is None or out_root is None:
        raise ValueError("checkpoint/init-style-adapter/out-root are required")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    style_names = list(config.get("data", {}).get("style_subdirs", []))
    if not style_names:
        raise ValueError("Checkpoint config has no data.style_subdirs")
    latent_root = _resolve_latent_root(config, args.latent_root)
    recipes = _parse_recipes(args.recipes)
    eval_batch_size = _memory_tier_eval_batch_size(args.device, args.eval_batch_size if args.eval_batch_size > 0 else None)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for recipe in recipes:
        rows.append(
            _run_recipe(
                recipe,
                checkpoint=checkpoint,
                init_style_adapter=init_style_adapter,
                latent_root=latent_root,
                out_root=out_root,
                style_names=style_names,
                eval_batch_size=eval_batch_size,
                vae_model=args.vae_model,
                seed=args.seed,
                device=args.device,
                skip_eval=bool(args.skip_eval),
            )
        )
    fields = sorted({key for row in rows for key in row.keys()})
    with (out_root / "style_memory_bank_adapter_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(out_root / "style_memory_bank_adapter_results.csv")


if __name__ == "__main__":
    main()
