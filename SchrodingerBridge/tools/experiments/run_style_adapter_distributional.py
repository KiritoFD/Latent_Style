import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ot_cost import SWDTransportCost  # noqa: E402
from run_style_embedding_distill import (  # noqa: E402
    _gradient_cosine_loss,
    _integrate_with_grad,
    _load_checkpoint_model,
    _memory_tier_eval_batch_size,
    _run_full_eval,
    _sample_latent_batch,
    _save_style_adapter,
    _style_latent_index,
    _tv_loss,
)
from run_style_embedding_mainline_calibration import (  # noqa: E402
    _apply_style_adapter,
    _highpass,
    _read_summary_metrics,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class DistributionalStyleEmbedding(nn.Module):
    """Low-variance style-code distribution with deterministic mean fallback."""

    def __init__(self, base_weight: torch.Tensor, init_log_std: float = -5.0) -> None:
        super().__init__()
        self.mu = nn.Parameter(base_weight.detach().clone())
        self.log_std = nn.Parameter(torch.full_like(base_weight.detach(), float(init_log_std)))
        self.embedding_dim = int(base_weight.shape[1])
        self.sample_enabled = True
        self.std_min = 0.0
        self.std_max = 0.08

    @property
    def weight(self) -> torch.Tensor:
        return self.mu

    def forward(self, style_id: torch.Tensor | int) -> torch.Tensor:
        if isinstance(style_id, int):
            style_id = torch.tensor([style_id], device=self.mu.device, dtype=torch.long)
        style_id = style_id.to(device=self.mu.device, dtype=torch.long)
        mu = F.embedding(style_id, self.mu)
        if not self.training or not self.sample_enabled:
            return mu
        std = F.embedding(style_id, self.log_std).exp().clamp(self.std_min, self.std_max)
        return mu + torch.randn_like(mu) * std


@dataclass(frozen=True)
class DistRecipe:
    name: str
    iters_per_style: int
    batch_size: int
    ode_steps: int
    lr_mu: float
    lr_log_std: float
    swd_weight: float
    anchor_weight: float
    consistency_weight: float
    grad_weight: float
    delta_tv_weight: float
    margin_weight: float
    margin_target: float
    response_margin_weight: float
    response_margin_target: float
    std_l2_weight: float
    std_floor_weight: float
    std_floor: float
    emb_l2_weight: float
    spatial_l2_weight: float
    optimize_spatial: bool
    highpass_kernel: int
    save_every: int = 0


RECIPES = [
    DistRecipe(
        name="dist_m02_fast_margin",
        iters_per_style=40,
        batch_size=8,
        ode_steps=6,
        lr_mu=1.1e-3,
        lr_log_std=4e-4,
        swd_weight=1.45,
        anchor_weight=0.12,
        consistency_weight=0.14,
        grad_weight=0.10,
        delta_tv_weight=0.04,
        margin_weight=0.28,
        margin_target=0.66,
        response_margin_weight=0.0,
        response_margin_target=0.75,
        std_l2_weight=0.030,
        std_floor_weight=0.012,
        std_floor=0.010,
        emb_l2_weight=0.016,
        spatial_l2_weight=0.008,
        optimize_spatial=True,
        highpass_kernel=5,
        save_every=20,
    ),
    DistRecipe(
        name="dist_m02_fast_hayao",
        iters_per_style=60,
        batch_size=8,
        ode_steps=6,
        lr_mu=1.0e-3,
        lr_log_std=4e-4,
        swd_weight=1.70,
        anchor_weight=0.09,
        consistency_weight=0.16,
        grad_weight=0.10,
        delta_tv_weight=0.04,
        margin_weight=0.36,
        margin_target=0.60,
        response_margin_weight=0.0,
        response_margin_target=0.72,
        std_l2_weight=0.032,
        std_floor_weight=0.014,
        std_floor=0.012,
        emb_l2_weight=0.012,
        spatial_l2_weight=0.007,
        optimize_spatial=True,
        highpass_kernel=5,
        save_every=30,
    ),
    DistRecipe(
        name="dist_m02_fast_hayao_response",
        iters_per_style=60,
        batch_size=8,
        ode_steps=6,
        lr_mu=1.0e-3,
        lr_log_std=4e-4,
        swd_weight=1.70,
        anchor_weight=0.09,
        consistency_weight=0.15,
        grad_weight=0.10,
        delta_tv_weight=0.04,
        margin_weight=0.26,
        margin_target=0.62,
        response_margin_weight=0.18,
        response_margin_target=0.62,
        std_l2_weight=0.032,
        std_floor_weight=0.014,
        std_floor=0.012,
        emb_l2_weight=0.012,
        spatial_l2_weight=0.007,
        optimize_spatial=True,
        highpass_kernel=5,
        save_every=30,
    ),
    DistRecipe(
        name="dist_m02_margin_lowvar",
        iters_per_style=120,
        batch_size=10,
        ode_steps=12,
        lr_mu=8e-4,
        lr_log_std=3e-4,
        swd_weight=1.45,
        anchor_weight=0.14,
        consistency_weight=0.12,
        grad_weight=0.12,
        delta_tv_weight=0.045,
        margin_weight=0.20,
        margin_target=0.70,
        response_margin_weight=0.0,
        response_margin_target=0.75,
        std_l2_weight=0.025,
        std_floor_weight=0.010,
        std_floor=0.010,
        emb_l2_weight=0.018,
        spatial_l2_weight=0.008,
        optimize_spatial=True,
        highpass_kernel=5,
        save_every=60,
    ),
    DistRecipe(
        name="dist_m02_hayao_margin",
        iters_per_style=140,
        batch_size=10,
        ode_steps=12,
        lr_mu=7e-4,
        lr_log_std=3e-4,
        swd_weight=1.65,
        anchor_weight=0.10,
        consistency_weight=0.16,
        grad_weight=0.10,
        delta_tv_weight=0.04,
        margin_weight=0.32,
        margin_target=0.62,
        response_margin_weight=0.12,
        response_margin_target=0.62,
        std_l2_weight=0.030,
        std_floor_weight=0.014,
        std_floor=0.012,
        emb_l2_weight=0.014,
        spatial_l2_weight=0.007,
        optimize_spatial=True,
        highpass_kernel=5,
        save_every=70,
    ),
]


def _parse_recipes(spec: str) -> list[DistRecipe]:
    if not spec.strip():
        return RECIPES
    keep = {item.strip() for item in spec.split(",") if item.strip()}
    selected = [recipe for recipe in RECIPES if recipe.name in keep]
    if not selected:
        raise ValueError(f"No matching recipes for {spec!r}")
    return selected


def _resolve_latent_root(config: dict, requested: Path | None) -> Path:
    if requested is not None:
        return requested if requested.is_absolute() else (ROOT / requested).resolve()
    data_root = str((config.get("data", {}) or {}).get("data_root", "")).strip()
    if data_root:
        p = Path(data_root)
        return p if p.is_absolute() else (ROOT / p).resolve()
    return ROOT.parent / "latent-256"


def _style_margin_loss(emb: torch.Tensor, target: float) -> torch.Tensor:
    normed = F.normalize(emb.float(), dim=1, eps=1e-6)
    sims = normed @ normed.t()
    mask = ~torch.eye(sims.shape[0], dtype=torch.bool, device=sims.device)
    return F.relu(sims[mask] - float(target)).square().mean()


def _response_margin_loss(pred: torch.Tensor, neg_pred: torch.Tensor, content: torch.Tensor, target: float) -> torch.Tensor:
    pos_delta = (pred.float() - content.float()).flatten(1)
    neg_delta = (neg_pred.float() - content.float()).flatten(1)
    cos = F.cosine_similarity(pos_delta, neg_delta, dim=1, eps=1e-6)
    return F.relu(cos - float(target)).square().mean()


def _effective_rank(emb: torch.Tensor) -> torch.Tensor:
    centered = emb.float() - emb.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    prob = singular.square() / singular.square().sum().clamp_min(1e-12)
    return torch.exp(-(prob * prob.clamp_min(1e-12).log()).sum())


def _adapter_stats(model) -> dict[str, float]:
    emb = model.style_emb.weight.detach().float()
    normed = F.normalize(emb, dim=1, eps=1e-6)
    sims = normed @ normed.t()
    mask = ~torch.eye(sims.shape[0], dtype=torch.bool, device=sims.device)
    std = model.style_emb.log_std.detach().float().exp().clamp(0.0, model.style_emb.std_max)
    return {
        "style_emb_dim": float(emb.shape[1]),
        "style_emb_effective_rank": float(_effective_rank(emb).item()),
        "style_emb_max_offdiag_cos": float(sims[mask].max().item()),
        "style_emb_mean_offdiag_cos": float(sims[mask].mean().item()),
        "style_std_mean": float(std.mean().item()),
        "style_std_max": float(std.max().item()),
    }


def _save_dist_adapter(path: Path, model, recipe: DistRecipe) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "style_adapter_format": "distributional_v1",
        "style_emb.weight": model.style_emb.mu.detach().cpu(),
        "style_emb.mu": model.style_emb.mu.detach().cpu(),
        "style_emb.log_std": model.style_emb.log_std.detach().cpu(),
        "style_emb.std_max": float(model.style_emb.std_max),
        "recipe": recipe.__dict__,
    }
    if hasattr(model, "style_spatial_id_16"):
        payload["style_spatial_id_16"] = model.style_spatial_id_16.detach().cpu()
    tokenizer = getattr(model, "style_tokenizer", None)
    if tokenizer is not None:
        payload["style_tokenizer.grammar_vocab.weight"] = tokenizer.grammar_vocab.weight.detach().cpu()
        payload["style_tokenizer.band_vocab.weight"] = tokenizer.band_vocab.weight.detach().cpu()
        identity = getattr(tokenizer, "identity_vocab", None)
        if torch.is_tensor(identity):
            payload["style_tokenizer.identity_vocab"] = identity.detach().cpu()
    torch.save(payload, path)


def _install_distributional_embedding(model, init_log_std: float) -> None:
    base = model.style_emb.weight.detach().clone()
    model.style_emb = DistributionalStyleEmbedding(base, init_log_std=init_log_std).to(base.device)


def _set_sampling(model, enabled: bool) -> None:
    if hasattr(model.style_emb, "sample_enabled"):
        model.style_emb.sample_enabled = bool(enabled)


def run_recipe(
    recipe: DistRecipe,
    *,
    checkpoint: Path,
    init_style_adapter: Path | None,
    latent_root: Path,
    out_root: Path,
    style_names: list[str],
    target_style_ids: list[int],
    eval_batch_size: int,
    vae_model: str,
    seed: int,
    device: str,
    skip_eval: bool,
    max_iters_per_style: int,
) -> dict:
    rng = random.Random(seed)
    model, config = _load_checkpoint_model(checkpoint, device)
    teacher, _ = _load_checkpoint_model(checkpoint, device)
    if init_style_adapter is not None:
        _apply_style_adapter(model, init_style_adapter, device)
        _apply_style_adapter(teacher, init_style_adapter, device)
    _install_distributional_embedding(model, init_log_std=-5.0)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(False)
    model.style_emb.mu.requires_grad_(True)
    model.style_emb.log_std.requires_grad_(True)
    if hasattr(model, "style_spatial_id_16"):
        model.style_spatial_id_16.requires_grad_(recipe.optimize_spatial)

    base_mu = model.style_emb.mu.detach().clone()
    base_spatial = (
        model.style_spatial_id_16.detach().clone()
        if recipe.optimize_spatial and hasattr(model, "style_spatial_id_16")
        else None
    )
    params = [
        {"params": [model.style_emb.mu], "lr": recipe.lr_mu},
        {"params": [model.style_emb.log_std], "lr": recipe.lr_log_std},
    ]
    if recipe.optimize_spatial and hasattr(model, "style_spatial_id_16"):
        params.append({"params": [model.style_spatial_id_16], "lr": recipe.lr_mu})
    optimizer = torch.optim.AdamW(params, weight_decay=0.0)

    latent_index = _style_latent_index(latent_root, style_names)
    content_pool = [p for style in style_names for p in latent_index[style]]
    transport = SWDTransportCost(config)
    recipe_dir = out_root / recipe.name
    losses: list[dict] = []
    start_time = time.time()
    iters_per_style = min(recipe.iters_per_style, max_iters_per_style) if max_iters_per_style > 0 else recipe.iters_per_style

    for style_id in target_style_ids:
        style_name = style_names[style_id]
        for iteration in range(1, iters_per_style + 1):
            content = _sample_latent_batch(content_pool, recipe.batch_size, device, rng)
            target = _sample_latent_batch(latent_index[style_name], recipe.batch_size, device, rng)
            sid = torch.full((recipe.batch_size,), style_id, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)

            model.train()
            _set_sampling(model, True)
            pred_a = _integrate_with_grad(model, content, style_id=sid, num_steps=recipe.ode_steps)
            pred_b = _integrate_with_grad(model, content, style_id=sid, num_steps=recipe.ode_steps)
            _set_sampling(model, False)
            pred_mean = _integrate_with_grad(model, content, style_id=sid, num_steps=recipe.ode_steps)
            response_margin = pred_mean.new_tensor(0.0)
            if recipe.response_margin_weight > 0.0 and len(style_names) > 2:
                neg_candidates = [idx for idx in range(1, len(style_names)) if idx != int(style_id)]
                if neg_candidates:
                    neg_style_id = int(neg_candidates[rng.randrange(len(neg_candidates))])
                    neg_sid = torch.full((recipe.batch_size,), neg_style_id, dtype=torch.long, device=device)
                    neg_pred = _integrate_with_grad(model, content, style_id=neg_sid, num_steps=recipe.ode_steps)
                    response_margin = _response_margin_loss(
                        pred_mean,
                        neg_pred,
                        content,
                        recipe.response_margin_target,
                    )
                    del neg_sid, neg_pred
            with torch.no_grad():
                teacher_pred = _integrate_with_grad(teacher, content, style_id=sid, num_steps=recipe.ode_steps)

            target_hp = _highpass(target, recipe.highpass_kernel)
            swd = 0.5 * (
                transport.aligned_cost(_highpass(pred_a, recipe.highpass_kernel), target_hp)
                + transport.aligned_cost(_highpass(pred_b, recipe.highpass_kernel), target_hp)
            )
            anchor = (pred_mean.float() - teacher_pred.float()).square().mean()
            consistency = (pred_a.float() - pred_b.float()).square().mean()
            grad = _gradient_cosine_loss(pred_mean, content) if recipe.grad_weight > 0.0 else pred_mean.new_tensor(0.0)
            tv = _tv_loss(pred_mean - content) if recipe.delta_tv_weight > 0.0 else pred_mean.new_tensor(0.0)
            margin = _style_margin_loss(model.style_emb.mu, recipe.margin_target)
            std = model.style_emb.log_std.exp().clamp(0.0, model.style_emb.std_max)
            std_l2 = std.square().mean()
            std_floor = F.relu(float(recipe.std_floor) - std).square().mean()
            emb_l2 = (model.style_emb.mu.float() - base_mu.float()).square().mean()
            spatial_l2 = (
                (model.style_spatial_id_16.float() - base_spatial.float()).square().mean()
                if base_spatial is not None and recipe.spatial_l2_weight > 0.0
                else pred_mean.new_tensor(0.0)
            )
            loss = (
                recipe.swd_weight * swd
                + recipe.anchor_weight * anchor
                + recipe.consistency_weight * consistency
                + recipe.grad_weight * grad
                + recipe.delta_tv_weight * tv
                + recipe.margin_weight * margin
                + recipe.response_margin_weight * response_margin
                + recipe.std_l2_weight * std_l2
                + recipe.std_floor_weight * std_floor
                + recipe.emb_l2_weight * emb_l2
                + recipe.spatial_l2_weight * spatial_l2
            )
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite loss in {recipe.name} style={style_name} iter={iteration}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for group in params for p in group["params"]],
                1.0,
            )
            optimizer.step()

            row = {
                "recipe": recipe.name,
                "style_id": style_id,
                "style_name": style_name,
                "iter": iteration,
                "loss": float(loss.detach().item()),
                "swd": float(swd.detach().item()),
                "anchor": float(anchor.detach().item()),
                "consistency": float(consistency.detach().item()),
                "margin": float(margin.detach().item()),
                "response_margin": float(response_margin.detach().item()),
                "std_mean": float(std.detach().mean().item()),
                "std_max": float(std.detach().max().item()),
                "grad": float(grad.detach().item()),
                "tv": float(tv.detach().item()),
            }
            losses.append(row)
            if iteration == 1 or iteration % 25 == 0 or iteration == iters_per_style:
                print(
                    f"[{recipe.name}] style={style_name} iter={iteration}/{iters_per_style} "
                    f"loss={row['loss']:.4f} swd={row['swd']:.4f} margin={row['margin']:.4f} "
                    f"resp={row['response_margin']:.4f} std={row['std_mean']:.5f} "
                    f"consistency={row['consistency']:.5f}"
                )
            if recipe.save_every > 0 and iteration % recipe.save_every == 0:
                _save_dist_adapter(recipe_dir / f"style_adapter_dist_style{style_id}_iter{iteration:04d}.pt", model, recipe)
            del content, target, sid, pred_a, pred_b, pred_mean, teacher_pred, loss
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    mean_adapter_path = recipe_dir / "style_adapter.pt"
    dist_adapter_path = recipe_dir / "style_adapter_dist.pt"
    _save_style_adapter(mean_adapter_path, model)
    _save_dist_adapter(dist_adapter_path, model, recipe)
    with (recipe_dir / "distributional_losses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(losses[0].keys()))
        writer.writeheader()
        writer.writerows(losses)
    stats = _adapter_stats(model)
    _write_json(
        recipe_dir / "distributional_config.json",
        {
            "recipe": recipe.__dict__,
            "checkpoint": str(checkpoint),
            "init_style_adapter": str(init_style_adapter or ""),
            "latent_root": str(latent_root),
            "style_names": style_names,
            "target_style_ids": target_style_ids,
            "effective_iters_per_style": iters_per_style,
            "elapsed_seconds": time.time() - start_time,
            "adapter_stats": stats,
            "notes": (
                "Distribution is a training/diagnostic device. The saved style_adapter.pt contains "
                "the deterministic mean and is the only artifact intended for baseline-comparable eval."
            ),
        },
    )

    eval_metrics = {}
    if not skip_eval:
        summary = _run_full_eval(
            checkpoint=checkpoint,
            style_adapter=mean_adapter_path,
            output_dir=recipe_dir / "full_eval_mean",
            batch_size=eval_batch_size,
            vae_model=vae_model,
        )
        _write_json(recipe_dir / "full_eval_mean_summary.json", summary)
        eval_metrics = _read_summary_metrics(summary)

    return {
        "recipe": recipe.name,
        "mean_adapter_path": str(mean_adapter_path),
        "dist_adapter_path": str(dist_adapter_path),
        **stats,
        **eval_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a distributional style adapter and export deterministic mean.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-style-adapter", type=Path, default=None)
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/style_adapter_distributional")
    parser.add_argument("--style-subdirs", type=str, default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--target-style-ids", type=str, default="1,2,3,4")
    parser.add_argument("--recipes", type=str, default="")
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--vae-model", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--max-iters-per-style", type=int, default=0)
    args = parser.parse_args()

    _, config = _load_checkpoint_model(args.checkpoint, args.device)
    latent_root = _resolve_latent_root(config, args.latent_root)
    style_names = [item.strip() for item in args.style_subdirs.split(",") if item.strip()]
    target_style_ids = [int(item.strip()) for item in args.target_style_ids.split(",") if item.strip()]
    recipes = _parse_recipes(args.recipes)
    eval_batch_size = _memory_tier_eval_batch_size(
        args.device,
        args.eval_batch_size if args.eval_batch_size > 0 else None,
    )
    rows = []
    args.out_root.mkdir(parents=True, exist_ok=True)
    for recipe in recipes:
        row = run_recipe(
            recipe,
            checkpoint=args.checkpoint,
            init_style_adapter=args.init_style_adapter,
            latent_root=latent_root,
            out_root=args.out_root,
            style_names=style_names,
            target_style_ids=target_style_ids,
            eval_batch_size=eval_batch_size,
            vae_model=args.vae_model,
            seed=args.seed,
            device=args.device,
            skip_eval=bool(args.skip_eval),
            max_iters_per_style=int(args.max_iters_per_style),
        )
        rows.append(row)

    summary_csv = args.out_root / "distributional_style_adapter_results.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(args.out_root / "summary.json", {"rows": rows})
    print(f"Saved distributional style-adapter summary to {summary_csv}")


if __name__ == "__main__":
    main()
