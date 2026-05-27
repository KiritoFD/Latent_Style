import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import torch
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
    _run_full_eval,
    _sample_latent_batch,
    _style_latent_index,
    _tv_loss,
)
from run_style_embedding_mainline_calibration import (  # noqa: E402
    _apply_style_adapter,
    _read_summary_metrics,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _highpass(x: torch.Tensor, kernel: int) -> torch.Tensor:
    k = max(1, int(kernel))
    if k <= 1:
        return x.float()
    if k % 2 == 0:
        k += 1
    low = F.avg_pool2d(x.float(), kernel_size=k, stride=1, padding=k // 2)
    return x.float() - low


def _lowpass(x: torch.Tensor, kernel: int) -> torch.Tensor:
    k = max(1, int(kernel))
    if k <= 1:
        return x.float()
    if k % 2 == 0:
        k += 1
    return F.avg_pool2d(x.float(), kernel_size=k, stride=1, padding=k // 2)


def _finite_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        value = float(value.detach().float().cpu().item())
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def _pairwise_cosine_stats(x: torch.Tensor) -> dict[str, float]:
    flat = x.float().flatten(1)
    if flat.shape[0] <= 1:
        return {
            "pair_cos_mean": float("nan"),
            "pair_cos_std": float("nan"),
            "pair_cos_max": float("nan"),
            "pair_cos_min": float("nan"),
            "pair_cos_margin": float("nan"),
        }
    normed = F.normalize(flat, dim=1, eps=1e-6)
    sims = normed @ normed.t()
    mask = ~torch.eye(sims.shape[0], dtype=torch.bool, device=sims.device)
    vals = sims[mask]
    return {
        "pair_cos_mean": _finite_float(vals.mean()),
        "pair_cos_std": _finite_float(vals.std(unbiased=False)),
        "pair_cos_max": _finite_float(vals.max()),
        "pair_cos_min": _finite_float(vals.min()),
        "pair_cos_margin": _finite_float(1.0 - vals.max()),
    }


def _embedding_stats(model, base_emb: torch.Tensor, base_spatial: torch.Tensor | None) -> dict[str, float]:
    emb = model.style_emb.weight.detach().float()
    base_emb = base_emb.to(device=emb.device, dtype=emb.dtype)
    centered_emb = emb - emb.mean(dim=0, keepdim=True)
    emb_singular = torch.linalg.svdvals(centered_emb)
    emb_var = emb_singular.square()
    emb_prob = emb_var / emb_var.sum().clamp_min(1e-12)
    emb_effective_rank = torch.exp(-(emb_prob * emb_prob.clamp_min(1e-12).log()).sum())
    rows: dict[str, float] = {
        "style_emb_num_styles": float(emb.shape[0]),
        "style_emb_dim": float(emb.shape[1]),
        "style_emb_centered_rank": float(torch.linalg.matrix_rank(centered_emb).item()),
        "style_emb_effective_rank": _finite_float(emb_effective_rank),
        "style_emb_norm_mean": _finite_float(emb.norm(dim=1).mean()),
        "style_emb_norm_std": _finite_float(emb.norm(dim=1).std(unbiased=False)),
        "style_emb_delta_l2": _finite_float((emb - base_emb).square().mean().sqrt()),
        "style_emb_delta_abs": _finite_float((emb - base_emb).abs().mean()),
    }
    rows.update({f"style_emb_{k}": v for k, v in _pairwise_cosine_stats(emb).items()})
    if hasattr(model, "style_spatial_id_16"):
        spatial = model.style_spatial_id_16.detach().float()
        centered_spatial = spatial.flatten(1) - spatial.flatten(1).mean(dim=0, keepdim=True)
        spatial_singular = torch.linalg.svdvals(centered_spatial)
        spatial_var = spatial_singular.square()
        spatial_prob = spatial_var / spatial_var.sum().clamp_min(1e-12)
        spatial_effective_rank = torch.exp(
            -(spatial_prob * spatial_prob.clamp_min(1e-12).log()).sum()
        )
        rows.update(
            {
                "style_spatial_dim": float(spatial.flatten(1).shape[1]),
                "style_spatial_centered_rank": float(torch.linalg.matrix_rank(centered_spatial).item()),
                "style_spatial_effective_rank": _finite_float(spatial_effective_rank),
                "style_spatial_norm_mean": _finite_float(spatial.flatten(1).norm(dim=1).mean()),
                "style_spatial_norm_std": _finite_float(spatial.flatten(1).norm(dim=1).std(unbiased=False)),
            }
        )
        rows.update({f"style_spatial_{k}": v for k, v in _pairwise_cosine_stats(spatial).items()})
        if base_spatial is not None:
            base_spatial = base_spatial.to(device=spatial.device, dtype=spatial.dtype)
            rows.update(
                {
                    "style_spatial_delta_l2": _finite_float((spatial - base_spatial).square().mean().sqrt()),
                    "style_spatial_delta_abs": _finite_float((spatial - base_spatial).abs().mean()),
                }
            )
    tokenizer = getattr(model, "style_tokenizer", None)
    if tokenizer is not None:
        grammar = tokenizer.grammar_vocab.weight.detach().float()
        band = tokenizer.band_vocab.weight.detach().float()
        rows.update(
            {
                "style_token_grammar_dim": float(grammar.shape[1]),
                "style_token_grammar_norm_mean": _finite_float(grammar.norm(dim=1).mean()),
                "style_token_grammar_norm_std": _finite_float(grammar.norm(dim=1).std(unbiased=False)),
                "style_token_band_dim": float(band.shape[1]),
                "style_token_band_norm_mean": _finite_float(band.norm(dim=1).mean()),
                "style_token_band_norm_std": _finite_float(band.norm(dim=1).std(unbiased=False)),
            }
        )
        rows.update({f"style_token_grammar_{k}": v for k, v in _pairwise_cosine_stats(grammar).items()})
        rows.update({f"style_token_band_{k}": v for k, v in _pairwise_cosine_stats(band).items()})
    return rows


def _band_energy(delta: torch.Tensor) -> dict[str, float]:
    low = _lowpass(delta, 9)
    mid = _lowpass(delta, 3) - low
    high = delta.float() - _lowpass(delta, 3)
    total = delta.float().abs().mean().clamp_min(1e-8)
    low_abs = low.abs().mean()
    mid_abs = mid.abs().mean()
    high_abs = high.abs().mean()
    return {
        "delta_low_abs": _finite_float(low_abs),
        "delta_mid_abs": _finite_float(mid_abs),
        "delta_high_abs": _finite_float(high_abs),
        "delta_low_ratio": _finite_float(low_abs / total),
        "delta_mid_ratio": _finite_float(mid_abs / total),
        "delta_high_ratio": _finite_float(high_abs / total),
    }


def _style_response_separation(
    model,
    content: torch.Tensor,
    target_style_ids: list[int],
    ode_steps: int,
) -> dict[str, float]:
    outputs = []
    with torch.no_grad():
        for style_id in target_style_ids:
            sid = torch.full((content.shape[0],), int(style_id), dtype=torch.long, device=content.device)
            outputs.append(_integrate_with_grad(model, content, sid, ode_steps).detach().float())
    if len(outputs) <= 1:
        return {
            "response_pair_l2_mean": float("nan"),
            "response_pair_cos_mean": float("nan"),
            "response_delta_l2_mean": float("nan"),
        }
    pair_l2 = []
    pair_cos = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            a = outputs[i].flatten(1)
            b = outputs[j].flatten(1)
            pair_l2.append((a - b).square().mean(dim=1).sqrt())
            pair_cos.append(F.cosine_similarity(a, b, dim=1, eps=1e-6))
    deltas = [out - content.float() for out in outputs]
    delta_l2 = torch.stack([d.flatten(1).square().mean(dim=1).sqrt() for d in deltas], dim=0)
    return {
        "response_pair_l2_mean": _finite_float(torch.cat(pair_l2).mean()),
        "response_pair_cos_mean": _finite_float(torch.cat(pair_cos).mean()),
        "response_delta_l2_mean": _finite_float(delta_l2.mean()),
    }


def _stochastic_style_sensitivity(
    model,
    content: torch.Tensor,
    target_style_ids: list[int],
    ode_steps: int,
    *,
    noise_std: float,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if noise_std <= 0.0 or samples <= 0:
        return {}
    generator = torch.Generator(device=content.device)
    generator.manual_seed(int(seed))
    base_weight = model.style_emb.weight.detach().clone()
    by_style_delta_l2: list[torch.Tensor] = []
    by_style_delta_tv: list[torch.Tensor] = []
    by_style_high_ratio: list[torch.Tensor] = []
    by_style_mean_shift: list[torch.Tensor] = []
    try:
        for style_id in target_style_ids:
            sid = torch.full((content.shape[0],), int(style_id), dtype=torch.long, device=content.device)
            with torch.no_grad():
                model.style_emb.weight.copy_(base_weight)
                reference = _integrate_with_grad(model, content, sid, ode_steps).detach().float()
                outputs = []
                for _ in range(int(samples)):
                    model.style_emb.weight.copy_(base_weight)
                    row = model.style_emb.weight[int(style_id)]
                    row_norm = row.detach().float().norm().clamp_min(1e-3)
                    noise = torch.randn(
                        row.shape,
                        generator=generator,
                        device=row.device,
                        dtype=row.dtype,
                    )
                    row.add_(noise * float(noise_std) * row_norm)
                    outputs.append(_integrate_with_grad(model, content, sid, ode_steps).detach().float())
                stack = torch.stack(outputs, dim=0)
                sample_mean = stack.mean(dim=0)
                centered = stack - sample_mean.unsqueeze(0)
                diff = stack - reference.unsqueeze(0)
                by_style_delta_l2.append(diff.flatten(2).square().mean(dim=2).sqrt().mean())
                by_style_delta_tv.append(torch.stack([_tv_loss(out - reference) for out in outputs]).mean())
                high_ratios = []
                for out in outputs:
                    ratios = _band_energy(out - reference)
                    high_ratios.append(content.new_tensor(float(ratios["delta_high_ratio"])))
                by_style_high_ratio.append(torch.stack(high_ratios).mean())
                by_style_mean_shift.append(centered.flatten(2).square().mean(dim=2).sqrt().mean())
    finally:
        with torch.no_grad():
            model.style_emb.weight.copy_(base_weight)
    return {
        "stoch_noise_std": float(noise_std),
        "stoch_samples": float(samples),
        "stoch_response_l2_mean": _finite_float(torch.stack(by_style_delta_l2).mean()),
        "stoch_delta_tv_mean": _finite_float(torch.stack(by_style_delta_tv).mean()),
        "stoch_high_ratio_mean": _finite_float(torch.stack(by_style_high_ratio).mean()),
        "stoch_centered_response_l2_mean": _finite_float(torch.stack(by_style_mean_shift).mean()),
    }


def _evaluate_latent_response(
    model,
    config: dict,
    latent_root: Path,
    style_names: list[str],
    target_style_ids: list[int],
    *,
    batch_size: int,
    num_batches: int,
    ode_steps: int,
    highpass_kernel: int,
    device: str,
    seed: int,
    noise_std: float,
    noise_samples: int,
) -> tuple[list[dict], dict[str, float]]:
    rng = random.Random(seed)
    latent_index = _style_latent_index(latent_root, style_names)
    content_pool = [p for style in style_names for p in latent_index[style]]
    transport = SWDTransportCost(config)

    by_style: list[dict] = []
    model.eval()
    for style_id in target_style_ids:
        style_name = style_names[int(style_id)]
        accum: dict[str, list[float]] = {}
        for _ in range(max(1, int(num_batches))):
            content = _sample_latent_batch(content_pool, batch_size, device, rng)
            target = _sample_latent_batch(latent_index[style_name], batch_size, device, rng)
            sid = torch.full((batch_size,), int(style_id), dtype=torch.long, device=device)
            with torch.no_grad():
                pred = _integrate_with_grad(model, content, sid, ode_steps)
                swd_before = transport.aligned_cost(content, target)
                swd_after = transport.aligned_cost(pred, target)
                swd_hp_before = transport.aligned_cost(_highpass(content, highpass_kernel), _highpass(target, highpass_kernel))
                swd_hp_after = transport.aligned_cost(_highpass(pred, highpass_kernel), _highpass(target, highpass_kernel))
                grad_cos = 1.0 - _gradient_cosine_loss(pred, content)
                delta = pred - content
                metrics = {
                    "target_style_id": float(style_id),
                    "swd_before": _finite_float(swd_before),
                    "swd_after": _finite_float(swd_after),
                    "swd_gain": _finite_float(swd_before - swd_after),
                    "swd_hp_before": _finite_float(swd_hp_before),
                    "swd_hp_after": _finite_float(swd_hp_after),
                    "swd_hp_gain": _finite_float(swd_hp_before - swd_hp_after),
                    "content_mse": _finite_float(delta.float().square().mean()),
                    "delta_abs": _finite_float(delta.float().abs().mean()),
                    "delta_max": _finite_float(delta.float().abs().max()),
                    "delta_tv": _finite_float(_tv_loss(delta)),
                    "gradient_cosine": _finite_float(grad_cos),
                }
                metrics.update(_band_energy(delta))
            for key, value in metrics.items():
                accum.setdefault(key, []).append(float(value))
            del content, target, sid, pred, delta
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        row = {
            "target_style": style_name,
            "target_style_id": int(style_id),
        }
        for key, values in accum.items():
            if key == "target_style_id":
                continue
            row[key] = sum(values) / max(len(values), 1)
        by_style.append(row)

    shared_content = _sample_latent_batch(content_pool, batch_size, device, rng)
    summary = _style_response_separation(model, shared_content, target_style_ids, ode_steps)
    summary.update(
        _stochastic_style_sensitivity(
            model,
            shared_content,
            target_style_ids,
            ode_steps,
            noise_std=float(noise_std),
            samples=int(noise_samples),
            seed=int(seed) + 7919,
        )
    )
    del shared_content
    for key in by_style[0].keys():
        if key in {"target_style", "target_style_id"}:
            continue
        vals = [float(row[key]) for row in by_style if key in row and math.isfinite(float(row[key]))]
        summary[f"mean_{key}"] = sum(vals) / max(len(vals), 1) if vals else float("nan")
    return by_style, summary


def _parse_adapter_specs(specs: list[str], globs: list[str]) -> list[tuple[str, Path | None]]:
    parsed: list[tuple[str, Path | None]] = []
    for spec in specs:
        text = spec.strip()
        if not text:
            continue
        if "=" in text:
            name, path_text = text.split("=", 1)
            name = name.strip()
            path_text = path_text.strip()
        else:
            path_text = text
            name = Path(text).parent.name if Path(text).name == "style_adapter.pt" else Path(text).stem
        if path_text.lower() in {"", "none", "base"}:
            parsed.append((name or "base", None))
        else:
            path = Path(path_text)
            if not path.is_absolute():
                path = (ROOT / path).resolve()
            parsed.append((name, path))
    for pattern in globs:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                parsed.append((path.parent.name, path.resolve()))
    if not parsed:
        parsed.append(("base", None))
    seen: set[str] = set()
    unique: list[tuple[str, Path | None]] = []
    for name, path in parsed:
        base = name or (path.parent.name if path else "base")
        candidate = base
        idx = 2
        while candidate in seen:
            candidate = f"{base}_{idx}"
            idx += 1
        seen.add(candidate)
        unique.append((candidate, path))
    return unique


def _resolve_latent_root(config: dict, requested: Path | None) -> Path:
    if requested is not None:
        return requested if requested.is_absolute() else (ROOT / requested).resolve()
    data_root = str((config.get("data", {}) or {}).get("data_root", "")).strip()
    if data_root:
        p = Path(data_root)
        return p if p.is_absolute() else (ROOT / p).resolve()
    return ROOT.parent / "latent-256"


def _read_existing_full_eval(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return _read_summary_metrics(summary)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate style_adapter.pt quality as a frozen-backbone style actuator: "
            "embedding geometry, latent response, optional full eval."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", action="append", default=[], help="Adapter spec: name=path, path, or base=none.")
    parser.add_argument("--adapter-glob", action="append", default=[], help="ROOT-relative glob for adapter files.")
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/diagnostics/style_adapter_quality")
    parser.add_argument("--style-subdirs", type=str, default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--target-style-ids", type=str, default="1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--ode-steps", type=int, default=12)
    parser.add_argument("--highpass-kernel", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Optional stochastic style-code perturbation scale for sensitivity diagnostics only.",
    )
    parser.add_argument(
        "--noise-samples",
        type=int,
        default=0,
        help="Number of perturbation samples per style for the stochastic sensitivity gate.",
    )
    parser.add_argument("--run-full-eval", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=6)
    parser.add_argument("--vae-model", type=str, default="auto")
    parser.add_argument("--existing-summary", action="append", default=[], help="Optional name=summary.json for reuse.")
    args = parser.parse_args()

    style_names = [item.strip() for item in args.style_subdirs.split(",") if item.strip()]
    target_style_ids = [int(item.strip()) for item in args.target_style_ids.split(",") if item.strip()]
    adapters = _parse_adapter_specs(args.adapter, args.adapter_glob)
    existing_summaries: dict[str, Path] = {}
    for item in args.existing_summary:
        if "=" in item:
            name, path_text = item.split("=", 1)
            path = Path(path_text.strip())
            existing_summaries[name.strip()] = path if path.is_absolute() else (ROOT / path).resolve()

    base_model, config = _load_checkpoint_model(args.checkpoint, args.device)
    latent_root = _resolve_latent_root(config, args.latent_root)
    base_emb = base_model.style_emb.weight.detach().cpu()
    base_spatial = (
        base_model.style_spatial_id_16.detach().cpu()
        if hasattr(base_model, "style_spatial_id_16")
        else None
    )
    del base_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    summary_rows: list[dict] = []
    by_style_rows: list[dict] = []
    args.out_root.mkdir(parents=True, exist_ok=True)

    for index, (name, adapter_path) in enumerate(adapters):
        if adapter_path is not None and not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found for {name}: {adapter_path}")
        model, _ = _load_checkpoint_model(args.checkpoint, args.device)
        if adapter_path is not None:
            _apply_style_adapter(model, adapter_path, args.device)
        model.eval()
        adapter_dir = args.out_root / name
        emb_stats = _embedding_stats(model, base_emb, base_spatial)
        by_style, latent_summary = _evaluate_latent_response(
            model,
            config,
            latent_root,
            style_names,
            target_style_ids,
            batch_size=max(1, int(args.batch_size)),
            num_batches=max(1, int(args.num_batches)),
            ode_steps=max(1, int(args.ode_steps)),
            highpass_kernel=max(1, int(args.highpass_kernel)),
            device=args.device,
            seed=int(args.seed) + index * 1009,
            noise_std=max(0.0, float(args.noise_std)),
            noise_samples=max(0, int(args.noise_samples)),
        )
        for row in by_style:
            by_style_rows.append({"adapter": name, "adapter_path": str(adapter_path or ""), **row})

        full_eval_metrics = {}
        if name in existing_summaries:
            full_eval_metrics = _read_existing_full_eval(existing_summaries[name])
        elif args.run_full_eval and adapter_path is not None:
            summary = _run_full_eval(
                checkpoint=args.checkpoint,
                style_adapter=adapter_path,
                output_dir=adapter_dir / "full_eval",
                batch_size=max(1, int(args.eval_batch_size)),
                vae_model=args.vae_model,
            )
            _write_json(adapter_dir / "full_eval_summary.json", summary)
            full_eval_metrics = _read_summary_metrics(summary)

        summary_row = {
            "adapter": name,
            "adapter_path": str(adapter_path or ""),
            **emb_stats,
            **latent_summary,
            **{f"full_eval_{k}": v for k, v in full_eval_metrics.items()},
        }
        summary_rows.append(summary_row)
        _write_json(adapter_dir / "summary.json", summary_row)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    _write_csv(args.out_root / "style_adapter_quality_summary.csv", summary_rows)
    _write_csv(args.out_root / "style_adapter_quality_by_style.csv", by_style_rows)
    _write_json(
        args.out_root / "manifest.json",
        {
            "checkpoint": str(args.checkpoint),
            "latent_root": str(latent_root),
            "style_names": style_names,
            "target_style_ids": target_style_ids,
            "batch_size": int(args.batch_size),
            "num_batches": int(args.num_batches),
            "ode_steps": int(args.ode_steps),
            "highpass_kernel": int(args.highpass_kernel),
            "noise_std": float(args.noise_std),
            "noise_samples": int(args.noise_samples),
            "adapters": [{"name": name, "path": str(path or "")} for name, path in adapters],
        },
    )
    print(f"Saved adapter quality report to {args.out_root}")


if __name__ == "__main__":
    main()
