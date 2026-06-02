from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _repo_src_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


SRC_PATH = str(_repo_src_path())
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from config_schema import ExperimentConfig  # noqa: E402
from model import build_model_from_config  # noqa: E402
from utils.inference import LGTInference  # noqa: E402
from utils.training import strip_compile_prefix  # noqa: E402


def _load_latent(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("latent", "z", "tensor"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload):
        raise TypeError(f"Unsupported latent payload: {path}")
    x = payload.float()
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected latent [C,H,W], got {tuple(x.shape)} from {path}")
    return x.contiguous()


def _style_names(latent_root: Path, raw: str) -> list[str]:
    if raw.strip():
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [p.name for p in sorted(latent_root.iterdir(), key=lambda x: x.name) if p.is_dir()]


def _spectral_ratio(x: torch.Tensor) -> dict[str, float]:
    # x: [N,C,H,W]
    fft = torch.fft.rfft2(x.float(), norm="ortho")
    amp = fft.abs().mean(dim=1)
    h, w = amp.shape[-2:]
    yy = torch.linspace(-1.0, 1.0, h, device=amp.device).view(h, 1)
    xx = torch.linspace(0.0, 1.0, w, device=amp.device).view(1, w)
    rr = torch.sqrt(yy.square() + xx.square())
    low = amp[..., rr <= 0.25].mean()
    mid = amp[..., (rr > 0.25) & (rr <= 0.55)].mean()
    high = amp[..., rr > 0.55].mean()
    return {
        "fft_low": float(low.item()),
        "fft_mid": float(mid.item()),
        "fft_high": float(high.item()),
        "fft_high_low_ratio": float((high / low.clamp_min(1e-8)).item()),
    }


def _latent_stats(latent_root: Path, names: list[str], max_per_style: int) -> tuple[list[dict], dict[str, torch.Tensor]]:
    rows: list[dict] = []
    means: dict[str, torch.Tensor] = {}
    for name in names:
        paths = sorted((latent_root / name).glob("*.pt"), key=lambda p: p.name)
        if max_per_style > 0:
            paths = paths[:max_per_style]
        if not paths:
            raise FileNotFoundError(f"No latents under {latent_root / name}")
        batch = torch.stack([_load_latent(path) for path in paths], dim=0)
        flat = batch.flatten(1)
        mean_vec = flat.mean(dim=0)
        means[name] = mean_vec
        stats = _spectral_ratio(batch)
        centered = flat - mean_vec
        cov_trace = centered.square().mean()
        rows.append(
            {
                "style": name,
                "count": len(paths),
                "latent_abs_mean": float(flat.abs().mean().item()),
                "latent_std": float(flat.std(unbiased=False).item()),
                "latent_cov_trace": float(cov_trace.item()),
                **stats,
            }
        )
    return rows, means


def _pair_rows(means: dict[str, torch.Tensor]) -> list[dict]:
    rows = []
    names = list(means)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            va = means[a]
            vb = means[b]
            rows.append(
                {
                    "style_a": a,
                    "style_b": b,
                    "latent_mean_l2": float((va - vb).norm().item()),
                    "latent_mean_cos": float(F.cosine_similarity(va.view(1, -1), vb.view(1, -1), dim=1).item()),
                }
            )
    return rows


def _effective_rank(matrix: torch.Tensor) -> tuple[float, list[float]]:
    _, svals, _ = torch.linalg.svd(matrix - matrix.mean(dim=0, keepdim=True), full_matrices=False)
    rank = float((svals.sum().square() / svals.square().sum().clamp_min(1e-8)).item())
    return rank, [float(v) for v in svals.tolist()]


def _tokenizer_rows(checkpoint: Path, names: list[str], device: str) -> tuple[list[dict], list[dict], dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ExperimentConfig.from_mapping(payload["config"])
    model = build_model_from_config(cfg.model, use_checkpointing=False).to(device)
    model.load_state_dict(strip_compile_prefix(payload["model_state_dict"]), strict=False)
    model.eval()
    ids = torch.arange(len(names), device=device, dtype=torch.long)
    with torch.no_grad():
        codes = model.encode_style_id(ids).float().detach().cpu()
    gram = F.normalize(codes, dim=1) @ F.normalize(codes, dim=1).T
    effective_rank, svals_list = _effective_rank(codes)
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "style": name,
                "style_code_norm": float(codes[i].norm().item()),
                "style_code_abs_mean": float(codes[i].abs().mean().item()),
                "nearest_code_cos": float(torch.cat([gram[i, :i], gram[i, i + 1 :]]).max().item()) if len(names) > 1 else 0.0,
            }
        )
    pair_rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names[i + 1 :], start=i + 1):
            va = codes[i]
            vb = codes[j]
            pair_rows.append(
                {
                    "style_a": a,
                    "style_b": b,
                    "style_code_l2": float((va - vb).norm().item()),
                    "style_code_cos": float(gram[i, j].item()),
                }
            )
    summary = {
        "tokenizer_rank_svals": svals_list,
        "tokenizer_effective_rank": effective_rank,
        "tokenizer_mean_offdiag_cos": float(((gram.sum() - torch.diagonal(gram).sum()) / max(1, len(names) * (len(names) - 1))).item()),
    }
    return rows, pair_rows, summary


def _content_paths_for_delta_probe(latent_root: Path, names: list[str], max_content_per_style: int) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for name in names:
        style_paths = sorted((latent_root / name).glob("*.pt"), key=lambda p: p.name)
        if max_content_per_style > 0:
            style_paths = style_paths[:max_content_per_style]
        paths.extend((name, p) for p in style_paths)
    if not paths:
        raise FileNotFoundError(f"No delta-probe latents under {latent_root}")
    return paths


@torch.no_grad()
def _generated_delta_rows(
    *,
    checkpoint: Path,
    latent_root: Path,
    names: list[str],
    device: str,
    max_content_per_style: int,
    batch_size: int,
    num_steps: int,
    step_size: float,
    style_strength: float,
) -> tuple[list[dict], list[dict], dict]:
    content_paths = _content_paths_for_delta_probe(latent_root, names, max_content_per_style)
    infer = LGTInference(
        str(checkpoint),
        device=device,
        num_steps=max(1, int(num_steps)),
        step_size=float(step_size),
        style_strength=float(style_strength),
    )
    batch_size = max(1, int(batch_size))
    target_sums = {name: None for name in names}
    target_abs_sums = {name: 0.0 for name in names}
    target_l2_sums = {name: 0.0 for name in names}
    target_counts = {name: 0 for name in names}

    for start in range(0, len(content_paths), batch_size):
        batch_meta = content_paths[start : start + batch_size]
        x = torch.stack([_load_latent(path) for _, path in batch_meta], dim=0).to(device)
        for target_id, target_name in enumerate(names):
            target_ids = torch.full((x.shape[0],), target_id, dtype=torch.long, device=device)
            y = infer.transfer_style(x, target_style_id=target_ids, num_steps=max(1, int(num_steps)))
            delta = (y.float() - x.float()).detach()
            flat = delta.flatten(1).cpu()
            delta_sum = flat.sum(dim=0)
            if target_sums[target_name] is None:
                target_sums[target_name] = delta_sum
            else:
                target_sums[target_name] = target_sums[target_name] + delta_sum
            target_abs_sums[target_name] += float(flat.abs().mean(dim=1).sum().item())
            target_l2_sums[target_name] += float(flat.norm(dim=1).sum().item())
            target_counts[target_name] += int(flat.shape[0])

    means: dict[str, torch.Tensor] = {}
    rows: list[dict] = []
    for target_name in names:
        count = max(1, target_counts[target_name])
        mean_vec = target_sums[target_name] / count
        means[target_name] = mean_vec
        rows.append(
            {
                "target_style": target_name,
                "content_count": int(target_counts[target_name]),
                "delta_mean_l2": float(mean_vec.norm().item()),
                "delta_sample_l2_mean": float(target_l2_sums[target_name] / count),
                "delta_sample_abs_mean": float(target_abs_sums[target_name] / count),
            }
        )

    pair_rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            va = means[a]
            vb = means[b]
            pair_rows.append(
                {
                    "target_a": a,
                    "target_b": b,
                    "delta_mean_l2": float((va - vb).norm().item()),
                    "delta_mean_cos": float(F.cosine_similarity(va.view(1, -1), vb.view(1, -1), dim=1).item()),
                }
            )
    matrix = torch.stack([means[name] for name in names], dim=0)
    delta_rank, delta_svals = _effective_rank(matrix)
    gram = F.normalize(matrix, dim=1) @ F.normalize(matrix, dim=1).T
    summary = {
        "generated_delta_content_count": len(content_paths),
        "generated_delta_effective_rank": delta_rank,
        "generated_delta_rank_svals": delta_svals,
        "generated_delta_mean_offdiag_cos": float(((gram.sum() - torch.diagonal(gram).sum()) / max(1, len(names) * (len(names) - 1))).item()),
    }
    return rows, pair_rows, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().mul(y.square().sum()).sqrt().clamp_min(1e-12)
    return float((x * y).sum().div(denom).item())


def _pair_geometry_correlations(
    latent_pairs: list[dict],
    tokenizer_pairs: list[dict],
    delta_pairs: list[dict],
) -> dict:
    if not delta_pairs:
        return {}
    latent = {tuple(sorted((r["style_a"], r["style_b"]))): r for r in latent_pairs}
    tokenizer = {tuple(sorted((r["style_a"], r["style_b"]))): r for r in tokenizer_pairs}
    delta = {tuple(sorted((r["target_a"], r["target_b"]))): r for r in delta_pairs}
    keys = sorted(set(latent) & set(tokenizer) & set(delta))
    if len(keys) < 2:
        return {}
    return {
        "corr_latent_l2_to_delta_l2": _pearson(
            [float(latent[k]["latent_mean_l2"]) for k in keys],
            [float(delta[k]["delta_mean_l2"]) for k in keys],
        ),
        "corr_latent_cos_to_delta_cos": _pearson(
            [float(latent[k]["latent_mean_cos"]) for k in keys],
            [float(delta[k]["delta_mean_cos"]) for k in keys],
        ),
        "corr_tokenizer_l2_to_delta_l2": _pearson(
            [float(tokenizer[k]["style_code_l2"]) for k in keys],
            [float(delta[k]["delta_mean_l2"]) for k in keys],
        ),
        "corr_tokenizer_cos_to_delta_cos": _pearson(
            [float(tokenizer[k]["style_code_cos"]) for k in keys],
            [float(delta[k]["delta_mean_cos"]) for k in keys],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-training probes for style latent/tokenizer geometry.")
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--classes", type=str, default="")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-style", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--delta-probe", action="store_true", help="Probe whether target style codes produce separated generated latent residuals.")
    parser.add_argument("--delta-probe-max-content-per-style", type=int, default=20)
    parser.add_argument("--delta-probe-batch-size", type=int, default=8)
    parser.add_argument("--delta-probe-num-steps", type=int, default=4)
    parser.add_argument("--delta-probe-step-size", type=float, default=1.0)
    parser.add_argument("--delta-probe-style-strength", type=float, default=1.0)
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    names = _style_names(args.latent_root, args.classes)
    style_rows, means = _latent_stats(args.latent_root, names, max_per_style=int(args.max_per_style))
    pair_rows = _pair_rows(means)
    token_rows: list[dict] = []
    token_pair_rows: list[dict] = []
    token_summary: dict = {}
    if args.checkpoint is not None:
        token_rows, token_pair_rows, token_summary = _tokenizer_rows(args.checkpoint, names, device=device)
    delta_rows: list[dict] = []
    delta_pair_rows: list[dict] = []
    delta_summary: dict = {}
    if args.delta_probe:
        if args.checkpoint is None:
            raise ValueError("--delta-probe requires --checkpoint")
        delta_rows, delta_pair_rows, delta_summary = _generated_delta_rows(
            checkpoint=args.checkpoint,
            latent_root=args.latent_root,
            names=names,
            device=device,
            max_content_per_style=int(args.delta_probe_max_content_per_style),
            batch_size=int(args.delta_probe_batch_size),
            num_steps=int(args.delta_probe_num_steps),
            step_size=float(args.delta_probe_step_size),
            style_strength=float(args.delta_probe_style_strength),
        )
    correlation_summary = _pair_geometry_correlations(pair_rows, token_pair_rows, delta_pair_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "style_latent_stats.csv", style_rows)
    _write_csv(args.output_dir / "style_latent_pairs.csv", pair_rows)
    if token_rows:
        _write_csv(args.output_dir / "tokenizer_code_stats.csv", token_rows)
        _write_csv(args.output_dir / "tokenizer_code_pairs.csv", token_pair_rows)
    if delta_rows:
        _write_csv(args.output_dir / "generated_delta_stats.csv", delta_rows)
        _write_csv(args.output_dir / "generated_delta_pairs.csv", delta_pair_rows)
    summary = {
        "latent_root": str(args.latent_root),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "classes": names,
        "style_latent_stats": style_rows,
        "style_latent_pairs": pair_rows,
        "tokenizer_code_stats": token_rows,
        "tokenizer_code_pairs": token_pair_rows,
        "generated_delta_stats": delta_rows,
        "generated_delta_pairs": delta_pair_rows,
        **token_summary,
        **delta_summary,
        **correlation_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
