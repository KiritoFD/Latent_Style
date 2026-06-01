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


def _tokenizer_rows(checkpoint: Path, names: list[str], device: str) -> tuple[list[dict], dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ExperimentConfig.from_mapping(payload["config"])
    model = build_model_from_config(cfg.model, use_checkpointing=False).to(device)
    model.load_state_dict(strip_compile_prefix(payload["model_state_dict"]), strict=False)
    model.eval()
    ids = torch.arange(len(names), device=device, dtype=torch.long)
    with torch.no_grad():
        codes = model.encode_style_id(ids).float().detach().cpu()
    gram = F.normalize(codes, dim=1) @ F.normalize(codes, dim=1).T
    _, svals, _ = torch.linalg.svd(codes - codes.mean(dim=0, keepdim=True), full_matrices=False)
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
    summary = {
        "tokenizer_rank_svals": [float(v) for v in svals.tolist()],
        "tokenizer_effective_rank": float((svals.sum().square() / svals.square().sum().clamp_min(1e-8)).item()),
        "tokenizer_mean_offdiag_cos": float(((gram.sum() - torch.diagonal(gram).sum()) / max(1, len(names) * (len(names) - 1))).item()),
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-training probes for style latent/tokenizer geometry.")
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--classes", type=str, default="")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-style", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    names = _style_names(args.latent_root, args.classes)
    style_rows, means = _latent_stats(args.latent_root, names, max_per_style=int(args.max_per_style))
    pair_rows = _pair_rows(means)
    token_rows: list[dict] = []
    token_summary: dict = {}
    if args.checkpoint is not None:
        token_rows, token_summary = _tokenizer_rows(args.checkpoint, names, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "style_latent_stats.csv", style_rows)
    _write_csv(args.output_dir / "style_latent_pairs.csv", pair_rows)
    if token_rows:
        _write_csv(args.output_dir / "tokenizer_code_stats.csv", token_rows)
    summary = {
        "latent_root": str(args.latent_root),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "classes": names,
        "style_latent_stats": style_rows,
        "style_latent_pairs": pair_rows,
        "tokenizer_code_stats": token_rows,
        **token_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
