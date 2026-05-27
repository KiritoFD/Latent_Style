from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_schema import ExperimentConfig  # noqa: E402
from ot_cost import SWDTransportCost  # noqa: E402


TOKEN_KEYS = {
    "grammar": "style_tokenizer.grammar_vocab.weight",
    "band": "style_tokenizer.band_vocab.weight",
    "identity": "style_tokenizer.identity_vocab",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported checkpoint format: {path}")
    return payload


def _state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload.get("model_state_dict") or payload.get("state_dict")
    if state is None and any(key in payload for key in TOKEN_KEYS.values()):
        state = payload
    if not isinstance(state, dict):
        raise ValueError("could not find model/tokenizer state dict")
    return state


def _load_tokens(path: Path, style_names: list[str]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = _load_payload(path)
    state = _state_dict(payload)
    tokens: dict[str, torch.Tensor] = {}
    for field, suffix in TOKEN_KEYS.items():
        match = next((key for key in state if str(key).endswith(suffix)), None)
        if match is not None:
            tensor = state[match].detach().float()
            if tensor.ndim == 2:
                tokens[field] = tensor[: len(style_names)]
    if "grammar" not in tokens or "band" not in tokens:
        raise ValueError(f"missing tokenizer grammar/band tensors in {path}")
    if "identity" not in tokens:
        n = tokens["grammar"].shape[0]
        identity = torch.eye(n, dtype=torch.float32)
        identity = identity - identity.mean(dim=0, keepdim=True)
        tokens["identity"] = F.normalize(identity, dim=1, eps=1e-6)
    return tokens, payload


def _resolve_latent_root(payload: dict[str, Any], requested: Path | None) -> Path:
    if requested is not None:
        return requested if requested.is_absolute() else (ROOT / requested).resolve()
    cfg = payload.get("config") if isinstance(payload, dict) else None
    data_root = ""
    if isinstance(cfg, dict):
        data_root = str((cfg.get("data", {}) or {}).get("data_root", "") or "")
    root = Path(data_root) if data_root else ROOT.parent / "latent-256"
    return root if root.is_absolute() else (ROOT / root).resolve()


def _sample_latents(root: Path, style: str, count: int, rng: random.Random) -> torch.Tensor:
    style_dir = root / style
    paths = sorted(style_dir.glob("*.pt")) + sorted(style_dir.glob("*.pth"))
    if not paths:
        raise FileNotFoundError(f"no latent tensors found in {style_dir}")
    chosen = [rng.choice(paths) for _ in range(max(1, int(count)))]
    tensors = []
    for path in chosen:
        item = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(item, dict):
            if "latent" in item:
                item = item["latent"]
            elif "z" in item:
                item = item["z"]
            else:
                tensor_values = [value for value in item.values() if torch.is_tensor(value)]
                if not tensor_values:
                    raise ValueError(f"no tensor found in {path}")
                item = tensor_values[0]
        if not torch.is_tensor(item):
            raise ValueError(f"latent file did not contain a tensor: {path}")
        if item.ndim == 3:
            item = item.unsqueeze(0)
        if item.ndim != 4:
            raise ValueError(f"expected latent shape BCHW/CHW, got {tuple(item.shape)} in {path}")
        tensors.append(item.float())
    return torch.cat(tensors, dim=0)


def _lowpass(x: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel <= 1:
        return x.float()
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    return F.avg_pool2d(F.pad(x.float(), (pad, pad, pad, pad), mode="reflect"), kernel_size=kernel, stride=1)


def _feature(x: torch.Tensor, branch: str, kernel: int) -> torch.Tensor:
    branch = str(branch).lower()
    low = _lowpass(x, kernel)
    if branch == "low":
        return low
    if branch == "high":
        return x.float() - low
    if branch == "abs_high":
        return (x.float() - low).abs()
    return x.float()


def _band_energy_vector(x: torch.Tensor, kernel: int) -> torch.Tensor:
    low = _lowpass(x, kernel)
    inner_kernel = max(3, int(kernel) // 2)
    if inner_kernel % 2 == 0:
        inner_kernel += 1
    inner = _lowpass(x, inner_kernel)
    mid = inner - low
    high = x.float() - inner
    vec = torch.stack(
        [
            low.float().var(unbiased=False),
            mid.float().var(unbiased=False),
            high.float().var(unbiased=False),
        ]
    )
    return vec / vec.sum().clamp_min(1e-12)


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return float("nan")
    tx = torch.tensor(x, dtype=torch.float64)
    ty = torch.tensor(y, dtype=torch.float64)
    tx = tx - tx.mean()
    ty = ty - ty.mean()
    denom = torch.sqrt(tx.square().sum() * ty.square().sum()).clamp_min(1e-12)
    return float((tx * ty).sum().div(denom).item())


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) * 0.5
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _pairwise_token_dist(tokens: torch.Tensor, names: list[str], field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dist = float((tokens[i].float() - tokens[j].float()).norm().item())
            rows.append({"style_a": names[i], "style_b": names[j], f"{field}_token_l2": dist})
    return rows


def _concat_tokens(tokens: dict[str, torch.Tensor], fields: list[str]) -> torch.Tensor:
    parts = [tokens[field].float() for field in fields if field in tokens]
    if not parts:
        raise ValueError(f"no token fields found among {fields}")
    return torch.cat(parts, dim=1)


def _cross_cov_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(int(a.shape[0]), int(b.shape[0]))
    if n < 2:
        return 0.0
    a = a[:n].float()
    b = b[:n].float()
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    cov = a.t().matmul(b) / float(n - 1)
    aa = a.t().matmul(a) / float(n - 1)
    bb = b.t().matmul(b) / float(n - 1)
    denom = torch.sqrt(aa.square().sum() * bb.square().sum()).clamp_min(1e-12)
    return float((cov.square().sum().sqrt() / denom.sqrt()).item())


def _effective_rank(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    s = s[s > 1e-8]
    if int(s.numel()) == 0:
        return 0.0
    p = s / s.sum().clamp_min(1e-8)
    return float(torch.exp(-(p * p.clamp_min(1e-8).log()).sum()).item())


def _default_config(payload: dict[str, Any]) -> ExperimentConfig:
    cfg = payload.get("config")
    if isinstance(cfg, dict):
        return ExperimentConfig.from_mapping(cfg)
    return ExperimentConfig()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose style tokenizer as a metric space.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--style-subdirs", default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--lowpass-kernel", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "exp/diagnostics/style_token_metric_space")
    args = parser.parse_args()

    names = [item.strip() for item in args.style_subdirs.split(",") if item.strip()]
    tokens, payload = _load_tokens(args.checkpoint, names)
    root = _resolve_latent_root(payload, args.latent_root)
    config = _default_config(payload)
    cost = SWDTransportCost(config)
    rng = random.Random(int(args.seed))
    device = torch.device(args.device)

    latents = {
        name: _sample_latents(root, name, int(args.sample_count), rng).to(device=device)
        for name in names
    }
    energy_vectors = {
        name: _band_energy_vector(latents[name], int(args.lowpass_kernel)).detach().cpu()
        for name in names
    }

    distance_rows: list[dict[str, Any]] = []
    branches = ["full", "low", "high", "abs_high"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            row: dict[str, Any] = {"style_a": a_name, "style_b": b_name}
            for branch in branches:
                a_feat = _feature(latents[a_name], branch, int(args.lowpass_kernel))
                b_feat = _feature(latents[b_name], branch, int(args.lowpass_kernel))
                with torch.no_grad():
                    row[f"data_{branch}_swd"] = float(cost.pairwise_cost(a_feat, b_feat).mean().detach().cpu().item())
            energy_a = energy_vectors[a_name]
            energy_b = energy_vectors[b_name]
            row["data_band_energy_l2"] = float((energy_a - energy_b).norm().item())
            row["data_log_band_energy_l2"] = float((energy_a.clamp_min(1e-8).log() - energy_b.clamp_min(1e-8).log()).norm().item())
            distance_rows.append(row)

    for field in ("identity", "grammar", "band"):
        token_rows = _pairwise_token_dist(tokens[field], names, field)
        lookup = {(row["style_a"], row["style_b"]): row for row in token_rows}
        for row in distance_rows:
            row.update(lookup[(row["style_a"], row["style_b"])])
    combo = _concat_tokens(tokens, ["identity", "grammar", "band"])
    lookup = {(row["style_a"], row["style_b"]): row for row in _pairwise_token_dist(combo, names, "all")}
    for row in distance_rows:
        row.update(lookup[(row["style_a"], row["style_b"])])

    token_metric_keys = ["identity_token_l2", "grammar_token_l2", "band_token_l2", "all_token_l2"]
    data_metric_keys = [f"data_{branch}_swd" for branch in branches] + [
        "data_band_energy_l2",
        "data_log_band_energy_l2",
    ]
    correlation_rows: list[dict[str, Any]] = []
    for token_key in token_metric_keys:
        tx = [float(row[token_key]) for row in distance_rows]
        for data_key in data_metric_keys:
            dy = [float(row[data_key]) for row in distance_rows]
            correlation_rows.append(
                {
                    "token_metric": token_key,
                    "data_metric": data_key,
                    "pearson": _pearson(tx, dy),
                    "spearman": _spearman(tx, dy),
                }
            )

    style_rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        z = latents[name]
        low = _feature(z, "low", int(args.lowpass_kernel))
        high = _feature(z, "high", int(args.lowpass_kernel))
        full_var = float(z.float().var(unbiased=False).detach().cpu().item())
        low_var = float(low.float().var(unbiased=False).detach().cpu().item())
        high_var = float(high.float().var(unbiased=False).detach().cpu().item())
        energy = energy_vectors[name]
        style_rows.append(
            {
                "style": name,
                "sample_count": int(z.shape[0]),
                "full_var": full_var,
                "low_var": low_var,
                "high_var": high_var,
                "high_to_low_var": high_var / max(low_var, 1e-12),
                "energy_low": float(energy[0].item()),
                "energy_mid": float(energy[1].item()),
                "energy_high": float(energy[2].item()),
                "grammar_norm": float(tokens["grammar"][idx].norm().item()),
                "band_norm": float(tokens["band"][idx].norm().item()),
                "identity_norm": float(tokens["identity"][idx].norm().item()),
            }
        )

    geometry_rows = [
        {
            "metric": "cross_cov_identity_grammar",
            "value": _cross_cov_norm(tokens["identity"], tokens["grammar"]),
        },
        {
            "metric": "cross_cov_identity_band",
            "value": _cross_cov_norm(tokens["identity"], tokens["band"]),
        },
        {
            "metric": "cross_cov_grammar_band",
            "value": _cross_cov_norm(tokens["grammar"], tokens["band"]),
        },
        {"metric": "effective_rank_identity", "value": _effective_rank(tokens["identity"])},
        {"metric": "effective_rank_grammar", "value": _effective_rank(tokens["grammar"])},
        {"metric": "effective_rank_band", "value": _effective_rank(tokens["band"])},
        {"metric": "effective_rank_all", "value": _effective_rank(combo)},
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "style_data_token_distances.csv", distance_rows)
    _write_csv(args.out_dir / "style_token_metric_correlations.csv", correlation_rows)
    _write_csv(args.out_dir / "style_token_geometry.csv", geometry_rows)
    _write_csv(args.out_dir / "style_frequency_statistics.csv", style_rows)

    best_rows = sorted(
        correlation_rows,
        key=lambda row: abs(float(row["spearman"])) if math.isfinite(float(row["spearman"])) else -1.0,
        reverse=True,
    )
    lines = [
        "# Style Token Metric-Space Diagnosis",
        "",
        f"checkpoint: `{args.checkpoint}`",
        f"latent_root: `{root}`",
        f"sample_count_per_style: `{int(args.sample_count)}`",
        "",
        "## Best Token/Data Distance Correlations",
        "",
        "| token metric | data metric | pearson | spearman |",
        "|---|---|---:|---:|",
    ]
    for row in best_rows[:10]:
        lines.append(
            f"| {row['token_metric']} | {row['data_metric']} | {float(row['pearson']):.4f} | {float(row['spearman']):.4f} |"
        )
    lines += [
        "",
        "## Geometry",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for row in geometry_rows:
        lines.append(f"| {row['metric']} | {float(row['value']):.6f} |")
    lookup_corr = {
        (str(row["token_metric"]), str(row["data_metric"])): row
        for row in correlation_rows
    }

    def corr_line(title: str, token_key: str, data_key: str) -> str:
        row = lookup_corr.get((token_key, data_key))
        if row is None:
            return f"| {title} | {token_key} | {data_key} | nan | nan | missing |"
        spearman = float(row["spearman"])
        pearson = float(row["pearson"])
        verdict = "pass" if math.isfinite(spearman) and abs(spearman) >= 0.50 else "weak"
        return f"| {title} | {token_key} | {data_key} | {pearson:.4f} | {spearman:.4f} | {verdict} |"

    lines += [
        "",
        "## Axiom Readout",
        "",
        "| axiom | token metric | data metric | pearson | spearman | verdict |",
        "|---|---|---|---:|---:|---|",
        corr_line("identity should preserve low-frequency measure", "identity_token_l2", "data_low_swd"),
        corr_line("grammar should preserve high-frequency measure", "grammar_token_l2", "data_high_swd"),
        corr_line("grammar should preserve abs high-frequency texture", "grammar_token_l2", "data_abs_high_swd"),
        corr_line("band should preserve energy allocation", "band_token_l2", "data_log_band_energy_l2"),
        corr_line("all fields should preserve full style measure", "all_token_l2", "data_full_swd"),
    ]
    lines += [
        "",
        "## Interpretation Rules",
        "",
        "- Low token/data correlation means the tokenizer is not an isometric map of the style data manifold.",
        "- High cross-covariance means the named subspaces are not algebraically disentangled.",
        "- A style with high data high/low variance but near-zero grammar or band norm is under-covered.",
    ]
    (args.out_dir / "style_token_metric_space_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "latent_root": str(root),
                "style_names": names,
                "sample_count": int(args.sample_count),
                "lowpass_kernel": int(args.lowpass_kernel),
                "outputs": [
                    "style_data_token_distances.csv",
                    "style_token_metric_correlations.csv",
                    "style_token_geometry.csv",
                    "style_frequency_statistics.csv",
                    "style_token_metric_space_readout.md",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.out_dir / "style_token_metric_space_readout.md")


if __name__ == "__main__":
    main()
