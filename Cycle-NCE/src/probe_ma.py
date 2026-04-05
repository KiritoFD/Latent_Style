from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dataset import _load_latent_file
from model import (
    AttentionBlock,
    CrossAttnAdaGN,
    NormFreeModulation,
    ResBlock,
    SpatialSelfAttention,
    StyleRoutingSkip,
    build_model_from_config,
)


PROBE_MODULE_TYPES = (
    CrossAttnAdaGN,
    ResBlock,
    AttentionBlock,
    SpatialSelfAttention,
    StyleRoutingSkip,
    NormFreeModulation,
)

EXTRA_PROBE_NAMES = {
    "enc_in",
    "down",
    "dec_up",
    "skip_fusion",
    "skip_up_proj",
    "skip_src_proj",
    "dec_post",
    "dec_out",
}


@dataclass
class ProbeBatch:
    content: torch.Tensor
    target: torch.Tensor
    source_style_id: int
    target_style_ids: torch.Tensor
    content_paths: list[Path]
    target_paths: list[Path]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe massive activations in Latent AdaCUT/LANCET checkpoints.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str((Path(__file__).resolve().parent.parent / "epoch_0060.pt").resolve()),
        help="Path to checkpoint .pt file",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional external config json. Defaults to checkpoint['config'].")
    parser.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu. Defaults to CUDA when available.")
    parser.add_argument("--num-samples", type=int, default=4, help="How many latent pairs to probe in one batch.")
    parser.add_argument("--source-style", type=str, default="photo", help="Source/content style name or integer id.")
    parser.add_argument("--target-style", type=str, default=None, help="Target style name or integer id. Defaults to first non-source style.")
    parser.add_argument("--content-latents", type=str, nargs="*", default=None, help="Explicit latent files for content batch.")
    parser.add_argument("--target-latents", type=str, nargs="*", default=None, help="Explicit latent files for target-style batch.")
    parser.add_argument("--top-k-layers", type=int, default=12, help="How many hottest layers to print.")
    parser.add_argument("--top-k-channels", type=int, default=6, help="How many channels per layer to print.")
    parser.add_argument("--top-k-spatial", type=int, default=6, help="How many spatial hotspots per layer to print.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON path for full probe dump.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic file sampling.")
    return parser.parse_args()


def _resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(checkpoint_path: Path, config_path: str | None) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if config_path is None:
        config = ckpt.get("config")
        if not isinstance(config, dict):
            raise KeyError("checkpoint does not contain a valid 'config' object")
        return config
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model(checkpoint_path: Path, model_cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model = build_model_from_config(model_cfg, use_checkpointing=False).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _resolve_data_root(config: dict[str, Any], checkpoint_path: Path) -> Path:
    data_cfg = config.get("data", {})
    data_root_raw = data_cfg.get("data_root")
    if not data_root_raw:
        raise KeyError("config.data.data_root is required for automatic latent sampling")
    data_root = Path(str(data_root_raw))
    if not data_root.is_absolute():
        candidates = [
            (checkpoint_path.parent / data_root).resolve(),
            (checkpoint_path.parent / "src" / data_root).resolve(),
            (Path(__file__).resolve().parent / data_root).resolve(),
            (Path.cwd() / data_root).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        data_root = candidates[0]
    return data_root.resolve()


def _coerce_style_id(style_arg: str | None, style_order: list[str], fallback_index: int | None = None) -> int:
    if style_arg is None:
        if fallback_index is None:
            raise ValueError("style argument is missing and no fallback index was provided")
        return int(fallback_index)
    raw = str(style_arg).strip()
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(style_order):
            return idx
        raise ValueError(f"style id {idx} out of range [0, {len(style_order) - 1}]")
    if raw in style_order:
        return style_order.index(raw)
    lowered = raw.lower()
    lowered_order = [name.lower() for name in style_order]
    if lowered in lowered_order:
        return lowered_order.index(lowered)
    raise ValueError(f"unknown style '{style_arg}', valid styles: {style_order}")


def _pick_default_target_style(source_id: int, style_order: list[str]) -> int:
    for idx in range(len(style_order)):
        if idx != source_id:
            return idx
    return source_id


def _resolve_latent_paths(
    explicit_paths: list[str] | None,
    *,
    style_dir: Path,
    count: int,
    seed: int,
) -> list[Path]:
    if explicit_paths:
        return [Path(p).resolve() for p in explicit_paths]
    files = sorted(list(style_dir.glob("*.pt")) + list(style_dir.glob("*.npy")))
    if not files:
        raise RuntimeError(f"no latent files found under {style_dir}")
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(files), generator=g)[: max(1, min(count, len(files)))].tolist()
    return [files[i].resolve() for i in indices]


def _stack_latents(paths: list[Path], device: torch.device) -> torch.Tensor:
    tensors = [_load_latent_file(path) for path in paths]
    batch = torch.stack(tensors, dim=0).to(device=device, dtype=torch.float32)
    return batch


def _build_probe_batch(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    num_samples: int,
    source_style: str | None,
    target_style: str | None,
    content_latents: list[str] | None,
    target_latents: list[str] | None,
    seed: int,
) -> tuple[ProbeBatch, list[str]]:
    data_cfg = config.get("data", {})
    style_order = list(data_cfg.get("style_subdirs", []))
    if not style_order:
        raise KeyError("config.data.style_subdirs is required for automatic latent sampling")

    source_id = _coerce_style_id(source_style, style_order, fallback_index=0)
    target_id = _coerce_style_id(target_style, style_order, fallback_index=_pick_default_target_style(source_id, style_order))
    data_root = _resolve_data_root(config, checkpoint_path)
    src_dir = data_root / style_order[source_id]
    tgt_dir = data_root / style_order[target_id]

    content_paths = _resolve_latent_paths(
        content_latents,
        style_dir=src_dir,
        count=num_samples,
        seed=seed,
    )
    target_paths = _resolve_latent_paths(
        target_latents,
        style_dir=tgt_dir,
        count=max(num_samples, len(content_paths)),
        seed=seed + 17,
    )

    batch_size = min(len(content_paths), len(target_paths), max(1, num_samples))
    content_paths = content_paths[:batch_size]
    target_paths = target_paths[:batch_size]

    content = _stack_latents(content_paths, device)
    target = _stack_latents(target_paths, device)
    target_style_ids = torch.full((batch_size,), target_id, device=device, dtype=torch.long)
    return ProbeBatch(
        content=content,
        target=target,
        source_style_id=source_id,
        target_style_ids=target_style_ids,
        content_paths=content_paths,
        target_paths=target_paths,
    ), style_order


def _classify_stage(name: str) -> str:
    if name.startswith("hires_body"):
        return "hires"
    if name.startswith("body"):
        return "body"
    if name.startswith("decoder_blocks") or name.startswith("dec_"):
        return "decoder"
    if name.startswith("skip_") or name.startswith("skip_router"):
        return "skip"
    if name.startswith("enc_in") or name.startswith("down"):
        return "stem"
    if name.startswith("dec_out"):
        return "output"
    return "other"


def _topk_pairs(values: torch.Tensor, k: int) -> list[tuple[int, float]]:
    if values.numel() == 0:
        return []
    topk = min(max(1, int(k)), values.numel())
    top_vals, top_idx = torch.topk(values, k=topk)
    return [(int(idx.item()), float(val.item())) for idx, val in zip(top_idx, top_vals)]


class MAProbe:
    def __init__(self, model: torch.nn.Module, *, top_k_spatial: int) -> None:
        self.model = model
        self.top_k_spatial = max(1, int(top_k_spatial))
        self.handles: list[Any] = []
        self.stats: dict[str, dict[str, Any]] = {}

    def _ensure_layer(self, name: str, output: torch.Tensor) -> dict[str, Any]:
        entry = self.stats.get(name)
        channels = int(output.shape[1])
        if entry is not None:
            return entry
        entry = {
            "name": name,
            "stage": _classify_stage(name),
            "shape": list(output.shape),
            "num_calls": 0,
            "count_per_channel": 0,
            "channel_abs_max": torch.zeros(channels, dtype=torch.float32),
            "channel_abs_sum": torch.zeros(channels, dtype=torch.float64),
            "channel_peak_batch": torch.full((channels,), -1, dtype=torch.long),
            "channel_peak_y": torch.full((channels,), -1, dtype=torch.long),
            "channel_peak_x": torch.full((channels,), -1, dtype=torch.long),
            "layer_hotspots": [],
        }
        self.stats[name] = entry
        return entry

    def _update_hotspots(self, entry: dict[str, Any], abs_output: torch.Tensor) -> None:
        bsz, channels, height, width = abs_output.shape
        flat = abs_output.reshape(-1)
        top_k = min(self.top_k_spatial, flat.numel())
        vals, idxs = torch.topk(flat, k=top_k)
        candidates = []
        stride_c = height * width
        stride_b = channels * stride_c
        for val, flat_idx in zip(vals, idxs):
            flat_value = int(flat_idx.item())
            batch_idx = flat_value // stride_b
            rem = flat_value % stride_b
            channel_idx = rem // stride_c
            rem = rem % stride_c
            y_idx = rem // width
            x_idx = rem % width
            candidates.append(
                {
                    "value": float(val.item()),
                    "batch": int(batch_idx),
                    "channel": int(channel_idx),
                    "y": int(y_idx),
                    "x": int(x_idx),
                }
            )
        merged = entry["layer_hotspots"] + candidates
        merged.sort(key=lambda item: item["value"], reverse=True)
        entry["layer_hotspots"] = merged[: self.top_k_spatial]

    def _hook_fn(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                return
            if output.ndim != 4:
                return
            with torch.no_grad():
                detached = output.detach()
                if detached.device.type != "cpu":
                    detached = detached.to(dtype=torch.float32).cpu()
                else:
                    detached = detached.float()
                entry = self._ensure_layer(name, detached)
                entry["num_calls"] += 1
                entry["shape"] = list(detached.shape)
                abs_output = detached.abs()
                channel_max = abs_output.amax(dim=(0, 2, 3))
                channel_sum = abs_output.sum(dim=(0, 2, 3), dtype=torch.float64)
                entry["channel_abs_max"] = torch.maximum(entry["channel_abs_max"], channel_max)
                entry["channel_abs_sum"] += channel_sum
                entry["count_per_channel"] += int(abs_output.shape[0] * abs_output.shape[2] * abs_output.shape[3])

                per_channel = abs_output.permute(1, 0, 2, 3).reshape(abs_output.shape[1], -1)
                peak_vals, peak_idx = per_channel.max(dim=1)
                current_best = entry["channel_abs_max"] <= peak_vals
                height, width = abs_output.shape[2], abs_output.shape[3]
                for ch in torch.nonzero(current_best, as_tuple=False).view(-1).tolist():
                    flat_idx = int(peak_idx[ch].item())
                    batch_idx = flat_idx // (height * width)
                    spatial_idx = flat_idx % (height * width)
                    y_idx = spatial_idx // width
                    x_idx = spatial_idx % width
                    entry["channel_peak_batch"][ch] = batch_idx
                    entry["channel_peak_y"][ch] = y_idx
                    entry["channel_peak_x"][ch] = x_idx
                self._update_hotspots(entry, abs_output)

        return hook

    def attach(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, PROBE_MODULE_TYPES) or name in EXTRA_PROBE_NAMES:
                self.handles.append(module.register_forward_hook(self._hook_fn(name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summarize(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for name, entry in self.stats.items():
            count = max(1, int(entry["count_per_channel"]))
            channel_abs_mean = (entry["channel_abs_sum"] / count).to(dtype=torch.float32)
            channel_abs_max = entry["channel_abs_max"]
            ma_ratio = channel_abs_max / channel_abs_mean.clamp_min(1e-8)
            summaries.append(
                {
                    "name": name,
                    "stage": entry["stage"],
                    "shape": entry["shape"],
                    "num_calls": int(entry["num_calls"]),
                    "max_ma_ratio": float(ma_ratio.max().item()),
                    "mean_ma_ratio": float(ma_ratio.mean().item()),
                    "channel_abs_max": [float(v) for v in channel_abs_max.tolist()],
                    "channel_abs_mean": [float(v) for v in channel_abs_mean.tolist()],
                    "ma_ratio": [float(v) for v in ma_ratio.tolist()],
                    "channel_peak_batch": [int(v) for v in entry["channel_peak_batch"].tolist()],
                    "channel_peak_y": [int(v) for v in entry["channel_peak_y"].tolist()],
                    "channel_peak_x": [int(v) for v in entry["channel_peak_x"].tolist()],
                    "layer_hotspots": entry["layer_hotspots"],
                }
            )
        summaries.sort(key=lambda item: item["max_ma_ratio"], reverse=True)
        return summaries


def _format_top_channels(layer: dict[str, Any], top_k_channels: int) -> list[str]:
    ratios = torch.tensor(layer["ma_ratio"], dtype=torch.float32)
    means = torch.tensor(layer["channel_abs_mean"], dtype=torch.float32)
    maxes = torch.tensor(layer["channel_abs_max"], dtype=torch.float32)
    lines = []
    for ch_idx, ratio in _topk_pairs(ratios, top_k_channels):
        lines.append(
            (
                f"    ch={ch_idx:>3} ratio={ratio:>8.2f} "
                f"max={float(maxes[ch_idx]):>8.4f} mean={float(means[ch_idx]):>8.4f} "
                f"peak=(b{layer['channel_peak_batch'][ch_idx]}, y{layer['channel_peak_y'][ch_idx]}, x{layer['channel_peak_x'][ch_idx]})"
            )
        )
    return lines


def _format_hotspots(layer: dict[str, Any]) -> list[str]:
    hotspots = layer.get("layer_hotspots", [])
    lines = []
    for spot in hotspots:
        lines.append(
            f"    val={spot['value']:.4f} ch={spot['channel']} at (b{spot['batch']}, y{spot['y']}, x{spot['x']})"
        )
    return lines


def _print_report(
    *,
    checkpoint_path: Path,
    device: torch.device,
    style_order: list[str],
    probe_batch: ProbeBatch,
    summaries: list[dict[str, Any]],
    top_k_layers: int,
    top_k_channels: int,
) -> None:
    target_id = int(probe_batch.target_style_ids[0].item())
    source_id = int(probe_batch.source_style_id)
    print(f"[probe] checkpoint: {checkpoint_path}")
    print(f"[probe] device: {device}")
    print(f"[probe] batch_size: {probe_batch.content.shape[0]}")
    print(f"[probe] source files: {[p.name for p in probe_batch.content_paths]}")
    print(f"[probe] target files: {[p.name for p in probe_batch.target_paths]}")
    print(f"[probe] source style: id={source_id} name={style_order[source_id]}")
    print(f"[probe] target style: id={target_id} name={style_order[target_id]}")
    print("")

    stage_buckets: dict[str, list[float]] = {}
    for layer in summaries:
        stage_buckets.setdefault(layer["stage"], []).append(float(layer["max_ma_ratio"]))
    print("[stage summary]")
    for stage in ("stem", "hires", "body", "skip", "decoder", "output", "other"):
        values = stage_buckets.get(stage)
        if not values:
            continue
        t = torch.tensor(values, dtype=torch.float32)
        print(
            f"  {stage:>7}: layers={len(values):>2} max={float(t.max()):>8.2f} "
            f"mean={float(t.mean()):>8.2f} median={float(t.median()):>8.2f}"
        )
    print("")

    print(f"[top {min(top_k_layers, len(summaries))} MA layers]")
    for rank, layer in enumerate(summaries[:top_k_layers], start=1):
        print(
            f"{rank:>2}. {layer['name']}  stage={layer['stage']} shape={tuple(layer['shape'])} "
            f"max_ratio={layer['max_ma_ratio']:.2f} mean_ratio={layer['mean_ma_ratio']:.2f}"
        )
        print("  top channels:")
        for line in _format_top_channels(layer, top_k_channels):
            print(line)
        print("  hotspots:")
        for line in _format_hotspots(layer):
            print(line)
        print("")


def main() -> None:
    args = _parse_args()
    torch.manual_seed(int(args.seed))

    checkpoint_path = Path(args.checkpoint).resolve()
    device = _resolve_device(args.device)
    config = _load_config(checkpoint_path, args.config)
    model = _load_model(checkpoint_path, config["model"], device)
    probe_batch, style_order = _build_probe_batch(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        num_samples=max(1, int(args.num_samples)),
        source_style=args.source_style,
        target_style=args.target_style,
        content_latents=args.content_latents,
        target_latents=args.target_latents,
        seed=int(args.seed),
    )

    probe = MAProbe(model, top_k_spatial=int(args.top_k_spatial))
    probe.attach()
    with torch.no_grad():
        _ = model(
            probe_batch.content,
            style_id=probe_batch.target_style_ids,
            target_style_latent=probe_batch.target,
        )
    probe.remove()

    summaries = probe.summarize()
    _print_report(
        checkpoint_path=checkpoint_path,
        device=device,
        style_order=style_order,
        probe_batch=probe_batch,
        summaries=summaries,
        top_k_layers=max(1, int(args.top_k_layers)),
        top_k_channels=max(1, int(args.top_k_channels)),
    )

    if args.json_out:
        output_path = Path(args.json_out).resolve()
        payload = {
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "styles": style_order,
            "source_style_id": int(probe_batch.source_style_id),
            "target_style_id": int(probe_batch.target_style_ids[0].item()),
            "content_paths": [str(p) for p in probe_batch.content_paths],
            "target_paths": [str(p) for p in probe_batch.target_paths],
            "summaries": summaries,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[probe] wrote json: {output_path}")


if __name__ == "__main__":
    main()
