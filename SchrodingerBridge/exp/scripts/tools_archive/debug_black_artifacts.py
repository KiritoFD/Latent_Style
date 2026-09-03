from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

_VAE_CACHE: dict[str, Any] = {}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_stats(x: torch.Tensor | None) -> dict[str, Any]:
    if x is None:
        return {
            "present": False,
            "shape": None,
            "finite_ratio": 1.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
        }
    y = x.detach().float()
    finite = torch.isfinite(y)
    safe = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "present": True,
        "shape": list(y.shape),
        "finite_ratio": float(finite.float().mean().item()) if y.numel() > 0 else 1.0,
        "min": float(safe.min().item()) if y.numel() > 0 else 0.0,
        "max": float(safe.max().item()) if y.numel() > 0 else 0.0,
        "mean": float(safe.mean().item()) if y.numel() > 0 else 0.0,
        "std": float(safe.std().item()) if y.numel() > 1 else 0.0,
        "mean_abs": float(safe.abs().mean().item()) if y.numel() > 0 else 0.0,
        "max_abs": float(safe.abs().amax().item()) if y.numel() > 0 else 0.0,
    }


def _summarize_activation(module_name: str, output: Any) -> dict[str, Any] | None:
    if not torch.is_tensor(output):
        if isinstance(output, (list, tuple)) and output and torch.is_tensor(output[0]):
            output = output[0]
        else:
            return None
    stats = _tensor_stats(output)
    stats["module"] = module_name
    return stats


def _module_probe_names(model: torch.nn.Module) -> list[str]:
    candidates = [
        "body_blocks.3.to_q",
        "body_blocks.3.to_k",
        "body_blocks.3.to_v",
        "body_blocks.3.gate_conv.2",
        "decoder_blocks.1.conv2",
        "dec_post.1",
        "dec_out",
    ]
    existing = {name for name, _ in model.named_modules()}
    return [name for name in candidates if name in existing]


def _gradient_norm(component: torch.Tensor, param: torch.nn.Parameter) -> float:
    if not component.requires_grad:
        return 0.0
    grad = torch.autograd.grad(component, param, retain_graph=True, allow_unused=True)[0]
    if grad is None:
        return 0.0
    grad = torch.nan_to_num(grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    return float(torch.linalg.vector_norm(grad).item())


def _topk_outliers(x: torch.Tensor, *, k: int = 8) -> list[dict[str, Any]]:
    if x.ndim != 4:
        return []
    y = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = y.abs().reshape(y.shape[0], -1)
    topk = min(k, flat.shape[1])
    vals, idx = torch.topk(flat, k=topk, dim=1)
    out: list[dict[str, Any]] = []
    channels, height, width = y.shape[1:]
    hw = height * width
    for batch_idx in range(y.shape[0]):
        samples = []
        for value, flat_idx in zip(vals[batch_idx], idx[batch_idx]):
            flat_i = int(flat_idx.item())
            c = flat_i // hw
            rem = flat_i % hw
            h = rem // width
            w = rem % width
            raw_value = float(y[batch_idx, c, h, w].item())
            samples.append(
                {
                    "abs_value": float(value.item()),
                    "value": raw_value,
                    "channel": c,
                    "y": h,
                    "x": w,
                }
            )
        out.append({"batch_index": batch_idx, "points": samples})
    return out


def _decode_latents(debug_dir: Path, latent: torch.Tensor) -> torch.Tensor:
    repo = debug_dir
    sys.path.insert(0, str((repo / "src").resolve()))
    from utils.inference import decode_latent, load_vae

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = _VAE_CACHE.get(device)
    if vae is None:
        vae = load_vae(device=device)
        _VAE_CACHE[device] = vae
    chunks: list[torch.Tensor] = []
    latent = latent.detach().contiguous()
    for start in range(0, latent.shape[0], 2):
        batch = latent[start : start + 2]
        decoded = decode_latent(vae, batch, device=device).detach().float().cpu()
        chunks.append(decoded)
    return torch.cat(chunks, dim=0)


def _decode_artifact_stats(imgs: torch.Tensor) -> dict[str, Any]:
    dark_mask = imgs < 0.02
    per_sample_dark_ratio = dark_mask.flatten(1).float().mean(dim=1)
    per_sample_min = imgs.flatten(1).min(dim=1).values
    per_sample_mean = imgs.flatten(1).mean(dim=1)
    darkest = int(torch.argmax(per_sample_dark_ratio).item())
    return {
        "per_sample_dark_ratio": [float(v.item()) for v in per_sample_dark_ratio],
        "per_sample_min_pixel": [float(v.item()) for v in per_sample_min],
        "per_sample_mean_pixel": [float(v.item()) for v in per_sample_mean],
        "darkest_sample_index": darkest,
        "darkest_sample_ratio": float(per_sample_dark_ratio[darkest].item()),
        "darkest_sample_min_pixel": float(per_sample_min[darkest].item()),
        "darkest_sample_mean_pixel": float(per_sample_mean[darkest].item()),
    }


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _samplewise_latent_stats(pred_velocity: torch.Tensor, pred_endpoint: torch.Tensor) -> list[dict[str, float]]:
    v = torch.nan_to_num(pred_velocity.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    e = torch.nan_to_num(pred_endpoint.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    out: list[dict[str, float]] = []
    for i in range(v.shape[0]):
        out.append(
            {
                "batch_index": int(i),
                "velocity_mean_abs": float(v[i].abs().mean().item()),
                "velocity_max_abs": float(v[i].abs().amax().item()),
                "endpoint_mean_abs": float(e[i].abs().mean().item()),
                "endpoint_max_abs": float(e[i].abs().amax().item()),
            }
        )
    return out


def _top_endpoint_candidates(samplewise_stats: list[dict[str, float]], *, top_k: int = 8) -> list[int]:
    ranked = sorted(samplewise_stats, key=lambda row: row["endpoint_max_abs"], reverse=True)
    return [int(row["batch_index"]) for row in ranked[: max(1, min(top_k, len(ranked)))]]


def _semantic_attention_diagnostics(attn: torch.Tensor | None) -> dict[str, Any]:
    if attn is None or not torch.is_tensor(attn):
        return {"present": False}
    a = torch.nan_to_num(attn.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    probs = a.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)
    max_entropy = float(np.log(max(probs.shape[-1], 2)))
    top1 = probs.max(dim=-1).values
    q_tokens = int(probs.shape[1])
    q_side = int(round(q_tokens ** 0.5))
    chosen = probs.argmax(dim=-1).float()
    neighbor_delta = None
    if q_side * q_side == q_tokens and q_side > 1:
        chosen_2d = chosen.view(chosen.shape[0], q_side, q_side)
        right = (chosen_2d[:, :, 1:] != chosen_2d[:, :, :-1]).float().mean()
        down = (chosen_2d[:, 1:, :] != chosen_2d[:, :-1, :]).float().mean()
        neighbor_delta = float(((right + down) * 0.5).item())
    return {
        "present": True,
        "query_tokens": q_tokens,
        "key_tokens": int(probs.shape[-1]),
        "query_side": q_side,
        "mean_entropy": float(entropy.mean().item()),
        "normalized_entropy": float(entropy.mean().item() / max(max_entropy, 1e-12)),
        "mean_top1_prob": float(top1.mean().item()),
        "min_top1_prob": float(top1.min().item()),
        "neighbor_argmax_change_ratio": neighbor_delta,
    }


def _grid_boundary_diagnostics(x: torch.Tensor, *, query_side: int | None) -> dict[str, Any]:
    if x.ndim != 4 or query_side is None or query_side <= 1:
        return {"present": False}
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if h % query_side != 0 or w % query_side != 0:
        return {"present": False, "reason": "shape_not_divisible"}
    cell_h = h // query_side
    cell_w = w // query_side
    if cell_h <= 0 or cell_w <= 0:
        return {"present": False, "reason": "invalid_cell"}
    y = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    dh = (y[:, :, 1:, :] - y[:, :, :-1, :]).abs().mean(dim=1)
    dw = (y[:, :, :, 1:] - y[:, :, :, :-1]).abs().mean(dim=1)

    h_positions = torch.arange(h - 1)
    w_positions = torch.arange(w - 1)
    h_boundary = ((h_positions + 1) % cell_h) == 0
    w_boundary = ((w_positions + 1) % cell_w) == 0
    h_inner = ~h_boundary
    w_inner = ~w_boundary

    def _masked_mean(arr: torch.Tensor, mask: torch.Tensor, dim: int) -> float:
        if mask.sum().item() == 0:
            return 0.0
        if dim == 0:
            vals = arr[:, mask, :]
        else:
            vals = arr[:, :, mask]
        return float(vals.mean().item())

    h_boundary_mean = _masked_mean(dh, h_boundary, 0)
    h_inner_mean = _masked_mean(dh, h_inner, 0)
    w_boundary_mean = _masked_mean(dw, w_boundary, 1)
    w_inner_mean = _masked_mean(dw, w_inner, 1)
    return {
        "present": True,
        "query_side": query_side,
        "cell_h": cell_h,
        "cell_w": cell_w,
        "horizontal_boundary_mean": h_boundary_mean,
        "horizontal_inner_mean": h_inner_mean,
        "horizontal_boundary_ratio": float(h_boundary_mean / max(h_inner_mean, 1e-6)),
        "vertical_boundary_mean": w_boundary_mean,
        "vertical_inner_mean": w_inner_mean,
        "vertical_boundary_ratio": float(w_boundary_mean / max(w_inner_mean, 1e-6)),
    }


def _save_focus_artifacts(
    output_path: Path,
    *,
    darkest_slot: int,
    darkest_batch_index: int,
    content_imgs: torch.Tensor,
    pred_imgs: torch.Tensor,
    target_imgs: torch.Tensor,
    pred_endpoint: torch.Tensor,
    pred_velocity: torch.Tensor,
) -> dict[str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")
    triptych_path = stem.parent / f"{stem.name}_darkest_triptych.png"
    endpoint_heat_path = stem.parent / f"{stem.name}_darkest_endpoint_abs.png"
    velocity_heat_path = stem.parent / f"{stem.name}_darkest_velocity_abs.png"
    darkmask_path = stem.parent / f"{stem.name}_darkest_darkmask.png"

    triptych = torch.cat(
        [
            content_imgs[0].clamp(0.0, 1.0),
            pred_imgs[darkest_slot].clamp(0.0, 1.0),
            target_imgs[0].clamp(0.0, 1.0),
        ],
        dim=-1,
    )
    save_image(triptych, triptych_path)

    endpoint_abs = pred_endpoint[darkest_batch_index].detach().float().abs().mean(dim=0, keepdim=True)
    endpoint_abs = endpoint_abs / endpoint_abs.amax().clamp_min(1e-6)
    save_image(endpoint_abs, endpoint_heat_path)

    velocity_abs = pred_velocity[darkest_batch_index].detach().float().abs().mean(dim=0, keepdim=True)
    velocity_abs = velocity_abs / velocity_abs.amax().clamp_min(1e-6)
    save_image(velocity_abs, velocity_heat_path)

    darkmask = (pred_imgs[darkest_slot] < 0.02).any(dim=0, keepdim=True).float()
    save_image(darkmask, darkmask_path)
    return {
        "darkest_triptych": str(triptych_path),
        "darkest_endpoint_abs": str(endpoint_heat_path),
        "darkest_velocity_abs": str(velocity_heat_path),
        "darkest_darkmask": str(darkmask_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace black-artifact origins on a trained bridge checkpoint.")
    parser.add_argument("--config", required=True, help="Config path.")
    parser.add_argument("--resume", default="", help="Checkpoint path to analyze.")
    parser.add_argument("--batch-index", type=int, default=0, help="Which dataloader batch to inspect.")
    parser.add_argument("--disable-amp", action="store_true", help="Run debug forward without AMP.")
    parser.add_argument("--detect-anomaly", action="store_true", help="Enable torch.autograd anomaly detection.")
    parser.add_argument("--decode", action="store_true", help="Decode predicted latents and score dark pixels.")
    parser.add_argument("--output", default="", help="Optional report output path.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str((repo / "src").resolve()))

    from config_loader import load_config
    from dataset import AdaCUTLatentDataset
    from trainer import SBTrainer

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.resume:
        config.setdefault("training", {})
        config["training"]["resume_checkpoint"] = str(Path(args.resume).resolve())

    seed = int(config.get("training", {}).get("seed", 42))
    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = SBTrainer(config=config, device=device, config_path=str(config_path))
    trainer.use_amp = bool(trainer.use_amp and not args.disable_amp)

    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.get("data_root", "../latent-256"),
        style_subdirs=data_cfg.get("style_subdirs", ["photo", "Hayao", "monet", "vangogh", "cezanne"]),
        allow_hflip=bool(data_cfg.get("allow_hflip", False)),
        identity_ratio=data_cfg.get("identity_ratio", None),
        batch_size_hint=int(train_cfg.get("batch_size", 32)),
        balance_target_styles_per_batch=bool(data_cfg.get("balance_target_styles_per_batch", True)),
        preload_to_gpu=False,
        preload_max_vram_gb=float(data_cfg.get("preload_max_vram_gb", 0.0)),
        preload_reserve_ratio=float(data_cfg.get("preload_reserve_ratio", 0.35)),
        virtual_length_multiplier=int(data_cfg.get("virtual_length_multiplier", 1)),
        device=str(device),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=bool(train_cfg.get("pin_memory", device.type == "cuda")),
    )

    iterator = iter(loader)
    batch = None
    for _ in range(max(0, int(args.batch_index)) + 1):
        batch = next(iterator)
    if batch is None:
        raise RuntimeError("Unable to fetch debug batch.")

    content = batch["content"].to(device, non_blocking=True)
    target_style = batch["target_style"].to(device, non_blocking=True)
    target_style_id = batch["target_style_id"].to(device, non_blocking=True)
    source_style_id = batch.get("source_style_id")
    if source_style_id is not None:
        source_style_id = source_style_id.to(device, non_blocking=True)

    activation_log: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hooks = []
    for module_name in _module_probe_names(trainer.model):
        module = dict(trainer.model.named_modules())[module_name]

        def _make_hook(name: str):
            def _hook(_module, _inputs, output):
                stats = _summarize_activation(name, output)
                if stats is not None:
                    activation_log[name].append(stats)

            return _hook

        hooks.append(module.register_forward_hook(_make_hook(module_name)))

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    trainer.model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast("cuda", enabled=trainer.use_amp, dtype=trainer.amp_dtype)
    else:
        autocast_ctx = torch.autocast("cpu", enabled=False)

    with autocast_ctx:
        debug_payload = trainer.loss_fn.compute_debug(
            trainer.model,
            content=content,
            target_style=target_style,
            target_style_id=target_style_id,
            source_style_id=source_style_id,
        )

    metrics = debug_payload["metrics"]
    components = debug_payload["components"]
    state = debug_payload["state"]

    param_probes = {
        "dec_out.weight": trainer.model.dec_out.weight,
    }
    named_params = dict(trainer.model.named_parameters())
    if "body_blocks.3.to_q.weight" in named_params:
        param_probes["body_blocks.3.to_q.weight"] = named_params["body_blocks.3.to_q.weight"]
    if "body_blocks.3.to_k.weight" in named_params:
        param_probes["body_blocks.3.to_k.weight"] = named_params["body_blocks.3.to_k.weight"]

    grad_balance: dict[str, dict[str, float]] = {}
    for comp_name, comp_value in components.items():
        grad_balance[comp_name] = {
            probe_name: _gradient_norm(comp_value, probe_param)
            for probe_name, probe_param in param_probes.items()
        }

    pred_velocity = state["pred_velocity"]
    pred_endpoint = state["pred_endpoint"]
    latent_outliers = {
        "pred_velocity": _topk_outliers(pred_velocity, k=8) if isinstance(pred_velocity, torch.Tensor) else [],
        "pred_endpoint": _topk_outliers(pred_endpoint, k=8) if isinstance(pred_endpoint, torch.Tensor) else [],
    }

    decoded_stats = None
    decoded_paths = None
    samplewise_stats = None
    grid_diag = {}
    attn_diag = _semantic_attention_diagnostics(state.get("semantic_attn"))
    query_side = attn_diag.get("query_side") if attn_diag.get("present") else None

    if isinstance(pred_endpoint, torch.Tensor):
        samplewise_stats = _samplewise_latent_stats(pred_velocity, pred_endpoint)
        grid_diag["pred_endpoint"] = _grid_boundary_diagnostics(pred_endpoint, query_side=query_side)
        grid_diag["pred_velocity"] = _grid_boundary_diagnostics(pred_velocity, query_side=query_side)

    if args.decode and isinstance(pred_endpoint, torch.Tensor):
        candidate_indices = _top_endpoint_candidates(samplewise_stats or [], top_k=4)
        candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long)
        pred_subset = pred_endpoint.index_select(0, candidate_tensor.to(pred_endpoint.device))
        pred_imgs = _decode_latents(repo, pred_subset)
        decoded_stats = _decode_artifact_stats(pred_imgs)
        decoded_stats["candidate_batch_indices"] = candidate_indices
        grid_diag["decoded_pred"] = _grid_boundary_diagnostics(pred_imgs, query_side=query_side)
        darkest_slot = int(decoded_stats["darkest_sample_index"])
        darkest_batch_index = int(candidate_indices[darkest_slot])
        decoded_stats["darkest_candidate_slot"] = darkest_slot
        decoded_stats["darkest_sample_index"] = darkest_batch_index
        one_idx = torch.tensor([darkest_batch_index], dtype=torch.long)
        content_imgs = _decode_latents(repo, state["content"].index_select(0, one_idx.to(state["content"].device)))
        target_imgs = _decode_latents(repo, state["target_style"].index_select(0, one_idx.to(state["target_style"].device)))
        grid_diag["decoded_content"] = _grid_boundary_diagnostics(content_imgs, query_side=query_side)
        grid_diag["decoded_target"] = _grid_boundary_diagnostics(target_imgs, query_side=query_side)
        decoded_paths = _save_focus_artifacts(
            Path(args.output).resolve() if args.output else (repo / "orthogonal_phase_space_sweep_debug" / "tmp.json"),
            darkest_slot=darkest_slot,
            darkest_batch_index=darkest_batch_index,
            content_imgs=content_imgs,
            pred_imgs=pred_imgs,
            target_imgs=target_imgs,
            pred_endpoint=pred_endpoint.detach().cpu(),
            pred_velocity=pred_velocity.detach().cpu(),
        )
        if samplewise_stats is not None:
            dark_ratio = decoded_stats["per_sample_dark_ratio"]
            endpoint_max = [samplewise_stats[i]["endpoint_max_abs"] for i in candidate_indices]
            velocity_max = [samplewise_stats[i]["velocity_max_abs"] for i in candidate_indices]
            decoded_stats["correlation_darkratio_endpointmax"] = _corr(dark_ratio, endpoint_max)
            decoded_stats["correlation_darkratio_velocitymax"] = _corr(dark_ratio, velocity_max)

    report = {
        "config": str(config_path),
        "resume_checkpoint": str(config.get("training", {}).get("resume_checkpoint", "")),
        "batch_index": int(args.batch_index),
        "device": str(device),
        "use_amp": bool(trainer.use_amp),
        "metrics": {
            key: float(torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).item())
            for key, value in metrics.items()
            if torch.is_tensor(value) and value.ndim == 0
        },
        "component_grad_balance": grad_balance,
        "state_stats": {
            key: _tensor_stats(value) if isinstance(value, torch.Tensor) or value is None else {"present": False}
            for key, value in state.items()
        },
        "activation_summary": {
            module_name: stats_list[-1]
            for module_name, stats_list in activation_log.items()
            if stats_list
        },
        "semantic_attention_diagnostics": attn_diag,
        "grid_boundary_diagnostics": grid_diag,
        "latent_outliers": latent_outliers,
        "samplewise_latent_stats": samplewise_stats,
        "decoded_artifact_stats": decoded_stats,
        "decoded_artifact_paths": decoded_paths,
        "target_style_ids": [int(v) for v in target_style_id.detach().cpu().tolist()],
        "source_style_ids": [int(v) for v in source_style_id.detach().cpu().tolist()] if source_style_id is not None else None,
    }

    for hook in hooks:
        hook.remove()

    output_path = Path(args.output).resolve() if args.output else (
        repo / "orthogonal_phase_space_sweep_debug" / f"black_artifact_report_batch{int(args.batch_index):02d}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
