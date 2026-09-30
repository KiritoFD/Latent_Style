from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def _style_cache_name(style_id: int, style: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(style)).strip("_") or f"style_{style_id}"
    return f"{style_id:02d}_{safe}.pt"


def _load_packed_manifest(latent_root: Path) -> dict[str, object] | None:
    manifest_path = latent_root / ".latent_cache" / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_style_examples(latent_root: Path, style: str, style_id: int, max_count: int) -> tuple[torch.Tensor, list[str]]:
    style_dir = latent_root / style
    raw_paths = sorted(style_dir.glob("*.pt"), key=lambda p: p.name)
    if raw_paths:
        if max_count > 0:
            raw_paths = raw_paths[:max_count]
        batch = torch.stack([_load_latent(path) for path in raw_paths], dim=0)
        refs = [str(path.relative_to(latent_root)) for path in raw_paths]
        return batch, refs

    manifest = _load_packed_manifest(latent_root)
    if manifest is None:
        raise FileNotFoundError(f"No raw latents or packed manifest found for style={style} under {latent_root}")
    styles = manifest.get("styles", {})
    if not isinstance(styles, dict) or style not in styles:
        raise FileNotFoundError(f"Style={style} missing from packed manifest under {latent_root}")
    style_payload = styles.get(style, {})
    packed_rel = style_payload.get("packed") if isinstance(style_payload, dict) else None
    packed_path = latent_root / ".latent_cache" / str(packed_rel or Path("packed") / _style_cache_name(style_id, style))
    if not packed_path.exists():
        raise FileNotFoundError(f"Packed latent cache missing for style={style}: {packed_path}")
    payload = torch.load(packed_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        latents = payload.get("latents", payload.get("data", payload.get("tensor")))
        refs = payload.get("files", [])
    else:
        latents = payload
        refs = []
    if not torch.is_tensor(latents):
        raise TypeError(f"Unsupported packed latent payload: {packed_path}")
    batch = latents.float()
    if batch.ndim != 4:
        raise ValueError(f"Expected packed latents [N,C,H,W], got {tuple(batch.shape)} from {packed_path}")
    refs = [str(item) for item in refs] if isinstance(refs, list) else []
    if max_count > 0:
        batch = batch[:max_count]
        refs = refs[:max_count]
    if not refs:
        refs = [f"{style}/packed_{idx:04d}.pt" for idx in range(batch.shape[0])]
    return batch.contiguous(), refs


def _style_names(latent_root: Path, raw: str) -> list[str]:
    if raw.strip():
        return [item.strip() for item in raw.split(",") if item.strip()]
    dir_names = [
        p.name
        for p in sorted(latent_root.iterdir(), key=lambda x: x.name)
        if p.is_dir() and not p.name.startswith(".")
    ]
    if dir_names:
        return dir_names
    manifest = _load_packed_manifest(latent_root)
    style_subdirs = manifest.get("style_subdirs", []) if isinstance(manifest, dict) else []
    if isinstance(style_subdirs, list):
        return [str(item) for item in style_subdirs if str(item).strip()]
    return []


def _git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            or None
        )
    except Exception:
        return None


def _runtime_metadata(requested_device: str, resolved_device: str) -> dict[str, str | None]:
    meta: dict[str, str | None] = {
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": str(bool(torch.cuda.is_available())),
        "cuda_device_name": None,
    }
    if resolved_device != "cpu" and torch.cuda.is_available():
        try:
            meta["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            meta["cuda_device_name"] = None
    return meta


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_model(checkpoint: Path, device: str) -> tuple[torch.nn.Module, ExperimentConfig, str]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ExperimentConfig.from_mapping(payload["config"])
    model = build_model_from_config(cfg.model, use_checkpointing=False).to(device)
    model.load_state_dict(strip_compile_prefix(payload["model_state_dict"]), strict=False)
    model.eval()
    objective_mode = str(cfg.bridge.objective_mode).strip().lower()
    return model, cfg, objective_mode


def _parse_run(raw: str) -> tuple[str, Path]:
    for sep in ("::", "="):
        if sep in raw:
            label, value = raw.split(sep, 1)
            label = label.strip()
            value = value.strip()
            if not label or not value:
                break
            return label, Path(value)
    raise ValueError(f"Expected --run label=checkpoint or label::checkpoint, got: {raw}")


def _content_records(latent_root: Path, names: list[str], max_content_per_style: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for style_id, name in enumerate(names):
        batch, refs = _load_style_examples(latent_root, name, style_id, max_content_per_style)
        for idx in range(batch.shape[0]):
            ref = refs[idx] if idx < len(refs) else f"{name}/packed_{idx:04d}.pt"
            records.append(
                {
                    "source_style": name,
                    "latent_ref": str(ref),
                    "latent": batch[idx].contiguous(),
                }
            )
    if not records:
        raise FileNotFoundError(f"No path-probe latents found under {latent_root}")
    return records


def _resolve_horizon(model: torch.nn.Module, *, step_size: float, style_strength: float | None) -> float:
    if hasattr(model, "_resolve_integration_horizon"):
        return float(model._resolve_integration_horizon(step_size=step_size, style_strength=style_strength))
    strength = 1.0 if style_strength is None else float(style_strength)
    return max(0.0, min(1.0, float(step_size) * strength))


def _resolve_rollout_mode(raw: str, objective_mode: str) -> str:
    mode = str(raw).strip().lower()
    if mode == "auto":
        return "field" if objective_mode == "omf" else "integrate"
    if mode not in {"field", "integrate"}:
        raise ValueError(f"Unsupported rollout mode: {raw}")
    return mode


@torch.no_grad()
def _probe_run(
    *,
    label: str,
    checkpoint: Path,
    latent_root: Path,
    names: list[str],
    device: str,
    max_content_per_style: int,
    batch_size: int,
    num_steps: int,
    step_size: float,
    style_strength: float | None,
    rollout_mode: str,
) -> tuple[list[dict], list[dict], dict]:
    model, cfg, objective_mode = _load_model(checkpoint, device)
    effective_mode = _resolve_rollout_mode(rollout_mode, objective_mode)
    content_records = _content_records(latent_root, names, max_content_per_style)
    steps = max(1, int(num_steps))
    horizon = _resolve_horizon(model, step_size=step_size, style_strength=style_strength)
    dt = horizon / float(steps) if steps > 0 else 0.0

    time_buckets: dict[tuple[str, int], dict[str, float]] = {}
    summary_buckets: dict[str, dict[str, float]] = {}
    for split in ("all", "identity", "transfer"):
        summary_buckets[split] = {
            "count": 0.0,
            "endpoint_disp_l2_sum": 0.0,
            "path_length_l2_sum": 0.0,
            "path_length_ratio_sum": 0.0,
            "peak_velocity_l2_sum": 0.0,
        }
        for step_idx in range(steps):
            time_buckets[(split, step_idx)] = {
                "count": 0.0,
                "vel_l2_sum": 0.0,
                "vel_l2_sum_sq": 0.0,
                "vel_abs_sum": 0.0,
                "step_disp_l2_sum": 0.0,
                "step_disp_l2_sum_sq": 0.0,
            }

    batch_size = max(1, int(batch_size))
    for start in range(0, len(content_records), batch_size):
        batch_meta = content_records[start : start + batch_size]
        x0 = torch.stack([record["latent"] for record in batch_meta], dim=0).to(device)
        src_styles = [str(record["source_style"]) for record in batch_meta]
        for target_id, target_name in enumerate(names):
            target_ids = torch.full((x0.shape[0],), target_id, dtype=torch.long, device=device)
            h = x0.clone() if effective_mode == "integrate" else x0
            path_length_l2 = torch.zeros(x0.shape[0], device=device)
            peak_velocity_l2 = torch.zeros(x0.shape[0], device=device)

            for step_idx in range(steps):
                t_value = horizon * ((step_idx + 0.5) / float(steps)) if steps > 0 else 0.0
                t_tensor = torch.full((x0.shape[0],), float(t_value), device=device, dtype=x0.dtype)
                x_query = h if effective_mode == "integrate" else x0
                vel = model.forward(x_query, t=t_tensor, style_id=target_ids).float()
                vel_l2 = vel.flatten(1).norm(dim=1)
                vel_abs = vel.abs().mean(dim=(1, 2, 3))
                step_disp_l2 = vel_l2 * dt
                path_length_l2 += step_disp_l2
                peak_velocity_l2 = torch.maximum(peak_velocity_l2, vel_l2)
                if effective_mode == "integrate":
                    h = h + vel * dt

                split_masks = {
                    "all": torch.ones(x0.shape[0], device=device, dtype=torch.bool),
                    "identity": torch.tensor([src == target_name for src in src_styles], device=device, dtype=torch.bool),
                    "transfer": torch.tensor([src != target_name for src in src_styles], device=device, dtype=torch.bool),
                }
                for split, mask in split_masks.items():
                    if not bool(mask.any()):
                        continue
                    bucket = time_buckets[(split, step_idx)]
                    bucket["count"] += float(mask.sum().item())
                    bucket["vel_l2_sum"] += float(vel_l2[mask].sum().item())
                    bucket["vel_l2_sum_sq"] += float(vel_l2[mask].square().sum().item())
                    bucket["vel_abs_sum"] += float(vel_abs[mask].sum().item())
                    bucket["step_disp_l2_sum"] += float(step_disp_l2[mask].sum().item())
                    bucket["step_disp_l2_sum_sq"] += float(step_disp_l2[mask].square().sum().item())

            if effective_mode == "integrate":
                endpoint = h
            else:
                endpoint = model.endpoint_map(
                    x0,
                    style_id=target_ids,
                    step_size=float(step_size),
                    style_strength=style_strength,
                ).float()
            endpoint_disp_l2 = (endpoint - x0.float()).flatten(1).norm(dim=1)
            path_length_ratio = path_length_l2 / endpoint_disp_l2.clamp_min(1e-8)

            split_masks = {
                "all": torch.ones(x0.shape[0], device=device, dtype=torch.bool),
                "identity": torch.tensor([src == target_name for src in src_styles], device=device, dtype=torch.bool),
                "transfer": torch.tensor([src != target_name for src in src_styles], device=device, dtype=torch.bool),
            }
            for split, mask in split_masks.items():
                if not bool(mask.any()):
                    continue
                bucket = summary_buckets[split]
                bucket["count"] += float(mask.sum().item())
                bucket["endpoint_disp_l2_sum"] += float(endpoint_disp_l2[mask].sum().item())
                bucket["path_length_l2_sum"] += float(path_length_l2[mask].sum().item())
                bucket["path_length_ratio_sum"] += float(path_length_ratio[mask].sum().item())
                bucket["peak_velocity_l2_sum"] += float(peak_velocity_l2[mask].sum().item())

    per_time_rows: list[dict] = []
    for split in ("all", "identity", "transfer"):
        for step_idx in range(steps):
            bucket = time_buckets[(split, step_idx)]
            count = max(1.0, bucket["count"])
            vel_mean = bucket["vel_l2_sum"] / count
            vel_var = max(0.0, bucket["vel_l2_sum_sq"] / count - vel_mean * vel_mean)
            step_mean = bucket["step_disp_l2_sum"] / count
            step_var = max(0.0, bucket["step_disp_l2_sum_sq"] / count - step_mean * step_mean)
            per_time_rows.append(
                {
                    "run_label": label,
                    "objective_mode": objective_mode,
                    "rollout_mode": effective_mode,
                    "split": split,
                    "step_idx": step_idx,
                    "t": horizon * ((step_idx + 0.5) / float(steps)) if steps > 0 else 0.0,
                    "count": int(bucket["count"]),
                    "velocity_l2_mean": vel_mean,
                    "velocity_l2_std": vel_var ** 0.5,
                    "velocity_abs_mean": bucket["vel_abs_sum"] / count,
                    "step_disp_l2_mean": step_mean,
                    "step_disp_l2_std": step_var ** 0.5,
                }
            )

    summary_rows: list[dict] = []
    for split in ("all", "identity", "transfer"):
        bucket = summary_buckets[split]
        count = max(1.0, bucket["count"])
        rows = [row for row in per_time_rows if row["split"] == split]
        third = max(1, len(rows) // 3)
        early = rows[:third]
        late = rows[-third:]
        early_vel = sum(float(row["velocity_l2_mean"]) for row in early) / max(1, len(early))
        late_vel = sum(float(row["velocity_l2_mean"]) for row in late) / max(1, len(late))
        summary_rows.append(
            {
                "run_label": label,
                "objective_mode": objective_mode,
                "rollout_mode": effective_mode,
                "split": split,
                "sample_count": int(bucket["count"]),
                "mean_endpoint_disp_l2": bucket["endpoint_disp_l2_sum"] / count,
                "mean_path_length_l2": bucket["path_length_l2_sum"] / count,
                "mean_path_length_ratio": bucket["path_length_ratio_sum"] / count,
                "mean_peak_velocity_l2": bucket["peak_velocity_l2_sum"] / count,
                "early_velocity_l2_mean": early_vel,
                "late_velocity_l2_mean": late_vel,
                "late_to_early_velocity_ratio": late_vel / max(early_vel, 1e-8),
            }
        )

    summary = {
        "run_label": label,
        "checkpoint": str(checkpoint),
        "objective_mode": objective_mode,
        "rollout_mode": effective_mode,
        "config_model_styles": int(cfg.model.num_styles),
        "path_probe_num_steps": steps,
        "path_probe_horizon": horizon,
        "path_probe_step_size": float(step_size),
        "path_probe_style_strength": None if style_strength is None else float(style_strength),
        "path_probe_max_content_per_style": int(max_content_per_style),
        "path_probe_batch_size": int(batch_size),
    }
    return per_time_rows, summary_rows, summary


def _plot_velocity_over_time(rows: list[dict], out_path: Path) -> Path | None:
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    split_to_ax = {"identity": axes[0], "transfer": axes[1]}
    for split, ax in split_to_ax.items():
        split_rows = [row for row in rows if row["split"] == split]
        labels = sorted({str(row["run_label"]) for row in split_rows})
        for label in labels:
            label_rows = [row for row in split_rows if str(row["run_label"]) == label]
            label_rows.sort(key=lambda row: float(row["t"]))
            xs = [float(row["t"]) for row in label_rows]
            ys = [float(row["velocity_l2_mean"]) for row in label_rows]
            ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.5, label=label)
        ax.set_title(split)
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Mean velocity L2")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle("Velocity Magnitude Over Time", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe velocity-field stability over time for matched LBM checkpoints.")
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--classes", type=str, default="")
    parser.add_argument("--run", action="append", required=True, help="label=checkpoint or label::checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-content-per-style", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--style-strength", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rollout-mode", type=str, default="auto", choices=["auto", "field", "integrate"])
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    names = _style_names(args.latent_root, args.classes)
    if not names:
        raise FileNotFoundError(f"No style directories found under {args.latent_root}")

    all_time_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    run_summaries: list[dict] = []
    for raw_run in args.run:
        label, checkpoint = _parse_run(raw_run)
        time_rows, summary_rows, summary = _probe_run(
            label=label,
            checkpoint=checkpoint,
            latent_root=args.latent_root,
            names=names,
            device=device,
            max_content_per_style=int(args.max_content_per_style),
            batch_size=int(args.batch_size),
            num_steps=int(args.num_steps),
            step_size=float(args.step_size),
            style_strength=float(args.style_strength),
            rollout_mode=args.rollout_mode,
        )
        all_time_rows.extend(time_rows)
        all_summary_rows.extend(summary_rows)
        run_summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_time_stats.csv", all_time_rows)
    _write_csv(args.output_dir / "run_summary.csv", all_summary_rows)
    figure_paths: dict[str, str] = {}
    velocity_fig = _plot_velocity_over_time(
        [row for row in all_time_rows if row["split"] in {"identity", "transfer"}],
        args.output_dir / "fig_velocity_over_time.pdf",
    )
    if velocity_fig is not None:
        figure_paths["fig_velocity_over_time"] = str(velocity_fig)
    summary = {
        "output_dir": str(args.output_dir),
        "latent_root": str(args.latent_root),
        "classes": names,
        "git_commit": _git_commit(),
        "runtime_metadata": _runtime_metadata(args.device, device),
        "runs": run_summaries,
        "run_summary": all_summary_rows,
        "per_time_stats": all_time_rows,
        "figure_paths": figure_paths,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
