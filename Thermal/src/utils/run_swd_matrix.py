import argparse
import copy
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


_DECISION_FIELDS = [
    "delta_margin_mean",
    "identity_degradation_ratio",
    "delta_hf_ratio_pct",
    "delta_diversity_lpips",
    "is_effective",
    "noise_shortcut",
    "structure_break",
]
_CKPT_RE = re.compile(r"^epoch_(\d+)\.pt$")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _progress_enabled(no_progress: bool) -> bool:
    return (not no_progress) and bool(sys.stderr.isatty()) and (_tqdm is not None)


def _iter_progress(iterable, *, desc: str, total: Optional[int], unit: str, leave: bool, enabled: bool):
    if not enabled:
        return iterable
    return _tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave)


def _progress_write(message: str, enabled: bool) -> None:
    if enabled and _tqdm is not None:
        _tqdm.write(message)
    else:
        print(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _deep_update(dst: Dict, src: Dict) -> Dict:
    out = copy.deepcopy(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_command(cmd: List[str], cwd: Path, log_path: Path, append: bool = True) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(log_path, mode, encoding="utf-8") as logf:
        logf.write(f"\n\n===== [{_now()}] CMD: {' '.join(cmd)} =====\n")
        proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}. See {log_path}")


def _decision_against_baseline(cur: Dict, base: Dict) -> Dict:
    delta_margin = float(cur["margin_mean"] - base["margin_mean"])
    identity_deg = float(
        (cur["identity_mse_latent_mean"] - base["identity_mse_latent_mean"])
        / (base["identity_mse_latent_mean"] + 1e-8)
    )
    delta_hf = float((cur["hf_ratio_mean"] - base["hf_ratio_mean"]) / (base["hf_ratio_mean"] + 1e-8))
    delta_div = float(cur["diversity_lpips_across_styles_mean"] - base["diversity_lpips_across_styles_mean"])
    is_effective = delta_margin >= 0.05 and identity_deg <= 0.10 and delta_hf <= 0.05 and delta_div > 0.0
    return {
        "delta_margin_mean": delta_margin,
        "identity_degradation_ratio": identity_deg,
        "delta_hf_ratio_pct": delta_hf,
        "delta_diversity_lpips": delta_div,
        "is_effective": bool(is_effective),
        "noise_shortcut": bool(delta_margin >= 0.05 and delta_hf > 0.05),
        "structure_break": bool(identity_deg > 0.10),
    }


def _prepare_config(
    base_config: Dict,
    exp_name: str,
    save_dir: Path,
    epochs: int,
    eval_every: int,
    seed: int,
    overrides: Dict,
    batch_size_override: Optional[int] = None,
    accumulation_steps_override: Optional[int] = None,
    num_workers_override: Optional[int] = None,
    preload_data_to_gpu: Optional[bool] = None,
    pin_memory_override: Optional[bool] = None,
    prefetch_factor_override: Optional[int] = None,
    save_initial_inference: Optional[bool] = None,
) -> Dict:
    cfg = copy.deepcopy(base_config)
    cfg = _deep_update(cfg, overrides)
    cfg.setdefault("training", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("checkpoint", {})

    cfg["training"]["num_epochs"] = int(epochs)
    cfg["training"]["save_interval"] = int(eval_every)
    cfg["training"]["eval_interval"] = int(10**9)
    cfg["training"]["full_eval_interval"] = int(10**9)
    cfg["training"]["eval_on_last_epoch"] = False
    cfg["training"]["full_eval_on_last_epoch"] = False
    cfg["training"]["seed"] = int(seed)
    cfg["training"]["resume_checkpoint"] = ""
    cfg["checkpoint"]["save_dir"] = str(save_dir.resolve())

    if batch_size_override is not None and batch_size_override > 0:
        cfg["training"]["batch_size"] = int(batch_size_override)
    if accumulation_steps_override is not None and accumulation_steps_override > 0:
        cfg["training"]["accumulation_steps"] = int(accumulation_steps_override)
    if num_workers_override is not None and num_workers_override >= 0:
        cfg["training"]["num_workers"] = int(num_workers_override)
    if preload_data_to_gpu is not None:
        cfg["training"]["preload_data_to_gpu"] = bool(preload_data_to_gpu)
    if pin_memory_override is not None:
        cfg["training"]["pin_memory"] = bool(pin_memory_override)
    if prefetch_factor_override is not None and prefetch_factor_override > 0:
        cfg["training"]["prefetch_factor"] = int(prefetch_factor_override)
    if save_initial_inference is not None:
        cfg["training"]["save_initial_inference"] = bool(save_initial_inference)

    cfg["loss"]["use_style_classifier"] = False
    cfg["loss"]["w_style_classifier"] = 0.0
    cfg["loss"]["w_style_classifier_transfer"] = 0.0
    cfg["loss"]["w_style_classifier_identity"] = 0.0
    cfg["loss"]["style_cls_interval"] = int(10**9)

    return cfg


def _stage_a_experiments() -> List[Dict]:
    return [
        {"name": "E0_no_swd", "overrides": {"loss": {"use_style_swd": False}}},
        {
            "name": "E1_p2",
            "overrides": {"loss": {"use_style_swd": True, "swd_scales": [2], "swd_scale_weights": [1.0], "swd_feature_levels": ["latent"]}},
        },
        {
            "name": "E1_p4",
            "overrides": {"loss": {"use_style_swd": True, "swd_scales": [4], "swd_scale_weights": [1.0], "swd_feature_levels": ["latent"]}},
        },
        {
            "name": "E1_p8",
            "overrides": {"loss": {"use_style_swd": True, "swd_scales": [8], "swd_scale_weights": [1.0], "swd_feature_levels": ["latent"]}},
        },
        {
            "name": "E1_p16",
            "overrides": {"loss": {"use_style_swd": True, "swd_scales": [16], "swd_scale_weights": [1.0], "swd_feature_levels": ["latent"]}},
        },
        {
            "name": "E1_ms248",
            "overrides": {"loss": {"use_style_swd": True, "swd_scales": [2, 4, 8], "swd_scale_weights": [1.0, 1.0, 1.0], "swd_feature_levels": ["latent"]}},
        },
    ]


def _stage_b_experiments(best_patch: int) -> List[Dict]:
    patch_cfg = {"swd_scales": [best_patch], "swd_scale_weights": [1.0]}
    return [
        {
            "name": "E2_low_bestpatch",
            "overrides": {"loss": {"use_style_swd": True, **patch_cfg, "swd_band_mode": "lowpass", "swd_feature_levels": ["latent"]}},
        },
        {
            "name": "E2_high_bestpatch",
            "overrides": {"loss": {"use_style_swd": True, **patch_cfg, "swd_band_mode": "highpass", "swd_feature_levels": ["latent"]}},
        },
        {
            "name": "E2_both_bestpatch",
            "overrides": {"loss": {"use_style_swd": True, **patch_cfg, "swd_band_mode": "both", "swd_feature_levels": ["latent"]}},
        },
        {
            "name": "E3_early_bestpatch",
            "overrides": {"loss": {"use_style_swd": True, **patch_cfg, "swd_band_mode": "highpass", "swd_feature_levels": ["early"]}},
        },
        {
            "name": "E3_mid_bestpatch",
            "overrides": {"loss": {"use_style_swd": True, **patch_cfg, "swd_band_mode": "highpass", "swd_feature_levels": ["mid"]}},
        },
        {
            "name": "E3_late_bestpatch",
            "overrides": {"loss": {"use_style_swd": True, **patch_cfg, "swd_band_mode": "highpass", "swd_feature_levels": ["late"]}},
        },
        {
            "name": "E3_all_bestpatch",
            "overrides": {"loss": {"use_style_swd": True, **patch_cfg, "swd_band_mode": "highpass", "swd_feature_levels": ["early", "mid", "late"]}},
        },
    ]

def _select_best_patch(final_rows: List[Dict], final_epoch: int) -> int:
    baseline = next((r for r in final_rows if r["experiment"] == "E0_no_swd" and r["epoch"] == final_epoch), None)
    if baseline is None:
        return 4

    candidates = []
    for patch in [2, 4, 8, 16]:
        name = f"E1_p{patch}"
        row = next((r for r in final_rows if r["experiment"] == name and r["epoch"] == final_epoch), None)
        if row is None:
            continue
        candidates.append((patch, _decision_against_baseline(row, baseline)))

    if not candidates:
        return 4

    effective = [x for x in candidates if x[1]["is_effective"]]
    if effective:
        effective.sort(key=lambda x: (x[1]["delta_margin_mean"], -x[1]["delta_hf_ratio_pct"]), reverse=True)
        return int(effective[0][0])

    candidates.sort(key=lambda x: (x[1]["delta_margin_mean"], -x[1]["delta_hf_ratio_pct"]), reverse=True)
    return int(candidates[0][0])


def _epoch_from_ckpt_name(path: Path) -> Optional[int]:
    m = _CKPT_RE.match(path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _find_latest_ckpt(ckpt_dir: Path) -> Tuple[Optional[Path], int]:
    best_path = None
    best_epoch = -1
    if not ckpt_dir.exists():
        return None, -1
    for p in ckpt_dir.glob("epoch_*.pt"):
        e = _epoch_from_ckpt_name(p)
        if e is None:
            continue
        if e > best_epoch:
            best_epoch = e
            best_path = p
    return best_path, best_epoch


def _summary_path(exp_root: Path, epoch: int) -> Path:
    return exp_root / "validation" / f"epoch_{epoch:04d}" / "summary.json"


def _row_from_summary(exp_name: str, epoch: int, ckpt_dir: Path, summary_path: Path) -> Optional[Dict]:
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return None

    ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
    checkpoint_value = str(ckpt_path if ckpt_path.exists() else summary.get("checkpoint", ""))
    row = {"experiment": exp_name, "epoch": int(epoch), "checkpoint": checkpoint_value}
    row.update(summary)
    return row


def _collect_existing_rows(experiments_dir: Path, eval_epochs: List[int]) -> Tuple[Dict[Tuple[str, int], Dict], Dict[int, Path]]:
    row_map: Dict[Tuple[str, int], Dict] = {}
    baseline_summary_by_epoch: Dict[int, Path] = {}
    if not experiments_dir.exists():
        return row_map, baseline_summary_by_epoch

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        ckpt_dir = exp_dir / "checkpoints"
        for epoch in eval_epochs:
            sp = _summary_path(exp_dir, epoch)
            row = _row_from_summary(exp_name, epoch, ckpt_dir, sp)
            if row is None:
                continue
            row_map[(exp_name, int(epoch))] = row
            if exp_name == "E0_no_swd":
                baseline_summary_by_epoch[int(epoch)] = sp

    return row_map, baseline_summary_by_epoch


def _is_exp_complete(exp_root: Path, ckpt_dir: Path, target_epoch: int, eval_epochs: List[int], allow_missing_final_ckpt: bool) -> bool:
    final_ckpt = ckpt_dir / f"epoch_{target_epoch:04d}.pt"
    if not final_ckpt.exists() and not allow_missing_final_ckpt:
        return False
    for epoch in eval_epochs:
        if not _summary_path(exp_root, epoch).exists():
            return False
    return True


def _rows_sorted(row_map: Dict[Tuple[str, int], Dict], order: List[str]) -> List[Dict]:
    rank = {name: idx for idx, name in enumerate(order)}
    return sorted(
        row_map.values(),
        key=lambda r: (rank.get(str(r.get("experiment", "")), 10**9), int(r.get("epoch", 0)), str(r.get("experiment", ""))),
    )


def _enrich_rows_with_decisions(rows: List[Dict]) -> List[Dict]:
    baseline_by_epoch = {int(r["epoch"]): r for r in rows if r.get("experiment") == "E0_no_swd"}
    out: List[Dict] = []
    for row in rows:
        x = dict(row)
        if x.get("experiment") == "E0_no_swd":
            for k in _DECISION_FIELDS:
                x[k] = ""
        else:
            base = baseline_by_epoch.get(int(x.get("epoch", 0)))
            if base is not None:
                x.update(_decision_against_baseline(x, base))
            else:
                for k in _DECISION_FIELDS:
                    x[k] = ""
        out.append(x)
    return out


def _write_matrix_csv(path: Path, rows: List[Dict]) -> None:
    fieldnames = [
        "experiment",
        "epoch",
        "checkpoint",
        "margin_mean",
        "margin_pos_rate",
        "diversity_lpips_across_styles_mean",
        "identity_mse_latent_mean",
        "content_lpips_mean",
        "edge_sobel_l1_mean",
        "hf_ratio_mean",
        "patch_consistency_var_mean",
        "delta_margin_mean",
        "identity_degradation_ratio",
        "delta_hf_ratio_pct",
        "delta_diversity_lpips",
        "is_effective",
        "noise_shortcut",
        "structure_break",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_delta_csv(path: Path, rows: List[Dict]) -> None:
    out_rows = []
    baseline_by_epoch = {row["epoch"]: row for row in rows if row["experiment"] == "E0_no_swd"}
    for row in rows:
        if row["experiment"] == "E0_no_swd":
            continue
        base = baseline_by_epoch.get(row["epoch"])
        if base is None:
            continue
        out_rows.append({"experiment": row["experiment"], "epoch": row["epoch"], **_decision_against_baseline(row, base)})

    fields = [
        "experiment",
        "epoch",
        "delta_margin_mean",
        "identity_degradation_ratio",
        "delta_hf_ratio_pct",
        "delta_diversity_lpips",
        "is_effective",
        "noise_shortcut",
        "structure_break",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

def _write_recommendation(path: Path, rows: List[Dict], final_epoch: int, best_patch: int) -> None:
    baseline = next((row for row in rows if row["experiment"] == "E0_no_swd" and row["epoch"] == final_epoch), None)

    lines = [
        "# SWD Matrix Recommendation",
        "",
        f"- Final epoch: {final_epoch}",
        f"- Selected best patch (from E1): {best_patch}",
        "",
    ]
    if baseline is not None:
        lines += [
            "## Baseline (E0)",
            "",
            f"- margin_mean: {baseline['margin_mean']:.6f}",
            f"- identity_mse_latent_mean: {baseline['identity_mse_latent_mean']:.6f}",
            f"- hf_ratio_mean: {baseline['hf_ratio_mean']:.6f}",
            "",
        ]

    lines += [
        "## Final-Epoch Deltas vs E0",
        "",
        "| Experiment | dMargin | dIdentity% | dHF% | dDiversity | Effective |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["experiment"] == "E0_no_swd" or row["epoch"] != final_epoch or baseline is None:
            continue
        d = _decision_against_baseline(row, baseline)
        lines.append(
            f"| {row['experiment']} | {d['delta_margin_mean']:.4f} | {100*d['identity_degradation_ratio']:.2f}% | "
            f"{100*d['delta_hf_ratio_pct']:.2f}% | {d['delta_diversity_lpips']:.4f} | {d['is_effective']} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- `noise_shortcut=true` means margin improved but high-frequency ratio overshot threshold.",
        "- `structure_break=true` means identity structure degraded by more than 10%.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_state(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SWD full matrix experiments")
    parser.add_argument("--base_config", type=str, default="Thermal/src/config.json")
    parser.add_argument("--output_root", type=str, default="Thermal/swd_matrix")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--latent_root", type=str, default="style_data/latents/test")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--subset_debug", action="store_true")
    parser.add_argument("--debug_checkpoint", type=str, default="Thermal/unet_lite/epoch_0100.pt")
    parser.add_argument("--max_control", type=int, default=0)
    parser.add_argument("--max_identity", type=int, default=0)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--batch_size", type=int, default=512, help="Matrix training batch size override; <=0 keeps base config")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation override; <=0 keeps base config")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers override; <0 keeps base config")
    parser.add_argument("--preload_data_to_gpu", type=int, default=1, choices=[0, 1], help="1 to preload dataset latents to GPU")
    parser.add_argument("--pin_memory", type=int, default=0, choices=[0, 1], help="Pin host memory in DataLoader")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor when num_workers>0")
    parser.add_argument("--save_initial_inference", type=int, default=0, choices=[0, 1], help="Save epoch_-1 inference snapshot for each experiment")

    parser.add_argument("--no_resume", action="store_true", help="Disable artifact reuse and always run fresh pipeline")
    parser.add_argument("--force_retrain", action="store_true", help="Always rerun train phase for each experiment")
    parser.add_argument("--force_reeval", action="store_true", help="Always rerun eval phase for each experiment/epoch")
    parser.add_argument("--state_file", type=str, default="", help="Optional custom path for resume state json")

    args = parser.parse_args()

    repo_root = _repo_root()
    src_dir = repo_root / "Thermal" / "src"
    run_script = src_dir / "run.py"
    build_evalset_script = src_dir / "utils" / "build_swd_evalset.py"
    validate_script = src_dir / "utils" / "run_swd_validation.py"

    base_config_path = _resolve(args.base_config, repo_root)
    output_root = _resolve(args.output_root, repo_root)
    configs_dir = output_root / "configs"
    experiments_dir = output_root / "experiments"
    logs_dir = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)

    progress_on = _progress_enabled(args.no_progress)
    resume_enabled = not args.no_resume

    state_path = _resolve(args.state_file, repo_root) if args.state_file else (output_root / "resume_state.json")
    state = _load_state(state_path) if resume_enabled else {}
    if not state:
        state = {"version": 1, "created_at": _now(), "experiments": {}}

    def save_state() -> None:
        state["updated_at"] = _now()
        _write_json(state_path, state)

    state["paths"] = {
        "base_config": str(base_config_path),
        "output_root": str(output_root),
    }
    state["params"] = {
        "epochs": int(args.epochs),
        "eval_every": int(args.eval_every),
        "seed": int(args.seed),
        "subset_debug": bool(args.subset_debug),
        "max_control": int(args.max_control),
        "max_identity": int(args.max_identity),
        "resume_enabled": bool(resume_enabled),
        "batch_size": int(args.batch_size),
        "accumulation_steps": int(args.accumulation_steps),
        "num_workers": int(args.num_workers),
        "preload_data_to_gpu": bool(args.preload_data_to_gpu),
        "pin_memory": bool(args.pin_memory),
        "prefetch_factor": int(args.prefetch_factor),
        "save_initial_inference": bool(args.save_initial_inference),
    }
    save_state()

    with open(base_config_path, "r", encoding="utf-8") as f:
        base_config = json.load(f)
    debug_checkpoint = _resolve(args.debug_checkpoint, repo_root)

    manifest_path = _resolve(args.manifest, repo_root) if args.manifest else (output_root / "evalset" / "manifest.json")
    if not manifest_path.exists():
        cmd = [
            args.python,
            str(build_evalset_script),
            "--latent_root",
            args.latent_root,
            "--seed",
            str(args.seed),
            "--output",
            str(manifest_path),
        ]
        _run_command(cmd, cwd=repo_root, log_path=logs_dir / "build_evalset.log")
    state["manifest"] = str(manifest_path)
    save_state()

    eval_epochs = list(range(args.eval_every, args.epochs + 1, args.eval_every))
    if args.subset_debug and not eval_epochs:
        eval_epochs = [args.epochs]

    stage_a = _stage_a_experiments()
    if args.subset_debug:
        stage_a = [x for x in stage_a if x["name"] in {"E0_no_swd", "E1_p2"}]

    order = [x["name"] for x in _stage_a_experiments()]
    if args.subset_debug:
        order = [x["name"] for x in stage_a]
    else:
        order.extend([x["name"] for x in _stage_b_experiments(4)])

    row_map: Dict[Tuple[str, int], Dict] = {}
    baseline_summary_by_epoch: Dict[int, Path] = {}

    if resume_enabled:
        loaded_rows, loaded_baseline = _collect_existing_rows(experiments_dir, eval_epochs)
        row_map.update(loaded_rows)
        baseline_summary_by_epoch.update(loaded_baseline)
        if loaded_rows:
            _progress_write(f"Loaded {len(loaded_rows)} existing eval rows from disk.", enabled=progress_on)
    _progress_write(
        "Matrix speed profile | "
        f"batch={args.batch_size}, accum={args.accumulation_steps}, workers={args.num_workers}, "
        f"preload_gpu={bool(args.preload_data_to_gpu)}, pin_memory={bool(args.pin_memory)}, "
        f"prefetch={args.prefetch_factor}, init_infer={bool(args.save_initial_inference)}",
        enabled=progress_on,
    )

    def upsert_row(row: Dict) -> None:
        row_map[(str(row["experiment"]), int(row["epoch"]))] = row

    def run_one_experiment(exp: Dict) -> None:
        exp_name = exp["name"]
        exp_root = experiments_dir / exp_name
        ckpt_dir = exp_root / "checkpoints"
        exp_state = state["experiments"].setdefault(exp_name, {})
        exp_state["status"] = "running"
        exp_state["started_at"] = _now()
        save_state()

        allow_missing_final_ckpt = args.subset_debug and debug_checkpoint.exists()
        if resume_enabled and (not args.force_retrain) and (not args.force_reeval):
            if _is_exp_complete(
                exp_root=exp_root,
                ckpt_dir=ckpt_dir,
                target_epoch=args.epochs,
                eval_epochs=eval_epochs,
                allow_missing_final_ckpt=allow_missing_final_ckpt,
            ):
                _progress_write(f"[{exp_name}] skip (already complete)", enabled=progress_on)
                completed = []
                for epoch in eval_epochs:
                    sp = _summary_path(exp_root, epoch)
                    row = _row_from_summary(exp_name, epoch, ckpt_dir, sp)
                    if row is None:
                        continue
                    upsert_row(row)
                    completed.append(int(epoch))
                    if exp_name == "E0_no_swd":
                        baseline_summary_by_epoch[int(epoch)] = sp
                exp_state["status"] = "complete_skipped"
                exp_state["train"] = {"skipped": True, "reason": "already_complete"}
                exp_state["eval"] = {
                    "target_epochs": list(eval_epochs),
                    "completed_epochs": sorted(completed),
                    "missing_epochs": [e for e in eval_epochs if e not in set(completed)],
                }
                exp_state["finished_at"] = _now()
                save_state()
                return

        try:
            _progress_write(f"[{exp_name}] preparing config/train/eval", enabled=progress_on)
            cfg = _prepare_config(
                base_config=base_config,
                exp_name=exp_name,
                save_dir=ckpt_dir,
                epochs=args.epochs,
                eval_every=args.eval_every,
                seed=args.seed,
                overrides=exp["overrides"],
                batch_size_override=int(args.batch_size),
                accumulation_steps_override=int(args.accumulation_steps),
                num_workers_override=int(args.num_workers),
                preload_data_to_gpu=bool(args.preload_data_to_gpu),
                pin_memory_override=bool(args.pin_memory),
                prefetch_factor_override=int(args.prefetch_factor),
                save_initial_inference=bool(args.save_initial_inference),
            )

            latest_ckpt, latest_epoch = _find_latest_ckpt(ckpt_dir)
            resume_ckpt = ""
            if args.subset_debug and debug_checkpoint.exists():
                resume_ckpt = str(debug_checkpoint)
            elif resume_enabled and latest_ckpt is not None and latest_epoch < args.epochs:
                resume_ckpt = str(latest_ckpt)
            if resume_ckpt:
                cfg["training"]["resume_checkpoint"] = resume_ckpt

            cfg_path = configs_dir / f"{exp_name}.json"
            _write_json(cfg_path, cfg)

            final_ckpt = ckpt_dir / f"epoch_{args.epochs:04d}.pt"
            train_needed = True
            train_skip_reason = ""
            if not args.force_retrain:
                if final_ckpt.exists():
                    train_needed = False
                    train_skip_reason = "final_checkpoint_exists"
                elif args.subset_debug and debug_checkpoint.exists():
                    train_needed = False
                    train_skip_reason = "subset_debug_uses_external_checkpoint"

            if train_needed:
                _progress_write(f"[{exp_name}] training...", enabled=progress_on)
                _run_command([args.python, str(run_script), "--config", str(cfg_path)], cwd=src_dir, log_path=logs_dir / f"{exp_name}.train.log")
            else:
                _progress_write(f"[{exp_name}] skip training ({train_skip_reason})", enabled=progress_on)

            latest_ckpt_after, latest_epoch_after = _find_latest_ckpt(ckpt_dir)
            exp_state["train"] = {
                "target_epoch": int(args.epochs),
                "latest_checkpoint": str(latest_ckpt_after) if latest_ckpt_after is not None else "",
                "latest_checkpoint_epoch": int(latest_epoch_after),
                "final_checkpoint_exists": bool(final_ckpt.exists()),
                "skipped": bool(not train_needed),
                "skip_reason": train_skip_reason,
                "resume_checkpoint_used": resume_ckpt,
            }
            save_state()

            completed_epochs = set()
            epoch_iter = _iter_progress(
                eval_epochs,
                desc=f"{exp_name} eval",
                total=len(eval_epochs),
                unit="epoch",
                leave=False,
                enabled=progress_on,
            )
            for epoch in epoch_iter:
                sp = _summary_path(exp_root, epoch)
                if resume_enabled and sp.exists() and (not args.force_reeval):
                    row = _row_from_summary(exp_name, epoch, ckpt_dir, sp)
                    if row is not None:
                        upsert_row(row)
                        completed_epochs.add(int(epoch))
                        if exp_name == "E0_no_swd":
                            baseline_summary_by_epoch[int(epoch)] = sp
                        exp_state["eval"] = {
                            "target_epochs": list(eval_epochs),
                            "completed_epochs": sorted(completed_epochs),
                            "missing_epochs": sorted(set(eval_epochs) - completed_epochs),
                        }
                        save_state()
                        continue

                ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
                if not ckpt_path.exists() and args.subset_debug and debug_checkpoint.exists():
                    ckpt_path = debug_checkpoint
                if not ckpt_path.exists():
                    exp_state["eval"] = {
                        "target_epochs": list(eval_epochs),
                        "completed_epochs": sorted(completed_epochs),
                        "missing_epochs": sorted(set(eval_epochs) - completed_epochs),
                    }
                    save_state()
                    continue

                eval_out = exp_root / "validation" / f"epoch_{epoch:04d}"
                eval_cmd = [
                    args.python,
                    str(validate_script),
                    "--checkpoint",
                    str(ckpt_path),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(eval_out),
                    "--seed",
                    str(args.seed),
                ]
                if args.max_control > 0:
                    eval_cmd += ["--max_control", str(args.max_control)]
                if args.max_identity > 0:
                    eval_cmd += ["--max_identity", str(args.max_identity)]
                baseline_summary = baseline_summary_by_epoch.get(int(epoch))
                if exp_name != "E0_no_swd" and baseline_summary is not None and baseline_summary.exists():
                    eval_cmd += ["--baseline_summary", str(baseline_summary)]

                _run_command(eval_cmd, cwd=repo_root, log_path=logs_dir / f"{exp_name}.eval.e{epoch:04d}.log")

                row = _row_from_summary(exp_name, epoch, ckpt_dir, _summary_path(exp_root, epoch))
                if row is not None:
                    upsert_row(row)
                    completed_epochs.add(int(epoch))
                    if exp_name == "E0_no_swd":
                        baseline_summary_by_epoch[int(epoch)] = _summary_path(exp_root, epoch)

                exp_state["eval"] = {
                    "target_epochs": list(eval_epochs),
                    "completed_epochs": sorted(completed_epochs),
                    "missing_epochs": sorted(set(eval_epochs) - completed_epochs),
                }
                save_state()

            exp_state["status"] = "done"
            exp_state["finished_at"] = _now()
            save_state()
            _progress_write(f"[{exp_name}] done", enabled=progress_on)
        except Exception as exc:
            exp_state["status"] = "failed"
            exp_state["error"] = str(exc)
            exp_state["failed_at"] = _now()
            save_state()
            raise

    stage_a_iter = _iter_progress(stage_a, desc="Stage-A", total=len(stage_a), unit="exp", leave=True, enabled=progress_on)
    for exp in stage_a_iter:
        run_one_experiment(exp)

    rows_for_patch = _rows_sorted(row_map, order)
    if not args.subset_debug:
        best_patch = _select_best_patch(rows_for_patch, final_epoch=args.epochs)
        _progress_write(f"Selected best patch from E1: {best_patch}", enabled=progress_on)
        state["best_patch"] = int(best_patch)
        save_state()

        stage_b = _stage_b_experiments(best_patch)
        stage_b_iter = _iter_progress(stage_b, desc="Stage-B", total=len(stage_b), unit="exp", leave=True, enabled=progress_on)
        for exp in stage_b_iter:
            run_one_experiment(exp)
    else:
        best_patch = 2
        state["best_patch"] = int(best_patch)
        save_state()

    rows = _rows_sorted(row_map, order)
    rows = _enrich_rows_with_decisions(rows)

    matrix_csv = output_root / "matrix_metrics.csv"
    delta_csv = output_root / "matrix_deltas_vs_E0.csv"
    reco_md = output_root / "final_recommendation.md"
    _write_matrix_csv(matrix_csv, rows)
    _write_delta_csv(delta_csv, rows)
    _write_recommendation(reco_md, rows, final_epoch=args.epochs, best_patch=best_patch)

    state["outputs"] = {
        "matrix_metrics_csv": str(matrix_csv),
        "matrix_deltas_csv": str(delta_csv),
        "final_recommendation_md": str(reco_md),
        "row_count": len(rows),
    }
    state["status"] = "done"
    save_state()

    print(f"Wrote matrix metrics: {matrix_csv}")
    print(f"Wrote deltas vs E0: {delta_csv}")
    print(f"Wrote recommendation: {reco_md}")
    print(f"Wrote resume state: {state_path}")


if __name__ == "__main__":
    main()
