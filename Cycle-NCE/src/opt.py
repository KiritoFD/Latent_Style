from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import optuna

SRC_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = SRC_DIR / "config_style_oa_5_lr5e4_wc2_swd60_id30_e120.json"

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _extract_style_metrics(summary: Dict[str, Any]) -> Tuple[float, float]:
    analysis = summary.get("analysis", {})
    style = analysis.get("style_transfer_ability", {})
    clip_style = style.get("clip_style", None)
    lpips = style.get("content_lpips", None)
    if clip_style is None or lpips is None:
        raise KeyError("Cannot find analysis.style_transfer_ability.clip_style/content_lpips in summary.json")
    return float(clip_style), float(lpips)

def _find_summary_file(exp_dir: Path, epoch: int) -> Path:
    exact = exp_dir / "full_eval" / f"epoch_{epoch:04d}" / "summary.json"
    if exact.exists():
        return exact

    candidates = sorted((exp_dir / "full_eval").glob(f"epoch_{epoch:04d}*/summary*.json"))
    if not candidates:
        raise FileNotFoundError(f"No summary json found under {exp_dir / 'full_eval'} for epoch {epoch:04d}")

    candidates.sort(key=lambda p: (p.name != "summary.json", p.stat().st_mtime))
    return candidates[-1]

def _build_trial_config(
    base_cfg: Dict[str, Any],
    *,
    lr: float,
    w_identity: int,
    scheduler: str,
    trial_dir: Path,
    epochs: int,
    full_eval_interval: int,
    w_color: float,
    w_swd: float,
    ada_mix_rank: int,
    multistep_milestones: list[int],
    multistep_gamma: float,
    onecycle_max_lr: float,
    onecycle_pct_start: float,
    onecycle_anneal_strategy: str,
    grad_direction_interval: int,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("training", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("model", {})
    cfg.setdefault("checkpoint", {})

    cfg["training"]["learning_rate"] = float(lr)
    cfg["training"]["scheduler"] = str(scheduler)
    cfg["training"]["num_epochs"] = int(epochs)
    cfg["training"]["save_interval"] = int(epochs)
    cfg["training"]["full_eval_interval"] = int(full_eval_interval)
    cfg["training"]["full_eval_on_last_epoch"] = True
    cfg["training"]["resume_checkpoint"] = ""
    cfg["training"]["multistep_milestones"] = [int(v) for v in multistep_milestones]
    cfg["training"]["multistep_gamma"] = float(multistep_gamma)
    cfg["training"]["onecycle_max_lr"] = float(onecycle_max_lr)
    cfg["training"]["onecycle_pct_start"] = float(onecycle_pct_start)
    cfg["training"]["onecycle_anneal_strategy"] = str(onecycle_anneal_strategy)
    # Fixed-shape throughput settings (hardcoded).
    cfg["training"]["use_amp"] = True
    cfg["training"]["cudnn_benchmark"] = "auto"
    cfg["training"]["amp_dtype"] = "bf16"
    cfg["training"]["allow_tf32"] = True
    cfg["training"]["channels_last"] = True
    cfg["training"]["use_tqdm"] = False
    cfg["training"]["log_interval"] = 20
    cfg["training"]["enable_profiler"] = False
    cfg["training"]["cuda_sync_debug"] = False
    cfg["training"]["strict_batch_sanity"] = False
    cfg["training"]["gc_collect_interval"] = 0
    cfg["training"]["grad_direction_interval"] = int(max(0, grad_direction_interval))

    cfg["loss"]["w_identity"] = float(w_identity)
    cfg["loss"]["w_color"] = float(w_color)
    cfg["loss"]["w_swd"] = float(w_swd)
    cfg["loss"]["swd_patch_sizes"] = [1, 3, 5, 9, 15, 25]
    # HF-SWD branch removed from training; scrub legacy keys from base configs.
    cfg["loss"].pop("swd_use_high_freq", None)
    cfg["loss"].pop("swd_hf_weight_ratio", None)
    # Fixed-shape SWD kernel-launch reduction (hardcoded).
    cfg["loss"]["swd_projection_chunk_size"] = 128
    cfg["loss"]["swd_cdf_sample_chunk_size"] = 256

    cfg["model"]["ada_mix_rank"] = int(ada_mix_rank)

    # One trial -> one isolated experiment directory, avoiding auto-resume cross-trial contamination.
    cfg["checkpoint"]["save_dir"] = str((trial_dir / "experiment").resolve())
    return cfg

def _run_trial_training(config_path: Path) -> None:
    cmd = [sys.executable, "run.py", "--config", str(config_path)]
    subprocess.run(cmd, cwd=str(SRC_DIR), check=True)

def _dump_trials_csv(study: optuna.Study, out_csv: Path) -> None:
    def _pick_param(params: Dict[str, Any], names: tuple[str, ...]) -> Any:
        for n in names:
            if n in params:
                return params.get(n)
        return ""

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial",
                "state",
                "values",
                "lr",
                "w_idt",
                "w_swd",
                "w_color",
                "ada_mix_rank",
                "scheduler",
                "clip_style",
                "lpips",
                "pareto_score",
                "peak_lr",
                "summary_path",
                "exp_dir",
            ],
        )
        writer.writeheader()
        for t in study.trials:
            val_str = str(t.values) if t.values is not None else (str(t.value) if t.value is not None else "")
            writer.writerow(
                {
                    "trial": t.number,
                    "state": str(t.state),
                    "values": val_str,
                    "lr": _pick_param(t.params, ("lr_v2", "lr")),
                    "w_idt": _pick_param(t.params, ("w_idt_v2", "w_idt")) or t.user_attrs.get("w_idt", ""),
                    "w_swd": _pick_param(t.params, ("w_swd",)),
                    "w_color": _pick_param(t.params, ("w_color",)),
                    "ada_mix_rank": _pick_param(t.params, ("ada_mix_rank",)) or t.user_attrs.get("ada_mix_rank", ""),
                    "scheduler": _pick_param(t.params, ("scheduler_v2", "scheduler")),
                    "clip_style": t.user_attrs.get("clip_style", ""),
                    "lpips": t.user_attrs.get("lpips", ""),
                    "pareto_score": t.user_attrs.get("pareto_score", ""),
                    "peak_lr": t.user_attrs.get("peak_lr", ""),
                    "summary_path": t.user_attrs.get("summary_path", ""),
                    "exp_dir": t.user_attrs.get("exp_dir", ""),
                }
            )

def _pick_trial_param(params: Dict[str, Any], names: tuple[str, ...]) -> Any:
    for n in names:
        if n in params:
            return params.get(n)
    return None

def _find_existing_distribution(study: optuna.Study, param_name: str) -> Any:
    for t in study.trials:
        dist = t.distributions.get(param_name)
        if dist is not None:
            return dist
    return None

def _resolve_param_name_for_float(
    study: optuna.Study,
    *,
    base_name: str,
    new_name: str,
    low: float,
    high: float,
    log: bool,
) -> str:
    dist = _find_existing_distribution(study, base_name)
    if dist is None:
        return base_name
    if (
        dist.__class__.__name__ == "FloatDistribution"
        and float(dist.low) == float(low)
        and float(dist.high) == float(high)
        and bool(dist.log) == bool(log)
    ):
        return base_name
    return new_name

def _resolve_param_name_for_int(
    study: optuna.Study,
    *,
    base_name: str,
    new_name: str,
    low: int,
    high: int,
) -> str:
    dist = _find_existing_distribution(study, base_name)
    if dist is None:
        return base_name
    if dist.__class__.__name__ == "IntDistribution" and int(dist.low) == int(low) and int(dist.high) == int(high):
        return base_name
    return new_name

def _resolve_param_name_for_categorical(
    study: optuna.Study,
    *,
    base_name: str,
    new_name: str,
    choices: list[str],
) -> str:
    dist = _find_existing_distribution(study, base_name)
    if dist is None:
        return base_name
    existing_choices = tuple(dist.choices) if dist.__class__.__name__ == "CategoricalDistribution" else tuple()
    if existing_choices == tuple(choices):
        return base_name
    return new_name


def _normalize_sqlite_storage_path(storage_uri: str) -> Path | None:
    if not storage_uri.startswith("sqlite:///"):
        return None
    return Path(storage_uri[len("sqlite:///") :])


def _storage_has_study(storage_uri: str, study_name: str) -> bool:
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage_uri)
        return any(s.study_name == study_name for s in summaries)
    except Exception:
        return False


def _discover_warm_start_source(
    *,
    warm_start_name: str,
    explicit_warm_start_storage: str,
    default_storage: str,
    workdir: Path,
) -> tuple[str, str] | None:
    name_candidates = []
    if warm_start_name:
        name_candidates.append(warm_start_name)
        if warm_start_name.endswith("_v2"):
            name_candidates.append(warm_start_name[: -len("_v2")])
    name_candidates = list(dict.fromkeys(name_candidates))

    storage_candidates: list[str] = []
    if explicit_warm_start_storage:
        storage_candidates.append(explicit_warm_start_storage)
    storage_candidates.append(default_storage)

    # Common legacy locations in this repo.
    candidate_dirs = [
        workdir,
        SRC_DIR / "optuna_hpo",
        SRC_DIR / "style_oa" / "optuna_hpo",
    ]
    for d in candidate_dirs:
        for n in name_candidates:
            storage_candidates.append(f"sqlite:///{(d / (n + '.db')).as_posix()}")

    # Also scan existing db files under candidate dirs.
    for d in candidate_dirs:
        if d.exists():
            for db in sorted(d.glob("*.db")):
                storage_candidates.append(f"sqlite:///{db.as_posix()}")

    storage_candidates = list(dict.fromkeys(storage_candidates))
    for st in storage_candidates:
        db_path = _normalize_sqlite_storage_path(st)
        if db_path is not None and not db_path.exists():
            continue
        for n in name_candidates:
            if _storage_has_study(st, n):
                return n, st
    return None


def _warm_start_from_study(
    dst_study: optuna.Study,
    src_study: optuna.Study,
    *,
    lr_param_name: str,
    lr_min: float,
    lr_max: float,
    w_swd_min: float,
    w_swd_max: float,
    fixed_w_swd: float,
    w_color_min: float,
    w_color_max: float,
    fixed_w_color: float,
    max_trials: int,
) -> tuple[int, int]:
    complete_state = optuna.trial.TrialState.COMPLETE
    lr_dist = optuna.distributions.FloatDistribution(low=float(lr_min), high=float(lr_max), log=True)
    w_swd_dist = optuna.distributions.FloatDistribution(low=float(w_swd_min), high=float(w_swd_max), log=False)
    w_color_dist = optuna.distributions.FloatDistribution(low=float(w_color_min), high=float(w_color_max), log=False)

    imported = 0
    skipped = 0
    source_trials = [t for t in src_study.trials if t.state == complete_state and (t.value is not None or t.values is not None)]
    if max_trials > 0:
        source_trials = source_trials[:max_trials]

    is_multi_obj = dst_study.directions == [optuna.study.StudyDirection.MAXIMIZE, optuna.study.StudyDirection.MINIMIZE]

    for t in source_trials:
        lr = _pick_trial_param(t.params, ("lr_v2", "lr"))
        if lr is None:
            skipped += 1
            continue

        lr = float(lr)

        params: Dict[str, Any] = {}
        dists: Dict[str, Any] = {}

        # Keep search dimensions aligned with current objective: w_swd and w_color are always searched.
        w_swd = _pick_trial_param(t.params, ("w_swd",))
        if w_swd is None:
            w_swd = fixed_w_swd
        w_swd = float(max(w_swd_min, min(w_swd_max, float(w_swd))))
        params["w_swd"] = w_swd
        dists["w_swd"] = w_swd_dist

        w_color = _pick_trial_param(t.params, ("w_color",))
        if w_color is None:
            w_color = fixed_w_color
        w_color = float(max(w_color_min, min(w_color_max, float(w_color))))
        params["w_color"] = w_color
        dists["w_color"] = w_color_dist

        if not (lr_min <= lr <= lr_max):
            skipped += 1
            continue
        params[lr_param_name] = lr
        dists[lr_param_name] = lr_dist

        user_attrs = dict(t.user_attrs)
        user_attrs["warm_started_from_study"] = src_study.study_name
        user_attrs["warm_started_from_trial"] = t.number

        values = None
        value = None
        
        if is_multi_obj:
            clip_style = user_attrs.get("clip_style")
            lpips = user_attrs.get("lpips")
            if clip_style is not None and lpips is not None:
                values = [float(clip_style), float(lpips)]
            else:
                skipped += 1
                continue
        else:
            if t.value is not None:
                value = float(t.value)
            elif t.values is not None:
                ps = user_attrs.get("pareto_score")
                if ps is not None:
                    value = float(ps)
                else:
                    skipped += 1
                    continue

        try:
            frozen = optuna.trial.create_trial(
                params=params,
                distributions=dists,
                value=value,
                values=values,
                state=complete_state,
                user_attrs=user_attrs,
            )
            dst_study.add_trial(frozen)
            imported += 1
        except Exception as e:
            print(f"Skipped trial {t.number} due to error: {e}")
            skipped += 1

    return imported, skipped

def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for style_oa config (epoch=60 default).")
    parser.add_argument("--base_config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--study_name", type=str, default="style_transfer_hpo_mo", help="Study name (default: style_transfer_hpo_mo for multi-objective)")
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--warm_start_study_name", type=str, default="style_transfer_hpo_e60_v2", help="Import COMPLETE trials from another study.")
    parser.add_argument("--warm_start_storage", type=str, default="", help="Storage URI of warm-start source study; defaults to --storage.")
    parser.add_argument("--warm_start_max_trials", type=int, default=0, help="Max source COMPLETE trials to import (0 = all).")
    parser.add_argument("--warm_start_onecycle_peak_scale", type=float, default=1.3, help="Fallback: old onecycle peak_lr ~= lr * scale.")
    parser.add_argument("--workdir", type=Path, default=SRC_DIR / "style_oa" / "optuna_hpo")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument(
        "--interrupt_as_fail",
        action="store_true",
        help="Treat KeyboardInterrupt inside a trial as a failed trial and continue optimization.",
    )
    
    parser.add_argument("--single_objective", action="store_true", help="Revert to single objective (pareto score).")
    parser.add_argument("--alpha", type=float, default=1.0, help="pareto = style_weight * clip_style + alpha * (1 - lpips)")
    parser.add_argument("--style_weight", type=float, default=4.0, help="Weight applied to clip_style in Pareto score.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--full_eval_interval", type=int, default=60)

    parser.add_argument("--lr_min", type=float, default=2e-4)
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--fixed_scheduler", type=str, default="cosine")
    parser.add_argument("--fixed_w_idt", type=int, default=30)
    parser.add_argument("--w_swd_min", type=float, default=40.0)
    parser.add_argument("--w_swd_max", type=float, default=100.0)
    
    parser.add_argument("--w_color_min", type=float, default=1.0)
    parser.add_argument("--w_color_max", type=float, default=5.0)

    parser.add_argument("--fixed_w_color", type=float, default=2.0)
    parser.add_argument("--fixed_w_swd", type=float, default=60.0)
    parser.add_argument("--fixed_ada_mix_rank", type=int, default=16)

    parser.add_argument(
        "--scheduler_choices",
        type=str,
        default="cosine,onecycle,multistep",
        help="Comma-separated scheduler candidates",
    )
    parser.add_argument("--multistep_milestones", type=str, default="45,55")
    parser.add_argument("--multistep_gamma", type=float, default=0.1)
    parser.add_argument("--onecycle_max_lr_scale", type=float, default=1.3)
    parser.add_argument("--onecycle_pct_start", type=float, default=0.3)
    parser.add_argument("--onecycle_anneal_strategy", type=str, default="cos")
    parser.add_argument(
        "--grad_direction_interval",
        type=int,
        default=20,
        help="Compute gradient cosine metrics every N steps (0 disables).",
    )
    args = parser.parse_args()

    base_config_path = args.base_config.resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    base_cfg = _load_json(base_config_path)

    storage = args.storage.strip() or f"sqlite:///{(workdir / (args.study_name + '.db')).as_posix()}"
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=False,
        warn_independent_sampling=False,
    )
    
    is_multi_obj = not args.single_objective
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=["maximize", "minimize"] if is_multi_obj else None,
        direction="maximize" if not is_multi_obj else None,
        load_if_exists=True,
        sampler=sampler,
    )

    fixed_scheduler = str(args.fixed_scheduler).strip().lower()
    if fixed_scheduler not in {"cosine", "onecycle", "multistep"}:
        fixed_scheduler = "cosine"

    lr_param_name = _resolve_param_name_for_float(
        study, base_name="lr", new_name="lr_v2", low=args.lr_min, high=args.lr_max, log=True
    )

    warm_start_name = args.warm_start_study_name.strip()
    if warm_start_name and warm_start_name != args.study_name:
        if len(study.trials) == 0:
            try:
                discovered = _discover_warm_start_source(
                    warm_start_name=warm_start_name,
                    explicit_warm_start_storage=args.warm_start_storage.strip(),
                    default_storage=storage,
                    workdir=workdir,
                )
                if discovered is None:
                    raise RuntimeError("no matching study found in candidate storages")
                discovered_name, src_storage = discovered
                src_study = optuna.load_study(study_name=discovered_name, storage=src_storage)
                imported, skipped = _warm_start_from_study(
                    study,
                    src_study,
                    lr_param_name=lr_param_name,
                    lr_min=args.lr_min,
                    lr_max=args.lr_max,
                    w_swd_min=float(args.w_swd_min),
                    w_swd_max=float(args.w_swd_max),
                    fixed_w_swd=float(args.fixed_w_swd),
                    w_color_min=float(args.w_color_min),
                    w_color_max=float(args.w_color_max),
                    fixed_w_color=float(args.fixed_w_color),
                    max_trials=args.warm_start_max_trials,
                )
                print(
                    f"Warm start imported {imported} trial(s) from '{src_study.study_name}' "
                    f"@ {src_storage} (skipped {skipped})"
                )
            except Exception as e:
                print(f"Warm start skipped: cannot load source study '{warm_start_name}' ({e})")
        else:
            print("Warm start skipped: destination study is not empty.")

    print(
        "Param mapping: "
        f"lr->{lr_param_name}"
    )
    raw_milestones = [x.strip() for x in str(args.multistep_milestones).split(",") if x.strip()]
    multistep_milestones = sorted({int(x) for x in raw_milestones}) if raw_milestones else [45, 55]

    def objective(trial: optuna.Trial) -> float | Tuple[float, float]:
        scheduler = fixed_scheduler
        base_lr = trial.suggest_float(lr_param_name, args.lr_min, args.lr_max, log=True)
        lr = float(base_lr)
        peak_lr = float(base_lr)
        onecycle_max_lr = float(base_lr)

        # Only search three variables: lr, w_swd, w_color.
        w_idt = int(args.fixed_w_idt)
        w_swd = trial.suggest_float("w_swd", args.w_swd_min, args.w_swd_max)
        w_color = trial.suggest_float("w_color", args.w_color_min, args.w_color_max)
        ada_mix_rank = int(args.fixed_ada_mix_rank)

        trial_dir = workdir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_cfg = _build_trial_config(
            base_cfg,
            lr=lr,
            w_identity=w_idt,
            scheduler=scheduler,
            trial_dir=trial_dir,
            epochs=args.epochs,
            full_eval_interval=args.full_eval_interval,
            w_color=w_color,
            w_swd=w_swd,
            ada_mix_rank=ada_mix_rank,
            multistep_milestones=multistep_milestones,
            multistep_gamma=float(args.multistep_gamma),
            onecycle_max_lr=onecycle_max_lr,
            onecycle_pct_start=float(args.onecycle_pct_start),
            onecycle_anneal_strategy=str(args.onecycle_anneal_strategy),
            grad_direction_interval=int(args.grad_direction_interval),
        )
        trial_cfg_path = trial_dir / "config.json"
        _save_json(trial_cfg_path, trial_cfg)

        try:
            _run_trial_training(trial_cfg_path)
        except KeyboardInterrupt as exc:
            if args.interrupt_as_fail:
                raise RuntimeError("Trial interrupted by KeyboardInterrupt") from exc
            raise

        exp_dir = Path(trial_cfg["checkpoint"]["save_dir"])
        summary_path = _find_summary_file(exp_dir, args.epochs)
        summary = _load_json(summary_path)
        clip_style, lpips = _extract_style_metrics(summary)
        pareto_score = float(args.style_weight * clip_style + args.alpha * (1.0 - lpips))

        trial.set_user_attr("clip_style", clip_style)
        trial.set_user_attr("lpips", lpips)
        trial.set_user_attr("pareto_score", pareto_score)
        trial.set_user_attr("scheduler", scheduler)
        trial.set_user_attr("peak_lr", peak_lr)
        trial.set_user_attr("w_idt", w_idt)
        trial.set_user_attr("ada_mix_rank", ada_mix_rank)
        trial.set_user_attr("summary_path", str(summary_path.resolve()))
        trial.set_user_attr("exp_dir", str(exp_dir.resolve()))

        if is_multi_obj:
            return clip_style, lpips
        else:
            return pareto_score

    # Keep search running even if a trial fails.
    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        catch=(subprocess.CalledProcessError, FileNotFoundError, KeyError, ValueError, RuntimeError),
    )

    _dump_trials_csv(study, workdir / f"{args.study_name}_trials.csv")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("No successful trials yet.")
        print(f"Trials CSV: {(workdir / f'{args.study_name}_trials.csv').resolve()}")
        print("Dashboard:")
        print(f"  optuna-dashboard \"{storage}\"")
        return

    print("=" * 60)
    print(f"Study: {args.study_name}")
    print(f"Storage: {storage}")
    
    if is_multi_obj:
        best_trials = study.best_trials
        print(f"Found {len(best_trials)} trials on the Pareto front.")
        for i, bt in enumerate(best_trials):
            print(f"  [{i+1}] Trial {bt.number} | clip_style: {bt.values[0]:.4f} | lpips: {bt.values[1]:.4f}")
    else:
        best = study.best_trial
        print(f"Best Trial: {best.number}")
        print(f"Best Pareto Score: {best.value:.6f}")
        print(f"Best Params: {best.params}")
        print(f"Best Attrs: clip_style={best.user_attrs.get('clip_style')} lpips={best.user_attrs.get('lpips')}")
        print(f"Best Summary: {best.user_attrs.get('summary_path')}")
        
    print("=" * 60)
    print(f"Trials CSV: {(workdir / f'{args.study_name}_trials.csv').resolve()}")
    print("Dashboard:")
    print(f"  optuna-dashboard \"{storage}\"")

if __name__ == "__main__":
    main()
