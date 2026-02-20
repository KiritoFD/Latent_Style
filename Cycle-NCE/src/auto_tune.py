from __future__ import annotations

import argparse
import csv
import json
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import optuna


def evaluate_score(content_ssim: float, style_swd: float) -> float:
    # Primary objective: minimize SWD (Optuna maximize => negate).
    score = -float(style_swd)

    # Constraint floor on SSIM with smooth linear penalty below threshold.
    target_ssim = 0.45
    if float(content_ssim) < target_ssim:
        penalty = (target_ssim - float(content_ssim)) * 10.0
        score -= penalty
    return score


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _extract_metrics(summary_history: dict) -> tuple[float, float]:
    latest = summary_history.get("latest", {})
    style_swd = latest.get("photo_to_art_style_swd", latest.get("transfer_style_swd"))
    content_ssim = latest.get("transfer_content_ssim")
    if style_swd is None or content_ssim is None:
        raise RuntimeError("Cannot parse style_swd/content_ssim from summary_history.json")
    return float(style_swd), float(content_ssim)


def _resolve_csv_path(args: argparse.Namespace) -> Path:
    if str(args.csv_path).strip():
        return Path(args.csv_path).resolve()
    root = Path(args.trials_root).resolve()
    return root / f"{args.study_name}_trials.csv"


def _append_trial_csv(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp",
        "study_name",
        "trial",
        "status",
        "elapsed_sec",
        "score",
        "style_swd",
        "content_ssim",
        "w_style",
        "w_structure",
        "w_moment",
        "w_identity",
        "batch_size",
        "num_epochs",
        "trial_dir",
        "train_log",
        "trial_config",
        "error",
    ]
    legacy_header = [
        "timestamp",
        "study_name",
        "trial",
        "status",
        "elapsed_sec",
        "score",
        "clip_style",
        "content_lpips",
        "w_style",
        "w_structure",
        "w_moment",
        "w_identity",
        "batch_size",
        "num_epochs",
        "trial_dir",
        "train_log",
        "trial_config",
        "error",
    ]
    exists = csv_path.exists()
    target_header = header
    write_row = dict(row)
    if exists:
        with open(csv_path, "r", encoding="utf-8", newline="") as rf:
            first = rf.readline().strip()
        if first:
            current = [x.strip() for x in first.split(",")]
            if current == legacy_header:
                target_header = legacy_header
                write_row["clip_style"] = write_row.get("style_swd", "")
                write_row["content_lpips"] = write_row.get("content_ssim", "")
            elif current != header:
                raise RuntimeError(f"CSV header mismatch for {csv_path}")
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=target_header)
        if not exists:
            writer.writeheader()
        writer.writerow({k: write_row.get(k, "") for k in target_header})


def _format_hparams(trial: optuna.Trial) -> str:
    return (
        f"w_style={trial.params['w_style']:.3f}, "
        f"w_structure={trial.params['w_structure']:.3f}, "
        f"w_moment={trial.params['w_moment']:.3f}, "
        f"w_identity={trial.params['w_identity']:.3f}"
    )


def _run_and_stream(cmd: list[str], *, cwd: Path, log_path: Path, prefix: str) -> None:
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            print(f"{prefix} {line.rstrip()}")
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)


def make_objective(args: argparse.Namespace, base_cfg: dict):
    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        cfg = json.loads(json.dumps(base_cfg))

        # Constrained search space.
        w_style = trial.suggest_float("w_style", 100.0, 300.0)
        w_structure = trial.suggest_float("w_structure", 10.0, 60.0)
        w_moment = trial.suggest_float("w_moment", 50.0, 200.0)
        w_identity = trial.suggest_float("w_identity", 100.0, 800.0)

        cfg.setdefault("loss", {})
        cfg["loss"].update(
            {
                "w_style": w_style,
                "w_structure": w_structure,
                "w_moment": w_moment,
                "w_identity": w_identity,
            }
        )

        cfg.setdefault("data", {})
        cfg["data"].update(
            {
                "data_root": args.data_root,
                "virtual_length_multiplier": 1000,
            }
        )

        cfg.setdefault("training", {})
        cfg["training"].update(
            {
                "batch_size": int(args.batch_size),
                "num_epochs": 1,
                "num_workers": 0,
                "persistent_workers": False,
                "scheduler": "cosine",
                "learning_rate": 5e-4,
                "min_learning_rate": 1e-6,
                "warmup_epochs": 0.1,
                "save_interval": 1,
                "full_eval_interval": 0,
                "full_eval_on_last_epoch": True,
                "use_tqdm": False,
                "log_interval": int(args.log_interval),
                "debug_grad_enabled": False,
            }
        )

        batch_size = int(cfg["training"]["batch_size"])
        num_epochs = int(cfg["training"]["num_epochs"])

        trial_dir = (Path(args.trials_root) / f"trial_{trial.number:04d}").resolve()
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)
        cfg.setdefault("checkpoint", {})
        cfg["checkpoint"]["save_dir"] = str(trial_dir)
        cfg["training"]["resume_checkpoint"] = ""

        trial_cfg_path = trial_dir / "config.trial.json"
        _write_json(trial_cfg_path, cfg)

        print(
            f"\n[Trial {trial.number}] START | {_format_hparams(trial)} | "
            f"batch={batch_size} epochs={num_epochs}"
        )
        log_path = trial_dir / "train.log"
        csv_path = _resolve_csv_path(args)
        try:
            # Let it crash: any non-zero training/eval exit marks the trial failed.
            _run_and_stream(
                ["uv", "run", "run.py", "--config", str(trial_cfg_path)],
                cwd=Path(__file__).resolve().parent,
                log_path=log_path,
                prefix=f"[T{trial.number:04d}]",
            )

            history_path = trial_dir / "full_eval" / "summary_history.json"
            history = _read_json(history_path)
            style_swd, content_ssim = _extract_metrics(history)
            score = evaluate_score(content_ssim, style_swd)
            elapsed = float(time.time() - t0)

            trial.set_user_attr("style_swd", style_swd)
            trial.set_user_attr("content_ssim", content_ssim)
            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("elapsed_sec", elapsed)

            _append_trial_csv(
                csv_path,
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "study_name": args.study_name,
                    "trial": trial.number,
                    "status": "completed",
                    "elapsed_sec": f"{elapsed:.3f}",
                    "score": f"{score:.8f}",
                    "style_swd": f"{style_swd:.8f}",
                    "content_ssim": f"{content_ssim:.8f}",
                    "w_style": f"{w_style:.8f}",
                    "w_structure": f"{w_structure:.8f}",
                    "w_moment": f"{w_moment:.8f}",
                    "w_identity": f"{w_identity:.8f}",
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "trial_dir": str(trial_dir),
                    "train_log": str(log_path),
                    "trial_config": str(trial_cfg_path),
                    "error": "",
                },
            )

            print(
                f"[Trial {trial.number}] DONE | score={score:.6f} "
                f"style_swd={style_swd:.6f} ssim={content_ssim:.6f} "
                f"time={elapsed:.1f}s"
            )
            print(f"[Trial {trial.number}] CSV -> {csv_path}")
            return score
        except Exception as exc:
            elapsed = float(time.time() - t0)
            _append_trial_csv(
                csv_path,
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "study_name": args.study_name,
                    "trial": trial.number,
                    "status": "failed",
                    "elapsed_sec": f"{elapsed:.3f}",
                    "score": "",
                    "style_swd": "",
                    "content_ssim": "",
                    "w_style": f"{w_style:.8f}",
                    "w_structure": f"{w_structure:.8f}",
                    "w_moment": f"{w_moment:.8f}",
                    "w_identity": f"{w_identity:.8f}",
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "trial_dir": str(trial_dir),
                    "train_log": str(log_path),
                    "trial_config": str(trial_cfg_path),
                    "error": repr(exc),
                },
            )
            print(f"[Trial {trial.number}] FAILED | wrote CSV row -> {csv_path}")
            raise

    return objective


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bayesian HPO (Optuna/TPE) for Cycle-NCE loss weights.")
    p.add_argument("--config", type=str, default="config.json")
    p.add_argument("--data-root", type=str, default="../../sdxl-256-overfit50")
    p.add_argument("--trials-root", type=str, default="../ssim-optun-nce")
    p.add_argument("--study-name", type=str, default="ssim-latent_style_hpo-nce")
    p.add_argument("--storage", type=str, default="sqlite:///optuna_latent_style.db")
    p.add_argument("--csv-path", type=str, default="", help="Trial CSV path. Default: <trials_root>/<study_name>_trials.csv")
    p.add_argument("--n-trials", type=int, default=1000)
    p.add_argument("--num-epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=0, help="Seconds. 0 means no timeout.")
    return p


def _print_best_snapshot(study: optuna.Study, trial: optuna.Trial) -> None:
    best = study.best_trial
    print(
        f"[Study] finished trial={trial.number} value={trial.value} state={trial.state.name} | "
        f"best_trial={best.number} best_value={study.best_value:.6f}"
    )


def main() -> None:
    args = build_argparser().parse_args()

    config_file = Path(args.config).resolve()
    backup_file = config_file.with_suffix(config_file.suffix + ".backup")
    shutil.copy2(config_file, backup_file)

    base_cfg = _read_json(config_file)
    sampler = optuna.samplers.TPESampler(seed=int(args.sampler_seed))
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=sampler,
        storage=args.storage,
        load_if_exists=True,
    )

    try:
        study.optimize(
            make_objective(args, base_cfg),
            n_trials=int(args.n_trials),
            timeout=None if int(args.timeout) <= 0 else int(args.timeout),
            callbacks=[_print_best_snapshot],
        )
    finally:
        # Always restore the user's base config file.
        shutil.move(str(backup_file), str(config_file))
    print(f"Trial CSV: {_resolve_csv_path(args)}")

    best = study.best_trial
    print("\n" + "=" * 48)
    print(f"Optimization Completed. Best Trial: {best.number}")
    print(f"Best Score: {study.best_value:.6f}")
    print("Best Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v:.6f}")
    print("Best Metrics:")
    print(f"  style_swd: {best.user_attrs.get('style_swd', float('nan')):.6f}")
    print(f"  content_ssim: {best.user_attrs.get('content_ssim', float('nan')):.6f}")
    print(f"  trial_dir: {best.user_attrs.get('trial_dir', '')}")
    print("=" * 48)


if __name__ == "__main__":
    main()
