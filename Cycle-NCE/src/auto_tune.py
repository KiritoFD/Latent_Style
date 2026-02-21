from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import optuna


def _safe_float(v) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _softplus(x: float) -> float:
    if x > 30.0:
        return x
    if x < -30.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires non-empty values")
    q = min(max(float(q), 0.0), 1.0)
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _build_score_context(content_refs: list[float], args: argparse.Namespace) -> dict:
    if not content_refs:
        raise ValueError("content_refs is empty")
    q_p = _quantile(content_refs, float(args.score_content_quantile))
    n = max(float(args.score_data_size), 1.0)
    n0 = max(float(args.score_data_size_n0), 1.0)
    delta = float(args.score_logdown_a) * math.log(n / n0)
    tau = q_p - delta
    iqr = _quantile(content_refs, 0.75) - _quantile(content_refs, 0.25)
    sigma_c = max(iqr / 1.349, float(args.score_sigma_min))
    return {
        "tau": tau,
        "sigma_c": sigma_c,
        "lambda": float(args.score_penalty_lambda),
        "alpha": float(args.score_content_bonus_alpha),
    }


def evaluate_score(clip_content_sim: float, style_swd: float, clip_style_sim: float, score_ctx: dict) -> float:
    del style_swd
    c = float(clip_content_sim)
    s = float(clip_style_sim)
    tau = float(score_ctx["tau"])
    sigma_c = float(score_ctx["sigma_c"])
    lam = float(score_ctx["lambda"])
    alpha = float(score_ctx["alpha"])
    z = (tau - c) / sigma_c
    return s - lam * _softplus(z) + alpha * max(0.0, (c - tau) / sigma_c)


def _load_trial_rows(csv_path: Path) -> tuple[list[str], list[dict]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    return header, rows


def _collect_completed_rows_with_clip(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        if str(row.get("status", "")).strip().lower() != "completed":
            continue
        c = _safe_float(row.get("clip_content_sim"))
        s = _safe_float(row.get("clip_style_sim"))
        if c is None or s is None:
            continue
        out.append({"row": row, "clip_content_sim": c, "clip_style_sim": s})
    return out


def _completed_content_refs(csv_path: Path) -> list[float]:
    if not csv_path.exists():
        return []
    _, rows = _load_trial_rows(csv_path)
    return [x["clip_content_sim"] for x in _collect_completed_rows_with_clip(rows)]


def _maybe_recompute_historical_scores(csv_path: Path, args: argparse.Namespace) -> dict[int, float]:
    if not csv_path.exists():
        return {}
    header, rows = _load_trial_rows(csv_path)
    if not header:
        return {}
    completed = _collect_completed_rows_with_clip(rows)
    if not completed:
        return {}

    score_ctx = _build_score_context([x["clip_content_sim"] for x in completed], args)
    first = completed[0]
    old_score = _safe_float(first["row"].get("score"))
    new_score = evaluate_score(first["clip_content_sim"], 0.0, first["clip_style_sim"], score_ctx)
    changed = old_score is None or abs(old_score - new_score) > float(args.score_recalc_eps)
    if not changed:
        return {}

    trial_score_map: dict[int, float] = {}
    for item in completed:
        sc = evaluate_score(item["clip_content_sim"], 0.0, item["clip_style_sim"], score_ctx)
        item["row"]["score"] = f"{sc:.8f}"
        tnum = _safe_float(item["row"].get("trial"))
        if tnum is not None:
            trial_score_map[int(tnum)] = float(sc)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Score] objective changed; recomputed historical scores in {csv_path} ({len(completed)} rows)")
    return trial_score_map


def _resolve_sqlite_db_path(storage: str, cwd: Path) -> Path | None:
    s = str(storage).strip()
    if not s.startswith("sqlite:///"):
        return None
    raw = s[len("sqlite:///") :]
    if not raw:
        return None
    if raw.startswith("/") and len(raw) >= 3 and raw[2] == ":":
        raw = raw[1:]
    p = Path(raw)
    if not p.is_absolute():
        p = (cwd / p).resolve()
    return p


def _recompute_db_scores(storage: str, study_name: str, trial_score_map: dict[int, float]) -> None:
    if not trial_score_map:
        return
    db_path = _resolve_sqlite_db_path(storage, Path.cwd())
    if db_path is None:
        print(f"[Score] skip DB update: non-sqlite storage ({storage})")
        return
    if not db_path.exists():
        print(f"[Score] skip DB update: sqlite file not found ({db_path})")
        return

    updated = 0
    with sqlite3.connect(str(db_path)) as con:
        cur = con.cursor()
        row = cur.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,)).fetchone()
        if row is None:
            print(f"[Score] skip DB update: study '{study_name}' not found in {db_path}")
            return
        study_id = int(row[0])
        trials = cur.execute(
            "SELECT trial_id, number FROM trials WHERE study_id = ? AND state = 'COMPLETE'",
            (study_id,),
        ).fetchall()
        for trial_id, number in trials:
            tnum = int(number)
            if tnum not in trial_score_map:
                continue
            cur.execute(
                "UPDATE trial_values SET value = ?, value_type = 'FINITE' WHERE trial_id = ? AND objective = 0",
                (float(trial_score_map[tnum]), int(trial_id)),
            )
            updated += int(cur.rowcount or 0)
        con.commit()
    print(f"[Score] updated DB values in {db_path} ({updated} rows)")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _extract_metrics(summary_history: dict) -> tuple[float, float, float]:
    latest = summary_history.get("latest", {})
    style_swd = latest.get("photo_to_art_style_swd", latest.get("transfer_style_swd"))
    clip_content_sim = latest.get("transfer_clip_content_sim")
    clip_style_sim = latest.get("photo_to_art_clip_style_sim", latest.get("transfer_clip_style_sim"))
    if style_swd is None or clip_content_sim is None or clip_style_sim is None:
        raise RuntimeError("Missing style_swd/transfer_clip_content_sim/transfer_clip_style_sim in summary_history.json")
    return float(style_swd), float(clip_content_sim), float(clip_style_sim)


def _extract_metrics_from_summary(summary: dict) -> tuple[float, float, float]:
    analysis = summary.get("analysis", {})
    transfer = analysis.get("style_transfer_ability", {})
    style_swd = transfer.get("style_swd")
    clip_content_sim = transfer.get("clip_content_sim")
    clip_style_sim = transfer.get("clip_style_sim")
    if style_swd is None or clip_content_sim is None or clip_style_sim is None:
        raise RuntimeError("Missing style_swd/clip_content_sim/clip_style_sim in analysis.style_transfer_ability")
    return float(style_swd), float(clip_content_sim), float(clip_style_sim)


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
        "clip_content_sim",
        "clip_style_sim",
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
    old_header = [
        "timestamp",
        "study_name",
        "trial",
        "status",
        "elapsed_sec",
        "score",
        "style_swd",
        "clip_content_sim",
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
    older_header = [
        "timestamp",
        "study_name",
        "trial",
        "status",
        "elapsed_sec",
        "score",
        "style_swd",
        "content_lf_ssim",
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
    oldest_header = [
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
                write_row["content_lpips"] = write_row.get("clip_content_sim", "")
            elif current == old_header:
                target_header = old_header
                write_row["clip_content_sim"] = write_row.get("clip_content_sim", "")
            elif current == older_header:
                target_header = older_header
                write_row["content_lf_ssim"] = write_row.get("clip_content_sim", "")
            elif current == oldest_header:
                target_header = oldest_header
                write_row["content_ssim"] = write_row.get("clip_content_sim", "")
            elif current != header:
                raise RuntimeError(f"CSV header mismatch for {csv_path}")
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=target_header)
        if not exists:
            writer.writeheader()
        writer.writerow({k: write_row.get(k, "") for k in target_header})


def _format_hparams(trial: optuna.Trial, w_structure: float) -> str:
    return (
        f"w_style={trial.params['w_style']:.3f}, "
        f"w_structure={w_structure:.3f}, "
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


def _run_inprocess(config_path: Path, *, cwd: Path, log_path: Path) -> None:
    import run as run_module

    old_cwd = Path.cwd()
    old_argv = list(sys.argv)
    try:
        os.chdir(str(cwd))
        sys.argv = ["run.py", "--config", str(config_path)]
        with open(log_path, "w", encoding="utf-8") as log_f:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout = log_f
                sys.stderr = log_f
                ret = run_module.main(["--config", str(config_path)])
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    finally:
        sys.argv = old_argv
        os.chdir(str(old_cwd))
    if ret not in (None, 0):
        raise subprocess.CalledProcessError(int(ret), ["inprocess", "run.py", "--config", str(config_path)])


def make_objective(args: argparse.Namespace, base_cfg: dict):
    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        cfg = json.loads(json.dumps(base_cfg))

        w_style = trial.suggest_float("w_style", 1000.0, 1000000.0, log=True)
        w_structure = 1.0
        w_moment = trial.suggest_float("w_moment", 0.1, 50.0, log=True)
        w_identity = trial.suggest_float("w_identity", 0.01, 10.0, log=True)

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
                "virtual_length_multiplier": 300,
            }
        )

        cfg.setdefault("training", {})
        cfg["training"].update(
            {
                "batch_size": int(args.batch_size),
                "num_epochs": 1,
                "num_workers": 0,
                "persistent_workers": False,
                "use_gradient_checkpointing": False,
                "use_compile": False,
                "scheduler": "cosine",
                "learning_rate": 5e-4,
                "min_learning_rate": 1e-6,
                "warmup_epochs": 0.1,
                "save_interval": 1,
                "full_eval_interval": 0,
                "full_eval_on_last_epoch": True,
                "full_eval_inprocess": bool(args.inprocess_run),
                "full_eval_batch_size": int(args.full_eval_batch_size),
                "full_eval_ref_feature_batch_size": int(args.full_eval_ref_feature_batch_size),
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
            f"\n[Trial {trial.number}] START | {_format_hparams(trial, w_structure)} | "
            f"batch={batch_size} epochs={num_epochs}"
        )
        log_path = trial_dir / "train.log"
        csv_path = _resolve_csv_path(args)
        try:
            # Let it crash: any non-zero training/eval exit marks the trial failed.
            if bool(args.inprocess_run):
                _run_inprocess(
                    trial_cfg_path,
                    cwd=Path(__file__).resolve().parent,
                    log_path=log_path,
                )
            else:
                _run_and_stream(
                    ["uv", "run", "run.py", "--config", str(trial_cfg_path)],
                    cwd=Path(__file__).resolve().parent,
                    log_path=log_path,
                    prefix=f"[T{trial.number:04d}]",
                )

            history_path = trial_dir / "full_eval" / "summary_history.json"
            if history_path.exists():
                history = _read_json(history_path)
                style_swd, clip_content_sim, clip_style_sim = _extract_metrics(history)
            else:
                latest_summary = trial_dir / "full_eval" / "epoch_0001" / "summary.json"
                if not latest_summary.exists():
                    raise FileNotFoundError(
                        f"Missing both {history_path} and fallback {latest_summary}. "
                        f"Check full-eval log under {trial_dir / 'logs'}."
                    )
                summary = _read_json(latest_summary)
                style_swd, clip_content_sim, clip_style_sim = _extract_metrics_from_summary(summary)
            refs = _completed_content_refs(csv_path)
            refs.append(float(clip_content_sim))
            score_ctx = _build_score_context(refs, args)
            score = evaluate_score(clip_content_sim, style_swd, clip_style_sim, score_ctx)
            elapsed = float(time.time() - t0)

            trial.set_user_attr("style_swd", style_swd)
            trial.set_user_attr("clip_content_sim", clip_content_sim)
            trial.set_user_attr("clip_style_sim", clip_style_sim)
            trial.set_user_attr("content_lf_ssim", clip_content_sim)  # backward-compatible alias
            trial.set_user_attr("content_ssim", clip_content_sim)  # backward-compatible alias
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
                    "clip_content_sim": f"{clip_content_sim:.8f}",
                    "clip_style_sim": f"{clip_style_sim:.8f}",
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
                f"style_swd={style_swd:.6f} clip_content={clip_content_sim:.6f} clip_style={clip_style_sim:.6f} "
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
                    "clip_content_sim": "",
                    "clip_style_sim": "",
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
    p.add_argument("--trials-root", type=str, default="../fp32-clip")
    p.add_argument("--study-name", type=str, default="fp32-clip")
    p.add_argument("--storage", type=str, default="sqlite:///fp32-clip.db")
    p.add_argument("--csv-path", type=str, default="", help="Trial CSV path. Default: <trials_root>/<study_name>_trials.csv")
    p.add_argument("--n-trials", type=int, default=1000)
    p.add_argument("--num-epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--full-eval-batch-size", type=int, default=16)
    p.add_argument("--full-eval-ref-feature-batch-size", type=int, default=32)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=0, help="Seconds. 0 means no timeout.")
    p.add_argument("--inprocess-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--score-content-quantile", type=float, default=0.20)
    p.add_argument("--score-penalty-lambda", type=float, default=2.5)
    p.add_argument("--score-content-bonus-alpha", type=float, default=0.10)
    p.add_argument("--score-logdown-a", type=float, default=0.010)
    p.add_argument("--score-data-size", type=float, default=50.0)
    p.add_argument("--score-data-size-n0", type=float, default=50.0)
    p.add_argument("--score-sigma-min", type=float, default=0.02)
    p.add_argument("--score-recalc-eps", type=float, default=1e-6)
    return p


def _print_best_snapshot(study: optuna.Study, trial: optuna.Trial) -> None:
    best = study.best_trial
    print(
        f"[Study] finished trial={trial.number} value={trial.value} state={trial.state.name} | "
        f"best_trial={best.number} best_value={study.best_value:.6f}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    csv_path = _resolve_csv_path(args)
    recalculated = _maybe_recompute_historical_scores(csv_path, args)
    _recompute_db_scores(args.storage, args.study_name, recalculated)

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
    print(f"Trial CSV: {csv_path}")

    best = study.best_trial
    print("\n" + "=" * 48)
    print(f"Optimization Completed. Best Trial: {best.number}")
    print(f"Best Score: {study.best_value:.6f}")
    print("Best Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v:.6f}")
    print("Best Metrics:")
    print(f"  style_swd: {best.user_attrs.get('style_swd', float('nan')):.6f}")
    print(f"  clip_content_sim: {best.user_attrs.get('clip_content_sim', best.user_attrs.get('content_lf_ssim', best.user_attrs.get('content_ssim', float('nan')))):.6f}")
    print(f"  clip_style_sim: {best.user_attrs.get('clip_style_sim', float('nan')):.6f}")
    print(f"  trial_dir: {best.user_attrs.get('trial_dir', '')}")
    print("=" * 48)


if __name__ == "__main__":
    main()
