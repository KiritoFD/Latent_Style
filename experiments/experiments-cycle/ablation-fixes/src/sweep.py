import argparse
import copy
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def deep_set(d: Dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


@dataclass
class Exp:
    name: str
    group: str
    overrides: List[Tuple[str, Any]]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_cmd(cmd: List[str], log_path: Path, cwd: Optional[Path] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        return p.wait()


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def make_experiments() -> List[Exp]:
    exps: List[Exp] = []

    # =================================================================
    # Group A: regularization balance
    # =================================================================
    exps += [
        Exp("A1_tv0.5_l2_0.01", "A", [("loss.w_delta_tv", 0.5), ("loss.w_delta_l2", 0.01)]),
        Exp("A2_tv0.1_l2_0.1", "A", [("loss.w_delta_tv", 0.1), ("loss.w_delta_l2", 0.1)]),
        Exp("A3_noTV_l2_0.05", "A", [("loss.w_delta_tv", 0.0), ("loss.w_delta_l2", 0.05)]),
        Exp("A4_tv0.5_outTV0.5", "A", [("loss.w_delta_tv", 0.5), ("loss.w_output_tv", 0.5)]),
    ]

    # =================================================================
    # Group B: style strength + projector
    # =================================================================
    exps += [
        Exp("B1_gram2.0_moment1.0", "B", [("loss.w_stroke_gram", 2.0), ("loss.w_color_moment", 1.0)]),
        Exp("B2_proj32ch", "B", [("model.loss_projector_channels", 32)]),
        Exp(
            "B3_soft_style",
            "B",
            [
                ("loss.w_stroke_gram", 2.0),
                ("loss.w_color_moment", 1.0),
                ("model.loss_projector_channels", 32),
            ],
        ),
    ]

    # =================================================================
    # Group C: structure keeping
    # =================================================================
    exps += [
        Exp("C1_struct1.0", "C", [("loss.w_struct", 1.0)]),
        Exp("C2_no_struct", "C", [("loss.w_struct", 0.0)]),
    ]

    # =================================================================
    # Group D: semigroup stabilizer
    # =================================================================
    exps += [
        Exp("D1_semigroup0.2", "D", [("loss.w_semigroup", 0.2)]),
        Exp("D2_semigroup0.5", "D", [("loss.w_semigroup", 0.5)]),
    ]

    return exps


def apply_overrides(base_cfg: Dict[str, Any], overrides: List[Tuple[str, Any]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for k, v in overrides:
        deep_set(cfg, k, v)
    return cfg


def apply_infra_profile(cfg: Dict[str, Any], profile: str) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    p = str(profile).strip().lower()
    if p in {"", "none"}:
        return out

    if "rtx30" in p:
        deep_set(out, "training.use_amp", True)
        deep_set(out, "training.amp_dtype", "fp16")
        deep_set(out, "training.use_grad_scaler", True)
        deep_set(out, "training.allow_tf32", True)
        deep_set(out, "training.use_compile", False)
        deep_set(out, "training.prefetch_factor", 2)
        deep_set(out, "training.persistent_workers", True)
        return out

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=str, default="config.json", help="Path to base config.json")
    ap.add_argument("--run_py", type=str, default="run.py", help="Path to training entry script")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="../ablation-fixes",
        help="Directory for generated configs/logs/checkpoints",
    )
    ap.add_argument("--epochs", type=int, default=60, help="Override training.num_epochs")
    ap.add_argument("--batch_size", type=int, default=-1, help="Override training.batch_size (-1 keeps base config)")
    ap.add_argument("--seed", type=int, default=42, help="Fixed seed")
    ap.add_argument("--groups", type=str, default="A,B,C,D", help="Groups to run")
    ap.add_argument("--filter", type=str, default="", help="Run only experiment names containing this text")
    ap.add_argument("--infra_profile", type=str, default="rtx3060_stable")
    ap.add_argument("--dry_run", action="store_true", help="Only write configs")
    ap.add_argument("--stop_on_error", action="store_true", help="Stop sweep when a run fails")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = (script_dir / base_path).resolve()

    if not base_path.exists():
        print(f"Error: Base config not found: {base_path}")
        print("Please make sure config.json exists in src/ or specify --base_config")
        return

    run_py = Path(args.run_py)
    if not run_py.is_absolute():
        run_py = (script_dir / run_py).resolve()

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = (script_dir / out_root).resolve()

    base_cfg = load_json(base_path)
    exps = make_experiments()

    group_set = {g.strip().upper() for g in args.groups.split(",") if g.strip()}
    exps = [e for e in exps if e.group.upper() in group_set]
    if args.filter:
        exps = [e for e in exps if args.filter in e.name]

    if not exps:
        print("No experiments matched.")
        return

    python_bin = sys.executable
    summary: List[Tuple[str, int, str]] = []

    for i, exp in enumerate(exps, 1):
        run_name = safe_name(exp.name)
        run_dir = out_root / run_name

        cfg = apply_overrides(base_cfg, exp.overrides)
        cfg = apply_infra_profile(cfg, args.infra_profile)
        deep_set(cfg, "training.seed", args.seed)
        deep_set(cfg, "training.num_epochs", int(args.epochs))
        deep_set(cfg, "training.resume_checkpoint", "")
        # Lock all runs to single-step to avoid multi-step artifact amplification.
        deep_set(cfg, "loss.train_num_steps_min", 1)
        deep_set(cfg, "loss.train_num_steps_max", 1)
        deep_set(cfg, "loss.train_step_size_min", 1.0)
        deep_set(cfg, "loss.train_step_size_max", 1.0)
        deep_set(cfg, "training.full_eval_num_steps", 1)
        deep_set(cfg, "training.full_eval_step_size", 1.0)
        deep_set(cfg, "inference.num_steps", 1)
        deep_set(cfg, "inference.step_size", 1.0)

        if int(args.batch_size) > 0:
            deep_set(cfg, "training.batch_size", int(args.batch_size))

        deep_set(cfg, "training.test_image_dir", "../../style_data/overfit50")

        ckpt_save = run_dir / "checkpoints"
        deep_set(cfg, "checkpoint.save_dir", str(ckpt_save))

        cfg_path = run_dir / "config.json"
        save_json(cfg_path, cfg)

        eff_bs = int(cfg.get("training", {}).get("batch_size", -1))
        print(f"[{i}/{len(exps)}] {exp.name}")
        print(f"  config: {cfg_path}")
        print(f"  batch_size: {eff_bs}")

        if args.dry_run:
            summary.append((exp.name, 0, "dry_run"))
            continue

        cmd = [python_bin, str(run_py), "--config", str(cfg_path)]
        log_path = run_dir / "train.log"
        print(f"  Logging to: {log_path}")
        code = run_cmd(cmd, log_path=log_path, cwd=script_dir)

        if code != 0:
            print(f"  FAILED code={code}")
            summary.append((exp.name, code, str(log_path)))
            if args.stop_on_error:
                break
            continue

        print("  OK")
        summary.append((exp.name, 0, str(log_path)))

    summary_path = out_root / "summary.json"
    payload = {
        "base_config": str(base_path),
        "run_py": str(run_py),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "groups": sorted(group_set),
        "filter": args.filter,
        "infra_profile": args.infra_profile,
        "batch_size": int(args.batch_size),
        "results": [{"name": n, "code": c, "log": l} for n, c, l in summary],
    }
    save_json(summary_path, payload)
    print("Done.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
