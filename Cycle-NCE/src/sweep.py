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
    epochs: Optional[int] = None


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


def make_experiments(base_cfg: Dict[str, Any]) -> List[Exp]:
    # Baseline Config (Default in overfit50.json):
    # SWD=20, Moment=2, Proj=128, Patch=[3,5], Id=50, Semi=1.0, L2=0.02

    exps: List[Exp] = []

    # Group P: projection precision
    exps.append(Exp("P01_Proj_32", "P", [("loss.swd_num_projections", 32)], epochs=40))
    exps.append(Exp("P02_Proj_128", "P", [("loss.swd_num_projections", 128)], epochs=40))
    exps.append(Exp("P03_Proj_256", "P", [("loss.swd_num_projections", 256)], epochs=50))

    # Group S: spatial scales
    exps.append(Exp("S01_Patch_3", "S", [("loss.swd_patch_sizes", [3])], epochs=30))
    exps.append(Exp("S02_Patch_3_5_9", "S", [("loss.swd_patch_sizes", [3, 5, 9])], epochs=30))

    # Group W: weight balance
    exps.append(Exp("W01_Strong_Moment", "W", [("loss.w_color_moment", 10.0)], epochs=40))
    exps.append(Exp("W02_No_Moment", "W", [("loss.w_color_moment", 0.0)], epochs=40))
    exps.append(Exp("W03_High_SWD", "W", [("loss.w_swd", 50.0)], epochs=40))

    # Group C: consistency/stability
    exps.append(Exp("C01_Semi_5", "C", [("loss.w_semigroup", 5.0)], epochs=50))
    exps.append(Exp("C02_Unconstrained", "C", [("loss.w_identity", 0.0), ("loss.w_semigroup", 0.0)], epochs=30))

    return exps


def apply_overrides(base_cfg: Dict[str, Any], overrides: List[Tuple[str, Any]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for k, v in overrides:
        deep_set(cfg, k, v)
    return cfg


def apply_infra_profile(cfg: Dict[str, Any], profile: str) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    p = str(profile).strip().lower()

    if "rtx30" in p:
        deep_set(out, "training.use_amp", True)
        deep_set(out, "training.amp_dtype", "fp16")
        deep_set(out, "training.use_grad_scaler", True)
        deep_set(out, "training.use_compile", False)
        deep_set(out, "training.batch_size", 32)
        deep_set(out, "training.allow_tf32", True)
        return out

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=str, default="overfit50.json")
    ap.add_argument("--run_py", type=str, default="run.py")
    ap.add_argument("--out_dir", type=str, default="../sweep_swd_reborn")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--groups", type=str, default="P,S,W,C")
    ap.add_argument("--infra_profile", type=str, default="rtx3060")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stop_on_error", action="store_true")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_path = (script_dir / args.base_config).resolve()
    run_py = (script_dir / args.run_py).resolve()
    out_root = (script_dir / args.out_dir).resolve()

    if not base_path.exists():
        print(f"Error: Base config not found: {base_path}")
        return

    base_cfg = load_json(base_path)
    deep_set(base_cfg, "model.loss_projector_use", False)

    exps = make_experiments(base_cfg)
    group_set = {g.strip().upper() for g in args.groups.split(",") if g.strip()}
    exps = [e for e in exps if e.group.upper() in group_set]
    if not exps:
        print("No experiments matched.")
        return

    python_bin = sys.executable
    summary = []

    for i, exp in enumerate(exps, 1):
        run_name = safe_name(exp.name)
        run_dir = out_root / run_name
        cfg = apply_overrides(base_cfg, exp.overrides)
        cfg = apply_infra_profile(cfg, args.infra_profile)

        deep_set(cfg, "training.seed", args.seed)
        final_epochs = exp.epochs if exp.epochs is not None else int(args.epochs)
        deep_set(cfg, "training.num_epochs", final_epochs)
        deep_set(cfg, "checkpoint.save_dir", str(run_dir / "checkpoints"))

        cfg_path = run_dir / "config.json"
        save_json(cfg_path, cfg)

        print(f"[{i}/{len(exps)}] {exp.name} (Group {exp.group}) | Epochs: {final_epochs}")

        if args.dry_run:
            summary.append((exp.name, 0))
            continue

        log_path = run_dir / "train.log"
        cmd = [python_bin, str(run_py), "--config", str(cfg_path)]
        code = run_cmd(cmd, log_path=log_path, cwd=script_dir)

        if code != 0:
            print(f"  FAILED code={code}")
            if args.stop_on_error:
                break

        summary.append((exp.name, code))

    save_json(out_root / "summary.json", {"experiments": summary})
    print("Done.")


if __name__ == "__main__":
    main()
