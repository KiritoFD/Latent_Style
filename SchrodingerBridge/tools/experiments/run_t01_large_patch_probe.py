from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
DEFAULT_BASE_CONFIG = ROOT / "exp" / "diffeomorphic_tangent_sweep" / "t01_ws0p03_g6_nl0p05" / "config.json"
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "t01_large_patch_probe"
DEFAULT_OUTPUT_ROOT = ROOT / "exp" / "t01_large_patch_probe"


@dataclass(frozen=True)
class Candidate:
    name: str
    patch_sizes: tuple[int, ...]
    micro_patch_max: int
    macro_patch_min: int
    note: str


def build_candidates() -> list[Candidate]:
    return [
        Candidate(
            name="lp00_small_macro_357_1516",
            patch_sizes=(3, 5, 7, 15, 16),
            micro_patch_max=7,
            macro_patch_min=15,
            note="Small structure anchors plus semi-global 9-patch macro style pressure.",
        ),
        Candidate(
            name="lp01_small_quad_357_21",
            patch_sizes=(3, 5, 7, 21),
            micro_patch_max=7,
            macro_patch_min=21,
            note="Small structure anchors plus four-patch K=21 global style pressure before one-patch collapse.",
        ),
        Candidate(
            name="lp02_small_degenerate_357_212931",
            patch_sizes=(3, 5, 7, 21, 29, 31),
            micro_patch_max=7,
            macro_patch_min=21,
            note="Small anchors plus large-patch奇效/负对照: combines four-patch K=21 with one-patch K=29/31 degeneracy.",
        ),
    ]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print(" ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _make_config(base_cfg: dict[str, Any], candidate: Candidate, *, save_dir: Path, num_epochs: int) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    bridge = cfg.setdefault("bridge", {})
    bridge["swd_patch_sizes"] = list(candidate.patch_sizes)
    bridge["swd_micro_patch_max"] = int(candidate.micro_patch_max)
    bridge["swd_macro_patch_min"] = int(candidate.macro_patch_min)
    bridge["swd_scale_invariant_patches"] = False
    bridge["swd_micro_weight"] = 1.0
    bridge["swd_macro_weight"] = 1.0

    training = cfg.setdefault("training", {})
    training["num_epochs"] = int(num_epochs)
    training["save_interval"] = 1

    checkpoint = cfg.setdefault("checkpoint", {})
    checkpoint["save_dir"] = "./" + save_dir.resolve().relative_to(ROOT).as_posix()

    cfg["ablation"] = {
        "name": candidate.name,
        "stage": "t01_large_patch_probe",
        "axis": "large_swd_patch_sizes",
        "patch_sizes": list(candidate.patch_sizes),
        "note": candidate.note,
    }
    return cfg


def _metrics_from_summary(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"clip_style": None, "clip_content": None, "content_lpips": None}
    payload = _load_json(path)
    overview = ((payload.get("analysis") or {}).get("all_pairs_overview") or {})
    return {
        "clip_style": float(overview.get("clip_style")) if overview.get("clip_style") is not None else None,
        "clip_content": float(overview.get("clip_content")) if overview.get("clip_content") is not None else None,
        "content_lpips": float(overview.get("content_lpips")) if overview.get("content_lpips") is not None else None,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "name",
        "epoch",
        "clip_style",
        "clip_content",
        "content_lpips",
        "patch_sizes",
        "run_dir",
        "summary",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k) for k in fields} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny t01 large-patch probe: 2-3 high-risk SWD patch experiments.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--eval-epochs", default="1,4,6,8")
    parser.add_argument("--max-total", type=int, default=3)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-num-steps", type=int, default=12)
    parser.add_argument("--eval-step-size", type=float, default=1.0)
    parser.add_argument("--eval-vae-decode-scale", type=float, default=0.191)
    parser.add_argument("--eval-residual-scale", type=float, default=1.0)
    parser.add_argument("--eval-force-integrate", action="store_true")
    parser.add_argument("--eval-modern-metrics", action="store_true")
    args = parser.parse_args()

    base_config = _load_json(args.base_config.resolve())
    config_root = args.config_root.resolve()
    output_root = args.output_root.resolve()
    eval_epochs = tuple(sorted({int(x) for x in args.eval_epochs.split(",") if x.strip()}))
    ledger_path = output_root / "large_patch_probe_ledger.jsonl"
    results_path = output_root / "large_patch_probe_results.csv"

    rows: list[dict[str, Any]] = []
    for candidate in build_candidates()[: int(args.max_total)]:
        run_dir = output_root / candidate.name
        config_path = config_root / f"{candidate.name}.json"
        cfg = _make_config(base_config, candidate, save_dir=run_dir, num_epochs=args.num_epochs)
        _write_json(config_path, cfg)

        checkpoints = [run_dir / f"epoch_{epoch:04d}.pt" for epoch in eval_epochs]
        if args.force_train or not all(path.exists() for path in checkpoints):
            _run([sys.executable, "src/run.py", "--config", str(config_path)], cwd=ROOT, dry_run=args.dry_run)

        for epoch in eval_epochs:
            ckpt = run_dir / f"epoch_{epoch:04d}.pt"
            eval_dir = run_dir / "full_eval" / f"epoch_{epoch:04d}"
            summary_path = eval_dir / "summary.json"
            if args.force_eval or not summary_path.exists():
                eval_cmd = [
                    sys.executable,
                    "src/utils/run_evaluation.py",
                    "--checkpoint",
                    str(ckpt),
                    "--output",
                    str(eval_dir),
                    "--num_steps",
                    str(args.eval_num_steps),
                    "--step_size",
                    str(args.eval_step_size),
                    "--vae_decode_scale",
                    str(args.eval_vae_decode_scale),
                    "--residual_scale",
                    str(args.eval_residual_scale),
                ]
                if args.eval_modern_metrics:
                    eval_cmd.append("--eval_enable_modern_metrics")
                if args.eval_force_integrate:
                    eval_cmd.append("--force_integrate")
                _run(eval_cmd, cwd=ROOT, dry_run=args.dry_run)

            row = {
                "name": candidate.name,
                "epoch": epoch,
                "patch_sizes": ",".join(str(x) for x in candidate.patch_sizes),
                "run_dir": run_dir.as_posix(),
                "summary": summary_path.as_posix(),
                **_metrics_from_summary(summary_path),
            }
            rows.append(row)
            _append_jsonl(ledger_path, row)
            _write_csv(results_path, sorted(rows, key=lambda r: float(r.get("clip_style") or -9999.0), reverse=True))

    print(json.dumps({"output_root": str(output_root), "results": str(results_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
