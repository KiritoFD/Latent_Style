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
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "t01_patch36"
DEFAULT_OUTPUT_ROOT = ROOT / "exp" / "t01_patch36"


@dataclass(frozen=True)
class Candidate:
    name: str
    patch_sizes: tuple[int, ...]
    micro_patch_max: int
    macro_patch_min: int
    micro_weight: float = 1.0
    macro_weight: float = 1.0
    use_dilation: bool = False
    dilation: int = 2
    note: str = ""


def build_candidates() -> list[Candidate]:
    return [
        # A. t01 and direct controls.
        Candidate("p00_t01_35715", (3, 5, 7, 15), 7, 15, note="Original t01 patch recipe under best inference."),
        Candidate("p01_small_357", (3, 5, 7), 7, 15, note="Remove macro anchor; tests whether style loss came from local bands only."),
        Candidate("p02_small_3578", (3, 5, 7, 8), 7, 8, note="Replace K=15 with K=8 texton candidate."),
        Candidate("p03_small_3579", (3, 5, 7, 9), 7, 9, note="Replace K=15 with K=9 texton candidate."),
        Candidate("p04_texton_35789", (3, 5, 7, 8, 9), 7, 8, note="Full local-to-texton band."),
        Candidate("p05_texton_3589", (3, 5, 8, 9), 5, 8, note="Drop K=7, emphasize K=8/9."),
        # B. Focused K=8/9 texton probes.
        Candidate("p06_578", (5, 7, 8), 7, 8, note="Compact texton band around K=8."),
        Candidate("p07_579", (5, 7, 9), 7, 9, note="Compact texton band around K=9."),
        Candidate("p08_789", (7, 8, 9), 7, 8, note="High local to texton only."),
        Candidate("p09_589", (5, 8, 9), 5, 8, note="Mid-small plus texton."),
        Candidate("p10_389", (3, 8, 9), 3, 8, note="Minimal micro plus texton."),
        Candidate("p11_k8_only", (8,), 7, 8, note="Single K=8 stress test."),
        Candidate("p12_k9_only", (9,), 7, 9, note="Single K=9 stress test."),
        Candidate("p13_89_only", (8, 9), 7, 8, note="Pure texton pair."),
        Candidate("p14_5789", (5, 7, 8, 9), 7, 8, note="No K=3; avoid too-local color matching."),
        Candidate("p15_3489", (3, 4, 8, 9), 4, 8, note="Use K=4 as smoother micro anchor."),
        # C. Texton plus macro meeting.
        Candidate("p16_3578_15", (3, 5, 7, 8, 15), 7, 8, note="T01 plus K=8 before K=15."),
        Candidate("p17_3579_15", (3, 5, 7, 9, 15), 7, 9, note="T01 plus K=9 before K=15."),
        Candidate("p18_3578_16", (3, 5, 7, 8, 16), 7, 8, note="K=16 macro companion for K=8."),
        Candidate("p19_35789_15", (3, 5, 7, 8, 9, 15), 7, 8, note="Full texton band with K=15 anchor."),
        Candidate("p20_5789_15", (5, 7, 8, 9, 15), 7, 8, note="No K=3, retain K=15 macro."),
        Candidate("p21_357_1516", (3, 5, 7, 15, 16), 7, 15, note="Semi-global macro control from large-patch probe."),
        # D. Conservative large-patch probes, always anchored by small/texton bands.
        Candidate("p22_357_21", (3, 5, 7, 21), 7, 21, note="Four-patch macro with small anchors."),
        Candidate("p23_3578_21", (3, 5, 7, 8, 21), 7, 8, note="K=21 with K=8 texton stabilizer."),
        Candidate("p24_35789_21", (3, 5, 7, 8, 9, 21), 7, 8, note="All texton bands plus K=21."),
        Candidate("p25_357_212931", (3, 5, 7, 21, 29, 31), 7, 21, note="Degenerate large-patch control with small anchors."),
        # E. Micro/macro tax on the most plausible texton recipes.
        Candidate("p26_35789_macro125", (3, 5, 7, 8, 9), 7, 8, 0.75, 1.25, note="Tilt the full texton band toward macro K=8/9."),
        Candidate("p27_35789_macro150", (3, 5, 7, 8, 9), 7, 8, 0.60, 1.50, note="Stronger macro tax for style push."),
        Candidate("p28_35789_macro200", (3, 5, 7, 8, 9), 7, 8, 0.50, 2.00, note="Aggressive macro tax, high-risk style push."),
        Candidate("p29_35789_micro125", (3, 5, 7, 8, 9), 7, 8, 1.25, 0.75, note="Opposite tax; content/LPIPS control."),
        Candidate("p30_3578_15_macro150", (3, 5, 7, 8, 15), 7, 8, 0.70, 1.50, note="Macro tax on K=8+15 meeting recipe."),
        Candidate("p31_3579_15_macro150", (3, 5, 7, 9, 15), 7, 9, 0.70, 1.50, note="Macro tax on K=9+15 meeting recipe."),
        Candidate("p32_5789_macro150", (5, 7, 8, 9), 7, 8, 0.70, 1.50, note="No K=3, macro-taxed texton band."),
        Candidate("p33_3589_macro150", (3, 5, 8, 9), 5, 8, 0.70, 1.50, note="Sparse texton band with macro tax."),
        # F. Dilated projections: larger effective field without one-patch collapse.
        Candidate("p34_357_dil2", (3, 5, 7), 7, 15, use_dilation=True, dilation=2, note="Dilated local projections as soft large patch."),
        Candidate("p35_3578_dil2", (3, 5, 7, 8), 7, 8, use_dilation=True, dilation=2, note="Dilated K=8 texton candidate."),
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


def _make_config(
    base_cfg: dict[str, Any],
    candidate: Candidate,
    *,
    save_dir: Path,
    num_epochs: int,
    train_batch_size: int | None,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    bridge = cfg.setdefault("bridge", {})
    bridge["swd_patch_sizes"] = list(candidate.patch_sizes)
    bridge["swd_micro_patch_max"] = int(candidate.micro_patch_max)
    bridge["swd_macro_patch_min"] = int(candidate.macro_patch_min)
    bridge["swd_micro_weight"] = float(candidate.micro_weight)
    bridge["swd_macro_weight"] = float(candidate.macro_weight)
    bridge["swd_scale_invariant_patches"] = False
    bridge["swd_use_dilated_projections"] = bool(candidate.use_dilation)
    bridge["swd_projection_dilation"] = int(candidate.dilation)

    training = cfg.setdefault("training", {})
    training["num_epochs"] = int(num_epochs)
    training["save_interval"] = 1
    if train_batch_size is not None:
        training["batch_size"] = int(train_batch_size)

    checkpoint = cfg.setdefault("checkpoint", {})
    checkpoint["save_dir"] = "./" + save_dir.resolve().relative_to(ROOT).as_posix()

    cfg["ablation"] = {
        "name": candidate.name,
        "stage": "t01_patch36",
        "axis": "swd_patch_design",
        "patch_sizes": list(candidate.patch_sizes),
        "micro_patch_max": candidate.micro_patch_max,
        "macro_patch_min": candidate.macro_patch_min,
        "micro_weight": candidate.micro_weight,
        "macro_weight": candidate.macro_weight,
        "use_dilation": candidate.use_dilation,
        "dilation": candidate.dilation,
        "note": candidate.note,
    }
    return cfg


def _metrics_from_summary(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"clip_style": None, "clip_content": None, "content_lpips": None, "clip_dir": None}
    payload = _load_json(path)
    overview = ((payload.get("analysis") or {}).get("all_pairs_overview") or {})
    return {
        "clip_style": float(overview.get("clip_style")) if overview.get("clip_style") is not None else None,
        "clip_content": float(overview.get("clip_content")) if overview.get("clip_content") is not None else None,
        "content_lpips": float(overview.get("content_lpips")) if overview.get("content_lpips") is not None else None,
        "clip_dir": float(overview.get("clip_dir")) if overview.get("clip_dir") is not None else None,
    }


def _score(row: dict[str, Any]) -> float:
    style = float(row.get("clip_style") or 0.0)
    lpips = float(row.get("content_lpips") or 9.0)
    content = float(row.get("clip_content") or 0.0)
    return style - 0.25 * max(0.0, lpips - 0.49) + 0.05 * max(0.0, content - 0.77)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "score",
        "name",
        "epoch",
        "clip_style",
        "clip_content",
        "content_lpips",
        "clip_dir",
        "patch_sizes",
        "micro_patch_max",
        "macro_patch_min",
        "micro_weight",
        "macro_weight",
        "use_dilation",
        "dilation",
        "run_dir",
        "summary",
    ]
    ranked = sorted(rows, key=_score, reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(ranked, start=1):
            out = {k: row.get(k) for k in fields}
            out["rank"] = idx
            out["score"] = _score(row)
            writer.writerow(out)


def _write_plan(path: Path, candidates: list[Candidate]) -> None:
    rows = [
        {
            "index": idx,
            "name": c.name,
            "patch_sizes": ",".join(str(x) for x in c.patch_sizes),
            "micro_patch_max": c.micro_patch_max,
            "macro_patch_min": c.macro_patch_min,
            "micro_weight": c.micro_weight,
            "macro_weight": c.macro_weight,
            "use_dilation": c.use_dilation,
            "dilation": c.dilation,
            "note": c.note,
        }
        for idx, c in enumerate(candidates)
    ]
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="36-run t01 SWD patch design sweep with best endpoint inference settings.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-epochs", default="4,8")
    parser.add_argument("--max-total", type=int, default=36)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-num-steps", type=int, default=12)
    parser.add_argument("--eval-step-size", type=float, default=1.0)
    parser.add_argument("--eval-vae-decode-scale", type=float, default=0.197)
    parser.add_argument("--eval-residual-scale", type=float, default=1.0)
    parser.add_argument("--eval-force-integrate", action="store_true")
    parser.add_argument("--eval-modern-metrics", action="store_true")
    args = parser.parse_args()

    base_config = _load_json(args.base_config.resolve())
    config_root = args.config_root.resolve()
    output_root = args.output_root.resolve()
    candidates = build_candidates()[: int(args.max_total)]
    eval_epochs = tuple(sorted({int(x) for x in args.eval_epochs.split(",") if x.strip()}))
    ledger_path = output_root / "t01_patch36_ledger.jsonl"
    results_path = output_root / "t01_patch36_results.csv"
    plan_path = output_root / "t01_patch36_plan.csv"
    _write_plan(plan_path, candidates)

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        run_dir = output_root / candidate.name
        config_path = config_root / f"{candidate.name}.json"
        cfg = _make_config(
            base_config,
            candidate,
            save_dir=run_dir,
            num_epochs=args.num_epochs,
            train_batch_size=args.train_batch_size,
        )
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
                "micro_patch_max": candidate.micro_patch_max,
                "macro_patch_min": candidate.macro_patch_min,
                "micro_weight": candidate.micro_weight,
                "macro_weight": candidate.macro_weight,
                "use_dilation": candidate.use_dilation,
                "dilation": candidate.dilation,
                "run_dir": run_dir.as_posix(),
                "summary": summary_path.as_posix(),
                **_metrics_from_summary(summary_path),
            }
            rows.append(row)
            _append_jsonl(ledger_path, row)
            _write_csv(results_path, rows)

    print(json.dumps({"output_root": str(output_root), "plan": str(plan_path), "results": str(results_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
