from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_schema import load_config  # noqa: E402


DEFAULT_BASE_CONFIG = ROOT / "configs" / "diffeomorphic_stroke_tangent_local.json"
DEFAULT_OUTPUT_ROOT = ROOT / "exp" / "diffeomorphic_tangent_head_sweep"
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "diffeomorphic_tangent_head_sweep"
DEFAULT_EVAL_EPOCHS = (4, 6, 8)


def _detect_gpu_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return float(props.total_memory) / (1024.0 ** 3)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT,
            text=True,
        ).strip().splitlines()
        if out:
            return float(out[0].strip()) / 1024.0
    except Exception:
        return None
    return None


def _tiered_batch_sizes() -> tuple[int, int]:
    mem_gb = _detect_gpu_memory_gb()
    if mem_gb is None:
        return 64, 8
    if mem_gb <= 9.0:
        return 64, 8
    return 160, 16


@dataclass(frozen=True)
class TangentCandidate:
    name: str
    warp_mode: str
    warp_strength: float
    texture_gate_strength: float
    normal_leak: float
    color_strength: float = 0.85
    batch_size: int = 64
    full_eval_batch_size: int = 6
    num_epochs: int = 8
    seed: int = 42


def _fmt(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(" ".join(str(part) for part in cmd), flush=True)
    env = os.environ.copy()
    python_path = str(SRC_DIR)
    env["PYTHONPATH"] = python_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _config_payload(candidate: TangentCandidate, *, base_config: Path, output_root: Path) -> dict[str, Any]:
    base_ref = Path("..") / ".." / Path(base_config).resolve().relative_to(ROOT)
    save_dir = "./" + Path(output_root / candidate.name).resolve().relative_to(ROOT).as_posix()
    default_batch_size, default_eval_batch_size = _tiered_batch_sizes()
    batch_size = int(os.environ.get("LANCET_BATCH_SIZE", str(default_batch_size)))
    eval_batch_size = int(os.environ.get("LANCET_EVAL_BATCH_SIZE", str(default_eval_batch_size)))
    return {
        "_base": base_ref.as_posix(),
        "model": {
            "use_diffeomorphic_stroke": True,
            "diffeomorphic_color_strength": candidate.color_strength,
            "diffeomorphic_warp_strength": candidate.warp_strength,
            "diffeomorphic_texture_gate_strength": candidate.texture_gate_strength,
            "diffeomorphic_normal_leak": candidate.normal_leak,
            "diffeomorphic_warp_mode": candidate.warp_mode,
        },
        "training": {
            "seed": candidate.seed,
            "batch_size": batch_size,
            "full_eval_batch_size": eval_batch_size,
            "num_epochs": candidate.num_epochs,
            "num_workers": 0,
            "persistent_workers": False,
            "save_interval": 1,
            "resume_checkpoint": "",
        },
        "checkpoint": {
            "save_dir": save_dir,
        },
        "ablation": {
            "name": candidate.name,
            "stage": "diffeomorphic_tangent_head_sweep",
            "axis": "texture_tangent_head_parameterization",
            "notes": "Compare projected 2D warp with scalar tangent/normal 5-channel parameterization.",
        },
    }


def build_candidates() -> list[TangentCandidate]:
    specs = [
        ("h00_projected_t00", "projected_xy", 0.03, 6.0, 0.0, 0.85),
        ("h01_scalar_t00", "scalar_tangent", 0.03, 6.0, 0.0, 0.85),
        ("h02_scalar_leak", "scalar_tangent", 0.03, 6.0, 0.03, 0.85),
        ("h03_scalar_wide", "scalar_tangent", 0.05, 6.0, 0.0, 0.85),
        ("h04_scalar_strong_color", "scalar_tangent", 0.025, 6.0, 0.0, 0.90),
        ("h05_projected_t01", "projected_xy", 0.03, 6.0, 0.05, 0.85),
        ("h06_scalar_t01_leak", "scalar_tangent", 0.03, 6.0, 0.05, 0.85),
        ("h07_scalar_gate8", "scalar_tangent", 0.03, 8.0, 0.03, 0.85),
    ]
    return [
        TangentCandidate(
            name=name,
            warp_mode=warp_mode,
            warp_strength=warp_strength,
            texture_gate_strength=gate_strength,
            normal_leak=normal_leak,
            color_strength=color_strength,
        )
        for name, warp_mode, warp_strength, gate_strength, normal_leak, color_strength in specs
    ]


def _load_summary_metrics(summary_path: Path) -> dict[str, float | None]:
    if not summary_path.exists():
        return {
            "clip_style_all": None,
            "content_lpips_all": None,
            "clip_content_all": None,
            "cmmd": None,
            "dino_structure": None,
            "gram_micro": None,
            "gram_macro": None,
        }
    payload = _load_json(summary_path)
    overview = ((payload.get("analysis") or {}).get("all_pairs_overview") or {})
    return {
        "clip_style_all": float(overview.get("clip_style")) if overview.get("clip_style") is not None else None,
        "content_lpips_all": float(overview.get("content_lpips")) if overview.get("content_lpips") is not None else None,
        "clip_content_all": float(overview.get("clip_content")) if overview.get("clip_content") is not None else None,
        "cmmd": float(overview.get("cmmd")) if overview.get("cmmd") is not None else None,
        "dino_structure": float(overview.get("dino_structure")) if overview.get("dino_structure") is not None else None,
        "gram_micro": float(overview.get("gram_micro")) if overview.get("gram_micro") is not None else None,
        "gram_macro": float(overview.get("gram_macro")) if overview.get("gram_macro") is not None else None,
    }


def _candidate_row(candidate: TangentCandidate, *, config_path: Path, run_dir: Path, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(
        eval_rows,
        key=lambda row: float(row.get("clip_style_all") or -9999.0),
        default={},
    )
    return {
        "name": candidate.name,
        "config": config_path.as_posix(),
        "run_dir": run_dir.as_posix(),
        "best_epoch": best.get("epoch"),
        "best_summary": best.get("summary"),
        "clip_style_all": best.get("clip_style_all"),
        "content_lpips_all": best.get("content_lpips_all"),
        "clip_content_all": best.get("clip_content_all"),
        "cmmd": best.get("cmmd"),
        "dino_structure": best.get("dino_structure"),
        "gram_micro": best.get("gram_micro"),
        "gram_macro": best.get("gram_macro"),
        "warp_strength": candidate.warp_strength,
        "warp_mode": candidate.warp_mode,
        "texture_gate_strength": candidate.texture_gate_strength,
        "normal_leak": candidate.normal_leak,
        "color_strength": candidate.color_strength,
    }


def _write_frontier(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "best_epoch",
        "clip_style_all",
        "content_lpips_all",
        "clip_content_all",
        "cmmd",
        "dino_structure",
        "gram_micro",
        "gram_macro",
        "warp_strength",
        "warp_mode",
        "texture_gate_strength",
        "normal_leak",
        "color_strength",
        "run_dir",
    ]
    ranked = sorted(rows, key=lambda row: float(row.get("clip_style_all") or -9999.0), reverse=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: row.get(key) for key in fields})


def _train_one(config_path: Path) -> Path:
    _run([sys.executable, "src/run.py", "--config", str(config_path)], cwd=ROOT)
    config = load_config(config_path)
    save_dir = Path(config["checkpoint"]["save_dir"]).resolve()
    ckpt = save_dir / f"epoch_{int(config['training']['num_epochs']):04d}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint after training: {ckpt}")
    return ckpt


def _eval_one(ckpt_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, default_eval_batch_size = _tiered_batch_sizes()
    eval_batch_size = os.environ.get("LANCET_EVAL_BATCH_SIZE", str(default_eval_batch_size))
    _run(
        [
            sys.executable,
            "src/utils/run_evaluation.py",
            "--checkpoint",
            str(ckpt_path),
            "--output",
            str(out_dir),
            "--batch_size",
            str(eval_batch_size),
            "--eval_enable_modern_metrics",
        ],
        cwd=ROOT,
    )
    summary = out_dir / "summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"missing eval summary: {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Small sweep for diffeomorphic tangent head parameterizations.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-epochs", type=str, default="4,6,8")
    parser.add_argument("--max-experiments", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    eval_epochs = tuple(int(part) for part in args.eval_epochs.split(",") if part.strip())
    candidates = build_candidates()[: max(0, args.max_experiments)]
    rows: list[dict[str, Any]] = []
    ledger_path = args.output_root / "tangent_head_ledger.jsonl"
    frontier_path = args.output_root / "tangent_head_frontier.csv"
    mem_gb = _detect_gpu_memory_gb()
    default_batch_size, default_eval_batch_size = _tiered_batch_sizes()

    print(f"[plan] candidates={len(candidates)} eval_epochs={eval_epochs}", flush=True)
    print(f"[base] {args.base_config}", flush=True)
    print(
        f"[gpu-tier] memory_gb={mem_gb if mem_gb is not None else 'unknown'} "
        f"default_train_batch={default_batch_size} default_eval_batch={default_eval_batch_size} "
        f"env_train={os.environ.get('LANCET_BATCH_SIZE') or ''} env_eval={os.environ.get('LANCET_EVAL_BATCH_SIZE') or ''}",
        flush=True,
    )

    for candidate in candidates:
        config_path = args.config_root / f"{candidate.name}.json"
        run_dir = (args.output_root / candidate.name).resolve()
        ckpt = run_dir / f"epoch_{candidate.num_epochs:04d}.pt"
        print(f"\n=== {candidate.name} ===", flush=True)
        print(
            f"[candidate] warp={candidate.warp_strength} gate={candidate.texture_gate_strength} "
            f"leak={candidate.normal_leak} color={candidate.color_strength} mode={candidate.warp_mode}",
            flush=True,
        )

        if not config_path.exists() or args.force_train or args.force_eval:
            _write_json(config_path, _config_payload(candidate, base_config=args.base_config, output_root=args.output_root))

        if args.dry_run:
            print(f"[dry-run] would train {config_path}", flush=True)
            continue

        if not ckpt.exists() or args.force_train:
            ckpt = _train_one(config_path)
        else:
            print(f"[skip] checkpoint exists: {ckpt}", flush=True)

        eval_rows: list[dict[str, Any]] = []
        for epoch in eval_epochs:
            eval_dir = run_dir / "full_eval" / f"epoch_{epoch:04d}"
            summary = eval_dir / "summary.json"
            if summary.exists() and not args.force_eval:
                metrics = _load_summary_metrics(summary)
                eval_rows.append({"epoch": epoch, "summary": summary.as_posix(), **metrics})
                print(f"[skip] eval exists: {summary}", flush=True)
                continue
            epoch_ckpt = run_dir / f"epoch_{epoch:04d}.pt"
            if not epoch_ckpt.exists():
                print(f"[warn] missing checkpoint for eval epoch {epoch}: {epoch_ckpt}", flush=True)
                continue
            summary = _eval_one(epoch_ckpt, eval_dir)
            metrics = _load_summary_metrics(summary)
            eval_rows.append({"epoch": epoch, "summary": summary.as_posix(), **metrics})

        row = _candidate_row(candidate, config_path=config_path, run_dir=run_dir, eval_rows=eval_rows)
        rows.append(row)
        _append_jsonl(ledger_path, {"candidate": asdict(candidate), "result": row, "eval_rows": eval_rows})
        _write_frontier(rows, frontier_path)
        print(
            f"[best] epoch={row.get('best_epoch')} style={row.get('clip_style_all')} "
            f"lpips={row.get('content_lpips_all')} content={row.get('clip_content_all')} "
            f"dino={row.get('dino_structure')}",
            flush=True,
        )

    _write_frontier(rows, frontier_path)
    print(f"\n[done] {frontier_path}", flush=True)
    print(f"[done] {ledger_path}", flush=True)


if __name__ == "__main__":
    main()
