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
DEFAULT_OUTPUT_ROOT = ROOT / "exp" / "self_topology_gate_sweep"
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "self_topology_gate_sweep"
DEFAULT_EVAL_EPOCHS = (4, 6, 8)


@dataclass(frozen=True)
class SelfTopologyCandidate:
    name: str
    self_topology_blend: float
    warp_strength: float = 0.03
    texture_gate_strength: float = 6.0
    normal_leak: float = 0.0
    color_strength: float = 0.85
    num_epochs: int = 8
    seed: int = 42


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
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def build_candidates() -> list[SelfTopologyCandidate]:
    return [
        SelfTopologyCandidate(name="stg01_t00_self_topology_b1p0", self_topology_blend=1.0),
        SelfTopologyCandidate(name="stg02_t00_self_topology_b0p5", self_topology_blend=0.5),
    ]


def _config_payload(candidate: SelfTopologyCandidate, *, base_config: Path, output_root: Path) -> dict[str, Any]:
    base_ref = Path("..") / ".." / Path(base_config).resolve().relative_to(ROOT)
    save_dir = "./" + Path(output_root / candidate.name).resolve().relative_to(ROOT).as_posix()
    batch_size = int(os.environ.get("LANCET_BATCH_SIZE", "64"))
    eval_batch_size = int(os.environ.get("LANCET_EVAL_BATCH_SIZE", "8"))
    return {
        "_base": base_ref.as_posix(),
        "model": {
            "use_diffeomorphic_stroke": True,
            "diffeomorphic_color_strength": candidate.color_strength,
            "diffeomorphic_warp_strength": candidate.warp_strength,
            "diffeomorphic_texture_gate_strength": candidate.texture_gate_strength,
            "diffeomorphic_normal_leak": candidate.normal_leak,
            "diffeomorphic_color_lowpass_kernel": 1,
            "diffeomorphic_color_edge_gamma": 0.0,
            "semantic_self_topology_gate": True,
            "semantic_self_topology_blend": candidate.self_topology_blend,
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
            "stage": "self_topology_gate_sweep",
            "axis": "content_self_similarity_gated_semantic_cross_attention",
            "notes": "Keep t00 tangent warp fixed; constrain semantic style payload by one content self-similarity diffusion step.",
        },
    }


def _load_summary_metrics(summary_path: Path) -> dict[str, float | None]:
    if not summary_path.exists():
        return {
            "clip_style_all": None,
            "content_lpips_all": None,
            "clip_content_all": None,
            "clip_dir_all": None,
        }
    payload = _load_json(summary_path)
    overview = ((payload.get("analysis") or {}).get("all_pairs_overview") or {})
    return {
        "clip_style_all": float(overview.get("clip_style")) if overview.get("clip_style") is not None else None,
        "content_lpips_all": float(overview.get("content_lpips")) if overview.get("content_lpips") is not None else None,
        "clip_content_all": float(overview.get("clip_content")) if overview.get("clip_content") is not None else None,
        "clip_dir_all": float(overview.get("clip_dir")) if overview.get("clip_dir") is not None else None,
    }


def _train_one(config_path: Path) -> Path:
    _run([sys.executable, "src/run.py", "--config", str(config_path)], cwd=ROOT)
    config = load_config(config_path)
    save_dir = Path(config["checkpoint"]["save_dir"])
    if not save_dir.is_absolute():
        save_dir = ROOT / save_dir
    save_dir = save_dir.resolve()
    ckpt = save_dir / f"epoch_{int(config['training']['num_epochs']):04d}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint after training: {ckpt}")
    return ckpt


def _eval_one(ckpt_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_batch_size = os.environ.get("LANCET_EVAL_BATCH_SIZE", "8")
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
            "--eval_lpips_chunk_size",
            "2",
        ],
        cwd=ROOT,
    )
    summary = out_dir / "summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"missing eval summary: {summary}")
    return summary


def _write_frontier(rows: list[dict[str, Any]], output_path: Path) -> None:
    fields = [
        "name",
        "best_epoch",
        "clip_style_all",
        "content_lpips_all",
        "clip_content_all",
        "clip_dir_all",
        "self_topology_blend",
        "run_dir",
    ]
    ranked = sorted(rows, key=lambda row: float(row.get("clip_style_all") or -9999.0), reverse=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-similarity topology gate sweep for semantic cross-attention.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-epochs", type=str, default="4,6,8")
    parser.add_argument("--max-experiments", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    eval_epochs = tuple(int(part) for part in args.eval_epochs.split(",") if part.strip())
    candidates = build_candidates()[: max(0, args.max_experiments)]
    rows: list[dict[str, Any]] = []
    ledger_path = args.output_root / "self_topology_gate_ledger.jsonl"
    frontier_path = args.output_root / "self_topology_gate_frontier.csv"

    for candidate in candidates:
        config_path = args.config_root / f"{candidate.name}.json"
        run_dir = (args.output_root / candidate.name).resolve()
        ckpt = run_dir / f"epoch_{candidate.num_epochs:04d}.pt"
        print(f"\n=== {candidate.name} ===", flush=True)
        print(
            f"[candidate] blend={candidate.self_topology_blend} "
            f"warp={candidate.warp_strength} gate={candidate.texture_gate_strength}",
            flush=True,
        )

        if not config_path.exists() or args.force_train or args.force_eval:
            _write_json(config_path, _config_payload(candidate, base_config=args.base_config, output_root=args.output_root))

        if args.dry_run:
            print(f"[dry-run] wrote {config_path}", flush=True)
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
                continue
            epoch_ckpt = run_dir / f"epoch_{epoch:04d}.pt"
            if not epoch_ckpt.exists():
                print(f"[warn] missing checkpoint for eval epoch {epoch}: {epoch_ckpt}", flush=True)
                continue
            summary = _eval_one(epoch_ckpt, eval_dir)
            metrics = _load_summary_metrics(summary)
            eval_rows.append({"epoch": epoch, "summary": summary.as_posix(), **metrics})

        best = max(eval_rows, key=lambda row: float(row.get("clip_style_all") or -9999.0), default={})
        row = {
            "name": candidate.name,
            "best_epoch": best.get("epoch"),
            "clip_style_all": best.get("clip_style_all"),
            "content_lpips_all": best.get("content_lpips_all"),
            "clip_content_all": best.get("clip_content_all"),
            "clip_dir_all": best.get("clip_dir_all"),
            "self_topology_blend": candidate.self_topology_blend,
            "run_dir": run_dir.as_posix(),
        }
        rows.append(row)
        _append_jsonl(ledger_path, {"candidate": asdict(candidate), "result": row, "eval_rows": eval_rows})
        _write_frontier(rows, frontier_path)
        print(
            f"[best] epoch={row.get('best_epoch')} style={row.get('clip_style_all')} "
            f"lpips={row.get('content_lpips_all')} content={row.get('clip_content_all')}",
            flush=True,
        )

    _write_frontier(rows, frontier_path)
    print(f"\n[done] {frontier_path}", flush=True)
    print(f"[done] {ledger_path}", flush=True)


if __name__ == "__main__":
    main()
