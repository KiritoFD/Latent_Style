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


DEFAULT_BASE_CONFIG = ROOT / "exp" / "diffeomorphic_tangent_sweep" / "t00_ws0p03_g6_nl0" / "config.json"
DEFAULT_OUTPUT_ROOT = ROOT / "exp" / "dynamic_metric_probe"
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "dynamic_metric_probe"
DEFAULT_EVAL_EPOCHS = (4,)
DEFAULT_TRAIN_EPOCHS = 4


@dataclass(frozen=True)
class ProbeCandidate:
    name: str
    model: dict[str, Any]
    notes: str


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


def _t00_payload(train_epochs: int) -> dict[str, Any]:
    payload = _load_json(DEFAULT_BASE_CONFIG)
    payload["checkpoint"]["save_dir"] = "./exp/dynamic_metric_probe/__placeholder__"
    payload["ablation"]["stage"] = "dynamic_metric_probe"
    payload["training"]["num_epochs"] = int(train_epochs)
    payload["training"]["save_interval"] = 1
    payload["training"]["resume_checkpoint"] = ""
    payload["training"]["num_workers"] = 0
    payload["training"]["persistent_workers"] = False
    return payload


def build_candidates(train_epochs: int) -> list[ProbeCandidate]:
    base = _t00_payload(train_epochs)["model"]
    return [
        ProbeCandidate(
            name="dm00_t00_baseline",
            model=dict(base),
            notes="Reference t00 tangent warp baseline.",
        ),
        ProbeCandidate(
            name="dm01_zero_init_only",
            model={**dict(base), "zero_init_output_head": True},
            notes="Isolate zero-init stabilization without changing operator form.",
        ),
        ProbeCandidate(
            name="dm02_zero_init_metric_z0",
            model={
                **dict(base),
                "zero_init_output_head": True,
                "diffeomorphic_metric_mask_gamma": 5.0,
                "diffeomorphic_metric_mask_smooth_kernel": 3,
                "diffeomorphic_metric_mask_use_z0": True,
            },
            notes="Add z0-anchored metric mask while keeping static conv head.",
        ),
        ProbeCandidate(
            name="dm03_dynamic_zero_metric",
            model={
                **dict(base),
                "dynamic_style_operator_head": True,
                "dynamic_style_operator_hidden_mult": 1.0,
                "zero_init_output_head": True,
                "diffeomorphic_metric_mask_gamma": 5.0,
                "diffeomorphic_metric_mask_smooth_kernel": 3,
                "diffeomorphic_metric_mask_use_z0": True,
            },
            notes="Full bundle: dynamic operator head + zero init + z0 metric mask.",
        ),
    ]


def _candidate_config(candidate: ProbeCandidate, output_root: Path, train_epochs: int) -> dict[str, Any]:
    payload = _t00_payload(train_epochs)
    payload["model"].update(candidate.model)
    payload["checkpoint"]["save_dir"] = "./" + (output_root / candidate.name).resolve().relative_to(ROOT).as_posix()
    payload["ablation"]["name"] = candidate.name
    payload["ablation"]["notes"] = candidate.notes
    return payload


def _load_summary_metrics(summary_path: Path) -> dict[str, float | None]:
    if not summary_path.exists():
        return {"clip_style_all": None, "content_lpips_all": None, "clip_content_all": None}
    payload = _load_json(summary_path)
    overview = ((payload.get("analysis") or {}).get("all_pairs_overview") or {})
    return {
        "clip_style_all": float(overview.get("clip_style")) if overview.get("clip_style") is not None else None,
        "content_lpips_all": float(overview.get("content_lpips")) if overview.get("content_lpips") is not None else None,
        "clip_content_all": float(overview.get("clip_content")) if overview.get("clip_content") is not None else None,
    }


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
    fields = ["name", "best_epoch", "clip_style_all", "content_lpips_all", "clip_content_all", "status", "run_dir"]
    ranked = sorted(rows, key=lambda row: float(row.get("clip_style_all") or -9999.0), reverse=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe dynamic head + zero-init + z0 metric mask on t00.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--eval-epochs", type=str, default="4")
    parser.add_argument("--train-epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--max-total", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    eval_epochs = tuple(int(x) for x in args.eval_epochs.split(",") if x.strip())
    candidates = build_candidates(args.train_epochs)[: max(1, int(args.max_total))]
    frontier_path = args.output_root / "dynamic_metric_probe_frontier.csv"
    ledger_path = args.output_root / "dynamic_metric_probe_ledger.jsonl"
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        config_path = args.config_root / f"{candidate.name}.json"
        run_dir = args.output_root / candidate.name
        payload = _candidate_config(candidate, args.output_root, args.train_epochs)
        _write_json(config_path, payload)
        if args.dry_run:
            print(f"[dry-run] {candidate.name} -> {config_path}", flush=True)
            continue

        status = "ok"
        eval_rows: list[dict[str, Any]] = []
        try:
            final_ckpt = run_dir / f"epoch_{int(payload['training']['num_epochs']):04d}.pt"
            if args.force_train or not final_ckpt.exists():
                _run([sys.executable, "src/run.py", "--config", str(config_path)], cwd=ROOT)
            for epoch in eval_epochs:
                epoch_ckpt = run_dir / f"epoch_{int(epoch):04d}.pt"
                if not epoch_ckpt.exists():
                    continue
                eval_dir = run_dir / "full_eval" / f"epoch_{int(epoch):04d}"
                summary = eval_dir / "summary.json"
                if args.force_eval or not summary.exists():
                    summary = _eval_one(epoch_ckpt, eval_dir)
                metrics = _load_summary_metrics(summary)
                eval_rows.append({"epoch": epoch, "summary": summary.as_posix(), **metrics})
        except Exception as exc:
            status = f"failed: {exc}"

        best = max(eval_rows, key=lambda row: float(row.get("clip_style_all") or -9999.0), default={})
        row = {
            "name": candidate.name,
            "best_epoch": best.get("epoch"),
            "clip_style_all": best.get("clip_style_all"),
            "content_lpips_all": best.get("content_lpips_all"),
            "clip_content_all": best.get("clip_content_all"),
            "status": status,
            "run_dir": run_dir.as_posix(),
        }
        rows.append(row)
        _append_jsonl(ledger_path, {"candidate": asdict(candidate), "result": row, "eval_rows": eval_rows})
        _write_frontier(rows, frontier_path)
        print(f"[result] {candidate.name} status={status} best={best}", flush=True)

    if args.dry_run:
        print(f"[dry-run] wrote {len(candidates)} configs under {args.config_root}", flush=True)
    else:
        print(f"[done] {frontier_path}", flush=True)
        print(f"[done] {ledger_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
