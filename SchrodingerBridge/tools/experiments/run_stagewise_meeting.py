from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUTPUT_ROOT = ROOT / "exp" / "stagewise_meeting"
CONFIG_ROOT = ROOT / "configs" / "stagewise_meeting"

TRAIN_MIN_PER_EPOCH = 0.87
EVAL_MIN_PER_CHECKPOINT = 1.33
TAIL_EVAL_EPOCHS = [8, 10, 12]


def _rel(path: str) -> Path:
    return (ROOT / path).resolve()


PAIR_TOP8 = {
    "pairing_cache_path": "eval_cache/offline_pairing/dinov2_small_train_pairing_top8.pt",
    "pairing_cache_topk": 8,
}

PAIR_TOP4 = {
    "pairing_cache_path": "eval_cache/offline_pairing/dinov2_small_train_pairing_top4.pt",
    "pairing_cache_topk": 4,
}

LED9 = {
    "coupling_feature_mode": "lowfreq_edge",
    "coupling_lowfreq_kernel": 9,
    "coupling_edge_weight": 0.25,
}

SPEC_BAL = {
    "terminal_swd_mode": "spectral_orthogonal",
    "spectral_swd_low_weight": 1.0,
    "spectral_swd_high_weight": 1.0,
    "spectral_swd_low_kernel": 5,
}

TAX_LIGHT = {
    "w_head_color_tv": 0.01,
    "w_head_color_energy": 0.001,
}

TAX_TINY = {
    "w_head_color_tv": 0.005,
    "w_head_color_energy": 0.0005,
}

BRIDGE_FLAT_003 = {
    "bridge_sigma": 0.03,
    "bridge_noise_mode": "style_highfreq_flat",
    "bridge_style_noise_kernel": 5,
    "bridge_style_noise_flat_gamma": 4.0,
}

BRIDGE_HF_003 = {
    "bridge_sigma": 0.03,
    "bridge_noise_mode": "style_highfreq",
    "bridge_style_noise_kernel": 5,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    source_run: Path
    resume_epoch: int
    total_epochs: int = 12
    eval_epochs: tuple[int, ...] = (8, 10, 12)
    model_updates: dict[str, Any] = field(default_factory=dict)
    bridge_updates: dict[str, Any] = field(default_factory=dict)
    data_updates: dict[str, Any] = field(default_factory=dict)
    training_updates: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


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


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        dst[key] = value


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


def _score(row: dict[str, Any]) -> float:
    style = float(row.get("clip_style_all") or 0.0)
    content = float(row.get("clip_content_all") or 0.0)
    lpips = float(row.get("content_lpips_all") or 1.0)
    return style + 0.35 * content - 0.25 * lpips


def _eval_one(ckpt_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_batch_size = os.environ.get("LANCET_EVAL_BATCH_SIZE", "6")
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


def _estimate_minutes(candidate: Candidate) -> float:
    extra_epochs = candidate.total_epochs - candidate.resume_epoch
    return extra_epochs * TRAIN_MIN_PER_EPOCH + len(candidate.eval_epochs) * EVAL_MIN_PER_CHECKPOINT


def _source_frontier_path() -> Path:
    return ROOT / "exp" / "frontier_decision_tree_8h" / "frontier_decision_tree_frontier.csv"


def _candidate_source_summary(candidate: Candidate, epoch: int) -> Path:
    return candidate.source_run / "full_eval" / f"epoch_{epoch:04d}" / "summary.json"


def _candidate_source_ckpt(candidate: Candidate) -> Path:
    return candidate.source_run / f"epoch_{candidate.resume_epoch:04d}.pt"


def _config_payload(candidate: Candidate) -> dict[str, Any]:
    source_config = _load_json(candidate.source_run / "config.json")
    payload = json.loads(json.dumps(source_config))
    _deep_update(payload["model"], candidate.model_updates)
    _deep_update(payload["bridge"], candidate.bridge_updates)
    _deep_update(payload["data"], candidate.data_updates)
    _deep_update(payload["training"], candidate.training_updates)
    payload["training"]["seed"] = 42
    payload["training"]["batch_size"] = int(os.environ.get("LANCET_BATCH_SIZE", str(payload["training"].get("batch_size", 48))))
    payload["training"]["full_eval_batch_size"] = int(os.environ.get("LANCET_EVAL_BATCH_SIZE", str(payload["training"].get("full_eval_batch_size", 6))))
    payload["training"]["num_epochs"] = int(candidate.total_epochs)
    payload["training"]["num_workers"] = 0
    payload["training"]["persistent_workers"] = False
    payload["training"]["save_interval"] = 1
    payload["training"]["resume_checkpoint"] = str(_candidate_source_ckpt(candidate))
    payload["checkpoint"]["save_dir"] = "./" + (OUTPUT_ROOT / candidate.name).resolve().relative_to(ROOT).as_posix()
    payload["ablation"]["name"] = candidate.name
    payload["ablation"]["stage"] = "stagewise_meeting"
    payload["ablation"]["axis"] = candidate.family
    payload["ablation"]["notes"] = candidate.notes
    return payload


def build_candidates() -> list[Candidate]:
    frontier_root = ROOT / "exp" / "frontier_decision_tree_8h"
    t01_anchor = frontier_root / "00_t01_anchor"
    t01_zero = frontier_root / "01_t01_zero"
    pair8_tax = frontier_root / "19_t01_pair8_tax"
    pair4_bridge_tax = frontier_root / "30_t01_pair4_bridge_tax"
    pair8_bridge_tax = frontier_root / "31_t01_pair8_bridge_tax"
    summit = frontier_root / "35_t01_pair8_led9_spec_tax_tail12"

    items: list[Candidate] = []

    def add(
        slug: str,
        family: str,
        *,
        source_run: Path,
        resume_epoch: int,
        model_updates: dict[str, Any] | None = None,
        bridge_updates: dict[str, Any] | None = None,
        data_updates: dict[str, Any] | None = None,
        training_updates: dict[str, Any] | None = None,
        notes: str = "",
    ) -> None:
        idx = len(items)
        items.append(
            Candidate(
                name=f"{idx:02d}_{slug}",
                family=family,
                source_run=source_run,
                resume_epoch=resume_epoch,
                model_updates=model_updates or {},
                bridge_updates=bridge_updates or {},
                data_updates=data_updates or {},
                training_updates=training_updates or {},
                notes=notes,
            )
        )

    # Structure-first -> style recovery.
    add(
        "p8tax_relax",
        "structure_to_style",
        source_run=pair8_tax,
        resume_epoch=7,
        bridge_updates=TAX_LIGHT,
        notes="Resume pair8_tax at epoch7 and relax head tax to recover style.",
    )
    add(
        "p8tax_relax_c090",
        "structure_to_style",
        source_run=pair8_tax,
        resume_epoch=7,
        model_updates={"diffeomorphic_color_strength": 0.90},
        bridge_updates=TAX_LIGHT,
        notes="pair8_tax plus lighter tax and slightly stronger color residual.",
    )
    add(
        "p8tax_relax_flat003",
        "structure_to_style",
        source_run=pair8_tax,
        resume_epoch=7,
        bridge_updates={**TAX_LIGHT, **BRIDGE_FLAT_003},
        notes="pair8_tax plus lighter tax and flat high-frequency bridge during late training.",
    )
    add(
        "p8tax_relax_spec",
        "structure_to_style",
        source_run=pair8_tax,
        resume_epoch=7,
        bridge_updates={**TAX_LIGHT, **SPEC_BAL},
        notes="pair8_tax plus lighter tax and spectral SWD in late training.",
    )
    add(
        "p8brtax_relax",
        "structure_to_style",
        source_run=pair8_bridge_tax,
        resume_epoch=7,
        bridge_updates={**BRIDGE_FLAT_003, **TAX_LIGHT},
        notes="Resume pair8_bridge_tax with a weaker bridge/tax package.",
    )
    add(
        "p8brtax_relax_c090",
        "structure_to_style",
        source_run=pair8_bridge_tax,
        resume_epoch=7,
        model_updates={"diffeomorphic_color_strength": 0.90},
        bridge_updates={**BRIDGE_FLAT_003, **TAX_LIGHT},
        notes="pair8_bridge_tax plus lighter bridge/tax and slightly stronger color.",
    )
    add(
        "p4brtax_relax_hf003",
        "structure_to_style",
        source_run=pair4_bridge_tax,
        resume_epoch=7,
        bridge_updates={**TAX_LIGHT, **BRIDGE_HF_003},
        notes="pair4_bridge_tax with late high-frequency bridge recovery.",
    )
    add(
        "p8brtax_relax_spec",
        "structure_to_style",
        source_run=pair8_bridge_tax,
        resume_epoch=7,
        bridge_updates={**TAX_LIGHT, **SPEC_BAL},
        notes="pair8_bridge_tax with light tax and spectral SWD late phase.",
    )

    # Style-first -> structure recovery.
    add(
        "t01_late_lighttax",
        "style_to_structure",
        source_run=t01_anchor,
        resume_epoch=7,
        bridge_updates=TAX_TINY,
        notes="Late light tax only on top of t01 anchor.",
    )
    add(
        "t01_late_pair8_lighttax",
        "style_to_structure",
        source_run=t01_anchor,
        resume_epoch=7,
        data_updates=PAIR_TOP8,
        bridge_updates=TAX_TINY,
        notes="Inject top8 pairing only in late training on top of t01.",
    )
    add(
        "t01_late_pair8_flat003_lighttax",
        "style_to_structure",
        source_run=t01_anchor,
        resume_epoch=7,
        data_updates=PAIR_TOP8,
        bridge_updates={**TAX_TINY, **BRIDGE_FLAT_003},
        notes="t01 late pairing plus flat bridge and tiny tax.",
    )
    add(
        "t01_late_pair8_spec_lighttax",
        "style_to_structure",
        source_run=t01_anchor,
        resume_epoch=7,
        data_updates=PAIR_TOP8,
        bridge_updates={**TAX_TINY, **SPEC_BAL},
        notes="t01 late pairing plus spectral SWD and tiny tax.",
    )
    add(
        "t01zero_late_c090",
        "style_to_structure",
        source_run=t01_zero,
        resume_epoch=7,
        model_updates={"diffeomorphic_color_strength": 0.90},
        notes="Continue t01_zero with stronger color only, testing pure style recovery.",
    )
    add(
        "t01zero_late_pair8_tinytax",
        "style_to_structure",
        source_run=t01_zero,
        resume_epoch=7,
        data_updates=PAIR_TOP8,
        bridge_updates=TAX_TINY,
        notes="Continue t01_zero with late pairing and tiny tax.",
    )

    # Summit refinements around the closest existing meeting recipe.
    add(
        "summit_relax_tax",
        "summit_refine",
        source_run=summit,
        resume_epoch=8,
        bridge_updates={**LED9, **SPEC_BAL, **TAX_LIGHT},
        notes="Resume the best mixed summit recipe and relax tax after epoch8.",
    )
    add(
        "summit_relax_tax_c090",
        "summit_refine",
        source_run=summit,
        resume_epoch=8,
        model_updates={"diffeomorphic_color_strength": 0.90},
        bridge_updates={**LED9, **SPEC_BAL, **TAX_LIGHT},
        notes="Summit refinement with lighter tax and slightly stronger color.",
    )
    add(
        "summit_relax_tax_flat003",
        "summit_refine",
        source_run=summit,
        resume_epoch=8,
        bridge_updates={**LED9, **SPEC_BAL, **TAX_LIGHT, **BRIDGE_FLAT_003},
        notes="Summit refinement with late flat bridge for style recovery.",
    )
    add(
        "summit_pair4_switch",
        "summit_refine",
        source_run=summit,
        resume_epoch=8,
        data_updates=PAIR_TOP4,
        bridge_updates={**LED9, **SPEC_BAL, **TAX_LIGHT},
        notes="Same summit recipe but switch to pair-top4 in late training.",
    )

    return items


def _write_plan_csv(candidates: list[Candidate], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "family",
                "source_run",
                "resume_epoch",
                "total_epochs",
                "eval_epochs",
                "estimated_minutes",
                "notes",
            ],
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "name": candidate.name,
                    "family": candidate.family,
                    "source_run": candidate.source_run.as_posix(),
                    "resume_epoch": candidate.resume_epoch,
                    "total_epochs": candidate.total_epochs,
                    "eval_epochs": ",".join(str(x) for x in candidate.eval_epochs),
                    "estimated_minutes": round(_estimate_minutes(candidate), 2),
                    "notes": candidate.notes,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Stagewise meeting experiments: resume from strong frontier checkpoints and continue with late-phase style/structure edits.")
    parser.add_argument("--family", choices=["all", "structure_to_style", "style_to_structure", "summit_refine"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    candidates = build_candidates()
    if args.family != "all":
        candidates = [c for c in candidates if c.family == args.family]

    plan_csv = OUTPUT_ROOT / "stagewise_meeting_plan.csv"
    _write_plan_csv(candidates, plan_csv)
    total_minutes = sum(_estimate_minutes(candidate) for candidate in candidates)
    print(f"[plan] candidates={len(candidates)} estimated_minutes={total_minutes:.1f} estimated_hours={total_minutes/60.0:.2f}", flush=True)
    print(f"[plan] {plan_csv}", flush=True)

    if args.dry_run:
        for candidate in candidates:
            print(
                f"  - {candidate.name} family={candidate.family} "
                f"resume={candidate.source_run.name}:e{candidate.resume_epoch} "
                f"-> e{candidate.total_epochs} est={_estimate_minutes(candidate):.1f}m",
                flush=True,
            )
        return 0

    ledger_path = OUTPUT_ROOT / "stagewise_meeting_ledger.jsonl"
    frontier_path = OUTPUT_ROOT / "stagewise_meeting_frontier.csv"
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        source_ckpt = _candidate_source_ckpt(candidate)
        if not source_ckpt.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {source_ckpt}")

        config_path = CONFIG_ROOT / f"{candidate.name}.json"
        run_dir = OUTPUT_ROOT / candidate.name
        payload = _config_payload(candidate)
        _write_json(config_path, payload)

        status = "ok"
        eval_rows: list[dict[str, Any]] = []
        try:
            final_ckpt = run_dir / f"epoch_{candidate.total_epochs:04d}.pt"
            if args.force_train or not final_ckpt.exists():
                _run([sys.executable, "src/run.py", "--config", str(config_path)], cwd=ROOT)

            for epoch in candidate.eval_epochs:
                epoch_ckpt = run_dir / f"epoch_{epoch:04d}.pt"
                if epoch_ckpt.exists():
                    eval_dir = run_dir / "full_eval" / f"epoch_{epoch:04d}"
                    summary = eval_dir / "summary.json"
                    if args.force_eval or not summary.exists():
                        summary = _eval_one(epoch_ckpt, eval_dir)
                    metrics = _load_summary_metrics(summary)
                    eval_rows.append({"epoch": epoch, "summary": summary.as_posix(), "origin": "stage2", **metrics})
                    continue

                source_summary = _candidate_source_summary(candidate, epoch)
                if source_summary.exists():
                    metrics = _load_summary_metrics(source_summary)
                    eval_rows.append({"epoch": epoch, "summary": source_summary.as_posix(), "origin": "source", **metrics})
        except Exception as exc:
            status = f"failed: {exc}"

        best = max(eval_rows, key=_score, default={})
        row = {
            "name": candidate.name,
            "family": candidate.family,
            "score": _score(best) if best else 0.0,
            "source_run": candidate.source_run.name,
            "resume_epoch": candidate.resume_epoch,
            "best_epoch": best.get("epoch"),
            "best_origin": best.get("origin"),
            "clip_style_all": best.get("clip_style_all"),
            "content_lpips_all": best.get("content_lpips_all"),
            "clip_content_all": best.get("clip_content_all"),
            "clip_dir_all": best.get("clip_dir_all"),
            "status": status,
            "run_dir": run_dir.as_posix(),
        }
        rows.append(row)
        _append_jsonl(
            ledger_path,
            {
                "candidate": {
                    "name": candidate.name,
                    "family": candidate.family,
                    "source_run": candidate.source_run.as_posix(),
                    "resume_epoch": candidate.resume_epoch,
                    "notes": candidate.notes,
                },
                "estimated_minutes": _estimate_minutes(candidate),
                "result": row,
                "eval_rows": eval_rows,
            },
        )

        frontier_path.parent.mkdir(parents=True, exist_ok=True)
        with frontier_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "family",
                    "score",
                    "source_run",
                    "resume_epoch",
                    "best_epoch",
                    "best_origin",
                    "clip_style_all",
                    "content_lpips_all",
                    "clip_content_all",
                    "clip_dir_all",
                    "status",
                    "run_dir",
                ],
            )
            writer.writeheader()
            for item in sorted(rows, key=lambda r: float(r.get("score") or 0.0), reverse=True):
                writer.writerow(item)

        print(f"[result] {candidate.name} status={status} best={best}", flush=True)

    print(f"[done] {frontier_path}", flush=True)
    print(f"[done] {ledger_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
