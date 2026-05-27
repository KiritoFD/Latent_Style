from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

from run_clip_style_decision_tree import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_BASELINE_SUMMARY,
    Candidate,
    _append_jsonl,
    _baseline_metrics,
    _load_existing_best,
    _load_existing_rows,
    _parse_epochs,
    _write_tables,
    train_and_eval,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "exp" / "physical_flow_matrix"
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "physical_flow_matrix"


def physical_candidates() -> list[Candidate]:
    base = dict(
        stage="physical_flow_matrix",
        hypothesis="physical_pde_constraints_rebuild_style_transport",
        w_kinetic=1.0,
        terminal_swd_weight=20.0,
        w_variance_penalty=0.0,
        residual_gain=1.0,
        semantic_attn_temperature=0.04,
        semantic_swd_num_projections=32,
        swd_num_projections=32,
        swd_scale_invariant_patches=True,
        swd_adaptive_highpass=True,
    )
    target_axis = [
        ("x0_rgb", "Vanilla target measure.", {}),
        ("x1_ret05", "Half Retinex pushforward: keep content illumination while partly matching reflectance statistics.", {"retinex_target_blend": 0.5}),
        ("x2_ret10", "Full Retinex pushforward: freeze content illumination and move reflectance statistics.", {"retinex_target_blend": 1.0}),
        (
            "x3_retfft",
            "Retinex target plus low-dose FFT amplitude pressure for phase-free texture energy.",
            {"retinex_target_blend": 1.0, "w_spectral_amplitude": 0.05},
        ),
    ]
    pde_axis = [
        ("y0_vanilla", "No extra PDE constraint.", {}),
        (
            "y1_aniso",
            "Anisotropic structure-tensor kinetic field: punish cross-edge flow, allow tangential brush motion.",
            {"w_anisotropic_kinetic": 0.08, "anisotropic_normal_weight": 25.0, "anisotropic_tangent_weight": 0.25},
        ),
        (
            "y2_stokes",
            "Stokes viscous flow: penalize velocity Laplacian to force coherent impasto-like color-block motion.",
            {"w_stokes_viscous": 0.20},
        ),
        (
            "y3_phase",
            "Cahn-Hilliard phase separation: discourage grey intermediate states while keeping a small interface cost.",
            {"w_phase_separation": 0.02, "phase_gradient_weight": 0.03},
        ),
    ]
    candidates: list[Candidate] = []
    for ix, (x_name, x_reason, x_cfg) in enumerate(target_axis):
        for iy, (y_name, y_reason, y_cfg) in enumerate(pde_axis):
            payload = {**base, **x_cfg, **y_cfg}
            candidates.append(
                Candidate(
                    name=f"p{ix}{iy}_{x_name}_{y_name}",
                    reason=f"{x_reason} {y_reason}",
                    **payload,
                )
            )
    return candidates


def _style_value(row: dict[str, Any]) -> float:
    try:
        return float(row.get("clip_style_all") or -9999.0)
    except (TypeError, ValueError):
        return -9999.0


def _write_physical_frontier(rows: list[dict[str, Any]], *, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "epoch",
        "clip_style_all",
        "content_lpips_all",
        "clip_content_all",
        "retinex_target_blend",
        "w_spectral_amplitude",
        "w_anisotropic_kinetic",
        "w_stokes_viscous",
        "w_phase_separation",
        "anisotropic_kinetic_train",
        "stokes_viscous_train",
        "phase_separation_train",
        "decision",
        "score",
        "summary",
    ]
    with (output_root / "physical_frontier.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=_style_value, reverse=True):
            writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="16-run Retinex/PDE physical flow matrix.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--eval-epochs", type=str, default="4,6,8")
    parser.add_argument("--max-experiments", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()
    if args.eval_root is None:
        args.eval_root = args.output_root / "full_eval"

    baseline = _baseline_metrics(args.baseline_summary)
    eval_epochs = _parse_epochs(args.eval_epochs)
    candidates = physical_candidates()[: max(0, args.max_experiments)]
    all_rows = _load_existing_rows(args.output_root)
    best_rows = _load_existing_best(args.output_root)
    tried = {str(row.get("name")) for row in best_rows}
    ledger_path = args.output_root / "physical_flow_ledger.jsonl"

    print(
        f"[baseline] style={baseline['clip_style_all']:.6f} "
        f"lpips={baseline['content_lpips_all']:.6f} content={baseline['clip_content_all']:.6f}",
        flush=True,
    )
    print(f"[plan] candidates={len(candidates)} eval_epochs={eval_epochs}", flush=True)

    for candidate in candidates:
        if candidate.name in tried:
            print(f"[skip] already completed: {candidate.name}", flush=True)
            continue
        print(f"\n=== Physical flow {len(tried) + 1}/{len(candidates)}: {candidate.name} ===", flush=True)
        print(f"[reason] {candidate.reason}", flush=True)
        rows, best = train_and_eval(
            candidate,
            epochs=eval_epochs,
            config_root=args.config_root,
            output_root=args.output_root,
            eval_root=args.eval_root,
            base_config=args.base_config,
            baseline=baseline,
            dry_run=args.dry_run,
            force_train=args.force_train,
            force_eval=args.force_eval,
        )
        if args.dry_run:
            continue
        tried.add(candidate.name)
        all_rows.extend(rows)
        best_rows.append(best)
        best_rows.sort(key=lambda row: float(row.get("score") or -9999.0), reverse=True)
        _write_tables(all_rows, best_rows, output_root=args.output_root)
        _write_physical_frontier(all_rows, output_root=args.output_root)
        _append_jsonl(
            ledger_path,
            {
                "candidate": asdict(candidate),
                "best_by_score": best,
                "best_by_style": max(rows, key=_style_value) if rows else None,
                "global_style_best": max(all_rows, key=_style_value) if all_rows else None,
            },
        )
        style_best = max(rows, key=_style_value) if rows else {}
        print(
            f"[best by style] e{style_best.get('epoch')} style={style_best.get('clip_style_all')} "
            f"lpips={style_best.get('content_lpips_all')} content={style_best.get('clip_content_all')}",
            flush=True,
        )

    if args.dry_run:
        print("\n[dry-run done] configs generated only", flush=True)
        return
    _write_tables(all_rows, best_rows, output_root=args.output_root)
    _write_physical_frontier(all_rows, output_root=args.output_root)
    print(f"\n[done] {args.output_root / 'decision_tree_results.csv'}", flush=True)
    print(f"[done] {args.output_root / 'decision_tree_best.csv'}", flush=True)
    print(f"[done] {args.output_root / 'physical_frontier.csv'}", flush=True)
    print(f"[done] {ledger_path}", flush=True)


if __name__ == "__main__":
    main()
