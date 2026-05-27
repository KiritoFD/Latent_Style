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
DEFAULT_OUTPUT_ROOT = ROOT / "exp" / "feature_hijack_ablation"
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "feature_hijack_ablation"


def hijack_candidates() -> list[Candidate]:
    """Small, mechanism-first sweep after the dilated FFT mutation underperformed."""

    base = dict(
        stage="feature_hijack_ablation",
        semantic_attn_temperature=0.04,
        semantic_swd_num_projections=32,
        swd_num_projections=32,
        swd_scale_invariant_patches=True,
        swd_adaptive_highpass=True,
        w_spectral_amplitude=0.05,
        terminal_swd_weight=20.0,
        w_kinetic=1.0,
        w_variance_penalty=0.0,
        residual_gain=1.0,
    )
    return [
        Candidate(
            name="h00_fft_guard_baseline",
            hypothesis="low_dose_fft_is_current_frontier",
            reason="Control run: low-dose spectral amplitude with scale-invariant SWD, no hijack mechanism.",
            **base,
        ),
        Candidate(
            name="h01_pre_moment_blend025",
            hypothesis="diagonal_bures_prefit_reduces_global_style_gap",
            reason="Gentle pre-integration moment map; should raise style if reference latents are available without forcing full AdaIN.",
            pre_integrate_moment_match=True,
            pre_integrate_moment_blend=0.25,
            **base,
        ),
        Candidate(
            name="h02_pre_moment_blend050",
            hypothesis="stronger_diagonal_bures_prefit_can_lift_clip_style",
            reason="Medium pre-integration moment map; checks whether global mean/std matching is useful or too destructive.",
            pre_integrate_moment_match=True,
            pre_integrate_moment_blend=0.50,
            **base,
        ),
        Candidate(
            name="h03_nonlocal_cost010",
            hypothesis="self_similarity_cost_guides_sinkhorn_without_attention",
            reason="Weak non-local structure term in the OT oracle; tests S2WAT-like self-similarity with low overhead.",
            w_nonlocal_structure=0.10,
            nonlocal_structure_pool=8,
            **base,
        ),
        Candidate(
            name="h04_nonlocal_cost030",
            hypothesis="stronger_self_similarity_cost_preserves_lpips_under_style_pressure",
            reason="Stronger non-local structure term; useful only if LPIPS improves without suppressing CLIP-S.",
            w_nonlocal_structure=0.30,
            nonlocal_structure_pool=8,
            **base,
        ),
        Candidate(
            name="h05_nonlocal_fft007",
            hypothesis="nonlocal_guard_allows_higher_fft_style_pull",
            reason="Non-local guard plus slightly stronger frequency amplitude pressure.",
            w_nonlocal_structure=0.10,
            nonlocal_structure_pool=8,
            w_spectral_amplitude=0.07,
            **{k: v for k, v in base.items() if k != "w_spectral_amplitude"},
        ),
        Candidate(
            name="h06_sb_noise002_flow",
            stage="feature_hijack_ablation",
            hypothesis="schrodinger_entropy_smooths_low_res_flow_matching",
            reason="Switch to path-flow objective and inject mild Brownian bridge noise.",
            objective_mode="flow",
            sb_noise_epsilon=0.02,
            terminal_swd_weight=20.0,
            w_kinetic=1.0,
            w_variance_penalty=0.0,
            residual_gain=1.0,
            semantic_attn_temperature=0.04,
            semantic_swd_num_projections=32,
            swd_num_projections=32,
            swd_scale_invariant_patches=True,
            swd_adaptive_highpass=True,
            w_spectral_amplitude=0.05,
        ),
        Candidate(
            name="h07_sb_noise005_flow",
            stage="feature_hijack_ablation",
            hypothesis="stronger_entropy_may_reduce_mapping_brittleness",
            reason="Higher Brownian bridge noise; likely slower/weaker, but diagnoses whether entropy helps content.",
            objective_mode="flow",
            sb_noise_epsilon=0.05,
            terminal_swd_weight=20.0,
            w_kinetic=1.0,
            w_variance_penalty=0.0,
            residual_gain=1.0,
            semantic_attn_temperature=0.04,
            semantic_swd_num_projections=32,
            swd_num_projections=32,
            swd_scale_invariant_patches=True,
            swd_adaptive_highpass=True,
            w_spectral_amplitude=0.05,
        ),
        Candidate(
            name="h08_premember_nonlocal010",
            hypothesis="global_moment_plus_self_similarity_is_complementary",
            reason="Combine gentle moment prefit and weak non-local structure cost.",
            pre_integrate_moment_match=True,
            pre_integrate_moment_blend=0.25,
            w_nonlocal_structure=0.10,
            nonlocal_structure_pool=8,
            **base,
        ),
        Candidate(
            name="h09_nonlocal_sb002_flow",
            stage="feature_hijack_ablation",
            hypothesis="self_similarity_plus_entropy_can_protect_content",
            reason="Path-flow objective with weak entropy and non-local cost.",
            objective_mode="flow",
            sb_noise_epsilon=0.02,
            w_nonlocal_structure=0.10,
            nonlocal_structure_pool=8,
            terminal_swd_weight=20.0,
            w_kinetic=1.0,
            w_variance_penalty=0.0,
            residual_gain=1.0,
            semantic_attn_temperature=0.04,
            semantic_swd_num_projections=32,
            swd_num_projections=32,
            swd_scale_invariant_patches=True,
            swd_adaptive_highpass=True,
            w_spectral_amplitude=0.05,
        ),
        Candidate(
            name="h10_guarded_style_floor",
            hypothesis="energy_floor_raises_style_without_full_variance_match",
            reason="A pragmatic guard rail: content/edge anchors plus one-sided style energy floor.",
            w_content_anchor=2.0,
            w_edge_anchor=0.5,
            w_style_energy_floor=0.25,
            style_energy_floor_ratio=0.55,
            **base,
        ),
        Candidate(
            name="h11_full_reasonable_combo",
            hypothesis="low_dose_fft_nonlocal_prefit_and_energy_floor_are_complementary",
            reason="The best reasonable combination before trying heavier critics or high-order geometry again.",
            pre_integrate_moment_match=True,
            pre_integrate_moment_blend=0.25,
            w_nonlocal_structure=0.10,
            nonlocal_structure_pool=8,
            w_content_anchor=2.0,
            w_edge_anchor=0.5,
            w_style_energy_floor=0.25,
            style_energy_floor_ratio=0.55,
            **base,
        ),
    ]


def _style_value(row: dict[str, Any]) -> float:
    try:
        return float(row.get("clip_style_all") or -9999.0)
    except (TypeError, ValueError):
        return -9999.0


def _write_mechanism_frontier(rows: list[dict[str, Any]], *, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "epoch",
        "objective_mode",
        "clip_style_all",
        "content_lpips_all",
        "clip_content_all",
        "w_spectral_amplitude",
        "pre_integrate_moment_match",
        "pre_integrate_moment_blend",
        "w_nonlocal_structure",
        "nonlocal_structure_pool",
        "sb_noise_epsilon",
        "w_content_anchor",
        "w_edge_anchor",
        "w_style_energy_floor",
        "decision",
        "score",
        "summary",
    ]
    ranked = sorted(rows, key=_style_value, reverse=True)
    with (output_root / "mechanism_frontier.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="8-16 run feature-hijack ablation sweep.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--eval-epochs", type=str, default="4,6,8")
    parser.add_argument("--max-experiments", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()
    if args.eval_root is None:
        args.eval_root = args.output_root / "full_eval"

    baseline = _baseline_metrics(args.baseline_summary)
    eval_epochs = _parse_epochs(args.eval_epochs)
    candidates = hijack_candidates()[: max(0, args.max_experiments)]
    all_rows = _load_existing_rows(args.output_root)
    best_rows = _load_existing_best(args.output_root)
    tried = {str(row.get("name")) for row in best_rows}
    ledger_path = args.output_root / "feature_hijack_ledger.jsonl"

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
        print(f"\n=== Feature hijack {len(tried) + 1}/{len(candidates)}: {candidate.name} ===", flush=True)
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
        _write_mechanism_frontier(all_rows, output_root=args.output_root)
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
    _write_mechanism_frontier(all_rows, output_root=args.output_root)
    print(f"\n[done] {args.output_root / 'decision_tree_results.csv'}", flush=True)
    print(f"[done] {args.output_root / 'decision_tree_best.csv'}", flush=True)
    print(f"[done] {args.output_root / 'mechanism_frontier.csv'}", flush=True)
    print(f"[done] {ledger_path}", flush=True)


if __name__ == "__main__":
    main()
