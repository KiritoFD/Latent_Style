# Style Ablation Plan

This document defines a full ablation protocol focused on style quality
(`clip_style`) and style consistency.

## 1. Objective

Primary target:
- `analysis.style_transfer_ability.clip_style >= 0.65`

Secondary targets:
- `analysis.photo_to_art_performance.clip_style` increases consistently
- classifier accuracy does not collapse
- visual block artifacts remain acceptable

## 2. Core Metrics

Read from `full_eval/epoch_XXXX/summary.json`:
- `analysis.style_transfer_ability.clip_style`
- `analysis.photo_to_art_performance.clip_style`
- `analysis.style_transfer_ability.classifier_acc`
- `matrix_breakdown.*.*.content_lpips`
- `classification_report.accuracy`

## 3. Ablation Axes (Style-Oriented)

### A. Style Injection Path (covered by script)
- `style_ref_gain`
- `style_spatial_pre_gain_16`
- `use_decoder_spatial_inject`
- `style_spatial_dec_gain_32`
- `style_texture_gain`
- `style_texture_mode`
- `use_delta_highpass_bias`
- `style_delta_lowfreq_gain`
- `highpass_last_step_only`
- `highpass_last_step_scale`
- `use_style_delta_gate`
- `use_decoder_adagn`
- `use_style_spatial_highpass`
- `normalize_style_spatial_maps`
- `use_output_style_affine`
- `style_gate_floor`
- `style_strength_step_curve`
- `use_style_spatial_blur`

### B. Style Objectives / Losses (covered by script)
- `w_distill`
- `w_code`
- `w_struct`
- `w_edge`
- `w_cycle`
- `w_stroke_gram`
- `w_color_moment`
- `w_style_spatial_tv`
- `w_nce`
- `w_push`
- `w_delta_tv`
- `w_semigroup`

### C. Style Training Dynamics (covered by script)
- `train_style_strength_min/max`
- `train_num_steps_min/max`
- `train_step_size_min/max`

### D. Bundle-Level Ablations (covered by script)
- style-loss bundle off/up
- content-guard relaxed/strict
- style-injection-heavy combo
- all-style-paths-off combo

## 4. Proposed Experiment Stages

## Stage S0: Baseline Reproduction
- Run exact base config once.
- Verify metric extraction and output folder structure.

## Stage S1: Component Knockout (single-factor)
- No pre spatial injection.
- No decoder spatial injection.
- No texture head gain.
- No highpass bias.
- No stroke gram loss.
- No color moment loss.

Goal: identify style-critical components and weak contributors.

## Stage S2: Strength Sweep (style-up / style-down)
- Style-up: stronger style gains + stronger style losses.
- Style-down: weaker style gains + weaker style losses.
- Style jitter: wider `train_style_strength` range.

Goal: estimate controllability and stability window for `clip_style`.

## Stage S3: Balance Sweep (style vs content)
- Keep best style setting from S2.
- Sweep (`w_struct`, `w_edge`, `w_nce`) down/up.

Goal: maximize style metrics without unacceptable content collapse.

## Stage S4: Long Run Confirmation
- Take top 1-2 candidates.
- Extend epochs (for example 2x baseline).
- Evaluate on fixed intervals + final epoch.

Goal: verify improvements are persistent, not short-run noise.

## 5. Recommended Decision Rules

- Promote candidate if:
  - `clip_style` improves by >= 0.01
  - classifier accuracy drop <= 0.03
  - visual artifacts do not clearly worsen

- Reject candidate if:
  - `clip_style` gains but severe block artifacts appear
  - classification collapses
  - improvement appears only on one transfer direction

## 6. Automation

Use `scripts/style_ablation.py` (or wrapper `scripts/style_ablation.sh`):
- Generates variant configs automatically from the current base config.
- Runs training sequentially.
- Aggregates summary metrics into:
  - `ablation_summary.csv`
  - `ablation_summary.md`

Summary includes both:
- final epoch metrics (`full_eval/epoch_XXXX/summary.json`)
- best historical metrics (`full_eval/summary_history.json`, when available)

This enables repeatable and auditable style-focused iteration.

## 7. Reporting Pipeline

1. Refresh run index:
   - `python scripts/analyze_experiments_cycle.py --root experiments-cycle --out-dir docs/experiments_cycle/data`
2. Generate separate + integrated reports:
   - `python scripts/generate_eval_reports.py --experiments-root experiments-cycle --runs-csv docs/experiments_cycle/data/runs_metrics.csv --history-csv docs/experiments_cycle/data/history_rounds.csv --out-dir docs/reports`
3. Read outputs:
   - separate historical report: `docs/reports/REPORT_EXPERIMENTS.md`
   - separate ablation report: `docs/reports/REPORT_ABLATION50.md`
   - integrated decision report: `docs/reports/REPORT_INTEGRATED.md`
