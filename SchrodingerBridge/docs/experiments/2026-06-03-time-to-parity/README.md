# Distinct5 Time-to-Parity Protocol

Date: 2026-06-03

This directory defines the reviewer-safe protocol for the normalized
time-to-parity artifact required by Gate C.

## Purpose

This artifact replaces vulnerable speedup rhetoric with a same-scope timing
comparison on Distinct5-512.

The goal is not to prove a universal efficiency theorem. The goal is to answer
one bounded question:

> under one explicit measurement protocol, how quickly do LBM, SaMAM, and
> SaMST reach comparable style/content operating regions on Distinct5-512?

## Scope

Primary benchmark:

- `Distinct5-512`

Primary outputs:

- `wall_clock -> clip_style`
- `wall_clock -> content_lpips`
- `wall_clock -> delta_idt`

Headline rule:

- Distinct5 is the speed headline;
- historical strict-750 timings are secondary operating-point records only.

## Artifact contract

Required durable outputs:

- `distinct5_time_to_parity_points.csv`
- `figures/distinct5_time_to_clip_style.pdf`
- `figures/distinct5_time_to_lpips.pdf`
- `figures/distinct5_time_to_delta_idt.pdf`
- one short provenance note in this README describing what was included and
  excluded in each timing column

## Measurement rules

### 1. Same-scope comparison

Allowed:

- same dataset split family: `Distinct5-512`
- same evaluation scope per plotted family:
  - either `full 5x5 / 750`
  - or explicitly labeled alternative scope

Not allowed:

- mixing historical strict-750 timings into the Distinct5 parity figure
- mixing `full` and `transfer-only` timings without explicit labels

### 2. Explicit clock definition

Each point must state whether `wall_seconds` includes:

- train only
- train + checkpoint save
- train + eval

Preferred default for parity plots:

- cumulative training wall time with evaluation excluded, plus a separate
  `eval_scope` / `eval_wall_seconds` field when available

Reason:

- this keeps the convergence curve interpretable while still recording eval
  overhead explicitly.

### 3. Stop-criterion discipline

Every method family must declare one of:

- `operating_point_record`
- `time_to_threshold`
- `full_curve_partial`

Definitions:

- `operating_point_record`:
  - one measured point selected after the fact
  - usable for context, not for headline parity
- `time_to_threshold`:
  - first point reaching a declared threshold
  - preferred for strong timing claims
- `full_curve_partial`:
  - incomplete curve used diagnostically while longer runs are pending

### 4. Abnormal-run policy

If a run completes under abnormal runtime conditions, it must be flagged.

Current example:

- the `SA-SWD random-axis` arm is currently documented as a quality-only run if
  it finishes, because its wall-clock behavior is abnormal on the remote 3060

Policy:

- abnormal runs may contribute quality points
- abnormal runs may not contribute normal-speed timing claims

## CSV schema

The canonical CSV is:

- `distinct5_time_to_parity_points.csv`

Each row should minimally contain:

- `date`
- `method`
- `variant`
- `dataset`
- `scope`
- `checkpoint_or_step`
- `wall_seconds`
- `timing_mode`
- `includes_eval`
- `eval_scope`
- `eval_wall_seconds`
- `clip_style`
- `content_lpips`
- `delta_idt_full`
- `delta_idt_transfer`
- `hardware`
- `status`
- `timing_quality_flag`
- `evidence_path`
- `note`

## Current evidence sources

LBM:

- `docs/experiments/aaai2027_master_experiment_log.csv`
- `docs/experiments/distinct5_512_20260602/`

SaMAM:

- `docs/experiments/2026-06-02-distinct5-512-lancet-representation-summary.zh.md`
- `docs/experiments/comparison_20260602/comparison_report.md`

SaMST:

- Distinct5 points remain incomplete and must be recorded explicitly as partial
  until the curve is available under the same protocol

## Acceptance gate

Gate C may be treated as closed only when:

1. the CSV contains the compared Distinct5 points with explicit clock meaning;
2. the vector figures are generated from that CSV;
3. the main paper timing language is backed by this artifact rather than by
   mixed operating-point anecdotes.
