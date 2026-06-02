# Smoke Surface Prune

Date: 2026-06-03

Purpose:

- remove low-value local smoke clutter that is no longer part of the active
  AAAI 2027 evidence graph;
- preserve current paper-facing summary and checkpoint paths;
- reduce confusion between durable experiment evidence and one-off local trial
  surfaces.

## Rule used

This prune followed:

- `docs/experiments/2026-06-03-exp-surface-classification.md`

Operational rule:

1. delete whole directories only when they have no current citation hits beyond
   the classification note itself;
2. for still-cited smoke surfaces, delete only generated `images/` payloads and
   keep:
   - `summary.json`
   - checkpoints
   - logs
   - config snapshots

## Deleted whole-directory smoke surfaces

### `exp/code_reset_smoke`

- status before deletion:
  - local smoke-only directory
  - no current doc, paper, or tool dependency beyond the generic clutter
    classification
- payload:
  - `29` files
  - about `2.06 MB`

### `exp/scitexture_512_smoke_local`

- status before deletion:
  - local dataset smoke surface
  - no current doc, paper, or tool dependency beyond the generic clutter
    classification
- payload:
  - `17` files
  - about `44.33 MB`

## Deleted payload-only smoke artifacts

Deleted the nested `images/` directories from:

- `exp/_smoke_distinct5_512_ema_baseline_vlen004/full_eval/epoch_0001_smoke25`
- `exp/_smoke_distinct5_512_ema_baseline_vlen004/full_eval/epoch_0001_smoke25_timing`

Why payload-only deletion was required here:

- these smoke outputs still have active summary-path references in:
  - `docs/experiments/2026-06-02-distinct5-512-lancet-representation-speed.md`
  - `docs/experiments/comparison_20260602/lancet_history_registry.csv`
  - `docs/experiments/comparison_20260602/lancet_history_registry.json`
- the `summary.json` files remain part of the current provenance graph;
- the generated image payloads are not required for the current paper surface.

## Runtime scratch cleanup

Deleted Python cache directories under:

- `SchrodingerBridge/__pycache__/`
- `SchrodingerBridge/src/__pycache__/`
- `SchrodingerBridge/src/utils/__pycache__/`
- `SchrodingerBridge/tools/__pycache__/`
- `SchrodingerBridge/tools/experiments/__pycache__/`

These were pure runtime byproducts and not part of research provenance.

## Net effect

Guaranteed reclaimed local disk from the fully removed smoke directories:

- at least `46.39 MB`

In addition, this pass removed:

- two smoke-eval image payload directories, and
- multiple Python cache trees

without touching any currently cited summary path.

## Verification

Post-prune checks confirmed:

- `exp/code_reset_smoke`:
  - deleted
- `exp/scitexture_512_smoke_local`:
  - deleted
- cited smoke summaries still present:
  - `exp/_smoke_distinct5_512_ema_baseline_vlen004/full_eval/epoch_0001_smoke25/summary.json`
  - `exp/_smoke_distinct5_512_ema_baseline_vlen004/full_eval/epoch_0001_smoke25_timing/summary.json`
