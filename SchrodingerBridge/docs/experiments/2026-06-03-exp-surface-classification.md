# `exp/` Surface Classification

Date: 2026-06-03

Purpose:

- classify the current `SchrodingerBridge/exp/` surface without breaking
  paper-facing evidence paths;
- separate active AAAI 2027 evidence from local smoke/probe clutter;
- define which directories may still be cited, which are frozen only for
  provenance, and which should stop growing.

## 1. Constraint

Many current notes, comparison tables, and ledger rows still point directly
into `exp/.../summary.json`, `remote_train.log`, or `full_eval/...`.

Therefore this pass does **not** do a mass physical move of existing
directories. The immediate safe action is:

1. classify;
2. freeze low-value surfaces;
3. require new paper-facing work to use the tighter naming/logging contract.

If an old directory is still cited by:

- `docs/experiments/...`,
- `docs/reviews/...`,
- `aaai2027_master_experiment_log.csv`,

it stays in place until the citation graph is migrated.

## 2. Current top-level classes

### A. Active paper-facing evidence

Keep discoverable and allowed in current writing:

- `exp/distinct5_512_ema_variant_*_remote`
  - Distinct5 representation variants already discussed in current paper notes
  - status:
    - frozen evidence bundles
- `exp/timing_20260601`
- `exp/timing_20260602`
  - timing and eval-throughput evidence
  - status:
    - timing-only evidence
- `exp/ablation_destructive_7epoch`
- `exp/kinetic_sweep`
- `exp/weight_sweep_40`
- `exp/orth12`
- `exp/legacy`
- `exp/review_additional_experiments`
  - historical evidence families already referenced by comparison notes or
    historical tables
  - status:
    - historical evidence; no further growth unless explicitly reopened

### B. Current AAAI 2027 claim-closing packets

These should remain the preferred pattern for new formal runs:

- `exp/aaai2027_*`
  - currently used by:
    - repaired endpoint-metric packet
    - SA-SWD semantic-vs-random packet
  - status:
    - formal remote evidence surface
  - rule:
    - every such run must have a paired note under `docs/experiments/`
    - every such run must have a row in
      `docs/experiments/aaai2027_master_experiment_log.csv`

### C. Frozen local exploratory probes

Do not cite directly in the paper unless promoted through a dated experiment
note and the master ledger:

- `exp/local_wsl_wikiart512_*`
- `exp/probes_20260601`
- `exp/style_representation_*`
- `exp/tokenizer_*`
- `exp/fisher_*`
- `exp/memory_*`
- `exp/style_memory_*`
- `exp/router_*`
- `exp/typed_*`
- `exp/reference_memory_generation_probe_full`
- `exp/remote_factorized_tokenizer_pull`
- `exp/seedream_distill_adapter`

Status:

- frozen exploratory evidence only

Interpretation:

- useful for internal theory review;
- not safe as first-class paper evidence without promotion into a dated packet.

### D. Local smoke / calibration clutter

These should stop growing and should never be cited directly:

- `exp/_smoke_*`
- `exp/scitexture_512_smoke_local`
- `exp/code_reset_smoke`

Status:

- retire from active surface

Retention rule:

- keep only while a later comparison note still depends on their summaries or
  timing outputs;
- otherwise they are the first deletion candidates.

### E. Utility/runtime support

- `exp/runs`
- `exp/analysis`
- `exp/inference`
- `exp/scripts`
- `exp/paper`
- `exp/video`
- `exp/wikiart_512_encode_logs`
- `exp/wikiart_512_transfer_logs`

Status:

- runtime/support surfaces

Rule:

- not paper evidence by default;
- promote outputs out of these directories before citing them.

## 3. Cleanup policy from now on

### New formal experiment rule

A new formal run is allowed to create a new top-level directory under `exp/`
only if all of the following are true:

1. it is remote or otherwise paper-facing;
2. it follows the `aaai2027_*` naming family or another explicitly documented
   family;
3. it has a paired launch/readme packet under `docs/experiments/`;
4. it receives a ledger row in `aaai2027_master_experiment_log.csv`.

### Low-value run rule

For local smoke or calibration runs:

1. do not create a new one-off top-level family if a matching retired surface
   already exists;
2. prefer reusing:
   - `_smoke_*`
   - `timing_*`
   - `probes_*`
3. if the run fails or teaches nothing new, record the conclusion in a note and
   stop citing the directory itself.

## 4. Deletion boundary

This pass makes no destructive deletions yet because too many current notes
still point into existing `exp/` paths.

Safe first deletion candidates, after path-audit confirmation, are:

1. generated `images/` under `_smoke_*` directories;
2. duplicate eval-image dumps under timing-only directories once the retained
   `summary.json`, `metrics.csv`, and wall-time note are confirmed;
3. local-only grids or PNG dumps that have no ledger row and no citation hit in
   `docs/`.

Unsafe to delete right now:

- anything under `exp/aaai2027_*`
- anything directly cited by the current working index, comparison report, or
  master experiment log
- historical `summary.json` paths still used by paper tables

## 5. Immediate repo rule for the next cycle

Until the current paper gate closes:

- keep the active evidence graph centered on:
  - `docs/experiments/`
  - `docs/reviews/`
  - `docs/aaai2027_working_index_20260602.md`
  - `docs/experiments/aaai2027_master_experiment_log.csv`
- treat `exp/` as the runtime/evidence backing store, not as the narrative
  surface.

That keeps the repo cleaner without invalidating the current citation graph.
