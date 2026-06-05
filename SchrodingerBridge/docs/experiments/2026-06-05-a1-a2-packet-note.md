# A1 / A2 Packet Note

Date: 2026-06-05

This note records the first concrete packet after the weekly priority reset.

## Purpose

Turn the corrected weekly plan into a launchable remote packet without
reopening low-value lanes.

The packet has two parts:

1. `A1` executor-side promotion on the current paper-facing `H` surface
2. `A2` narrow same-family softening sweep

## Why these are the first two lanes

The current reviewer ledger says:

- executor-side refresh is the strongest landed localization signal on the
  current tokenizer line
- same-family kinetic weakening already failed cleanly, so the next stability
  sweep should not reduce `w_kinetic`
- endpoint-only and semantic-vs-random lanes are already closed in the
  negative direction and should not absorb more budget

Therefore the first new packet should:

- keep the family close to the current paper surface
- avoid new theory debt
- target improvement levers that still have upside

## A1 exact read

Config:

- `configs/aaai2027/executor_promotion_h_e1_seed42_b44.json`

Read:

- load the `H e1` style-side control branch
- freeze that branch
- train only the executor side from fresh parameters

If this packet loses clearly to `H e1` on both `delta_idt` and `LPIPS`, the
executor-refresh idea should not be promoted into a larger branch family.

## A2 exact read

Configs:

- `configs/aaai2027/mainline_h_softterm18_sem010_seed42_b44.json`
- `configs/aaai2027/mainline_h_softterm18_sem012_seed42_b44.json`
- `configs/aaai2027/mainline_h_softterm16_sem012_seed42_b44.json`

Shared hypothesis:

- the current `H` line may still be slightly over-driven at the endpoint
- a mild reduction in terminal pressure plus softer routing could improve the
  content side without reopening the path-instability failure mode seen when
  kinetic is weakened

Important boundary:

- this is **not** a claim that semantic axes are novel or necessary
- this is only a stability-oriented mainline improvement sweep

## Expected keep rule

Promote only if a packet gives one of:

1. better `LPIPS` at near-flat or improved `delta_idt`
2. visibly cleaner outputs with no major collapse on style movement

Otherwise:

- retain the result as a negative or neutral sweep note
- do not keep it alive in the paper-facing queue

## Landing checklist

When `A1` or any `A2` arm lands, collect the same evidence set before writing a
keep/drop conclusion:

1. `remote_train.log`
2. retained checkpoints for all completed epochs
3. `full_eval/<epoch>/summary.json`
4. `full_eval/<epoch>/metrics.csv`
5. `full_eval/<epoch>/aggregate_targetwise_artfid.json`
6. first-health GPU memory read from the watcher or reporter
7. exact queue position and whether the next packet auto-started

Minimum metric fields to copy back into the ledger:

- `clip_style`
- `content_lpips`
- `delta_idt_full`
- `delta_idt_transfer`
- selected `artfid_target_mean` if present in the summary payload

## Result writeback rule

For every landed `A1/A2` packet:

1. update `docs/experiments/aaai2027_master_experiment_log.csv`
2. append the latest queue state to:
   - `docs/experiments/2026-06-06-remote-autonomy-status.md`
3. update:
   - `docs/experiments/2026-06-05-weekly-model-improvement-tracker.md`
4. state one of:
   - `keep`
   - `neutral`
   - `drop`

Do not mark a packet `paper_safe` from launch logs alone. The packet becomes
paper-safe only after the landed eval artifacts above exist and the summary
numbers have been copied into the ledger.
