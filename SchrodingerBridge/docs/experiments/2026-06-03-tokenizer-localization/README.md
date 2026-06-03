# Tokenizer Localization Packet

Date: 2026-06-03

Purpose:

- localize the remaining tokenizer bottleneck after the landed `L`-family
  execution-alignment successor packet;
- separate style-side weakness from executor-side weakness with a matched
  freeze-direction probe on the current Distinct5-512 paper-facing surface.

## Packet status

Current state:

- protocol drafted:
  - `docs/experiments/2026-06-03-tokenizer-localization-probe-protocol.md`
- legacy256 preflight explicitly superseded and replaced by the Distinct5 note:
  - `docs/experiments/2026-06-03-tokenizer-localization-remote-preflight.md`
- Distinct5 `L e1` matched configs prepared:
  - `configs/aaai2027/tokenizer_localization_l_e1_seed42_b44_base.json`
  - `configs/aaai2027/tokenizer_localization_l_e1_stylebranch_seed42_b44.json`
  - `configs/aaai2027/tokenizer_localization_l_e1_executoronly_seed42_b44.json`
- remote launch state:
  - updated Distinct5 `L e1` preflight passed on remote
  - style-branch arm completed training and recovered all three full-eval
    summaries
  - executor-only arm completed training after the queue handoff
  - executor-only auto full-eval crashed on the remote eval surface, then was
    recovered on the same remote machine
  - both arms now have:
    - `epoch_0001`
    - `epoch_0002`
    - `epoch_0003`
    summaries landed under their original output trees
- durable packet readout:
  - `docs/experiments/2026-06-03-tokenizer-localization/readout_20260603.csv`
- launch contract:
  - `docs/experiments/2026-06-03-tokenizer-localization/launch_manifest_20260603.md`
  - live recovery / remote truth note:
    - `docs/experiments/2026-06-03-tokenizer-localization-remote-preflight.md`

## Why this packet exists

The landed `L`-family successor packet narrowed the current tokenizer story:

- raw code geometry is only a partial predictor of executed geometry;
- executed movement tracks no-op-adjusted style gain more closely than raw code
  geometry;
- but that evidence still does not say whether the main remaining weakness is
  on the style-side control branch or on the executor side.

This packet is the next clean mechanism experiment because it directly attacks
 that identification question.

## Matched arms

### Arm A: fresh style branch, frozen executor

- config:
  - `configs/aaai2027/tokenizer_localization_l_e1_stylebranch_seed42_b44.json`
- intent:
  - keep the reviewed `L e1` executor fixed;
  - reinitialize `style_tokenizer.*` and `style_spatial_id_16`;
  - train only the style-side branch.

### Arm B: frozen style branch, fresh executor

- config:
  - `configs/aaai2027/tokenizer_localization_l_e1_executoronly_seed42_b44.json`
- intent:
  - load only `style_tokenizer.*` and `style_spatial_id_16` from reviewed
    `L e1`;
  - leave the executor random;
  - train only the executor side through the new `executor_only` freeze mode.

## Readout contract

Both arms must land:

- `remote_train.log`
- `full_eval/epoch_0001/summary.json`
- `full_eval/epoch_0002/summary.json`
- `full_eval/epoch_0003/summary.json`

Current landed status:

- style-branch:
  - `epoch_0001/summary.json` landed
  - `epoch_0002/summary.json` landed
  - `epoch_0003/summary.json` landed
- executor-only:
  - `epoch_0001/summary.json` landed
  - `epoch_0002/summary.json` landed
  - `epoch_0003/summary.json` landed

The paper-facing interpretation must use:

- `clip_style`
- `content_lpips`
- `clip_dir`
- `delta_idt` or equivalent no-op-adjusted style gain
- tokenizer/executed geometry diagnostics

## Exact baseline used for `delta_idt`

The packet readout uses the same Distinct5 unchanged-image reference as the
existing metric-stress notes:

- baseline summary:
  - `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\no_op_identity_5x5_summary.json`
- baseline values:
  - `all_pairs_overview.clip_style = 0.6801226128737131`
  - `style_transfer_ability.clip_style = 0.6399208252628644`

`delta_idt` is read as:

- `delta_idt_full = all_pairs_overview.clip_style - 0.6801226128737131`
- `delta_idt_transfer = style_transfer_ability.clip_style - 0.6399208252628644`

## Full per-epoch readout

The durable machine-readable table is:

- `docs/experiments/2026-06-03-tokenizer-localization/readout_20260603.csv`

Compact human-readable view:

| arm | epoch | full `clip_style` | full `LPIPS` | `delta_idt_full` | transfer `clip_style` | transfer `LPIPS` | `delta_idt_transfer` | identity `clip_style` | identity `LPIPS` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| style-branch | `e1` | `0.690304` | `0.319396` | `+0.010181` | `0.657570` | `0.319615` | `+0.017650` | `0.821238` | `0.318519` |
| style-branch | `e2` | `0.690575` | `0.319515` | `+0.010453` | `0.657863` | `0.319992` | `+0.017942` | `0.821426` | `0.317607` |
| style-branch | `e3` | `0.690619` | `0.319595` | `+0.010497` | `0.657971` | `0.320181` | `+0.018050` | `0.821213` | `0.317251` |
| executor-only | `e1` | `0.696258` | `0.321755` | `+0.016135` | `0.664571` | `0.329355` | `+0.024650` | `0.823006` | `0.291354` |
| executor-only | `e2` | `0.691628` | `0.328816` | `+0.011505` | `0.660810` | `0.336322` | `+0.020889` | `0.814898` | `0.298792` |
| executor-only | `e3` | `0.695719` | `0.335366` | `+0.015596` | `0.663743` | `0.343879` | `+0.023822` | `0.823620` | `0.301315` |

## Outcome on the current `L e1` surface

The matched packet now supports one narrow factual conclusion:

- among the two one-sided refresh arms, `executor-only` is stronger than
  `style-branch` on no-op-adjusted style movement under both scopes;
- this is not a marginal one-epoch fluke:
  - best `style-branch delta_idt_full = +0.010497` at `e3`
  - best `executor-only delta_idt_full = +0.016135` at `e1`
  - best `style-branch delta_idt_transfer = +0.018050` at `e3`
  - best `executor-only delta_idt_transfer = +0.024650` at `e1`
- the corresponding LPIPS trade-off is modest in the full view
  (`0.321755` vs `0.319595`) and clearer in the transfer-only view
  (`0.329355` vs `0.320181`);
- identity-block LPIPS is also lower for executor-only, although raw identity
  `clip_style` stays below the Distinct5 unchanged-image identity reference for
  both arms.

Safe reading:

- on the current matched Distinct5 `L e1` localization packet, the stronger
  recoverable direction is executor-side refresh rather than style-side refresh
  alone.

Unsafe reading:

- tokenizer design is solved;
- tokenizer geometry no longer matters;
- this closes the broader tokenizer theory;
- this restores blocked `H`-family continuity.

## Claim boundary

Safe:

- if one arm wins clearly, it strengthens the localization story for this
  specific Distinct5 `L e1` packet.

Unsafe:

- turning one packet into a family-generic theorem;
- claiming tokenizer theory is closed;
- claiming the correct next tokenizer factorization is proven.
