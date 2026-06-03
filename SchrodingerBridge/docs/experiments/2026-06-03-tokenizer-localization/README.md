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
- legacy256 preflight explicitly superseded:
  - `docs/experiments/2026-06-03-tokenizer-localization-remote-preflight.md`
- Distinct5 `L e1` matched configs prepared:
  - `configs/aaai2027/tokenizer_localization_l_e1_seed42_b44_base.json`
  - `configs/aaai2027/tokenizer_localization_l_e1_stylebranch_seed42_b44.json`
  - `configs/aaai2027/tokenizer_localization_l_e1_executoronly_seed42_b44.json`
- remote launch readiness:
  - pending updated preflight against the Distinct5 `L e1` packet and the new
    `executor_only` freeze mode

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

The paper-facing interpretation must use:

- `clip_style`
- `content_lpips`
- `clip_dir`
- `delta_idt` or equivalent no-op-adjusted style gain
- tokenizer/executed geometry diagnostics

## Claim boundary

Safe:

- if one arm wins clearly, it strengthens the localization story for this
  specific Distinct5 `L e1` packet.

Unsafe:

- turning one packet into a family-generic theorem;
- claiming tokenizer theory is closed;
- claiming the correct next tokenizer factorization is proven.
