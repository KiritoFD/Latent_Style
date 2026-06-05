# Projection-Count Sensitivity Packet

Date: 2026-06-06

Purpose:

- prepare the cheap `C2` reviewer control from the weekly plan
- answer the question:
  - is the SA-SWD terminal loss overly brittle to the chosen number of semantic
    projections?

## Packet design

Config:

- [projection_count_h_sem32_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/projection_count_h_sem32_seed42_b44.json)

Base surface:

- [mainline_h_seed42_b44_base.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/mainline_h_seed42_b44_base.json)

Control logic:

- keep the paper-facing `H` family train root, eval root, batch size, pairing
  cache, VAE line, and full-eval contract fixed
- keep `terminal_swd_axis_source = "semantic"`
- change only the semantic projection budget inside SA-SWD

Explicit change relative to the active `H` surface:

- `semantic_swd_num_projections: 64 -> 32`

Budget choice:

- `training.num_epochs = 2`

Reason:

- this is a sensitivity check, not a new headline family
- we already have a negative closure on `semantic` vs `random` axis source, so
  this packet must not reopen that claim boundary
- the goal is only to test whether a cheaper semantic-axis budget produces a
  materially different style/content tradeoff under the same Distinct5-512
  contract

## Comparison target

Primary anchor:

- current paper-facing `H` family under the same Distinct5-512 latent/eval
  surface

What should remain fixed:

- style domains
- latent/VAE contract
- pairing-cache contract
- full-eval `12`-step contract
- batch size `44`
- eval root `/mnt/i/wikiart_distinct5_samam_512_classview/test`

## Expected readout

If projection count is robust:

- the `32`-projection control should remain in the same broad region on:
  - full `clip_style`
  - transfer `clip_style`
  - `content_lpips`
  - targetwise `ArtFID`

If the gap is large:

- the paper should say the semantic terminal loss is somewhat budget-sensitive
  and keep the implementation details explicit

## Launch readiness

This packet is prepared for the reviewed remote launcher:

- [launch_remote_aaai2027_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_aaai2027_packet.py)

Dry-run example:

```bash
python SchrodingerBridge/tools/experiments/launch_remote_aaai2027_packet.py \
  --config SchrodingerBridge/configs/aaai2027/projection_count_h_sem32_seed42_b44.json \
  --dry-run
```
