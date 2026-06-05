# Pairing-Cache Sensitivity Packet

Date: 2026-06-06

Purpose:

- prepare the cheap `C1` reviewer control from the weekly plan
- answer the question:
  - does the prototype-aware pairing cache matter, or would simple random
    cross-style pairing do essentially the same thing?

## Packet design

Config:

- [pairing_cache_h_randompair_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/pairing_cache_h_randompair_seed42_b44.json)

Base surface:

- [mainline_h_seed42_b44_base.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/mainline_h_seed42_b44_base.json)

Control logic:

- keep the paper-facing `H` family train root, eval root, batch size, VAE line,
  and full-eval contract fixed
- remove only the offline prototype-aware pairing cache
- let the dataset fall back to simple random cross-style target sampling

Explicit changes relative to the active `H` surface:

- `pairing_cache_path = ""`
- `pairing_cache_topk = 1`
- `pairing_cache_active_topk = 0`
- `pairing_cache_sample_mode = "uniform_topk"`
- `pairing_cache_rank_schedule = "fixed"`
- `pairing_cache_min_topk = 1`
- `pairing_cache_rank_power = 1.0`
- `pairing_cache_explore_prob = 0.0`
- `pairing_cache_explore_topk = 0`
- `pairing_cache_dual_target_mix = 0.0`
- `pairing_cache_dual_target_topk = 0`
- `pairing_cache_aux_target_topk = 0`

Budget choice:

- `training.num_epochs = 2`

Reason:

- this is a reviewer control, not a new headline candidate family
- the goal is to test whether removing the pairing cache causes a clear drop in
  style movement, LPIPS tradeoff, or visible stability under the same
  Distinct5-512 contract

## Comparison target

Primary anchor:

- current paper-facing `H` family under the same Distinct5-512 latent/eval
  surface

What should remain fixed:

- style domains
- latent/VAE contract
- full-eval `12`-step contract
- batch size `44`
- eval root `/mnt/i/wikiart_distinct5_samam_512_classview/test`

## Expected readout

If the cache matters:

- the no-cache control should lose on at least one of:
  - `delta_idt`
  - transfer `clip_style`
  - `content_lpips`
  - targetwise `ArtFID`

If the gap is small:

- we should write the paper conservatively:
  - the pairing cache is a mild helper for endpoint target selection, not a
    make-or-break theorem claim

## Launch readiness

This packet is prepared for the reviewed remote launcher:

- [launch_remote_aaai2027_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_aaai2027_packet.py)

Dry-run example:

```bash
python SchrodingerBridge/tools/experiments/launch_remote_aaai2027_packet.py \
  --config SchrodingerBridge/configs/aaai2027/pairing_cache_h_randompair_seed42_b44.json \
  --dry-run
```
