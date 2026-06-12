# sde_i2sb_sigma_0p25 Closure

- Status: closed as `stopped_lpips_fail`
- Governing rule:
  - under Phase 2, `content_lpips >= 0.70` is an immediate failure
- Corrected runtime closure point:
  - `rtfix epoch_0001`
  - transfer `0.724444 / 0.712723`
  - all-pairs `0.724472 / 0.707551`
- Decision:
  - preserve this family as true-I2SB implementation evidence
  - remove it from the active Distinct5 remote training queue
  - do not spend more formal remote training time trying to “wait for LPIPS to come down”
