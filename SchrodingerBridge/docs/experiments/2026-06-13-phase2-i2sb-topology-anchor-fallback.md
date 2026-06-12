# Phase 2: true-I2SB topology-anchor fallback

Date: 2026-06-13

## Why This Exists

- the current `velocity + topology anchor` packet remains the first active read
- but its first settled point only reached:
  - transfer `0.674077 / 0.393103`
  - all-pairs `0.700842 / 0.390843`
- that is still in-band, yet still below the earlier velocity shelf
- meanwhile the Phase 2 board logic remains:
  - velocity is better at keeping structure in-band
  - endpoint / true-I2SB is better at finding style headroom
- so if velocity keeps plateauing, the clean next question is no longer "more solver hacks"
- it is:
  - can true-I2SB recover enough structure if we directly anchor its endpoint topology during training?

## Candidate Packet

- config:
  - [phase2_i2sb_topo_anchor_sigma0p25_seed42_b22a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_topo_anchor_sigma0p25_seed42_b22a1.json)
- smoke:
  - [phase2_i2sb_topo_anchor_sigma0p25_seed42_b22a1_smoke.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_i2sb_topo_anchor_sigma0p25_seed42_b22a1_smoke.json)

## Contract

- `tokenizer_family = pure_latent_spatial`
- `transport_prediction_mode = endpoint`
- `solver_family = solver_i2sb`
- `objective_mode = i2sb_endpoint`
- `bridge_noise_schedule = exact_brownian`
- `bridge_sigma = 0.25`

This is a real `true I2SB` packet, not the historical delayed-window variant.

## Structure Rescue Terms

- `w_content_lowpass_anchor = 0.35`
- `w_content_edge_anchor = 0.15`
- `content_anchor_lowpass_kernel = 9`

Interpretation:

- lowpass anchor:
  - keeps coarse latent topology tied to the source
- edge anchor:
  - keeps the source edge skeleton present without forcing texture identity

This is intentionally cleaner than old heuristic stacks:

- no DINO
- no solver-only post-hoc rescue
- no anisotropic or stokes penalties in the first fallback packet
- no velocity-side kinetic suppression

Why this is the right fallback:

- the active velocity topology-anchor packet already showed that the new anchors are live
- but its numeric debug also showed the remaining dominant structure-side term was still `kinetic_energy`
- `true I2SB` removes that velocity bottleneck entirely and lets the topology anchors act directly on the endpoint prediction path

## Resource Read

- conservative launch target:
  - `batch_size = 22`
  - `accumulation_steps = 1`
- rationale:
  - current paper-facing 3060 policy still treats `< 11 GiB` as the hard ceiling
  - the packet should start from the same safe band logic as the current velocity lane, then be recalibrated only if the first health read proves it is too far under-band

## Read Rule

- promote only if:
  - style moves toward `0.72+`
  - while `LPIPS` stays strictly below `0.40`
- immediate fail:
  - if the first settled point lands at `LPIPS >= 0.70`
- archival only:
  - if the first settled point lands in `0.40 <= LPIPS < 0.70`

## Current Status

- local smoke passed
  - `status = ok`
  - `objective_mode = i2sb_endpoint`
  - `solver_family = solver_i2sb`
  - `bridge_sigma = 0.25`
  - first grad hit `structured_style_tokenizer.universal_keys`
- remote status:
  - not launched yet
  - blocked only by the fact that the active formal lane is still the velocity topology-anchor read
