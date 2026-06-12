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

- initial config:
  - [phase2_i2sb_topo_anchor_sigma0p25_seed42_b22a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_topo_anchor_sigma0p25_seed42_b22a1.json)
- formal relaunch config:
  - [phase2_i2sb_topo_anchor_sigma0p25_seed42_b30a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_topo_anchor_sigma0p25_seed42_b30a1.json)
- current active subfamily:
  - [phase2_i2sb_pattn_topo_anchor_sigma0p10_warm_vel2_seed42_b22a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_pattn_topo_anchor_sigma0p10_warm_vel2_seed42_b22a1.json)
- smokes:
  - [phase2_i2sb_topo_anchor_sigma0p25_seed42_b22a1_smoke.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_i2sb_topo_anchor_sigma0p25_seed42_b22a1_smoke.json)
  - [phase2_i2sb_topo_anchor_sigma0p25_seed42_b30a1_smoke.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_i2sb_topo_anchor_sigma0p25_seed42_b30a1_smoke.json)
  - [phase2_i2sb_pattn_topo_anchor_sigma0p10_warm_vel2_seed42_b22a1_smoke.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_i2sb_pattn_topo_anchor_sigma0p10_warm_vel2_seed42_b22a1_smoke.json)

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

## Subfamily Read

### A. endpoint-topology only

- `sigma=0.25`, `b30`:
  - first settled point:
    - transfer `0.719704 / 0.728506`
    - all-pairs `0.719743 / 0.725755`
  - decision:
    - immediate `fail_stop`
- `sigma=0.10`, warm-start from velocity parent, `b30`:
  - first settled point:
    - transfer `0.701320 / 0.712179`
    - all-pairs `0.702178 / 0.711280`
  - decision:
    - immediate `fail_stop`

Interpretation:

- direct endpoint topology anchors alone do not rescue structure enough
- simply lowering `sigma` is not sufficient on its own

### B. add internal proximal rescue

- hypothesis:
  - endpoint prediction still needs an internal content-respecting correction path
  - `crossattn_texture` is acceptable here because it is still an internal latent mechanism, not an external prior
- calibration ladder:
  - `b26`:
    - runtime guard killed the packet at about `11.21 GiB`
    - over cap, so archival only
  - `b24`:
    - runtime guard killed the packet at about `11.11 GiB`
    - still over cap
  - `b22`:
    - this is the current active formal lane
    - health read `10419 MiB`

## Resource Read

- conservative first launch:
  - `batch_size = 22`
  - `accumulation_steps = 1`
- first calibration result:
  - training was valid but under-band
  - health read was only `6256 MiB`
  - later runtime read was about `7348 MiB`
  - so `b22` is archived as a calibration miss, not as a paper-facing result
- formal launch target:
  - `batch_size = 30`
  - `accumulation_steps = 1`
- rationale:
  - current paper-facing 3060 policy still treats `< 11 GiB` as the hard ceiling
  - `b30` moved the lane back into the intended band without overstepping the cap

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
  - `b22` launch:
    - valid but under-band calibration miss
    - remote PID `55787` was stopped before the first settled checkpoint
  - `b30` launch:
    - run name `aaai2027_phase2_i2sb_topo_anchor_sigma0p25_seed42_b30a1`
    - first settled point was `fail_stop`
    - lane closed
  - `sigma=0.10 warm_vel2 b30`:
    - first settled point was also `fail_stop`
    - lane closed
  - `sigma=0.10 warm_vel2 b26`:
    - over-cap runtime calibration miss
    - lane closed before settle
  - `sigma=0.10 warm_vel2 b24`:
    - over-cap runtime calibration miss
    - lane closed before settle
  - `sigma=0.10 warm_vel2 b22`:
    - run name `aaai2027_phase2_i2sb_pattn_topo_anchor_sigma0p10_warm_vel2_seed42_b22a1`
    - health read `10419 MiB`
    - current live state `training_before_first_settled_eval`
    - this is the current active formal phase2 lane
