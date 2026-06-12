# Phase 2: topology-anchor follow-up

Date: 2026-06-13

## Trigger

- the first pure-tokenizer velocity lane stayed inside `LPIPS < 0.40`
- but it plateaued around:
  - best `epoch_0002` all-pairs `0.701666 / 0.381724`
  - latest `epoch_0006` all-pairs `0.698086 / 0.367844`
- the solver-only reuse probe also failed to rescue structure:
  - `eval_only_pc_solver`
  - transfer `0.729014 / 0.621056`
  - all-pairs `0.735295 / 0.611310`

## Hypothesis

- the current velocity line is not failing because structure is unconstrained
- it is failing because the only active structure control is still dominated by transport-energy suppression
- that keeps LPIPS safe, but also caps style motion too early
- the clean next move is:
  - reduce kinetic pressure slightly
  - replace part of that safety budget with a direct latent topology anchor on the predicted endpoint

## New Loss Switches

- added to `bridge`:
  - `w_content_lowpass_anchor`
  - `w_content_edge_anchor`
  - `content_anchor_lowpass_kernel`
- behavior:
  - `content_lowpass_anchor`
    - `L1( lowpass(endpoint), lowpass(content) )`
    - preserves coarse layout and low-frequency latent topology
  - `content_edge_anchor`
    - `L1( gradmag(lowpass(endpoint)), gradmag(lowpass(content)) )`
    - preserves the structural edge skeleton without forcing texture identity
- design intent:
  - cheap enough for the formal `3060` lane
  - no external priors
  - reusable by both `velocity` and `true I2SB`

## First Candidate Packet

- config:
  - [phase2_vel_pattn_topo_anchor_k075_seed42_b22a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_pattn_topo_anchor_k075_seed42_b22a1.json)
- deltas vs the closed velocity packet:
  - `w_kinetic: 1.0 -> 0.75`
  - `w_content_lowpass_anchor = 0.25`
  - `w_content_edge_anchor = 0.10`
  - `content_anchor_lowpass_kernel = 9`
- constants retained:
  - `tokenizer_family = pure_latent_spatial`
  - `transport_prediction_mode = velocity`
  - `solver_family = euler_legacy`
  - `proximal_mode = crossattn_texture`
  - `kinetic_penalty_mode = manifold_adaptive_split`
  - batch stays at `22` until a new VRAM read proves otherwise

## Read Rule

- success target remains unchanged:
  - style `>= 0.72`
  - `LPIPS <= 0.30`
- practical read:
  - if style does not clear the old `0.701666` shelf while LPIPS stays in-band, the anchor is too weak
  - if LPIPS jumps into `0.40+`, the anchor is not sufficient to replace the missing kinetic pressure
  - if style lifts while LPIPS remains `< 0.40`, this becomes the first real Phase 2 continuation path after the plateaued parent

## Sequence

1. local smoke and compile on the new switches
2. launch the new velocity packet on the remote `3060`
3. first health check within `30s`
4. let checkpoint-level `CLIP-S + LPIPS` decide whether the lane survives

## Current Status

- local compile:
  - `py_compile` passed for `config_schema.py` and `losses.py`
- local smoke:
  - [phase2_vel_pattn_topo_anchor_k075_seed42_b22a1_smoke.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_vel_pattn_topo_anchor_k075_seed42_b22a1_smoke.json)
  - status `ok`
  - first gradient hit `structured_style_tokenizer.universal_keys`
- remote launch:
  - run name `aaai2027_phase2_vel_pattn_topo_anchor_k075_seed42_b22a1`
  - PID `53058`
  - live state `training_after_settled_eval`
  - `30s` health read `10211 MiB`
  - follow-up status read `10347 MiB`
  - this is inside the paper-facing `< 11 GiB` rule
- first settled checkpoint:
  - `epoch_0001` at `2026-06-13 04:46:35`
  - transfer `0.674077 / 0.393103`
  - all-pairs `0.700842 / 0.390843`
  - eval wall `210.28s`
  - generation `114.78s`
  - VAE decode `54.18s`
- read:
  - this is still in-band because both transfer and all-pairs remain below `0.40`
  - but it does **not** beat the closed velocity shelf:
    - prior best all-pairs `0.701666 / 0.381724`
    - current `epoch_0001` all-pairs `0.700842 / 0.390843`
  - so the topology anchor is not yet a win, only a live candidate
- training-side read:
  - epoch-level CSV after `epoch_0001`:
    - `kinetic_energy = 0.0790`
    - `loss = 1.0504`
  - numeric debug during `epoch_0002`:
    - `content_lowpass_anchor ≈ 0.051-0.059`
    - `content_edge_anchor ≈ 0.019-0.024`
    - `kinetic_energy ≈ 0.083-0.089`
  - interpretation:
    - the topology anchors are active
    - but kinetic is still the larger structure-side force
    - if this packet keeps plateauing, the cleaner next move is not “even more velocity regularization”
    - it is to move the same topology anchors onto `true I2SB`, where endpoint style capacity is already higher
- current decision:
  - the lane is now closed as `archival only`
  - reason:
    - `epoch_0002` improved style to all-pairs `0.706132`
    - but LPIPS also rose to `0.413976`
    - that crosses the Phase 2 continuation ceiling `LPIPS < 0.40`
  - final read for this packet:
    - `epoch_0001`: all-pairs `0.700842 / 0.390843`
    - `epoch_0002`: all-pairs `0.706132 / 0.413976`
  - interpretation:
    - the topology anchor did buy some style
    - but not enough to keep the line promotable
    - so this packet is evidence that velocity + lighter kinetic still leaks structure before reaching the paper target
  - parallel preparation:
    - if this packet stalls, the next cleaner candidate should be a `true I2SB + pure_latent_spatial + topology anchor` retry
    - reason:
      - velocity still under-expresses style
      - endpoint/I2SB already has style headroom, but needs structure rescue more than more solver hacks
