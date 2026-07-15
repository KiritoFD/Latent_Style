# Pure Latent I2SB Design

Date: 2026-06-12

Objective:

- pivot away from the DINO-heavy tokenizer direction unless it later proves overwhelmingly superior
- move the model toward:
  - a true endpoint-regression I2SB bridge with exact posterior stepping
  - a true latent-native spatial tokenizer that does not require external DINO patches or style-bank sidecars

## Current Gap

- current `endpoint` mode is not a true bridge solver
  - if `transport_prediction_mode == endpoint`, the runtime simply predicts the terminal endpoint and returns it
  - this bypasses the whole intermediate Schrödinger-bridge posterior chain
- current tokenizer families are still externally anchored
  - `tok_a/tok_b/tok_c/tok_d` all depend on DINO or VLM-like routing paths
  - even the best tokenizer signal so far (`tok_b_cross_image`) comes from that external route
- consequence:
  - the system is not yet a clean latent-only bridge model
  - the design story remains vulnerable to the critique that the hard part is outsourced to external priors

## Recommended Hard Pivot

- keep DINO routes only as archived ablations
- stop treating them as the main line
- introduce a new pair of first-class switches:
  - tokenizer family:
    - `pure_latent_spatial`
  - solver family:
    - `solver_i2sb`

## Tokenizer Design

- keep the closed-set `style_id` interface for compatibility
- replace DINO patch routing with latent-native routing directly from the input `z0`
- architecture:
  - local query extractor:
    - lightweight `3x3 conv -> SiLU -> 3x3 conv`
  - universal latent keys:
    - shared learnable cluster centers
  - style-specific values:
    - embedding table indexed by `style_id`
  - outputs:
    - `global_code`
    - `spatial_map`
    - optional `gate_map` and `mask_map`
- intended effect:
  - the tokenizer self-organizes spatial style routing directly from latent topology
  - no RGB foundation-model prior is required
- important runtime rule:
  - the tokenizer must preserve time conditioning
  - in the active implementation, the tokenizer adds a learned style-global residual on top of the incoming time-conditioned base code instead of replacing it

## Bridge / Solver Design

- training target:
  - sample Brownian-bridge states
  - regress the terminal endpoint `x_1`
- inference:
  - do not jump directly to `t=1`
  - step from `t_curr` to `t_next` using the exact posterior:
    - posterior mean from current state and predicted endpoint
    - posterior variance from `bridge_sigma`
- this becomes a real I2SB path rather than an endpoint shortcut

## Objective Correction

- an additional gap was found during implementation:
  - the first generated round-2 configs were still inheriting `bridge.objective_mode = omf`
  - that kept training on the legacy fixed-endpoint objective
  - it also caused the inference wrapper to route to `endpoint_map` instead of the integrated `solver_i2sb` path
- corrected mainline contract:
  - `bridge.objective_mode = i2sb_endpoint`
  - `transport_prediction_mode = endpoint`
  - `solver_family = solver_i2sb`
  - `w_flow = 1.0`
  - clean line zeros inherited `w_kinetic`, teacher terms, and structure heuristics
- corrected tokenizer control contract:
  - `tok_baseline_global` now sets `ablation_disable_spatial_prior = true`
  - this makes the tokenizer baseline genuinely global-only instead of secretly inheriting an id-spatial map

## Remote Eval Contract

- remote evaluation must happen on the authoritative `3060` path
- evaluation cannot rely on concurrent training-plus-second-python execution if that risks crossing the `11.0 GiB` cap
- the active round-2 direction is:
  - save checkpoint
  - offload trainer state from CUDA
  - run remote `CLIP-S + LPIPS` eval on the saved checkpoint
  - restore trainer state and continue

## First Implementation Slice

- completed in code now:
  - new tokenizer family label:
    - `pure_latent_spatial`
  - new solver family label:
    - `solver_i2sb`
  - first latent-native tokenizer module:
    - `PureLatentSpatialTokenizer`
  - runtime support:
    - structured tokenizer path no longer has to require DINO sidecars when family is `pure_latent_spatial`
    - pure tokenizer query extraction now reads the input latent `z0` and resizes into the body grid only after latent query formation
    - pure tokenizer global code now preserves the incoming time-conditioned base code
  - bridge support:
    - endpoint integration now supports a true posterior stepping branch for `solver_i2sb`
    - the training objective now exposes an explicit `i2sb_endpoint` mode for sampled-bridge endpoint regression
  - round-2 sweep registry:
    - [round2_registry.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/round2_registry.py)
  - round-2 config materializer:
    - [prepare_round2_pure_sde_configs.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/prepare_round2_pure_sde_configs.py)
  - runnable config scaffold:
    - [aaai2027_round2_pure_latent_i2sb_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round2_pure_latent_i2sb/aaai2027_round2_pure_latent_i2sb_seed42_b8a2.json)
  - generic config smoke tool:
    - [smoke_experiment_config.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/smoke_experiment_config.py)
  - first smoke note:
    - [2026-06-12-pure-latent-i2sb-foundation-smoke.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-12-pure-latent-i2sb-foundation-smoke.md)

## Immediate Next Steps

1. create a clean config family using:
   - `tokenizer_family = pure_latent_spatial`
   - `solver_family = solver_i2sb`
   - `transport_prediction_mode = endpoint`
   - `objective_mode = i2sb_endpoint`
   - `semantic_supervision_family = legacy_terminal_swd`
   - `dino_masked_swd_weight = 0.0`
2. remove DINO dependency from the active training plan
   - no more `tok_a/tok_b/tok_c/tok_d` as main-lane priorities
3. materialize the round-2 sweep configs and doc scaffold
   - manifest:
     - [round2_family_manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round2_pure_sde/round2_family_manifest.csv)
4. run a small local switch smoke on the new pair
5. launch a new remote lane only after the new config passes build/forward/backward and the `9.0-10.8 GiB` VRAM band is re-checked

## Compatibility Note

- the legacy `style_tokenizer` object is still instantiated in the model for checkpoint/state-dict compatibility
- on the `pure_latent_spatial` path it is no longer part of the active gradient path
- the authoritative tokenizer parameters are now under `structured_style_tokenizer.*`
- updated trainer contract:
  - the legacy tokenizer no longer performs latent-stat initialization on the `pure_latent_spatial` path
  - its parameters are now compatibility-only and excluded from trainable parameters on new pure-latent runs

## Decision Rule

- DINO is no longer the default path
- a DINO family only comes back if it later shows a clearly dominant gain on the real board
- until then, the paper-facing main line is:
  - pure latent tokenizer
  - exact posterior I2SB solver
