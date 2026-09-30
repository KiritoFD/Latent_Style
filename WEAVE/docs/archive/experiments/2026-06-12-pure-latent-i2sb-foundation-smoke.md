# Pure Latent I2SB Foundation Smoke

Date: 2026-06-12

Purpose:

- verify that the first pure-latent tokenizer plus true I2SB bridge slice is runnable
- confirm that the new mainline no longer requires DINO supervision to build and backpropagate

## Config

- config:
  - [aaai2027_round2_pure_latent_i2sb_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round2_pure_latent_i2sb/aaai2027_round2_pure_latent_i2sb_seed42_b8a2.json)

Key switches:

- `tokenizer_family = pure_latent_spatial`
- `solver_family = solver_i2sb`
- `transport_prediction_mode = endpoint`
- `bridge.objective_mode = i2sb_endpoint`
- `bridge.loss_type = mse`
- `semantic_supervision_family = legacy_terminal_swd`
- `dino_masked_swd_weight = 0.0`
- `bridge_sigma = 0.5`
- `bridge_noise_schedule = exact_brownian`
- `tokenizer_num_clusters = 32`
- `w_flow = 1.0`
- `w_kinetic = 0.0`
- `structure_penalty_mode = off`
- all inherited heuristic structure penalties in this config are zeroed for the first clean run

## Smoke Result

- artifact:
  - [round2_pure_latent_i2sb_smoke_latest.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round2_pure_latent_i2sb_smoke_latest.json)
- status:
  - `ok`
- shapes:
  - `forward = [1, 4, 32, 32]`
  - `endpoint = [1, 4, 32, 32]`
  - `integrated = [1, 4, 32, 32]`
- optimization sanity:
  - `loss = 3.3295`
  - `flow = 2.6355`
  - `terminal_swd = 0.0386`
  - `t_mean = 0.5020`
  - first gradient parameter:
    - `structured_style_tokenizer.universal_keys`
  - first gradient abs-mean:
    - `0.0861`

## Gradient Audit

- inactive legacy parameters:
  - `style_spatial_id_16 = null`
  - `style_tokenizer.concept_atoms = null`
  - `style_tokenizer.atom_logits.weight = null`
- active pure-tokenizer parameters:
  - `structured_style_tokenizer.universal_keys = 0.6799`
  - `structured_style_tokenizer.style_global_code.weight = 0.0268`
  - `structured_style_tokenizer.style_values.weight = 0.1727`
  - `structured_style_tokenizer.query_extractor.0.weight = 0.0998`

## What This Proves

- the codebase now contains:
  - a latent-native structured tokenizer path
  - a true posterior-stepping endpoint solver path
- the corrected config now also uses the true bridge training objective:
  - `objective_mode = i2sb_endpoint`
  - sampled Brownian-bridge training states are active
  - the Brownian factor is no longer window-gated on the true-I2SB path
  - `t_mean` is no longer pinned to `1.0`
- the pure tokenizer is now genuinely latent-native:
  - query extraction starts from `z0`
  - the query extractor receives non-zero gradients
- the pure tokenizer no longer drops the solver's time conditioning:
  - the structured global code is added on top of the incoming time-conditioned base code
- the new pair can:
  - build
  - run direct forward
  - run endpoint prediction
  - run integrated transport
  - compute the corrected sampled-bridge objective
  - backpropagate
- no DINO cache or DINO conditioning is required for this smoke path
- the legacy factorized tokenizer remains only as a compatibility anchor and does not receive gradients on this path
- on current trainer code for future pure-latent launches:
  - legacy tokenizer latent init is skipped
  - legacy tokenizer trainable parameter count is `0`

## Immediate Next Step

- create the first remote training lane from the generated round-2 sweep family
- keep DINO families out of the main queue unless they later show a clearly dominant empirical gain
