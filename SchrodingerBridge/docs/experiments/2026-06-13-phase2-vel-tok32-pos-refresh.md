# Phase 2: vel_tok32_pos_refresh

Date: 2026-06-13

## Goal

- keep the Distinct5 formal lane on the `velocity` side
- reuse the last safe parent instead of reopening endpoint / I2SB
- strengthen the pure-latent tokenizer before adding another structure-side training patch
- target board:
  - style `>= 0.72`
  - `content_lpips <= 0.30`

## Why This Packet Exists

- the last safe velocity packet closed at:
  - best `epoch_0002`
    - transfer `0.673934 / 0.384340`
    - all-pairs `0.701666 / 0.381724`
- that line stayed inside the `LPIPS < 0.40` continuation band, but it plateaued around the `0.70x / 0.38x` shelf
- the exact-I2SB fallback ladder is now closed:
  - the residual endpoint retry improved LPIPS relative to the absolute endpoint path
  - but it still landed at `all-pairs 0.697686 / 0.569086`
  - that is still archival-only under the current phase-2 rules
- so the clean next move is not another endpoint variant
- it is a tokenizer-capacity refresh on top of the last safe velocity parent

## Config

- candidate config:
  - [phase2_vel_tok32_pos_refresh_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_pos_refresh_seed42_b20a1.json)
- base config:
  - [phase2_vel_pattn_enhanced_tok_seed42_b22a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_pattn_enhanced_tok_seed42_b22a1.json)

## Changed Knobs

- keep unchanged:
  - `transport_prediction_mode = velocity`
  - `solver_family = euler_legacy`
  - `tokenizer_family = pure_latent_spatial`
  - `proximal_mode = crossattn_texture`
  - `kinetic_penalty_mode = manifold_adaptive_split`
- tokenizer refresh:
  - `tokenizer_query_dim = 96`
  - `tokenizer_query_num_blocks = 5`
  - `tokenizer_pe_temperature = 0.75`
  - `tokenizer_global_gate_hidden_dim = 192`
  - `tokenizer_global_gate_scale = 1.1`
  - `tokenizer_structured_temperature = 0.08`
- launch hygiene:
  - `batch_size = 20`
  - warm-start from the last safe parent:
    - `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_pattn_enhanced_tok_seed42_b22a1/epoch_0002.pt`
  - `resume_model_strict = false`
  - `resume_optimizer = false`
  - `resume_training_state = false`
  - `resume_prefer_local_checkpoint = false`

## Hypothesis

- the previous enhanced tokenizer path already proved the model can stay in-band while carrying style to `0.701666`
- the remaining gap looks more like routing capacity / style specificity than solver correctness
- this refresh therefore tries to:
  - sharpen routing with lower tokenizer temperature
  - increase semantic query capacity without moving back to endpoint prediction
  - let the global style code read more from the pooled spatial evidence instead of staying near the static embedding
- if the first settled checkpoint still lands near `0.70x / 0.38x`, the next move should be training-side structure control on top of this same velocity family, not another tokenizer-only retry

## Promotion Contract

- paper-facing success target:
  - style `>= 0.72`
  - `content_lpips <= 0.30`
- continue-to-train gate:
  - settled checkpoints must remain in `content_lpips < 0.40`
- archival gate:
  - `0.40 <= content_lpips < 0.70`
- fail-stop gate:
  - `content_lpips >= 0.70`

## Local Prep

- config inheritance:
  - switched from cloned large JSON to `_base` override form
  - this keeps future packet edits small and auditable
- code support:
  - pure-latent tokenizer hyperparameters are now explicit config switches instead of hidden constants
  - the 2D positional encoding implementation was also corrected to a proper sine/cosine construction
- launch state:
  - local smoke completed
  - remote formal launch not started yet

## Smoke

- local synthetic smoke:
  - output:
    - [phase2_vel_tok32_pos_refresh_seed42_b20a1_smoke.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_vel_tok32_pos_refresh_seed42_b20a1_smoke.json)
  - status `ok`
  - `objective_mode = bridge_velocity`
  - `tokenizer_family = pure_latent_spatial`
  - `solver_family = euler_legacy`
  - `transport_prediction_mode = velocity`
  - no DINO runtime required
  - tensor shapes:
    - forward `[1, 4, 32, 32]`
    - endpoint `[1, 4, 32, 32]`
    - integrated `[1, 4, 32, 32]`
  - loss read:
    - `loss = 2.303087`
    - `flow = 2.067408`
    - `terminal_swd = 0.009933`
    - `t_mean = 0.456406`
  - first grad:
    - `structured_style_tokenizer.universal_keys`
    - abs mean `0.005062`

## Run Log

- remote launch:
  - launcher:
    - `launch_remote_experiment_train.py`
  - task:
    - `exp-phase2_vel_tok32_pos_refresh_seed42_b20a1-train`
  - launch time:
    - `2026-06-13 07:49`
  - remote log:
    - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_phase2_vel_tok32_pos_refresh_seed42_b20a1_train.log`
- first health check:
  - `30s health = 10073 MiB`
  - accepted into the formal `9.0-10.8 GiB` band
- current remote status:
  - run name:
    - `aaai2027_phase2_vel_tok32_pos_refresh_seed42_b20a1`
  - live state:
    - `training_before_first_settled_eval`
  - runtime memory:
    - latest read `9870 / 12288 MiB`
  - live process:
    - `/home/xy/venvs/samam312/bin/python SchrodingerBridge/src/run.py --config /mnt/i/Github/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_pos_refresh_seed42_b20a1.json`
  - eval state:
    - no checkpoint has settled yet
    - first `CLIP-S + LPIPS` authority point is still pending
- warm-start read:
  - partial resume from:
    - `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_pattn_enhanced_tok_seed42_b22a1/epoch_0002.pt`
  - loader summary:
    - `loaded=241`
    - `skipped=29`
    - `missing=35`
    - `unexpected=0`
