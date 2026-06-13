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
  - `resume_prefer_local_checkpoint = true`

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
  - latest observed failure:
    - `epoch_0001` trained and checkpointed successfully
    - training then offloaded to CPU for remote full eval
    - the launcher runtime guard misread eval/offload as an under-band training failure and killed the process:
      - `=== RUNTIME_UNDER_BAND_STOP ... used=2101MiB floor=9216MiB elapsed=1165s consecutive=3 ===`
  - consequence:
    - `epoch_0001.pt` exists
    - `full_eval/epoch_0001/` was created
    - but the first settled summary never finished, so there is still no authority point
  - fix decision:
    - patch the launcher so `_base` configs are resolved before deciding whether `full_eval_each_epoch` requires `runtime_guard_min_mode=warn`
    - relaunch this same run and prefer the local latest checkpoint
  - relaunch result:
    - launcher now prints:
      - `switch runtime_guard_min_mode stop -> warn because config uses epoch-end remote full eval with trainer offload`
    - local checkpoint recovery now works:
      - local latest `epoch_0001.pt` was preferred over the original parent warm-start
      - partial resume read:
        - `loaded=276`
        - `skipped=0`
        - `missing=0`
        - `unexpected=0`
      - resumed at:
        - `epoch=2`
        - `global_step=944`
    - relaunch 30s health:
      - `10151 MiB`
    - current state after relaunch:
      - live training resumed successfully
      - first settled eval landed at `epoch_0002`
- watcher:
  - `watch_phase2_velocity_handoff.py`
  - mode:
    - `stop_only`
  - rationale:
    - this packet should obey the same LPIPS hard gates and plateau rule
    - but it should not auto-handoff into the old `solver_pc` review path
  - runtime note:
    - local watcher output now flushes poll JSON into the watcher log during the wait window

## Settled Curve

- current settled authority point:
  - `epoch_0002`
  - transfer `0.673024 / 0.390256`
  - all-pairs `0.700342 / 0.387609`
  - identity `0.809617 / 0.377019`
  - eval wall `217.09s`
  - generation `119.21s`
  - VAE decode `54.55s`
- second settled authority point:
  - `epoch_0003`
  - transfer `0.668702 / 0.364875`
  - all-pairs `0.698072 / 0.361798`
  - identity `0.815554 / 0.349488`
  - eval wall `220.31s`
  - generation `121.90s`
  - VAE decode `55.88s`
- third settled authority point:
  - `epoch_0004`
  - transfer `0.673399 / 0.376463`
  - all-pairs `0.701161 / 0.374695`
  - identity `0.812208 / 0.367622`
  - eval wall `234.73s`
  - generation `129.62s`
  - VAE decode `57.91s`
- fourth settled authority point:
  - `epoch_0005`
  - transfer `0.670604 / 0.375912`
  - all-pairs `0.699187 / 0.373331`
  - identity `0.813521 / 0.363008`
  - eval wall `232.11s`
  - generation `127.19s`
  - VAE decode `57.15s`
- convergence read:
  - `row_count = 4`
  - `best_epoch = epoch_0004`
  - `best_in_newest_2 = true`
  - `tail_flat = false`
  - `converged = false`
- unresolved backlog:
  - `epoch_0001` is now treated as `stale_pending`
    - the earlier guard bug interrupted that eval mid-run
    - it no longer masks the live lane as an active eval-pending run in the status reporter
  - the first valid authority point is therefore `epoch_0002`, not `epoch_0001`
  - current runtime read:
    - `live_state = training_after_settled_eval`
    - `pending_checkpoint_epochs = []`
    - `stale_pending_checkpoint_epochs = [epoch_0001]`

## Read

- gate decision:
  - continue running
  - both settled points remain inside the formal continuation band `LPIPS < 0.40`
- promotion read:
  - not promoted
  - compared with the previous safe velocity parent best `all-pairs 0.701666 / 0.381724`, this packet is currently:
    - `epoch_0002` is lower on style and worse on LPIPS
    - `epoch_0003` is lower on style but better on LPIPS
    - `epoch_0004` is still slightly lower on style, but now also better on LPIPS
    - `epoch_0005` is lower on style than `epoch_0004`, but improves LPIPS again
- interpretation:
  - the tokenizer refresh still has not produced the desired breakout above the old shelf
  - but `epoch_0004` does create a strictly stronger in-band point than `epoch_0003`
  - it is also the current best point of this packet on both transfer and all-pairs Pareto authority
  - `epoch_0005` adds another in-band Pareto point rather than a domination update
  - the line is therefore still alive as a real improvement path rather than a flat plateau
  - the packet is currently training in `epoch_6`, with the next meaningful decision deferred to the next settled authority point
  - keep the formal lane alive until either:
    - a better in-band point appears
    - or the same plateau logic closes it later
- warm-start read:
  - partial resume from:
    - `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_pattn_enhanced_tok_seed42_b22a1/epoch_0002.pt`
  - loader summary:
    - `loaded=241`
    - `skipped=29`
    - `missing=35`
    - `unexpected=0`
