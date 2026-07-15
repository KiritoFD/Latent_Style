# Phase 2: vel_tok32_pos_refresh

Date: 2026-06-13

## Goal

- keep the Distinct5 formal lane on the `velocity` side
- reuse the last safe parent instead of reopening endpoint / I2SB
- strengthen the pure-latent tokenizer before adding another structure-side training patch
- stage targets:
  - first beat the safe shelf `all-pairs 0.701666 / 0.381724`
  - then reach `all-pairs style >= 0.705` with `content_lpips <= 0.380`
  - long-horizon target remains `style >= 0.72` with `content_lpips <= 0.35`

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
- if this packet closes while still sitting near `0.70x / 0.38x`, the next move is a safe-family rescan inside the same velocity tokenizer line
- topology-anchor or other structure-side reentry is now deferred until we first prove a stronger in-band parent

## Promotion Contract

- stage-A success:
  - exceed `all-pairs 0.701666 / 0.381724`
- stage-B success:
  - `all-pairs style >= 0.705`
  - `content_lpips <= 0.380`
- paper-facing long success:
  - style `>= 0.72`
  - `content_lpips <= 0.35`
- continue-to-train gate:
  - settled checkpoints must remain in `content_lpips < 0.40`
- archival gate:
  - `0.40 <= content_lpips < 0.70`
- fail-stop gate:
  - `content_lpips >= 0.70`
- closure preference:
  - if later checkpoints only trade tiny LPIPS reductions for flat-or-lower style, close the lane and hand off to a safe-family rescan

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
      - the packet later trained through `epoch_0006`
      - the watcher then closed the lane on the first confirmed below-shelf plateau
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
- fifth settled authority point:
  - `epoch_0006`
  - transfer `0.671522 / 0.385051`
  - all-pairs `0.699725 / 0.381878`
  - identity `0.812538 / 0.369186`
  - eval wall `226.56s`
  - generation `126.01s`
  - VAE decode `54.94s`
- convergence read:
  - `row_count = 5`
  - `best_epoch = epoch_0004`
  - `best_in_newest_2 = false`
  - `tail_flat = false`
  - `since_best = 2`
  - `since_last_pareto = 1`
  - `converged = false`
- current interpretation:
  - the packet stayed valid in the LPIPS sense because every settled point remained in-band
  - but it never achieved the Stage-A shelf break
  - `epoch_0006` confirmed that style was still below the old shelf while LPIPS drifted back upward
  - this is therefore a formal closure into safe-family rescan, not a topology-anchor escalation
- unresolved backlog:
  - `epoch_0001` is now treated as `stale_pending`
    - the earlier guard bug interrupted that eval mid-run
    - it no longer masks the live lane as an active eval-pending run in the status reporter
  - the first valid authority point is therefore `epoch_0002`, not `epoch_0001`
  - current runtime read:
    - `live_state = settled_no_live_process`
    - `latest_checkpoint_epoch = epoch_0006`
    - `latest_settled_epoch = epoch_0006`
    - `pending_checkpoint_epochs = []`
    - `stale_pending_checkpoint_epochs = [epoch_0001]`

## Read

- gate decision:
  - close the formal lane
  - all settled points remained inside the formal continuation band `LPIPS < 0.40`, but the line failed the style-breakout test
- promotion read:
  - not promoted
  - compared with the previous safe velocity parent best `all-pairs 0.701666 / 0.381724`, this packet is currently:
    - `epoch_0002` is lower on style and worse on LPIPS
    - `epoch_0003` is lower on style but better on LPIPS
    - `epoch_0004` is still slightly lower on style, but now also better on LPIPS
    - `epoch_0005` is lower on style than `epoch_0004`, but improves LPIPS again
    - `epoch_0006` is still below the old shelf and also gives back part of the LPIPS gain
- interpretation:
  - the tokenizer refresh still did not produce the desired breakout above the old shelf
  - `epoch_0004` remains the best point of this packet on both transfer and all-pairs authority
  - `epoch_0005` added only another small LPIPS-side Pareto point
  - `epoch_0006` then failed to recover style and also moved LPIPS back toward the old shelf
  - this is enough evidence to stop burning the only formal lane on this packet
  - next formal candidate:
    - [2026-06-13-phase2-vel-tok32-safe-rescan-r1.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-rescan-r1.md)
    - [phase2_vel_tok32_safe_rescan_r1_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r1_seed42_b20a1.json)
    - launch status:
      - launched at `2026-06-13 10:24`
      - 30s health `10142 MiB`
- warm-start read:
  - partial resume from:
    - `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_pattn_enhanced_tok_seed42_b22a1/epoch_0002.pt`
  - loader summary:
    - `loaded=241`
    - `skipped=29`
    - `missing=35`
    - `unexpected=0`
