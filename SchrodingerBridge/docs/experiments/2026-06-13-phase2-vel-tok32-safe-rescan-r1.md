# Phase 2: vel_tok32_safe_rescan_r1

Date: 2026-06-13

## Role

- first formal follow-up after `vel_tok32_pos_refresh` closed inside-band but below the old shelf
- stays inside the same `velocity + pure_latent_spatial + crossattn_texture + manifold_adaptive_split` family
- does not reopen endpoint / I2SB
- does not spend the next formal slot on topology-anchor yet

## Why This Exists

- `vel_tok32_pos_refresh` proved the refreshed tokenizer stack is safe but not yet promotable:
  - best point stayed at `epoch_0004`
  - transfer `0.673399 / 0.376463`
  - all-pairs `0.701161 / 0.374695`
- the closure point at `epoch_0006` confirmed the packet was no longer improving style:
  - transfer `0.671522 / 0.385051`
  - all-pairs `0.699725 / 0.381878`
- that means the next question is not “can a stronger structure patch rescue it?”
- the next question is:
  - can we get a cleaner in-band style lift by only retuning the safe tokenizer / kinetic knobs around the current best parent?

## Config

- config:
  - [phase2_vel_tok32_safe_rescan_r1_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r1_seed42_b20a1.json)
- parent packet:
  - [phase2_vel_tok32_pos_refresh_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_pos_refresh_seed42_b20a1.json)
- warm-start parent checkpoint:
  - `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_pos_refresh_seed42_b20a1/epoch_0004.pt`

## Deltas

- keep unchanged:
  - `tokenizer_family = pure_latent_spatial`
  - `transport_prediction_mode = velocity`
  - `solver_family = euler_legacy`
  - `proximal_mode = crossattn_texture`
  - `kinetic_penalty_mode = manifold_adaptive_split`
  - `batch_size = 20`
  - deeper query extractor + 2D positional encoding + 32 clusters
- safe-family overrides:
  - `tokenizer_structured_temperature: 0.08 -> 0.07`
  - `tokenizer_global_gate_scale: 1.1 -> 1.2`
  - `w_kinetic: 1.0 -> 0.9`

## Hypothesis

- `tok32_pos_refresh` likely improved LPIPS because the tokenizer became more coherent, but the family still did not spend enough style budget to cross the old shelf
- this packet therefore makes three small, same-family moves:
  - sharpen routing a bit more
  - let pooled spatial evidence bias the global style code a bit harder
  - loosen the kinetic brake slightly without jumping to a new structure family
- if this still cannot beat `0.701666 / 0.381724`, then the safe-family sweep itself is weak evidence against more tokenizer-only retuning

## Promotion Contract

- Stage-A success:
  - exceed `all-pairs 0.701666 / 0.381724`
- Stage-B success:
  - `all-pairs style >= 0.705`
  - `content_lpips <= 0.380`
- continue-to-train gate:
  - all settled checkpoints must remain in `content_lpips < 0.40`
- archival gate:
  - `0.40 <= content_lpips < 0.70`
- fail-stop gate:
  - `content_lpips >= 0.70`
- closure preference:
  - if the line again flattens below the old shelf, close it before spending the formal slot on a second tokenizer-only sweep

## Smoke

- local synthetic smoke:
  - [phase2_vel_tok32_safe_rescan_r1_seed42_b20a1_smoke.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_vel_tok32_safe_rescan_r1_seed42_b20a1_smoke.json)
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
    - `loss = 2.297992`
    - `flow = 2.067853`
    - `terminal_swd = 0.009936`
    - `t_mean = 0.456406`
  - first grad:
    - `structured_style_tokenizer.universal_keys`
    - abs mean `0.005186`
- local observability verification:
  - a direct random forward/backward confirms `structured_style_tokenizer.last_debug` is now populated
  - current keys include:
    - `attn_entropy`
    - `attn_max`
    - `num_clusters`
    - `pe_temp`
    - `query_dim`
    - `query_num_blocks`
    - `global_gate_scale`
  - trainer-side scalar extraction also returns:
    - `structured_style_tokenizer_attn_entropy`
    - `structured_style_tokenizer_attn_max`
    - `structured_style_tokenizer_global_gate_scale`
  - this is the diagnostic channel that will be used to judge whether the tokenizer is actually self-organizing rather than only improving LPIPS by accident

## Run Log

- remote launch:
  - launcher:
    - `launch_remote_experiment_train.py`
  - task:
    - `exp-phase2_vel_tok32_safe_rescan_r1_seed42_b20a1-train`
  - initial launch time:
    - `2026-06-13 10:24`
  - remote log:
    - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_phase2_vel_tok32_safe_rescan_r1_seed42_b20a1_train.log`
  - initial 30s health:
    - `10142 MiB`
    - accepted into the formal `9.0-10.8 GiB` band
  - resume read:
    - partial model resume from `tok32_pos_refresh epoch_0004`
    - `loaded=276 skipped=0 missing=0 unexpected=0`
  - early relaunch:
    - before the first checkpoint settled, the packet was intentionally restarted once
    - reason:
      - enable tokenizer observability on the live formal lane
      - `semantic_tokenizer.last_debug` now really persists
      - scalar tokenizer diagnostics now enter trainer-side numeric debug and epoch metrics
    - relaunch time:
      - `2026-06-13 10:35`
    - relaunch 30s health:
      - `10140 MiB`
  - current remote status:
    - run name:
      - `aaai2027_phase2_vel_tok32_safe_rescan_r1_seed42_b20a1`
    - live state:
      - `training_after_settled_eval`
    - remote PID:
      - `23197`
    - current GPU read:
      - `9792 MiB`
    - checkpoint / eval state:
      - `epoch_0001.pt` has been saved
      - `epoch_0001` full eval has settled
      - the run has already resumed into `epoch_2`
    - latest epoch-end train read before eval:
      - `loss=0.9618`
      - `flow=0.6029`
      - `kin=0.0931`
      - `ot=0.0979`
      - `tswd=0.0148`
      - remote log also emitted tokenizer-side aggregate `ent=1.384`
      - this is the first formal-lane proof that tokenizer observability is landing in runtime logs
- watcher:
  - `watch_phase2_velocity_handoff.py`
  - mode:
    - `stop_only`
  - current wait state:
    - waiting for settled epoch `>= 6`
    - latest settled epoch `epoch_0001`
    - current pending checkpoint list:
      - none

## Settled Curve

- first settled authority point:
  - `epoch_0001`
  - transfer `0.672934 / 0.384740`
  - all-pairs `0.700686 / 0.383351`
  - identity `0.811691 / 0.377796`
  - eval wall `224.17s`
  - generation `122.75s`
  - VAE decode `55.88s`
- current interpretation:
  - still safely in-band
  - slightly stronger than the earlier `tok32_pos_refresh epoch_0002` warm-start region on LPIPS
  - but still below both:
    - the old shelf `0.701666 / 0.381724`
    - the direct parent best `epoch_0004 = 0.701161 / 0.374695`
  - concretely:
    - vs old shelf:
      - style `-0.000980`
      - LPIPS `+0.001627`
    - vs direct parent best:
      - style `-0.000475`
      - LPIPS `+0.008656`
  - so `epoch_0001` is jointly dominated by both references, not merely “still below target”
  - so this is not a breakout point
  - but it is also not an early fail-stop or archival read, so the lane should keep running
  - because this first point is jointly weaker than both the old shelf and the direct parent, the packet now enters a short-screen mode:
    - if by `epoch_0003` it still has no shelf break
    - and the newest settled points do not recover style
    - close the lane early instead of waiting for the longer generic patience

## Intended Read

- success:
  - style lifts above the old velocity shelf while staying clearly below `0.40`
- acceptable negative read:
  - LPIPS remains safe but the line still sits under `0.701666 / 0.381724`
  - in that case, the evidence says safe-family tokenizer retuning alone is too weak
- hard negative read:
  - the line leaves the continuation band even under these mild deltas
  - in that case, later structure-side reentry must start from the older safer parent rather than from this packet
