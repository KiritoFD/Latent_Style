# Round 1 Full Sweep Master

Date: 2026-06-10

Purpose:

- track the round-1 tokenizer / backbone / solver rebuild requested from `docs/tokenizer.md` and `docs/attn.md`
- keep one authoritative document for:
  - current promoted internal anchor
  - external board status against `SaMAM-2250` and `Seedream`
  - active remote lanes
  - local deep-review status
  - next family to launch

Current parent:

- `LBM-Knee e13`
- config:
  - [inmortal_knee_e13_spatial_carriergate_bodydecoder_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_seed42_b8a2.json)

Current external visual anchors:

- `SaMAM-2250`
- `Seedream-4.5 repaired750`

Round-1 folders:

- configs:
  - [round1_full_sweep](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep)
- docs:
  - [round1_full_sweep](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep)
- switch smoke:
  - [2026-06-10-family-switch-smoke.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/2026-06-10-family-switch-smoke.md)
- theory coverage:
  - [2026-06-11-theory-coverage-matrix.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/2026-06-11-theory-coverage-matrix.md)
- tokenizer warmstart:
  - [prepare_round1_tokenizer_warmstart_config.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/prepare_round1_tokenizer_warmstart_config.py)
  - [launch_remote_round1_tokenizer_warmstart.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_tokenizer_warmstart.py)
  - prepared configs currently exist for:
    - `tok_a_dino_dict`
    - `tok_b_cross_image`
    - `tok_c_residual_adapter`
    - `tok_d_vlm_prompt`
- tokenizer reconstruction-pretrain:
  - [prepare_round1_tokenizer_reconstruction_pretrain_config.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/prepare_round1_tokenizer_reconstruction_pretrain_config.py)
  - [launch_remote_round1_tokenizer_reconstruction_pretrain.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_tokenizer_reconstruction_pretrain.py)
  - prepared configs currently exist for:
    - `tok_a_dino_dict`
    - `tok_b_cross_image`
    - `tok_c_residual_adapter`
    - `tok_d_vlm_prompt`

Dataset note:

- larger five-style train pool:
  - [2026-06-10-wikiarts5-full-notest.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-wikiarts5-full-notest.md)
- local WSL baseline repro:
  - [2026-06-10-wikiarts5-baseline-repro.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-wikiarts5-baseline-repro.md)
- wikiarts5 page-1 read:
  - [2026-06-10-wikiarts5-page1-read.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-wikiarts5-page1-read.md)
- remote new-data latent prep:
  - `wikiarts5-latent-prep`
  - target latent root:
    - `/mnt/i/wikiarts_5_full_notest_latents_ema/train`
  - completion artifacts:
    - `/mnt/i/wikiarts_5_full_notest_latents_ema/train/.latent_cache/manifest.json`
    - `/mnt/i/wikiarts_5_full_notest_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`

Current execution preference:

- remote `3060`:
  - single-lane training only while local baseline repro owns the local GPU
  - launcher/status gate calibration has now been corrected to match the stated GiB targets exactly:
    - `9.0 GiB -> 9216 MiB`
    - `10.8 GiB -> 11059 MiB`
    - `11.3 GiB -> 11571 MiB`
- local GPU:
  - WSL `SaMAM patch8` segmented repro with `250-step` `CLIP-S + LPIPS` checkpoints
  - plus a detached convergence watcher that can stop the run once the curve really flattens
  - live baseline status is now auto-refreshed in:
    - [2026-06-10-wikiarts5-baseline-repro.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-wikiarts5-baseline-repro.md)
    - [baseline_live_status.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/baseline_live_status.json)
  - baseline segmented control is no longer allowed to die early at the historical fixed cap:
    - the segmented shell now defaults to `run until converged`
    - a detached resume watcher is armed on the current result root in case the already-running old controller exits before reading the new logic
  - baseline convergence authority has also been tightened:
    - use `transfer_clip_style / transfer_lpips`
    - not the raw all-pairs mean alone
  - master note no longer duplicates exact baseline point reads here:
    - use the live baseline note / json above as the single authoritative read surface
  - deferred local family wakeups are now safe to leave armed together:
    - the actual local fast-eval / review execution path is serialized by `local_gpu_lock`
    - this is to prevent the earlier `two local python jobs at once` failure from coming back when the baseline finally releases the GPU
  - deferred fast-eval launchers are now also family-status gated:
    - only families still marked `running` are allowed to auto-start local fast-eval after the baseline releases the GPU
    - this prevents a paused / recalibration-needed family such as `attn_gw_ot` from preempting the current formal lane
  - deferred stage-close launchers now use the same family-status gating:
    - a family that has been downgraded from `running` cannot later auto-enter bestfew review / frozen VLM closure by mistake
  - runtime status watchers can also be launched with the same gating:
    - once a family is no longer `running`, its live remote status poller can self-exit instead of continuing to refresh stale state forever
- generic queue now defaults to:
  - `remote train + remote fast-eval watcher`
  - correction after the `attn_gated_spade` miss:
    - `CLIP-S / LPIPS` convergence authority must exist on the same remote side as training
    - local GPU is reserved for heavy review, not as the sole authority path for fast `CLIP-S / LPIPS`
    - when queue uses remote fast-eval, it must not also auto-arm the old local deferred fast-eval / stageclose chain by default
  - current safe override while baseline repro is active:
    - `--skip-fast-eval-launch`
  - followup launcher:
    - [launch_round1_family_followups_detached.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_round1_family_followups_detached.py)
    - future family launches from the queue now auto-arm runtime watch, deferred fast-eval, and deferred stage-close instead of requiring manual followup setup
    - the helper is now idempotent for a family:
      - it stops older same-family watcher processes before starting replacements
      - this avoids silently stacking duplicate runtime/deferred watchers across recalibration or relaunch cycles
  - direct family launchers now also refuse foreign-running overlap by default:
    - `launch_remote_round1_family_train.py`
    - `launch_remote_round1_family_fast_eval.py`
    - this is a second safety net in case someone bypasses the queue and tries to start another family while a formal lane is already `running`
- shared fast-eval infra:
  - `run_evaluation.py` now persists both reference-side and source-side caches under `eval_cache`
  - the source-side cache stores resized source tensors and, when needed, source CLIP embeddings for reuse across checkpoint sweeps
  - `summary.json` now also records cache observability:
    - reference/source cache path
    - whether the cache was loaded, loaded-after-wait, or rebuilt
    - how many entries were available in each cache
- tokenizer + DINO policy for round 1:
  - DINO-supervised families stay in tokenizer-only update mode first
  - backbone remains frozen through `freeze_mode=style_branch`
  - if direct tokenizer-family training is still unstable after cache alignment is fixed, try a tokenizer-only DINO warm-start / pretrain stage before reopening the full family lane
  - a dedicated teacher/distill-based tokenizer warm-start packet now exists for tokenizer families:
    - prepare config:
      - `prepare_round1_tokenizer_warmstart_config.py`
    - optional remote launch wrapper:
      - `launch_remote_round1_tokenizer_warmstart.py`
  - remote main-lane priority is now:
    - non-DINO families first
    - DINO-related tokenizer families explicitly moved to the tail of the round-1 queue

Idle-time local work rule:

- when the remote formal lane is healthy, use local idle time for:
  - file/result organization
  - directory simplification
  - code cleanup / decoupling
  - doc summary refresh
  - short-horizon plan updates
  - theory notes for the next mechanism decision
- do not let local cleanup block:
  - remote train continuity
  - remote fast `CLIP-S / LPIPS` convergence authority
  - local heavy review already queued for shortlisted checkpoints

Required closure per family:

1. converged remote training
2. all-ckpt `CLIP-S + LPIPS` fast curve
3. shortlisted `IntroStyle + DINO`
4. frozen-snapshot `VLM`
5. closure note
6. decision note

Current remote lane status:

- active formal lane:
  - `solver_tangent_rk`
  - current formal launch setting:
    - `batch=16`
  - latest remote live sample:
    - `9510 MiB / 12288 MiB`
    - `band_status=in_band`
    - `formal_status=formal_in_band`
    - `epoch 13/24`
    - `step 1075/1180`
  - settled remote fast-eval points currently pulled:
    - `epoch_0001`
    - `epoch_0002`
    - `epoch_0003`
    - `epoch_0004`
    - `epoch_0005`
    - `epoch_0006`
    - `epoch_0007`
    - `epoch_0008`
    - `epoch_0009`
    - `epoch_0010`
    - `epoch_0011`
    - `epoch_0012`
    - `epoch_0013`
  - current fast read:
    - best transfer style remains `epoch_0001`:
      - `0.6999 / 0.5295`
    - best transfer LPIPS has now moved to `epoch_0013`:
      - `0.6935 / 0.4713`
    - best all-pairs style remains on the same high frontier and the newest best point is now:
      - `epoch_0013`
      - `0.7152 / 0.4604`
    - latest settled point is now `epoch_0013`:
      - `0.6935 / 0.4713`
    - interpretation:
      - `epoch_0005` was a true rollback point
      - `epoch_0007` first overtook the old `epoch_0004` frontier
      - `epoch_0008-0012` oscillated below that frontier
      - `epoch_0013` has now re-entered the Pareto frontier with the best transfer LPIPS so far
      - so the line remains alive and is still capable of late frontier updates
      - the line is still below external-board promotion level on transfer style
      - and because the newest settled point is again Pareto-active, closure is still premature
      - so this family remains alive and unconverged, but is not close to promotion
  - solver-family next-step policy after the `epoch_0004 -> epoch_0005` rollback:
    - do not interrupt the active in-flight tangent run
    - but future solver-family launches / continuations should use:
      - `virtual_length_multiplier = 0.5`
      - `num_epochs = 48`
    - rationale:
      - finer epoch granularity for early-knee capture
      - similar total optimization budget overall
  - resumed continuation on `2026-06-11`:
    - the remote train process had died while manifest status still remained `running`
    - direct segmented recovery was used instead of waiting for the queue:
      - resume checkpoint:
        - `epoch_0017.pt`
      - resumed target:
        - `epoch_0024`
    - first recovered health sample:
      - `9342 MiB / 12288 MiB`
      - `band_status=in_band`
      - `formal_status=formal_in_band`
      - `epoch 18/24`
      - `step 25/1180`
    - read:
      - the active remote lane is healthy again
      - so local time can shift back to cleanup/doc work until the next fast-eval packet lands
- implementation audit:
  - all `11` round-1 family configs now pass one reusable local switch smoke:
    - model build
    - direct forward
    - transport integration
    - objective compute
    - backward
  - artifact:
    - [round1_family_switch_smoke_20260610.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_family_switch_smoke_20260610.json)
  - execution rule:
    - `launch_remote_round1_family_train.py` now runs the same smoke gate before formal remote launch
    - a family that fails smoke cannot consume the remote 3060 lane
    - a direct successful launcher run now also writes `decision_status=running` back into the manifest immediately and refreshes family docs
    - a direct successful launcher run now also arms the detached runtime watcher automatically
    - a direct successful launcher run now also arms the remote fast-eval watcher automatically by default
    - family followups now also arm the local remote-fast-eval sync watcher automatically, so settled remote epochs surface into local docs without manual packet pulls
    - family followups now also arm a queue-idle watcher automatically, so once no family remains `running`, the next `planned` family can be launched through the existing round1 queue path without manual polling
    - the runtime watcher can now auto-transition a family from `running` to `reviewing` once:
      - its fast curve is marked `converged=true`
      - and the remote live train signal is gone
  - queue rule:
    - `run_round1_family_queue.py` now prefers `switch_smoke_status=ok`
    - and skips `switch_smoke_status=failed` by default
  - current auto-handoff state:
    - the local `watch_launch_round1_queue_when_idle.py` watcher is now armed against the round1 manifest
    - so once no family remains `running`, the next `planned` family will be launched through the existing queue path automatically
    - current next queue candidate:
      - `solver_pc`
- `attn_gated_spade` was downgraded on `2026-06-10`:
  - retained fast-eval evidence through `epoch_0022`
  - but process-local memory stayed under the requested band
  - the train pid disappeared mid `epoch 23`
  - status is now `recalibration_needed`, not `running`
- `attn_pnp_selfinject` remains `recalibration_needed`, not a restored formal lane:
  - opening batch is raised to `22` after the `gated_spade batch19` under-band read
  - update after the first launch:
    - `batch=22` overshot to `11939MiB` in `epoch 2`
    - the next relaunch target is now `batch=20`
  - update after the second launch:
    - `batch=20` itself is viable for training memory
    - but concurrent remote fast-eval still pushed the host above the guard cap
    - this family therefore needs segmented non-concurrent remote train/eval orchestration before the next retry
  - segmented orchestration entrypoint:
    - [run_remote_round1_family_segmented.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_remote_round1_family_segmented.py)
    - intended policy:
      - launch one bounded train segment
      - wait for remote train to exit cleanly
      - run remote fast-eval only after the train segment is no longer resident
      - then continue to the next segment from the latest retained checkpoint
    - current detached launch:
      - [round1_attn_pnp_selfinject_segmented_20260610.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_pnp_selfinject_segmented_20260610.stdout.log)
      - [round1_attn_pnp_selfinject_segmented_20260610.stderr.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_pnp_selfinject_segmented_20260610.stderr.log)
    - current observed behavior:
      - it already waited through one busy-GPU window
      - then completed one bounded train segment
      - but that bounded `batch=20` segment still died before writing a new retained checkpoint
      - a later segmented launch using the updated canonical `batch=18` config still failed the 30-second health check as under-band at about `8321 MiB`
      - next retry target is now `batch=19`
      - and automatically handed off into remote fast-eval
    - current authoritative read after stale-state cleanup:
      - remote GPU is idle again
      - canonical run root currently has no `epoch_*.pt`
      - synced remote packet now shows `train pid count = 0`, `fast-eval pid count = 0`
      - so `attn_pnp_selfinject` is currently `recalibration_needed`, not an active formal lane
    - latest calibration continuation:
      - segmented controller was extended to pass `min_runtime_slack_mib`
      - current active retry uses:
        - `batch=19`
        - `min_runtime_memory_mib=9000`
        - `min_runtime_slack_mib=256`
      - 30-second health read:
        - `8858 MiB`
      - current purpose:
        - determine whether this near-band lane can finish one bounded epoch, write a fresh retained checkpoint, and complete remote fast-eval handoff without crossing the hard cap
      - latest outcome:
        - the bounded segment advanced to about `739 / 994` steps, roughly `74%` of the epoch
        - then still died on persistent under-band:
          - `used=8858MiB floor=9000MiB elapsed=321s consecutive=3`
        - no fresh retained checkpoint landed
        - so this retry also remains `recalibration_needed`
      - infra follow-up already landed:
        - segmented control now skips remote fast-eval launch automatically if the bounded train segment exits without producing a new retained checkpoint
      - current nonformal continuation:
        - a new `batch=19 + slack256 + warn` detached calibration run is active
        - first launch was correctly deferred once because the remote host was externally occupied above the single-lane idle threshold
        - after auto-retry it launched and reached a live sampled state around:
          - `epoch 1/1`
          - `step 572 / 994`
          - `8752 MiB`
        - keep treating this as calibration evidence only, not a restored formal lane
        - but this continuation has now produced one real canonical checkpoint and fast-eval point:
          - `epoch_0001.pt`
          - transfer `0.6976 / 0.4750`
          - all-pairs `0.7181 / 0.4712`
          - wall `111.06s`
        - current follow-up:
          - an `epoch_0002` segmented continuation is now alive from canonical `epoch_0001.pt`
          - 30-second health read on that continuation:
            - `8903 MiB`
          - this lets the family build a second real curve point before deciding whether the nonformal line deserves further formal-band rescue
        - latest read:
          - the continuation is still alive
          - canonical run root still only has `epoch_0001.pt`
          - live memory remains in the narrow `~8.9G` zone
        - latest confirmed second point:
          - `epoch_0002.pt` landed and fast-eval completed
          - transfer moved to `0.6591 / 0.4656`
          - all-pairs moved to `0.6876 / 0.4585`
          - read:
            - LPIPS improved
            - style scores fell
          - current interpretation:
            - the family is still trending toward structure-preserving drift rather than a clean external-board win
        - third point update:
          - `epoch_0003.pt` landed and fast-eval completed
          - transfer moved to `0.6910 / 0.4544`
          - all-pairs moved to `0.7146 / 0.4491`
          - read:
            - style scores recovered strongly
            - LPIPS improved again
          - current interpretation:
            - the line is no longer monotonic style-collapse
            - it now merits one more canonical point before closure
        - current third-point continuation:
          - resumed from canonical `epoch_0002.pt`
          - now uses the corrected GiB-derived gate values together with `slack=512`
          - 30-second health read:
            - `8974 MiB`
          - current role:
            - determine whether the line keeps the same “style down / lpips up” drift at a third point or stabilizes
          - latest live read:
            - around `epoch 3`, `step 187 / 994`
            - sampled memory `9956 MiB`
            - on the corrected GiB-derived thresholds this sampled point is inside the requested formal band
        - current fourth-point continuation:
          - resumed from canonical `epoch_0003.pt`
          - live sampled memory is around `10097 MiB`
          - still in-band under the corrected GiB-derived thresholds
          - `epoch_0004.pt` has now landed
          - fourth fast-eval point is still pending at this read
        - current fifth-point continuation:
          - resumed from canonical `epoch_0004.pt`
          - 30-second health read:
            - `10016 MiB`
          - still in-band under the corrected GiB-derived thresholds
          - current landed-state read:
            - `epoch_0004.pt` is already present
            - but the fourth fast-eval point is still pending while the fifth segment is already running
        - fifth point update:
          - `epoch_0005.pt` landed and fast-eval completed
          - transfer moved to `0.6929 / 0.4534`
          - all-pairs moved to `0.7164 / 0.4476`
          - read:
            - style recovered slightly over `epoch_0004`
            - LPIPS softened slightly
          - current interpretation:
            - the line remains Pareto-active, so closure is still premature
        - current pulled five-point read:
          - best transfer style remains `epoch_0001`:
            - `0.6980 / 0.4747`
          - best transfer LPIPS is `epoch_0004`:
            - `0.6899 / 0.4504`
          - best all-pairs style remains `epoch_0001`:
            - `0.7194 / 0.4689`
          - current interpretation:
            - this family improved its low-LPIPS frontier materially
            - but it still reads as a recalibration line, not a promote-ready round-1 winner
        - current sixth-point continuation:
          - the `epoch_0006` controller is alive
          - initial launch retries are correctly waiting for the same-run fast-eval to release the remote GPU below the one-lane idle threshold
        - fourth point update:
          - `epoch_0004.pt` landed and fast-eval completed
          - transfer moved to `0.6899 / 0.4504`
          - all-pairs moved to `0.7140 / 0.4453`
          - read:
            - LPIPS improved again
            - style scores dipped only slightly from `epoch_0003`
          - current interpretation:
            - the line is still not a promote signal
            - but it still has active movement on the Pareto surface, so closure is premature

Closed family:

- `attn_sa_mod`
  - config:
    - [aaai2027_round1_attn_sa_mod_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_attn_sa_mod_seed42_b8a2.json)
  - remote run root:
    - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_attn_sa_mod_seed42_b8a2`
  - remote train log:
    - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_attn_sa_mod_seed42_b8a2_train.log`
  - remote fast-eval log:
    - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_attn_sa_mod_seed42_b8a2_fast_eval.log`
  - first health:
    - initial bootstrap line exposed two infra issues:
      - inherited `freeze_mode=injection_only`
      - too-light VRAM band
    - backbone family config generation was then tightened to:
      - `freeze_mode=attention_only`
      - formal `num_epochs = 24`
      - backbone `batch_size = 14`
    - the bootstrap and under-band attempts were stopped
    - the bootstrap remote run root was archived under:
      - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_attn_sa_mod_seed42_b8a2_bootstrap4e_20260610`
    - current authoritative formal run health sample: `9698 MiB / 12288 MiB`
  - note:
    - the first fast-eval watcher exited before the first retained checkpoint existed
    - `watch_round1_family_fast_eval.py` was then fixed to keep polling on empty-checkpoint cycles and relaunched
    - the currently authoritative live lane is the formal 24-epoch relaunch after the freeze-policy and VRAM-band fixes
    - future round-1 train launches now support launcher-level health-band enforcement for the requested VRAM range
    - the under-band formal attempt was cleaned before the current authoritative relaunch
    - fast all-ckpt `CLIP-S + LPIPS` is now being shifted off the remote `3060` to a local detached watcher so the training lane no longer shares remote GPU with eval
  - first retained fast-eval:
    - bootstrap-only evidence was already pulled locally
    - do not treat the old `epoch_0001 / epoch_0002` fast curve as formal round-1 closure evidence
  - current formal checkpoint pull observed locally:
    - through `epoch_0013.pt`
  - local fast-eval watcher:
    - root:
      - [round1_attn_sa_mod_fast_local](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_fast_local)
    - stdout:
      - [local_fast_eval.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_fast_local/local_fast_eval.stdout.log)
    - stderr:
      - [local_fast_eval.stderr.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_fast_local/local_fast_eval.stderr.log)
  - first authoritative formal local fast point:
    - `epoch_0001`
    - transfer `CLIP-S / LPIPS = 0.6955 / 0.4686`
    - wall total `= 58.06s`
    - this is materially faster than the earlier bootstrap fast-eval timing and now serves as the live all-ckpt trend surface
    - current convergence state:
      - `best_in_newest_2 = false`
      - `converged = false`
  - current formal fast screen through settled `epoch_0012`:
    - best transfer `CLIP-S` is still `epoch_0001`
    - best transfer `LPIPS` is currently `epoch_0008`:
      - `0.6920 / 0.4416`
    - best all-pairs `CLIP-S` is currently `epoch_0003`:
      - `0.7180 / 0.4509`
    - latest settled local fast point is `epoch_0012`:
      - transfer `0.6937 / 0.4516`
      - full `0.7169 / 0.4478`
    - `epoch_0013.pt` has already been pulled locally and is being evaluated now
    - this line is still not converged
  - local GPU concurrency rule:
    - local fast watcher and local deep-review jobs are now guarded by a shared local GPU lock
    - the lock now tracks the real child eval pid on Windows instead of only the parent wrapper pid
    - this is to prevent the exact `two local python jobs at once` failure
  - local GPU status after the fix:
    - the old pre-fix detached `IntroStyle` job has been cleared from the machine
    - the fast watcher was used to close the all-ckpt fast curve through `epoch_0024`
    - refreshed local deep review landed `IntroStyle + DINO` on `epoch_0001 / epoch_0008 / epoch_0003 / epoch_0024`
    - local GPU is now reserved for the WSL baseline repro
    - frozen `VLM` has been moved onto a local `CPU / network` detached chain
    - current frozen-snapshot launch:
      - `round1_attn_sa_mod_vlm_snapshot205_20260610.py`
      - target compare pairs:
        - `AttnSA_e08 vs Seedream vs SaMAM`
        - `AttnSA_e24 vs Seedream vs SaMAM`
      - stdout:
        - [round1_attn_sa_mod_vlm_snapshot205_20260610.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_vlm_snapshot205_20260610.stdout.log)
      - stderr:
        - [round1_attn_sa_mod_vlm_snapshot205_20260610.stderr.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_vlm_snapshot205_20260610.stderr.log)
      - partial-summary watcher:
        - [watch_vlm_snapshot_summaries.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_vlm_snapshot_summaries.py)
      - current frozen external board:
        - [round1_attn_sa_mod_vlm_snapshot205_board_20260610.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_vlm_snapshot205_board_20260610.md)
  - remote lane status:
    - formal `attn_sa_mod` training is complete through `epoch_0024`
    - family is now stage-closed as a reject under the current evidence stack
    - remaining frozen `VLM` rows are confirmatory only, not gating the queue any more
  - next-family launch status:
    - attempted `tok_a_dino_dict` remote launch after moving `attn_sa_mod` to `reviewing`
    - launcher refused because remote prelaunch GPU usage was already `8968 MiB`
    - this exceeds the single-lane idle threshold `7000 MiB`, so the refusal was correct
    - later repeated samples oscillated between about `5356 MiB` and `8968 MiB`
    - launcher-side prelaunch gating was tightened to the stricter effective ceiling `min(requested_prelaunch, max_runtime - min_runtime) = 2300 MiB`
    - current blocker for the next family is external remote GPU occupancy, not config readiness
  - queue order after the DINO deferral change:
    - the next non-DINO family is now `attn_gw_ot`
    - `tok_a/tok_b/tok_c/tok_d` no longer jump ahead of the mainline backbone / solver sweep

Recalibration-needed family:

- `attn_gw_ot`
  - current status:
    - `recalibration_needed`
  - why:
    - the stopped lane accumulated repeated `under_band` runtime samples
    - a stale concurrent remote training/eval lane was also found on the same `3060`
    - that means the run is useful for directional signal only, not formal paper-facing evidence
  - retained evidence:
    - [remote_run.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/attn_gw_ot/remote_run.md)
    - [closure.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/attn_gw_ot/closure.md)
    - [decision.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/attn_gw_ot/decision.md)

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Active Status

- Running families:
  - `solver_tangent_rk`
- Active family: `solver_tangent_rk`
- Decision status: `running`
- Batch / epochs / patience: `16 / 24 / 6`
- Remote GPU live: `9349 / 12288 MiB`, `util=89%`, `band=in_band`
- Best transfer `CLIP-S`: `epoch_0001` -> `0.6999 / 0.5295`
- Best transfer `LPIPS`: `epoch_0019` -> `0.6909 / 0.4498`
- Best all-pairs `CLIP-S`: `epoch_0007` -> `0.7159 / 0.4675`
- Latest settled fast point: `epoch_0023` -> transfer `0.6906 / 0.5189`
- Convergence: `row_count=23, since_best=22, tail_flat=False, closure_band=open, converged=False`
<!-- ROUND1_AUTO_STATUS:END -->













































































































































































































































































































































































































































































































































































