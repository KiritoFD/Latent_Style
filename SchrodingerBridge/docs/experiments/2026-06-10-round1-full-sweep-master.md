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

Dataset note:

- larger five-style train pool:
  - [2026-06-10-wikiarts5-full-notest.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-wikiarts5-full-notest.md)
- local WSL baseline repro:
  - [2026-06-10-wikiarts5-baseline-repro.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-wikiarts5-baseline-repro.md)
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
- local GPU:
  - WSL `SaMAM patch8` segmented repro with `250-step` `CLIP-S + LPIPS` checkpoints
  - plus a detached convergence watcher that can stop the run once the curve really flattens
- generic queue now defaults to:
  - `remote train + local detached fast watcher`
  - current safe override while baseline repro is active:
    - `--skip-fast-eval-launch`
- tokenizer + DINO policy for round 1:
  - DINO-supervised families stay in tokenizer-only update mode first
  - backbone remains frozen through `freeze_mode=style_branch`
  - if direct tokenizer-family training is still unstable after cache alignment is fixed, try a tokenizer-only DINO warm-start / pretrain stage before reopening the full family lane
  - remote main-lane priority is now:
    - non-DINO families first
    - DINO-related tokenizer families explicitly moved to the tail of the round-1 queue

Required closure per family:

1. converged remote training
2. all-ckpt `CLIP-S + LPIPS` fast curve
3. shortlisted `IntroStyle + DINO`
4. frozen-snapshot `VLM`
5. closure note
6. decision note

Current active remote lane:

- `attn_gw_ot`
  - config:
    - [aaai2027_round1_attn_gw_ot_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_attn_gw_ot_seed42_b8a2.json)
  - remote train log:
    - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_attn_gw_ot_seed42_b8a2_train.log`
  - data root:
    - `/mnt/i/wikiarts_5_full_notest_latents_ema/train`
  - launch notes:
    - first new-data formal attempt at `batch_size = 13` was correctly rejected as under-band at `7921 MiB`
    - second attempt at `batch_size = 15` hit `8992 MiB`, only `8 MiB` below the requested floor
    - host-side launcher now applies `128 MiB` slack only to the minimum-band check to ignore integer-MiB health noise; the `11.0 GiB` hard cap remains strict
    - the first accepted `batch_size = 15` lane later drifted to `11979 MiB / 12288 MiB`
    - that over-cap lane was stopped immediately and is not formal evidence
    - remote launch infra now includes a continuous runtime VRAM guard inside the generated WSL launcher
    - current authoritative relaunch uses `batch_size = 12`
    - current authoritative health sample: `10712 MiB / 12288 MiB`
    - fast-eval watcher launch was intentionally skipped because local GPU is reserved for the WSL baseline repro
    - a deferred local fast-eval launcher is now armed:
      - [launch_local_round1_fast_eval_after_wsl_idle.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_local_round1_fast_eval_after_wsl_idle.py)
      - it waits for `SaMAM` convergence plus WSL process exit, then auto-starts the `attn_gw_ot` local fast watcher
    - a second deferred stage-close launcher is also armed:
      - [run_round1_family_stageclose_when_ready.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_round1_family_stageclose_when_ready.py)
      - it waits for the local `attn_gw_ot` fast curve to converge, then runs bestfew rerun/review and external-baseline `VLM`

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

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Active Status

- Running families:
  - `attn_gw_ot`
- Active family: `attn_gw_ot`
- Decision status: `running`
- Batch / epochs / patience: `12 / 24 / 4`
<!-- ROUND1_AUTO_STATUS:END -->
































