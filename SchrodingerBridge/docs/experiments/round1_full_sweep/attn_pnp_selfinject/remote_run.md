# attn_pnp_selfinject Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_attn_pnp_selfinject_seed42_b8a2`
- Launch read on `2026-06-10`:
  - opening batch `22`
  - 30-second health check sample:
    - about `9979 MiB`
    - correctly inside the requested formal band
  - retained `epoch_0001.pt` landed
  - remote fast-eval watcher immediately started and is evaluating `epoch_0001`
- Failure read on `2026-06-10`:
  - during `epoch 2`, the remote runtime guard logged:
    - `used=11939MiB`
    - `cap=11000MiB`
  - the train lane was then killed with `rc=143`
  - this run therefore exceeded the hard paper-facing cap and cannot remain `running`
  - next recalibration target:
    - lower opening batch to `20`
- Relaunch read on `2026-06-10`:
  - the overshoot root was archived under:
    - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_attn_pnp_selfinject_seed42_b8a2_b22_overshoot_20260610_1825`
  - the fresh canonical run has now been relaunched at `batch_size = 20`
  - new 30-second health sample:
    - about `9206 MiB`
    - back inside the requested formal band
  - remote train and remote fast-eval watcher are both alive again
- second failure read on `2026-06-10`:
  - even after reducing remote fast-eval to:
    - `batch=1`
    - `vae_decode_batch_size=4`
    - `target_chunk_size=1`
  - the train log still ended with:
    - `RUNTIME_GUARD 2026-06-10T18:52:53+08:00 used=11016MiB cap=11000MiB`
  - direct remote process inspection after that stop showed:
    - no matching train process alive
    - only the remote fast-eval processes remained
  - conclusion:
    - this family now needs non-concurrent segmented train/eval orchestration on the remote `3060`
- gate-calibration correction:
  - the earlier `11000MiB` runtime guard was stricter than the stated experiment rule
  - the actual user-facing spec is:
    - preferred band `9.0-10.8 GiB`
    - hard cap `11.3 GiB`
  - exact binary-unit conversions are approximately:
    - `9.0 GiB = 9216 MiB`
    - `10.8 GiB = 11059 MiB`
    - `11.3 GiB = 11571 MiB`
  - implication for historical reads:
    - the older `used=11016MiB cap=11000MiB` stop was above the old conservative code cap
    - but still below the actual `10.8 GiB` soft ceiling
- segmented orchestration launch on `2026-06-10`:
  - detached controller:
    - [run_remote_round1_family_segmented.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_remote_round1_family_segmented.py)
  - current invocation:
    - `--family-id attn_pnp_selfinject --segment-epochs 1`
  - detached logs:
    - [round1_attn_pnp_selfinject_segmented_20260610.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_pnp_selfinject_segmented_20260610.stdout.log)
    - [round1_attn_pnp_selfinject_segmented_20260610.stderr.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_pnp_selfinject_segmented_20260610.stderr.log)
- first segmented cycle read:
  - controller first waited because remote prelaunch memory was still above the single-lane idle threshold
  - after remote GPU usage dropped, it successfully launched a bounded `1 epoch` train segment
  - the segment passed the 30-second health check in-band at about `10547 MiB`
  - after train exit, the controller automatically ran:
    - remote scalar sync
    - remote fast-eval launch
  - current state after that handoff:
    - no remote train pid alive
    - only the remote fast-eval watcher remains active
  - postmortem:
    - the bounded `batch=20` train segment itself still ended with a runtime-guard stop
    - a later segmented launch using the updated canonical `batch=18` config then failed the 30-second health check as under-band at about `8321 MiB`
    - therefore the next relaunch target is now `batch=19`
- current authoritative read after stale-sync cleanup:
  - remote GPU is idle again at about `537 MiB`
  - canonical run root currently has:
    - no `epoch_*.pt`
    - no live train pid
    - no live fast-eval pid
  - local sync packet now correctly reports:
    - `waiting_for_first_remote_fast_eval_epoch`
    - with remote `train=0`, `fast_eval=0`
  - so this family must remain `recalibration_needed`, not `running`
- current live recalibration retry:
  - launcher path was extended so segmented control can pass:
    - `min_runtime_slack_mib`
  - active detached controller:
    - [round1_attn_pnp_selfinject_segmented_b19slack256_retry_20260610_201746.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_pnp_selfinject_segmented_b19slack256_retry_20260610_201746.stdout.log)
  - active launch parameters:
    - `batch=19`
    - `min_runtime_memory_mib=9000`
    - `min_runtime_slack_mib=256`
  - 30-second health read:
    - `health_gpu_memory_used_mib = 8858`
  - immediate interpretation:
    - this crossed the current effective health floor and the segment continued into the bounded train wait phase
    - it is still a calibration run until it actually writes a fresh retained checkpoint and hands off to remote fast-eval successfully
  - final read for this retry:
    - the bounded segment later reached about `739 / 994` steps
    - then the runtime guard logged:
      - `RUNTIME_UNDER_BAND_STOP ... used=8858MiB floor=9000MiB elapsed=321s consecutive=3`
    - train exited with `rc=143`
    - controller correctly advanced into:
      - remote scalar sync
      - remote fast-eval launch
    - but the canonical run root still had:
      - no `epoch_*.pt`
      - therefore no actual fast-eval epoch could start
    - the stale empty-wait fast-eval watcher has since been cleaned
  - infra fix added after this read:
    - [run_remote_round1_family_segmented.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_remote_round1_family_segmented.py)
    - segmented control now re-scans for a newly landed retained checkpoint after train exit
    - if no new checkpoint exists, it skips remote fast-eval launch for that cycle instead of spawning an empty watcher
- active nonformal calibration after the stop-mode failure:
  - detached controller:
    - [round1_attn_pnp_selfinject_segmented_b19slack256warn_20260610_203251.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_pnp_selfinject_segmented_b19slack256warn_20260610_203251.stdout.log)
  - launch policy:
    - `batch=19`
    - `min_runtime_slack_mib=256`
    - `runtime_guard_min_mode=warn`
  - prelaunch behavior:
    - first retry was correctly deferred because remote idle memory was temporarily about `9342 MiB`, above the single-lane prelaunch threshold
    - the same controller then auto-retried and launched once the host dropped back near idle
  - current live read:
    - 30-second health sample:
      - `8752 MiB`
    - mid-run sampled state:
      - about `epoch 1/1`
      - about `step 572 / 994`
      - loss about `9.3182`
      - `tswd` about `4.8125`
  - status meaning:
    - this is intentionally being treated as `nonformal_under_band`
    - the purpose is to see whether the family can at least produce a complete bounded-segment checkpoint/eval read when under-band stop is relaxed to warn
  - latest outcome:
    - this warn-policy segment did produce:
      - canonical retained checkpoint `epoch_0001.pt`
      - remote fast-eval `epoch_0001`
    - first remote fast-eval read:
      - transfer `CLIP-S / LPIPS = 0.6976 / 0.4750`
      - all-pairs `CLIP-S / LPIPS = 0.7181 / 0.4712`
      - wall `= 111.06s`
    - so the family now has a real canonical trend point under the calibration path even though it is still not a paper-facing formal lane
  - current `epoch_0002` continuation:
    - detached controller:
      - [round1_attn_pnp_selfinject_segmented_b19slack256warn_e2retry_20260610_210225.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_pnp_selfinject_segmented_b19slack256warn_e2retry_20260610_210225.stdout.log)
    - key recovery fix:
      - segmented continuation now resumes model / epoch state from `epoch_0001.pt`
      - but no longer requires optimizer state to match exactly
    - current live read:
      - resumed at `epoch=2`, `global_step=497`
      - 30-second health sample:
        - `8903 MiB`
      - train is currently alive in `Epoch 2/2`
    - interpretation:
      - the family now has real continuing canonical training beyond the first point
      - this is still calibration, not a formal promoted lane
  - latest live read:
    - `epoch_0002` segment is still alive
    - canonical run root still shows:
      - only `epoch_0001.pt`
    - current sampled GPU memory is around:
      - `8886-8903 MiB`
    - so the run is still living in the narrow sub-band zone while trying to push a second real curve point
  - second-point outcome:
    - canonical run root now contains:
      - `epoch_0001.pt`
      - `epoch_0002.pt`
    - fast-eval also now contains:
      - `epoch_0001`
      - `epoch_0002`
    - `epoch_0002` fast read:
      - transfer `CLIP-S / LPIPS = 0.6591 / 0.4656`
      - all-pairs `CLIP-S / LPIPS = 0.6876 / 0.4585`
      - wall `= 171.56s`
    - direct trend read from `epoch_0001 -> epoch_0002`:
      - style scores fell
      - LPIPS improved
      - this looks like structure-preserving drift rather than a clear style-transfer win
  - third-point outcome:
    - canonical run root now also contains:
      - `epoch_0003.pt`
    - `epoch_0003` fast read:
      - transfer `CLIP-S / LPIPS = 0.6910 / 0.4544`
      - all-pairs `CLIP-S / LPIPS = 0.7146 / 0.4491`
      - wall `= 158.09s`
    - trend read from `epoch_0002 -> epoch_0003`:
      - style scores recovered strongly
      - LPIPS improved again
      - this changed the line from a simple structure-drift narrative into a potentially recoverable path
  - infra note:
    - duplicate remote fast-eval watcher instances were observed during the `epoch_0002` handoff
    - [launch_remote_round1_family_fast_eval.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_family_fast_eval.py)
    - now skips launch if a same-run fast-eval process is already alive, so future segmented cycles stay single-instance
  - current `epoch_0003` continuation:
    - detached controller:
      - [round1_attn_pnp_selfinject_segmented_b19slack512warn_e3_20260610_212658.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_pnp_selfinject_segmented_b19slack512warn_e3_20260610_212658.stdout.log)
    - launch-time gate calibration now uses:
      - `9.0 GiB -> 9216 MiB`
      - `10.8 GiB -> 11059 MiB`
      - `11.3 GiB -> 11571 MiB`
    - active nonformal continuation parameters:
      - `batch=19`
      - `min_runtime_slack_mib=512`
      - `runtime_guard_min_mode=warn`
    - current live read:
      - resumed from canonical `epoch_0002.pt`
      - resumed at `epoch=3`, `global_step=994`
      - 30-second health sample:
        - `8974 MiB`
      - train is currently alive in `Epoch 3/3`
    - latest live sample:
      - about `epoch 3`
      - about `step 187 / 994`
      - loss about `8.1171`
      - `tswd` about `5.5312`
      - sampled GPU memory about `9956 MiB`
    - read:
      - under the corrected GiB-derived thresholds, this sampled point is now inside the requested formal band
  - current `epoch_0004` continuation:
    - detached controller:
      - [round1_attn_pnp_selfinject_segmented_b19slack512warn_e4_20260610_214343.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_pnp_selfinject_segmented_b19slack512warn_e4_20260610_214343.stdout.log)
    - current read:
      - `epoch_0003` fast-eval is finished
      - `epoch_0004` bounded segment is now alive
      - current sampled GPU memory is around `10097 MiB`
    - interpretation:
      - the line is still inside the corrected formal band during this sampled point
      - one more canonical point is being collected to test whether the `epoch_0003` recovery is stable
  - current `epoch_0005` continuation:
    - detached controller:
      - [round1_attn_pnp_selfinject_segmented_b19slack512warn_e5_20260610_220341.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_pnp_selfinject_segmented_b19slack512warn_e5_20260610_220341.stdout.log)
    - current read:
      - resumed from canonical `epoch_0004.pt`
      - resumed at `epoch=5`, `global_step=1988`
      - 30-second health sample:
        - `10016 MiB`
      - train is currently alive in `Epoch 5/5`
    - read:
      - under the corrected GiB-derived thresholds this sampled point is still in-band
    - current landed-state read:
      - `epoch_0004.pt` and its fast-eval point are already settled
      - so the line is already training the next canonical segment from the updated four-point frontier
    - completed fifth-point read:
      - `epoch_0005.pt` is now present
      - `epoch_0005` fast-eval is also now settled
      - transfer `CLIP-S / LPIPS = 0.6929 / 0.4534`
      - all-pairs `CLIP-S / LPIPS = 0.7164 / 0.4476`
      - wall `= 201.54s`
    - direct trend read from `epoch_0004 -> epoch_0005`:
      - style scores recovered slightly
      - LPIPS softened slightly
      - the line remained on the Pareto surface
  - current `epoch_0006` continuation:
    - detached controller:
      - [round1_attn_pnp_selfinject_segmented_b19slack512warn_e6_20260610_222434.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_pnp_selfinject_segmented_b19slack512warn_e6_20260610_222434.stdout.log)
    - current read:
      - launch retries are being deferred because same-run fast-eval still occupies the remote GPU above the single-lane idle threshold
      - sampled prelaunch memory during these retries is around `3042-3050 MiB`
      - this is expected serialization behavior, not a new train failure
    - current later read:
      - `epoch_0005.pt` and its fast-eval point are now both settled
      - same-run fast-eval watcher is still the only active remote process family state
      - the sixth controller will keep retrying until that watcher releases the remote GPU below the one-lane idle threshold
  - fourth-point outcome:
    - canonical run root now also contains:
      - `epoch_0004.pt`
    - `epoch_0004` fast read:
      - transfer `CLIP-S / LPIPS = 0.6899 / 0.4504`
      - all-pairs `CLIP-S / LPIPS = 0.7140 / 0.4453`
      - wall `= 161.98s`
    - direct trend read from `epoch_0003 -> epoch_0004`:
      - style scores softened slightly
      - LPIPS improved again
      - this keeps the line on a structure-improving frontier while no longer collapsing style the way `epoch_0002` did

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `attn_pnp_selfinject`
- Run name: `aaai2027_round1_attn_pnp_selfinject_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_attn_pnp_selfinject_seed42_b8a2`
- Config: [aaai2027_round1_attn_pnp_selfinject_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_attn_pnp_selfinject_seed42_b8a2.json)
- Manifest status: `recalibration_needed`
- Local fast root: [round1_attn_pnp_selfinject_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_pnp_selfinject_fast_local)
- Local review root: [round1_attn_pnp_selfinject_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_pnp_selfinject_localreview)
- Prelaunch switch smoke: `ok`
- Switch smoke artifact: [round1_family_switch_smoke_20260610.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_family_switch_smoke_20260610.json)
- Switch smoke row count: `11`
- Remote train log: `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_attn_pnp_selfinject_seed42_b8a2_train.log`
- Remote train pid: not alive
<!-- ROUND1_AUTO_STATUS:END -->






























































































































