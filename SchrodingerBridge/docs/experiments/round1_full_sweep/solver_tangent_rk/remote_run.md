# solver_tangent_rk Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_solver_tangent_rk_seed42_b8a2`
- First formal launch read on `2026-06-10`:
  - opening batch `8`
  - 30-second health sample:
    - `5463 MiB`
  - interpretation:
    - this is far below the requested `9.0-10.8 GiB` formal band
    - so `batch=8` is an obvious under-band opening and should not remain the canonical launch setting
- Recalibration action:
  - default batch was first raised to `14`
  - second formal launch read on `2026-06-10`:
    - health sample:
      - `8534 MiB`
    - interpretation:
      - still below the corrected `9.0 GiB` floor
      - so `batch=14` is also under-band
  - next recalibration target:
    - `batch=15`
- Follow-up recalibration read on `2026-06-10`:
  - `batch=15`
  - 30-second health sample:
    - `9067 MiB`
  - interpretation:
    - this is only `21 MiB` below the current effective launch floor `9088 MiB`
    - so the next recalibration target should move one notch to `batch=16` rather than widening the formal band
- Current formal launch read on `2026-06-10`:
  - `batch=16`
  - 30-second health sample:
    - `9432 MiB`
  - interpretation:
    - this is inside the corrected `9.0-10.8 GiB` formal band
    - so `batch=16` is the first solver-tangent opening that satisfies the remote VRAM lane rule
- Current live lane:
  - remote train is alive under the same `batch=16` setting
  - same-run remote fast-eval watcher has already been launched and is waiting for the first retained checkpoint to settle
  - at this read there is still no pulled `epoch_0001` fast point locally yet
- First pulled fast-eval point:
  - `epoch_0001`
  - transfer `CLIP-S / LPIPS = 0.6999 / 0.5295`
  - all-pairs `CLIP-S / LPIPS = 0.7142 / 0.5222`
  - wall `= 186.74s`
  - immediate read:
    - style score is competitive with the current internal range
    - but LPIPS is materially weak
    - so this line is alive and formally in-band, but still far from any promote decision
- Current live continuation:
  - the remote lane has now produced and settled:
    - `epoch_0002.pt`
  - second pulled fast-eval point:
    - `epoch_0002`
    - transfer `CLIP-S / LPIPS = 0.6969 / 0.5230`
    - all-pairs `CLIP-S / LPIPS = 0.7129 / 0.5136`
    - wall `= 191.73s`
  - immediate read:
    - style scores softened slightly versus `epoch_0001`
    - LPIPS improved slightly
    - both points remain Pareto-active
    - so the line is alive, unconverged, and still needs more retained checkpoints
- Current live continuation:
  - remote training has already advanced into:
    - `epoch 4/24`
  - current live sampled state:
    - `9443 MiB / 12288 MiB`
    - `band_status=in_band`
    - `formal_status=formal_in_band`
    - `step 58/1180`
    - `loss=7.4893`
    - `tswd=5.1875`
  - remote fast-eval lag is currently healthy rather than broken:
    - `epoch_0003.pt` has now fully settled
    - third pulled fast-eval point:
      - `epoch_0003`
      - transfer `CLIP-S / LPIPS = 0.6948 / 0.5286`
      - all-pairs `CLIP-S / LPIPS = 0.7107 / 0.5190`
      - wall `= 189.16s`
    - immediate read:
      - this point is worse than `epoch_0002` on both transfer style and transfer LPIPS
      - so the line has not improved for one retained checkpoint
      - but with `patience=6` and `best_in_newest_2=true`, closure is still premature
- Fourth pulled fast-eval point:
  - `epoch_0004`
  - transfer `CLIP-S / LPIPS = 0.6944 / 0.4998`
  - all-pairs `CLIP-S / LPIPS = 0.7143 / 0.4886`
  - wall `= 178.91s`
  - immediate read:
    - transfer style slipped slightly again
    - but LPIPS improved sharply
    - all-pairs style also recovered to the best point so far
    - this point is back on the joint Pareto frontier, so the line clearly remains open
- Fifth pulled fast-eval point:
  - `epoch_0005`
  - transfer `CLIP-S / LPIPS = 0.6893 / 0.5371`
  - all-pairs `CLIP-S / LPIPS = 0.7052 / 0.5271`
  - wall `= 179.66s`
  - immediate read:
    - both transfer style and LPIPS regressed versus `epoch_0004`
    - all-pairs style also fell back materially
    - so `epoch_0004` remains the current best structure-preserving fast point
    - but this is only one later non-improving checkpoint under `patience=6`, so closure is still too early
- Sixth and seventh-point recovery:
  - `epoch_0006`
    - transfer `0.6929 / 0.5121`
    - all-pairs `0.7114 / 0.5017`
  - `epoch_0007`
    - transfer `0.6951 / 0.4787`
    - all-pairs `0.7159 / 0.4675`
  - read:
    - `epoch_0007` overtook the old `epoch_0004` frontier
    - so the mid-run rollback was not closure evidence
- Eighth pulled fast-eval point:
  - `epoch_0008`
  - transfer `CLIP-S / LPIPS = 0.6901 / 0.5243`
  - all-pairs `CLIP-S / LPIPS = 0.7082 / 0.5130`
  - wall `= 178.39s`
  - immediate read:
    - this point rolled back materially from `epoch_0007`
    - but `epoch_0007` still sits inside the newest 2 settled checkpoints
    - therefore the solver-family stop rule is still not satisfied

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `solver_tangent_rk`
- Run name: `aaai2027_round1_solver_tangent_rk_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_solver_tangent_rk_seed42_b8a2`
- Config: [aaai2027_round1_solver_tangent_rk_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_solver_tangent_rk_seed42_b8a2.json)
- Manifest status: `running`
- Local fast root: [round1_solver_tangent_rk_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_tangent_rk_fast_local)
- Local review root: [round1_solver_tangent_rk_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_tangent_rk_localreview)
- Prelaunch switch smoke: `ok`
- Switch smoke artifact: [round1_family_switch_smoke_20260610.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_family_switch_smoke_20260610.json)
- Switch smoke row count: `11`
- Remote GPU live sample:
  - `9516 MiB / 12288 MiB`, `util=86%`
  - `band_status=in_band`
  - `formal_status=formal_in_band`
- Remote train log: `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round1_solver_tangent_rk_seed42_b8a2_train.log`
- Remote train progress:
  - `epoch 10/24`
  - `step 279/1180`
  - `loss=7.6947`
  - `tswd=5.9688`
<!-- ROUND1_AUTO_STATUS:END -->













































