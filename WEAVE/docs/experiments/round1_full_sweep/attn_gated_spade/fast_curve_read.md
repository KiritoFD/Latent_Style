# attn_gated_spade Fast Curve Read

- Curve CSV: `clip_lpips_curve.csv`
- authority correction on `2026-06-10`:
  - the earlier `local deferred fast-eval only` setup was wrong
  - this family now uses remote-side fast `CLIP-S / LPIPS` during the run
  - current pulled remote fast root:
    - [round1_attn_gated_spade_remote_full_eval_pull](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_gated_spade_remote_full_eval_pull)
  - current pulled curve CSV:
    - [clip_lpips_curve.csv](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_gated_spade_remote_full_eval_pull/clip_lpips_curve.csv)
- current settled remote fast points:
  - `epoch_0001`
    - transfer `CLIP-S / LPIPS = 0.6929 / 0.4501`
    - all-pairs `CLIP-S / LPIPS = 0.7158 / 0.4464`
    - wall `= 96.43s`
  - `epoch_0002`
    - transfer `0.6917 / 0.4354`
    - all-pairs `0.7155 / 0.4319`
    - wall `= 92.51s`
  - `epoch_0003`
    - transfer `0.6909 / 0.4360`
    - all-pairs `0.7148 / 0.4326`
    - wall `= 93.03s`
  - `epoch_0004`
    - transfer `0.6917 / 0.4333`
    - all-pairs `0.7156 / 0.4298`
    - wall `= 92.73s`
  - `epoch_0005`
    - transfer `0.6916 / 0.4355`
    - all-pairs `0.7155 / 0.4319`
    - wall `= 93.99s`
  - `epoch_0006`
    - transfer `0.6903 / 0.4401`
    - all-pairs `0.7141 / 0.4363`
    - wall `= 92.92s`
- current early read:
  - structure improved through `epoch_0004`, then partially gave back at `epoch_0005/0006`
  - style is roughly flat to slightly down over `epoch_0001 -> epoch_0006`
  - the best transfer LPIPS so far is still `epoch_0004`
  - this is still a `style-holding / lpips-improving first, then wobbling` early drift, not a style breakout yet
  - current remote snapshot is still materially below the internal `attn_sa_mod` best-transfer opening point on transfer `CLIP-S`
  - but it is already competitive on LPIPS versus that family's early epochs, so the line is still live

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Authority root:
  - [round1_attn_gated_spade_remote_full_eval_pull](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_gated_spade_remote_full_eval_pull)
- Pulled curve CSV:
  - [clip_lpips_curve.csv](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_gated_spade_remote_full_eval_pull/clip_lpips_curve.csv)
- Settled pulled rows:
  - `22`
- Settled remote epochs:
  - `epoch_0001, epoch_0002, epoch_0003, epoch_0004, epoch_0005, epoch_0006, epoch_0007, epoch_0008, epoch_0009, epoch_0010, epoch_0011, epoch_0012, epoch_0013, epoch_0014, epoch_0015, epoch_0016, epoch_0017, epoch_0018, epoch_0019, epoch_0020, epoch_0021, epoch_0022`
- Best transfer `CLIP-S`:
  - `epoch_0001`
  - `style / lpips = 0.6929 / 0.4501`
- Best transfer `LPIPS`:
  - `epoch_0022`
  - `style / lpips = 0.6910 / 0.4252`
- Best all-pairs `CLIP-S`:
  - `epoch_0011`
  - `style / lpips = 0.7172 / 0.4220`
- Latest settled point:
  - `epoch_0022`
  - transfer `style / lpips = 0.6910 / 0.4252`
  - full `style / lpips = 0.7156 / 0.4220`
  - wall `= 91.85s`
- Convergence snapshot:
  - `best_epoch = epoch_0001`
  - `since_last_pareto = 0`
  - `best_in_newest_2 = True`
  - `tail_flat = True`
  - `closure_band = open`
  - `criterion = joint_transfer_allpairs_pareto`
  - `converged = False`
- Sync summary:
  - [sync_summary.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_gated_spade_remote_full_eval_pull/sync_summary.json)
<!-- ROUND1_AUTO_STATUS:END -->





























































































































































































































