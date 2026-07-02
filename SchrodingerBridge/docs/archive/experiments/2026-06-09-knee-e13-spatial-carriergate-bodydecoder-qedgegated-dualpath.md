# `LBM-Knee e13 + Spatial Carrier-Gate Body+Decoder + Quantile Edge-Gated Structure Leash + DualPathTexture Proximal`

Date: 2026-06-09

Why this packet exists:

- `QEdgePattn` already proved the direction is better than `LBM-Knee`
- but it also proved the current late style branch is still too weak
- the next round therefore has to be a branch-capacity round, not another schedule round

Mechanism:

- keep:
  - `Knee e13`
  - `spatial_carrier_gate`
  - `body_decoder`
  - `quantile_edge_gated_anisotropic_plus_stokes`
- replace the single late branch with:
  - `proximal_mode = dualpath_texture`

Dual-path idea:

- coarse late style branch:
  - `NormFreeModulation + conv head`
  - low-pass constrained
- texture late style branch:
  - cross-attention texture residual
  - high-pass constrained
- final residual:
  - `coarse_low * proximal_coarse_gain`
  - `+ texture_high * proximal_texture_gain`

Config:

- [inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_seed42_b8a2.json)

Launcher:

- [launch_remote_knee_spatial_carriergate_bodydecoder_qedgegated_dualpath_train.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_knee_spatial_carriergate_bodydecoder_qedgegated_dualpath_train.py)

## Launch state

Current launch read:

- remote task started successfully
- current remote run log:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/knee_spatial_carriergate_bodydecoder_qedgegated_dualpath_train.log`
- first-health passed
- matching post-train image-backed watcher is also now launched:
  - `knee-spatial-carriergate-bodydecoder-qedgegated-dualpath-fresh-eval-watcher`

Measured first-health facts:

- remote launcher pid alive
- training log is growing
- dataset and pairing cache loaded cleanly
- partial resume loaded from the earlier `AnisoStokesQueue` parent checkpoint
- current freeze mode:
  - `injection_only`
- current health GPU memory:
  - `8133 MiB`
- current live runtime check:
  - train process alive
  - watcher process alive
  - combined live GPU memory still around `8142 MiB`
  - latest checked train progress:
    - `epoch_0001.pt`
    - `epoch_0002.pt`
    - `epoch_0003.pt`
    - `epoch_0004.pt`
    - `epoch_0005.pt`
    - `epoch_0006.pt`
    - `epoch_0007.pt`
    - `epoch_0008.pt`
    - `epoch_0009.pt`
    - `epoch_0010.pt`
    - `epoch_0011.pt`
    - current live step is already beyond `epoch 11` launch stage
  - latest logged epoch summaries:
    - `epoch 1 loss = 8.4354`
    - `epoch 2 loss = 8.4194`
    - `epoch 3 loss = 8.2766`
    - `epoch 4 loss = 8.2084`
    - `epoch 5 loss = 8.3885`
    - `epoch 6 loss = 8.4321`
    - `epoch 7 loss = 8.3132`
    - `epoch 8 loss = 8.2275`
    - `epoch 9 loss = 8.2772`
    - `epoch 10 loss = 8.3743`
    - `epoch 11 loss = 8.2241`
    - `epoch 9 loss = 8.2772`
    - `epoch 10 loss = 8.3743`

Interpretation:

- this is comfortably below the formal `< 11.0 GiB` cap
- so the new branch-capacity line is formally machine-safe on the reviewed remote surface
- the remote post-train closure path is now already armed, so this line should not need another manual rescue after training exits
- the line is now beyond launch-only status:
  - it is a real live training lane with retained checkpoints already forming

Current early train read:

- `flow` has already improved:
  - `0.8279 -> 0.7975`
- `terminal_swd` improved, then oscillated:
  - `5.8750` at `epoch 5`
  - `6.3438` at `epoch 6`
  - `5.9688` at `epoch 9`
  - `6.1875` at `epoch 10`
  - `6.1563` at `epoch 11`
- `proximal_residual_abs` is still tiny:
  - about `0.0083`

Current interpretation:

- nothing is exploding numerically
- the new dual-path branch is not immediately bypassing the transport field with a huge residual
- the post-epoch-5 rebound is not permanent
- but the line is still oscillatory rather than clearly settled
- this is enough to keep running, but not enough yet to claim it has beaten `QEdgePattn`

Remote-to-local closure path is also now prepared:

- remote best-few pull:
  - [pull_remote_dualpath_bestfew_localreview.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/pull_remote_dualpath_bestfew_localreview.py)
- local best-few review:
  - [run_local_dualpath_bestfew_review.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_local_dualpath_bestfew_review.py)

This means the current branch-capacity round is now operationally complete:

- train launcher exists
- post-train watcher exists
- local best-few pull exists
- local `IntroStyle / DINO` review entry exists
- local review-state note now also exists:
  - [2026-06-09-dualpath-localreview-progress.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-dualpath-localreview-progress.md)

## First eval-side read

The first completed training-side `full_eval` points are now available locally:

- [dualpath_full_eval_curve_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_full_eval_curve_20260609.csv)
- [full_eval_bestfew_handoff.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_full_eval_progress_20260609/full_eval_bestfew_handoff.csv)
- current local progress root:
  - [dualpath_full_eval_progress_20260609](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_full_eval_progress_20260609)

Current completed points:

- `epoch_0001 = 0.6927 / 0.4037`
- `epoch_0002 = 0.6919 / 0.4218`
- `epoch_0003 = 0.6918 / 0.4306`
- `epoch_0004 = 0.6921 / 0.4348`
- `epoch_0005 = 0.6917 / 0.4348`
- `epoch_0006 = 0.6922 / 0.4377`

Current local handoff read:

- best transfer `LPIPS`
  - `epoch_0001`
- current latest pulled point
  - `epoch_0005`

Important distinction:

- the current scalar comparison above still comes from training-side `full_eval`
- but the first image-backed local packet has now also started to exist under:
  - `full_eval_fresh_localreview`

Current image-backed local state:

- local image-backed packet already present for:
  - `epoch_0001`
  - `epoch_0004`
  - `epoch_0009`
- current best-style handoff row now points to:
  - `epoch_0009`
- the first image-backed local CPU-only `VLM` triplet has now started for:
  - `DualPathFresh_e01 vs Knee vs Seedream`
- current dualpath local-review state note:
  - [2026-06-09-dualpath-localreview-progress.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-dualpath-localreview-progress.md)
- current compact staging board:
  - [dualpath_vlm_triplets_compare_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_vlm_triplets_compare_20260609.md)
- first local non-CLIP read from that packet is now available:
  - `DualPathFresh_e01` already records sparse real wins over `LBM-Knee`
  - and has now moved into a nontrivial local batch:
    - `141` completed cases
- the next image-backed local point has also been seeded into the same CPU-only VLM path:
  - `DualPathFresh_e09`
  - and that point now also has a first local non-CLIP summary
- the next local image-backed point has also been seeded into the same CPU-only VLM path:
  - `DualPath_e04`

Immediate comparison versus the earlier `QEdgePattn` family:

- `QEdgePattn epoch_0001` was about:
  - `0.7048 / 0.4521`
- `DualPath epoch_0001` is therefore:
  - clearly lower on cheap style
  - clearly better on `LPIPS`
- compact epoch-matched comparison table:
  - [dualpath_vs_qedgepattn_early_curve_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_vs_qedgepattn_early_curve_20260609.md)

Interpretation:

- the new branch-capacity family still has not reopened style strongly in the first completed eval-side points
- but it does look more geometry-conservative than `QEdgePattn`
- and the first three points are consistent with the same geometry-conservative basin:
  - style stays roughly flat
  - `LPIPS` stays much lower than the earlier `QEdgePattn` family
- the later completed points through `epoch_0006` do not yet change that story:
  - style remains pinned near `0.692`
  - `LPIPS` drifts upward but stays well below the old `QEdgePattn` line
- the epoch-matched comparison now makes that trade explicit:
  - roughly `0.012 to 0.018` lower cheap style
  - but roughly `0.048 to 0.067` lower `LPIPS`
- so the early eval-side signal is:
  - promising on structure/cost-to-style tradeoff
  - still unproven on target-style recovery ceiling

## Early train read

Current early train-only read:

- `flow` has already improved:
  - `0.8279 -> 0.8029`
- `terminal_swd` has also improved:
  - it improved as low as `5.8750` at `epoch 5`
  - but rose back to `6.3438` at `epoch 6`
- `proximal_residual_abs` is still tiny:
  - about `0.0083`

Current interpretation:

- nothing is exploding numerically
- the new dual-path branch is not immediately bypassing the transport field with a huge residual
- but the `terminal_swd` rebound at `epoch 6` means the line is not yet clearly smoother or better than `QEdgePattn`
- this is only a training-side stability read, not a quality verdict
- but it is already a cleaner start than a launch-only smoke check

## Latest checked state

Checked again after the first launch window:

- remote `run.py` is still alive
- matching fresh-eval watcher is still alive
- remote live GPU memory is about:
  - `8142 MiB / 12288 MiB`
- retained checkpoints still currently visible:
  - `epoch_0001.pt`
  - `epoch_0002.pt`
  - `epoch_0003.pt`
  - `epoch_0004.pt`
  - `epoch_0005.pt`
  - `epoch_0006.pt`
  - `epoch_0007.pt`
  - `epoch_0008.pt`
  - `epoch_0009.pt`
  - `epoch_0010.pt`
  - `epoch_0011.pt`
  - `epoch_0012.pt`
- latest available logged epoch rows remain:
  - `epoch 1 loss = 8.4354`
  - `epoch 2 loss = 8.4194`
  - `epoch 3 loss = 8.2766`
  - `epoch 4 loss = 8.2084`
  - `epoch 5 loss = 8.3885`
  - `epoch 6 loss = 8.4321`
  - `epoch 7 loss = 8.3132`
  - `epoch 8 loss = 8.2275`
  - `epoch 9 loss = 8.2772`
  - `epoch 10 loss = 8.3743`
  - `epoch 11 loss = 8.2241`
  - `epoch 12 loss = 8.0421`

Additional read:

- `flow` does not yet show a clean monotone improvement after `epoch 4`
  - `epoch 5` moved back up to about `0.8150`
- `epoch 6` moved back down again to about:
  - `0.8029`
- `epoch 7` moved back up again to about:
  - `0.8087`
- `epoch 8` moved back down again to about:
  - `0.7965`
- `epoch 9` improved slightly further to about:
  - `0.7905`
- `epoch 10` moved back up again to about:
  - `0.8076`
- `epoch 11` moved back down again to about:
  - `0.7975`
- `epoch 12` improved a little further to about:
  - `0.7959`
- `terminal_swd` at `epoch 5` is about:
  - `5.8750`
- but `epoch 6` also reopened it upward to about:
  - `6.3438`
- `epoch 7` eased it back to about:
  - `6.0625`
- `epoch 8` improved it slightly further to about:
  - `6.0000`
- `epoch 9` improved it a little further to about:
  - `5.9688`
- `epoch 10` reopened it upward to about:
  - `6.1875`
- `epoch 11` softened slightly to about:
  - `6.1563`
- `epoch 12` improved it further to about:
  - `5.7188`
- `proximal_residual_abs` is still tiny:
  - about `0.0083`

Operational read:

- the line remains comfortably below the formal `< 11.0 GiB` cap
- the packet is still a real active training lane, not a stale launcher residue
- no image-backed closure exists yet for this family
- but the training-side `full_eval` directory has now appeared, so the packet is no longer pure train-only
- current state is still:
  - train-plus-deferred-eval transition
  - still too early to claim a quality verdict
  - not yet ready for local `IntroStyle / DINO / VLM`
- still waiting for the first retained eval packet
- `epoch 8/9` were slightly cleaner than the earlier rebound point
- but `epoch 10` is another reminder that the line has not stabilized yet
- `epoch 11` is mildly cleaner than `epoch 10`, but still not enough to call the line stable

## First retained eval read

The first training-side retained eval point is now locally mirrored under:

- [dualpath_full_eval_progress_20260609](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_full_eval_progress_20260609)

Current first `CLIP/LPIPS` read:

- `epoch_0001`
  - `transfer = 0.6927 / 0.4037`
  - `all-pairs = 0.7177 / 0.4006`
- `epoch_0002`
  - `transfer = 0.6919 / 0.4218`
  - `all-pairs = 0.7157 / 0.4186`
- `epoch_0003`
  - `transfer = 0.6918 / 0.4306`
  - `all-pairs = 0.7149 / 0.4273`
- `epoch_0004`
  - `transfer = 0.6921 / 0.4348`
  - `all-pairs = 0.7150 / 0.4313`
- `epoch_0005`
  - `transfer = 0.6917 / 0.4348`
  - `all-pairs = 0.7149 / 0.4311`
- `epoch_0006`
  - `transfer = 0.6919 / 0.4332`
  - `all-pairs = 0.7148 / 0.4293`
- `epoch_0007`
  - `transfer = 0.6927 / 0.4389`
  - `all-pairs = 0.7155 / 0.4352`
- `epoch_0008`
  - `transfer = 0.6926 / 0.4379`
  - `all-pairs = 0.7157 / 0.4341`
- `epoch_0009`
  - `transfer = 0.6927 / 0.4387`
  - `all-pairs = 0.7157 / 0.4349`
- `epoch_0010`
  - `transfer = 0.6927 / 0.4399`
  - `all-pairs = 0.7156 / 0.4362`
- `epoch_0011`
  - `transfer = 0.6927 / 0.4402`
  - `all-pairs = 0.7155 / 0.4364`

Interpretation:

- relative to the earlier `QEdgePattn` family, this first retained point is:
  - lower on style
  - better on LPIPS
- the next few completed training-side retained points keep the same basic character:
  - style stays low
  - LPIPS stays relatively conservative
- by `epoch_0011`, the line still has not shown a late switch into a stronger style regime
- no training-side sign yet of a late transition into a stronger style regime
- that means the current dual-path branch may be reopening a more conservative operating point first
- this is still only a training-side `full_eval` read
- local `IntroStyle / DINO / VLM` remains blocked on image-backed `fresh_localreview`
- because the current training-side pulled `images/` directory is empty

## First image-backed local review status

The first image-backed local packet is now available under:

- [dualpath_fresh_localreview_progress_20260609](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_fresh_localreview_progress_20260609)

Current first image-backed point:

- `epoch_0001`
  - `transfer = 0.6925 / 0.4036`
  - `all-pairs = 0.7176 / 0.4005`

Current first image-backed local read:

- `epoch_0001`
  - `transfer = 0.6925 / 0.4036`
  - `IntroStyle target = 0.1076`
  - `IntroStyle source = 0.1490`
  - `IntroStyle best_non_target = 0.1585`
  - `IntroStyle margin = -0.0509`
  - `identity_target = 0.1468`
  - `DINO = 0.0263`

Operational implication:

- the first image-backed non-CLIP read is now available
- current read still looks negative for promotion:
  - style specificity is clearly wrong
  - and structure is still worse than the current balanced anchor band
- relative to `QEdgePattn e01`, this first `DualPath` image-backed point is not an obvious rescue:
  - `IntroStyle` target is still only modest
  - `IntroStyle` margin is even more negative
  - `DINO` is slightly better than `QEdgePattn e01`, but still far from the balanced anchor band

## First retained eval read

The first training-side retained eval point is now locally mirrored under:

- [dualpath_full_eval_progress_20260609](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_full_eval_progress_20260609)

Current first `CLIP/LPIPS` read:

- `epoch_0001`
  - `transfer = 0.6927 / 0.4037`
  - `all-pairs = 0.7177 / 0.4006`

Interpretation:

- relative to the earlier `QEdgePattn` family, this first retained point is:
  - lower on style
  - better on LPIPS
- that means the current dual-path branch may be reopening a more conservative operating point first
- but this is still only a training-side `full_eval` read
- local `IntroStyle / DINO / VLM` is still blocked on image-backed `fresh_localreview`
- because the current training-side pulled `images/` directory is empty

## Intended read

This packet is a win only if it changes more than cheap `CLIP` style.

What it must show:

- stronger target-style recovery than `QEdgePattn`
- without repeating the same `LPIPS` blow-up pattern too aggressively
- and with at least a plausible path to a stronger local `IntroStyle / VLM` read

If it fails:

- the conclusion becomes sharper:
  - a stronger branch alone is not enough
  - we may need a still more target-specific or multi-stage late branch family
