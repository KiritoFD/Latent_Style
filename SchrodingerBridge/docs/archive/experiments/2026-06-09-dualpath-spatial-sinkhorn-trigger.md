# DualPath Spatial Sinkhorn Trigger

Date: 2026-06-09

This note records the promotion of the next mechanism branch after the negative
to mixed read on plain `dualpath_spatialtexture`.

Promoted packet:

- `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_sinkhorn_seed42_b8a2`

Mechanism delta relative to the previous spatialtexture branch:

- keep:
  - `spatial_carrier_gate`
  - `body_decoder`
  - `quantile_edge_gated_anisotropic_plus_stokes`
  - `dual low/high proximal split`
- change:
  - proximal late-style routing
  - from diffuse `softmax`
  - to near doubly-stochastic `sinkhorn`

Current hypothesis:

- the plain spatial dual-path branch already proved that more late-branch
  capacity alone is not enough
- the remaining bottleneck is that target-style evidence is still being routed
  too diffusely across latent locations
- `sinkhorn` routing should force more selective and more distributed localized
  assignment of target-style texture evidence

Launch state:

- remote formal train launched
- first-health passed
- observed early runtime memory:
  - about `7823 MiB`
- this is safely below the current formal cap
- detached fresh-eval watcher also launched

Current live read during the early training phase:

- training has already advanced through `epoch 1` and deep into `epoch 2`
- latest checked progress was already well into the back half of `epoch 2`
- the first retained checkpoint now exists:
  - `epoch_0001.pt`
- the second retained checkpoint now also exists:
  - `epoch_0002.pt`
- the third retained checkpoint now also exists:
  - `epoch_0003.pt`
- the fourth retained checkpoint now also exists:
  - `epoch_0004.pt`
- the fifth retained checkpoint now also exists:
  - `epoch_0005.pt`
- the sixth retained checkpoint now also exists:
  - `epoch_0006.pt`
- the seventh retained checkpoint now also exists:
  - `epoch_0007.pt`
- the eighth retained checkpoint now also exists:
  - `epoch_0008.pt`
- the detached watcher has not started fresh eval yet, which is expected:
  - it is still correctly waiting for train completion
  - at the latest check, `full_eval_fresh_localreview/` still did not exist yet
  - this matches the watcher log, which still reports `train_alive=True`
- at the latest check, training still held high GPU occupancy and had not yet
  transitioned into the eval phase
- recent train log slices remain numerically stable:
  - `flow` about `0.823 to 0.836`
  - `kin` about `0.088 to 0.095`
  - `tswd` about `6.03 to 7.38`
- current GPU read remains:
  - earlier checks sat around `7828 / 12288 MiB`
  - later checks briefly rose to about `11057 / 12288 MiB`
  - so the live band is now better described as roughly `7.8 to 11.1 GiB`
  - utilization still high, recently about `52% to 99%`

Immediate expected artifacts:

1. remote train log
2. retained checkpoints
3. post-train fresh eval curve
4. later `IntroStyle` / local visual follow-up if the curve is promotable

## First full-eval read

The packet has now crossed the first real evaluation threshold through the
separate `full_eval/` tree.

Currently closed point:

- `full_eval/epoch_0001`
  - `summary.json`
  - `metrics.csv`

First usable read from `epoch_0001`:

- transfer:
  - `clip_style = 0.6932`
  - `LPIPS = 0.4016`
- all-pairs:
  - `clip_style = 0.7186`
  - `LPIPS = 0.3980`

Direct comparison against predecessor `dualpath_spatialtexture epoch_0001`:

- predecessor transfer:
  - `0.6929 / 0.4015`
- sinkhorn transfer:
  - `0.6932 / 0.4016`
- predecessor all-pairs:
  - `0.7183 / 0.3978`
- sinkhorn all-pairs:
  - `0.7186 / 0.3980`

Current interpretation:

- the first landed sinkhorn point is effectively a near-tie with the earlier
  spatialtexture `epoch_0001`
- this is not a decisive win
- but it is also not a collapse
- so the next information-bearing question is whether later sinkhorn retained
  points diverge upward on style specificity or just settle back into the same
  conservative basin

Current caution:

- later entries under the remote `full_eval/` tree were not yet safe to read as
  fully closed when this note was updated
- at the latest check:
  - `epoch_0005` had partial artifacts
  - but not yet a completed `summary.json`

## First closed sweep read: `epoch_0001` through `epoch_0005`

The first five closed `full_eval` points are now readable.

Sinkhorn branch:

- `epoch_0001`
  - transfer `0.6932 / 0.4016`
  - all-pairs `0.7186 / 0.3980`
- `epoch_0002`
  - transfer `0.6924 / 0.4194`
  - all-pairs `0.7166 / 0.4157`
- `epoch_0003`
  - transfer `0.6922 / 0.4282`
  - all-pairs `0.7157 / 0.4243`
- `epoch_0004`
  - transfer `0.6927 / 0.4328`
  - all-pairs `0.7159 / 0.4288`
- `epoch_0005`
  - transfer `0.6920 / 0.4333`
  - all-pairs `0.7155 / 0.4292`

Predecessor spatialtexture branch:

- `epoch_0001`
  - transfer `0.6929 / 0.4015`
  - all-pairs `0.7183 / 0.3978`
- `epoch_0002`
  - transfer `0.6922 / 0.4193`
  - all-pairs `0.7164 / 0.4156`
- `epoch_0003`
  - transfer `0.6920 / 0.4281`
  - all-pairs `0.7155 / 0.4242`
- `epoch_0004`
  - transfer `0.6924 / 0.4327`
  - all-pairs `0.7157 / 0.4287`
- `epoch_0005`
  - transfer `0.6916 / 0.4332`
  - all-pairs `0.7151 / 0.4290`

Current interpretation of the first closed sweep:

- across `epoch_0001` to `epoch_0005`, sinkhorn remains extremely close to the
  predecessor branch
- the direction is slightly positive on `clip_style`
- but the magnitude is tiny
- there is still no meaningful early breakaway in either:
  - transfer style
  - all-pairs style
- and there is no compensating LPIPS improvement

So the current read upgrades from:

- `epoch_0001 is a near-tie`

to:

- `the first closed 5-point sinkhorn sweep is still basically a near-tie family`

This keeps the branch alive, but does not yet support a claim that
`Sinkhorn proximal routing` has materially reopened target-specific style.

Prepared follow-up paths already landed:

- best-few pull to local:
  - [pull_remote_dualpath_spatial_sinkhorn_bestfew_localreview.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/pull_remote_dualpath_spatial_sinkhorn_bestfew_localreview.py)
- remote `IntroStyle` best-few probe:
  - [launch_remote_dualpath_spatial_sinkhorn_introstyle_bestfew_probe.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_dualpath_spatial_sinkhorn_introstyle_bestfew_probe.py)
- local CPU-only bestfew review entrypoint:
  - [run_local_dualpath_spatial_sinkhorn_bestfew_review.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_local_dualpath_spatial_sinkhorn_bestfew_review.py)

Current local review blocker:

- the pulled `full_eval` bestfew packet is sufficient for:
  - `summary.json`
  - `metrics.csv`
  - curve reading
- but not yet sufficient for image-backed non-CLIP review
- current pulled epoch dirs do not contain usable generated-image contents for:
  - local `IntroStyle`
  - local `DINO`
  - local `VLM`
- so the current local bestfew follow-up is blocked on an image-backed eval
  bundle, not on missing scripts

## Image-backed rerun lane

To unblock the local non-CLIP follow-up, a dedicated remote image-backed rerun
lane was added for the current sinkhorn bestfew points.

Purpose:

- regenerate actual image files for the most informative sinkhorn points
- keep the current remote packet root
- produce a local-review-safe eval bundle for:
  - `IntroStyle`
  - `DINO`
  - later local `VLM` if needed

Current launcher:

- [launch_remote_dualpath_spatial_sinkhorn_imagebacked_bestfew_rerun.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_dualpath_spatial_sinkhorn_imagebacked_bestfew_rerun.py)

Current rerun policy:

- rerun `epoch_0001`
- rerun `epoch_0009`
- save generated images
- write into:
  - `full_eval_imagebacked_bestfew/`

Current live status:

- remote rerun launched successfully
- first-health passed
- observed prelaunch memory:
  - about `5030 MiB`
- observed first-health memory:
  - about `6893 MiB`

Landed remote image-backed outputs now include:

- `full_eval_imagebacked_bestfew/epoch_0001`
- `full_eval_imagebacked_bestfew/epoch_0009`

This means the previous local blocker is no longer:

- `no path exists to generate image-backed sinkhorn bestfew`

It is now only:

- `wait for the launched image-backed rerun outputs to land, then run local
  IntroStyle / DINO review`

## Local image-backed follow-up status

The first image-backed sinkhorn packet is now locally available for:

- `epoch_0001`

Current local status:

- pulled image-backed `epoch_0001` contains real generated image files
- pulled image-backed `epoch_0009` now also contains real generated image files
- local `DINO` review on that packet has completed
- local `IntroStyle` review has been started on CPU
- the original foreground CPU run was too slow for a synchronous turn budget
- the `IntroStyle` probe is now relaunched as a detached local CPU background
  job with stdout/stderr logs under:
  - `aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/introstyle_epoch1_only.stdout.log`
  - `aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/introstyle_epoch1_only.stderr.log`
- the current detached run is not stalled:
  - local logs show steady progress through the image chunks
- at the latest check it had already advanced to about:
    - `chunk 678 / 750`
- the final `introstyle_epoch1_only.csv/json` outputs have now landed

So the current blocker has been reduced again:

- no longer `missing image-backed packet`
- no longer `no non-CLIP review has started`
- the first local image-backed `IntroStyle` review for `epoch_0001` has now
  landed
- after that, extend the same local non-CLIP read from `epoch_0001` to the
  later `epoch_0009` image-backed point

## First local non-CLIP partial read: `epoch_0001`

What is already landed locally:

- image-backed `epoch_0001`
- local `DINO` review

Current local `DINO` row:

- `epoch_0001`
  - source-aligned pairs:
    - `745`
  - `DINO structure = 0.02617`

Reference anchors:

- `LBM-Knee full750 DINO = 0.02171`
- `Seedream full750 DINO = 0.0291`

Current structure-side interpretation:

- sinkhorn `epoch_0001` is structurally worse than `LBM-Knee`
- but it is still slightly better than the old style-heavy `LBM-PS-v2` regime
- and it sits closer to the current style-pushing families than to the geometry
  anchor

What is still pending:

- current pending follow-up is now:
  - extend the same local non-CLIP review to later sinkhorn points such as
    `epoch_0009`

Cross-reference:

- [2026-06-10-dualpath-spatial-sinkhorn-epoch1-local-nonclip-read.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-dualpath-spatial-sinkhorn-epoch1-local-nonclip-read.md)

## Second local non-CLIP partial read: `epoch_0009`

The later sinkhorn point now also has a real local image-backed review chain.

What is already landed locally:

- image-backed `epoch_0009`
- local `DINO` review

Current local `DINO` row:

- `epoch_0009`
  - source-aligned pairs:
    - `745`
  - `DINO structure = 0.02741`

Immediate read:

- this is slightly worse than `sinkhorn epoch_0001`
  - `0.02741` vs `0.02617`
- so the later point does not currently rescue the structure side

Current `IntroStyle` status for `epoch_0009`:

- detached local CPU job has now completed
- landed outputs now exist:
  - [full_eval_imagebacked_bestfew_introstyle_epoch9_only.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_introstyle_epoch9_only.csv)
  - [full_eval_imagebacked_bestfew_introstyle_epoch9_only.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_introstyle_epoch9_only.json)
  - [full_eval_imagebacked_bestfew_introstyle_epoch9_only.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_introstyle_epoch9_only.md)

Landed `epoch_0001` IntroStyle outputs:

- [full_eval_imagebacked_bestfew_introstyle_epoch1_only.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_introstyle_epoch1_only.csv)
- [full_eval_imagebacked_bestfew_introstyle_epoch1_only.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_introstyle_epoch1_only.json)
- [full_eval_imagebacked_bestfew_introstyle_epoch1_only.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_introstyle_epoch1_only.md)

Cross-reference for the completed first point:

- [2026-06-10-dualpath-spatial-sinkhorn-epoch1-local-nonclip-read.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-dualpath-spatial-sinkhorn-epoch1-local-nonclip-read.md)
- [2026-06-10-dualpath-spatial-sinkhorn-epoch9-local-nonclip-read.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-dualpath-spatial-sinkhorn-epoch9-local-nonclip-read.md)
