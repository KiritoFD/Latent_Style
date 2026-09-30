# `LBM-Knee e13 + Spatial Carrier-Gate Body+Decoder + Quantile Edge-Gated Structure Leash + CrossAttnTexture Proximal`

Date: 2026-06-09

Why this packet exists:

- the plain spatial-carrier line showed:
  - a little extra style
  - but too much structure damage
- the first edge-gated line showed:
  - slightly better `DINO`
  - but lost part of the style gain
- the `qedgegated` follow-up is now testing whether a more selective leash alone is enough

If that still stays on a `style-up / LPIPS-worse` track, the likely next problem is:

- the carrier branch is still being asked to do too much style work by itself

Mechanism:

- keep:
  - `Knee e13`
  - `spatial_carrier_gate`
  - `body_decoder`
  - `quantile_edge_gated_anisotropic_plus_stokes`
- add:
  - `proximal_mode = crossattn_texture`

Intent:

- let the carrier branch and the structure leash stabilize geometry
- let the explicit cross-attention texture residual head carry more of the style-specific high-frequency burden

Config:

- [inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_pattn_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_pattn_seed42_b8a2.json)

Intended read:

- if this works:
  - `IntroStyle` should rise more than qedgegated alone
  - without reopening the full `DINO` damage of the plain spatial-carrier line
- if it fails:
  - that means this family may need a still stronger or more decoupled style head than the current proximal branch

## Live status

Checked through the latest local/remote audit on `2026-06-09`:

- remote `run.py` training has already exited on the reviewed `3060 WSL`
- current run dir:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_pattn_seed42_b8a2`
- current checkpoint progress:
  - `epoch_0001.pt` through `epoch_0012.pt` already landed
- current watcher state:
  - the first watcher launch fired too early at `11:51 +08:00`
  - it briefly saw `train_alive=False`
  - then failed inside `infer_fresh_epochs_from_latest_training_log.py`
  - a second watcher launch at `11:56 +08:00` is the valid active one and is now polling correctly
  - the watcher script has now been hardened so it:
    - requires at least one observed `train_alive=True` before treating the line as finished
    - waits for a short post-exit settle window
    - retries fresh-epoch inference instead of failing on the first race
  - the stale old watcher instance was explicitly killed
  - only the refreshed single watcher instance is left alive

Current training read from `logs/training_20260609_115647.csv`:

- epochs logged so far:
  - `1..12`
- recent endpoint trend:
  - `flow`: `0.8289 -> 0.8013`
  - `final_endpoint_abs`: `0.5959 -> 0.5738`
- `proximal_residual_abs`: stable near `0.124 to 0.128`
- runtime band:
  - `epoch_time_sec`: about `145s`
  - `samples_per_sec`: about `34`
  - `cuda_peak_reserved_gb`: about `3.22`

Operational implication:

- remote GPU is not idle
- the packet has finished the planned `12 epoch` training loop
- deferred in-process `full_eval` has now landed through:
  - `epoch_0012`
- the original watcher stayed alive, but `full_eval_fresh_localreview` still did not start promptly enough after train exit
- to avoid leaving the remote `3060` idle, a manual image-backed closure is now running:
  - `rerun_full_eval_for_run.py`
  - `output_subdir = full_eval_fresh_localreview`
  - `save_generated_images = true`
  - `epochs = 1..12`
- inherited training budget for this family is `12 epoch`
- current packet is therefore at `12/12 epoch` and has now moved into explicit post-train image-backed closure
- next audit gate is:
  - let `full_eval_fresh_localreview` land enough epochs to pick best-few
  - then pull best-few to local review
  - then compare against the already closed `qedgegated` curve
- this handoff has now started:
  - local best-few image-backed packet already contains:
    - `epoch_0001`
    - `epoch_0003`
  - current local handoff root:
    - [qedgegated_pattn_bestfew_localreview_20260609](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedgegated_pattn_bestfew_localreview_20260609)
- current fresh best-few handoff CSV:
  - [full_eval_fresh_localreview_bestfew_handoff.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedgegated_pattn_bestfew_localreview_20260609/full_eval_fresh_localreview_bestfew_handoff.csv)
- current fresh IntroStyle manifest:
  - [full_eval_fresh_localreview_bestfew_introstyle_manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedgegated_pattn_bestfew_localreview_20260609/full_eval_fresh_localreview_bestfew_introstyle_manifest.csv)
- current local CPU-only VLM triplet manifests are now also prepared for the early pair:
  - [vlm_manifest_qedgepattn_e01_vs_knee_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_qedgepattn_e01_vs_knee_vs_seedream_20260609.csv)
  - [vlm_manifest_qedgepattn_e03_vs_knee_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_qedgepattn_e03_vs_knee_vs_seedream_20260609.csv)
- corresponding CPU-only VLM jobs have been launched locally for:
  - `QEdgePattn_e01 vs Knee vs Seedream`
  - `QEdgePattn_e03 vs Knee vs Seedream`
- consolidated local-review progress note:
  - [2026-06-09-qedgepattn-localreview-progress.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-qedgepattn-localreview-progress.md)

Remote-to-local handoff entry prepared:

- once `full_eval_fresh_localreview` lands, use:
  - [pull_remote_qedgegated_pattn_bestfew_localreview.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/pull_remote_qedgegated_pattn_bestfew_localreview.py)
- this script does:
  - pull remote `clip_lpips_curve.csv`
  - build a compact best-few handoff locally
  - pull the selected best-few epoch directories to local review storage
- local best-few review entry prepared:
  - [run_local_qedgegated_pattn_bestfew_review.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_local_qedgegated_pattn_bestfew_review.py)
  - this script does:
    - pull remote best-few locally
    - build the local IntroStyle manifest
    - run local low-VRAM `IntroStyle`
    - run local `DINO` structure review
  - validated local source root:
    - `G:\GitHub\Latent_Style\Dataset\distinct5_512\test`

## Early full-eval progress read

A stage-only local pull from training-side `full_eval` is now available under:

- [qedgegated_pattn_full_eval_progress_20260609](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedgegated_pattn_full_eval_progress_20260609)

Current `CLIP/LPIPS` curve read from the first landed epochs:

- `epoch_0001 = 0.7048 / 0.4521`
- `epoch_0002 = 0.7056 / 0.4699`
- `epoch_0003 = 0.7063 / 0.4791`
- `epoch_0004 = 0.7074 / 0.4878`
- `epoch_0005 = 0.7084 / 0.4944`
- `epoch_0006 = 0.7093 / 0.4991`
- `epoch_0007 = 0.7102 / 0.5024`
- `epoch_0008 = 0.7101 / 0.5036`
- `epoch_0009 = 0.7102 / 0.5055`
- `epoch_0010 = 0.7105 / 0.5068`

Current best points on the training-side `full_eval` curve:

- best transfer `LPIPS`
  - `epoch_0001 = 0.7048 / 0.4521`
- best transfer `CLIP-style`
  - `epoch_0010 = 0.7105 / 0.5068`

Current best points on the image-backed `full_eval_fresh_localreview` curve so far:

- best transfer `LPIPS`
  - `epoch_0001 = 0.7047 / 0.4519`
- best transfer `CLIP-style`
  - `epoch_0011 = 0.7106 / 0.5073`

Current remote drain state:

- deferred training-side `full_eval` has now completed through:
  - `epoch_0012`
- the latest locally re-pulled stable curve currently contains completed summaries through:
  - `epoch_0010`
- so the current stage verdict is now based on a materially longer completed early curve, not only the first four points

Updated remote status at the latest check:

- remote `full_eval` has now stably landed summaries through:
  - `epoch_0012`
- `full_eval_fresh_localreview` is now present and actively running
- `full_eval_fresh_localreview` has now effectively closed through `epoch_0012`
- first visible image-backed epoch:
  - `epoch_0001`
- current locally pulled image-backed best-few already includes:
  - `epoch_0001`
  - `epoch_0003`
  - `epoch_0007`
- therefore the next meaningful gate is now:
  - `decide whether the completed image-backed curve plus growing non-CLIP local review is enough to close this family as non-promoted`

Current interpretation:

- the early training-side read is still consistent with the old warning pattern:
  - `style-up`
  - `LPIPS-worse`
- this read is now materially stronger than a 2- or 4-point hint:
  - the completed training-side curve through `epoch_0010` is monotone enough to treat the pattern as stable unless later image-backed evidence contradicts it
- but this is still not the paper-facing verdict
- reason:
  - the decision stack still needs the image-backed local review path
  - local `IntroStyle / DINO / VLM` should use the fresh image-backed closure rather than the raw in-process training-side eval packet

## First image-backed local read

The first image-backed local packet is now available under:

- [qedgegated_pattn_fresh_localreview_progress_20260609](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedgegated_pattn_fresh_localreview_progress_20260609)

First reviewed image-backed epoch:

- `epoch_0001`

Current `CLIP/LPIPS` read:

- `transfer = 0.7047 / 0.4519`

Current `IntroStyle` read:

- `transfer_target_style_score = 0.1073`
- `transfer_source_style_score = 0.1397`
- `transfer_best_non_target_score = 0.1480`
- `transfer_style_margin = -0.0407`
- `identity_target_style_score = 0.1457`

Current `DINO` read:

- `dino_structure = 0.0290`

Interpretation:

- the first image-backed non-CLIP read does not rescue the family
- style specificity is clearly wrong:
  - the average best non-target score is higher than the target score
  - so the mean style margin is strongly negative
- structure is also still far to the worse side of the current balanced anchor band
- so the current read remains:
  - `not promotable`
  - unless later image-backed epochs improve both specificity and structure materially

## Best-few image-backed closure

Best-few image-backed local review is now available for:

- `epoch_0001`
- `epoch_0011`
- `epoch_0012`

Current `IntroStyle` read:

- `epoch_0001`
  - `target = 0.1076`
  - `source = 0.1400`
  - `best_non_target = 0.1489`
  - `margin = -0.0413`
  - `identity_target = 0.1445`
- `epoch_0011`
  - `target = 0.1067`
  - `source = 0.1265`
  - `best_non_target = 0.1422`
  - `margin = -0.0354`
  - `identity_target = 0.1289`
- `epoch_0012`
  - `target = 0.1064`
  - `source = 0.1260`
  - `best_non_target = 0.1404`
  - `margin = -0.0340`
  - `identity_target = 0.1321`

Current `DINO` read:

- `epoch_0001 = 0.0290`
- `epoch_0011 = 0.0330`
- `epoch_0012 = 0.0330`

Closure read:

- the later style-heavy points do not rescue the family
- `IntroStyle` margins remain clearly negative
- `DINO` worsens further on the late points
- this closes the family as:
  - `not promotable`
  - `negative for mainline promotion`
