# Status Snapshot: Remote Live, Local CPU-Only

Date: 2026-06-09

This note freezes the current execution state after the local review split was locked.

- remote `3060 WSL` remains the only formal GPU surface
- local does not take GPU
- local review continues only on `CPU / network` workloads such as `VLM`

## Remote state

Current active formal remote train packet:

- `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_sinkhorn_seed42_b8a2`

Immediate predecessor packet:

- `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_seed42_b8a2`

Current promoted mechanism:

- keep:
  - `LBM-Knee e13`
  - `spatial_carrier_gate`
  - `body_decoder`
  - `quantile_edge_gated_anisotropic_plus_stokes`
  - `dualpath_spatialtexture`
- add:
  - `sinkhorn` proximal routing

Previous packet mechanism:

- base:
  - `LBM-Knee e13`
  - `spatial_carrier_gate`
  - `body_decoder`
  - `quantile_edge_gated_anisotropic_plus_stokes`
- added:
  - `dualpath_spatialtexture` proximal head

Current verified remote state:

- the new `sinkhorn` packet has been launched
- first-health passed
- observed early runtime memory:
  - about `7823 MiB`
- this is safely below the formal cap
- a detached post-train fresh-eval watcher was also launched for the same run
- the run has already progressed through `epoch 1` and deep into `epoch 2`
- the first retained checkpoint already exists:
  - `epoch_0001.pt`
- later retained checkpoints now also exist:
  - `epoch_0002.pt`
  - `epoch_0003.pt`
  - `epoch_0004.pt`
  - `epoch_0005.pt`
  - `epoch_0006.pt`
  - `epoch_0007.pt`
  - `epoch_0008.pt`
- `fresh_eval` still has not started yet:
  - `full_eval_fresh_localreview/` does not exist at the latest check
  - watcher still reports `train_alive=True`
- runtime memory is no longer just in the earlier `~7.8 GiB` band:
  - later checks briefly rose to about `11.1 GiB`
  - so this packet remains alive but is now clearly operating near the formal cap
- a separate remote `full_eval/` tree has now appeared with at least one closed
  point:
  - `epoch_0001`
  - `summary.json`
  - `metrics.csv`
- first sinkhorn full-eval read is a near-tie with predecessor
  `spatialtexture epoch_0001`, not a clear early breakaway
- the first closed `epoch_0001` through `epoch_0005` sweep keeps the same read:
  - sinkhorn remains extremely close to predecessor spatialtexture
  - any style gain is currently tiny
  - there is still no material early breakaway
- remote image-backed rerun outputs have now landed for:
  - `epoch_0001`
  - `epoch_0009`
- the corresponding local image-backed packets have been pulled back, so the
  non-CLIP image asset blocker has been resolved
- first local non-CLIP image-backed read for sinkhorn `epoch_0001` is now
  available:
  - `IntroStyle target = 0.1104`
  - `IntroStyle delta-IDT = -0.0434`
  - `IntroStyle margin = -0.0483`
  - `DINO = 0.02617`
- second local non-CLIP image-backed read for sinkhorn `epoch_0009` is now
  available:
  - `IntroStyle target = 0.1056`
  - `IntroStyle delta-IDT = -0.0326`
  - `IntroStyle margin = -0.0465`
  - `DINO = 0.02741`
- current read:
  - both image-backed sinkhorn points remain negative-to-mixed
  - still not a target-specific style reopening
  - not enough to promote the family
- later sinkhorn point `epoch_0009` now also has:
  - image-backed local packet
  - local `DINO = 0.02741`
  - local `IntroStyle` CPU review running, outputs pending

What is already present under the predecessor run root:

- checkpoints:
  - `epoch_0001` through `epoch_0012`
- fresh eval rows:
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
  - `epoch_0012`

Current eval-side state for the predecessor packet:

- fresh curve:
  - [dualpath_spatial_fresh_curve_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_fresh_curve_20260609.csv)
- early-read note:
  - [2026-06-09-dualpath-spatialtexture-early-read.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-dualpath-spatialtexture-early-read.md)
- current remote post-train bestfew probe state:
  - handoff CSV exists
  - IntroStyle manifest exists
  - remote `IntroStyle` bestfew probe has completed
  - landed outputs:
    - [dualpath_spatial_introstyle_bestfew_probe_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_introstyle_bestfew_probe_20260609.csv)
    - [dualpath_spatial_introstyle_bestfew_probe_20260609.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_introstyle_bestfew_probe_20260609.json)

Current remote read on the predecessor packet:

- this packet is no longer just machine-safe; it already has a real early curve
- but that curve is still conservative:
  - transfer style stays around `0.6916 to 0.6929`
  - `LPIPS` rises from about `0.401` toward `0.440`
- the landed `IntroStyle` bestfew probe also remains negative-to-mixed:
  - `epoch_0001 margin = -0.05031`
  - `epoch_0012 margin = -0.04673`
- current remote evidence still looks like:
  - a cleaner / safer continuation
  - not a clear target-style ceiling unlock
- consolidated current synthesis:
  - [2026-06-09-current-round-read-spatialtexture.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-current-round-read-spatialtexture.md)

Current trigger note for the new active packet:

- [2026-06-09-dualpath-spatial-sinkhorn-trigger.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-dualpath-spatial-sinkhorn-trigger.md)

Additional follow-up lane now active:

- `sinkhorn image-backed bestfew rerun`
- purpose:
  - regenerate images for local `IntroStyle / DINO / VLM` review on the bestfew
    sinkhorn points
- first-health already passed under a safer memory band than the original
  training line

## Local CPU-only review state

Local GPU is intentionally untouched in this phase.

Current main local blind comparison:

- `QEdgePattn e01 vs DualPath e01 vs Seedream`

Authoritative artifacts:

- manifest:
  - [vlm_manifest_qedgee01_vs_dualpathe01_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_qedgee01_vs_dualpathe01_vs_seedream_20260609.csv)
- method summary:
  - [vlm_qedgee01_vs_dualpathe01_vs_seedream_20260609.method_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_qedgee01_vs_dualpathe01_vs_seedream_20260609.method_summary.csv)
- compact board:
  - [qedge_vs_dualpath_interim_board_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedge_vs_dualpath_interim_board_20260609.md)
- interpretation note:
  - [2026-06-09-qedge-vs-dualpath-vlm-direct-interim.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-qedge-vs-dualpath-vlm-direct-interim.md)

Current local blind read:

- completed cases:
  - `557`
- overall wins:
  - `Seedream = 547 / 557`
  - `DualPath e01 = 10 / 557`
  - `QEdgePattn e01 = 0 / 557`
- mean local scores:
  - `DualPath e01`
    - style `2.162`
    - structure `3.512`
    - artifact `2.621`
  - `QEdgePattn e01`
    - style `1.968`
    - structure `3.345`
    - artifact `2.379`

Current local read:

- `DualPath e01` remains ahead on blind perceptual means
- `QEdgePattn e01` still collects more discrete structure-side subwins
- both remain very far below `Seedream`

Expanded external-baseline local blind lines are also active:

- `LBM-PS-v2 vs Seedream vs SaMST e15`
- `LBM-PS-v2 vs Seedream vs SaMAM-2250`

Authoritative merged board:

- [vlm_external_baseline_board_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_external_baseline_board_20260609.csv)
- [vlm_external_baseline_board_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_external_baseline_board_20260609.md)

Current expanded read:

- `LBM-PS-v2 vs Seedream vs SaMST e15`
  - valid cases: `469`
  - wins:
    - `Seedream = 386`
    - `SaMST = 72`
    - `LBM-PS-v2 = 11`
- `LBM-PS-v2 vs Seedream vs SaMAM-2250`
  - valid cases: `482`
  - wins:
    - `Seedream = 326`
    - `SaMAM = 143`
    - `LBM-PS-v2 = 13`

Current implication:

- comparing only against `Seedream` was too narrow
- the currently promoted `LBM-PS-v2` point is still visually behind both
  `SaMST e15` and `SaMAM-2250` on the accumulated local audit
- this is now supported by a meaningfully larger local `VLM` sample, not just
  an early tiny batch
- this raises the bar for the next remote mechanism branch:
  - not just safer / cleaner continuation
  - but a mechanism that visibly closes the style-specificity gap

## Current decision

Do now:

1. keep local blind `VLM` accumulating on the current successor pair
2. keep using `QEdgePattn` as the style-up comparison baseline
3. treat the current `dualpath_spatialtexture` curve as a real negative-to-mixed mechanism read until later evidence overturns it
4. keep external-baseline `VLM` comparisons explicitly widened beyond `Seedream`:
   - always include `SaMST`
   - and `SaMAM`

Do not do now:

- do not spend local GPU
- do not reopen old hold/schedule branches
- do not claim that `spatialtexture` has opened the target-style ceiling from the current early curve

