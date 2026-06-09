# Experiments Plan

Date: 2026-06-08

## Objective

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- current target:
  - break the current style/content ceiling
  - especially toward `0.72 / 0.30` on `transfer CLIP-style / LPIPS`
- evaluation policy:
  - fast screening:
    - `LPIPS`
    - optional cheap `CLIP-S` triage only
  - paper-facing audit:
    - `IntroStyle` as the preferred style axis
    - non-CLIP style classifier as fallback/support
    - structure metric
    - visual / pairwise audit
  - `ArtFID` remains out of the inner mechanism loop
- execution policy:
  - single active GPU lane only
  - any materially better point should be committed immediately
  - every experiment must have a reflection / closure note

## Evaluation Axes

Current evaluation is explicitly three-axis:

1. `style axis`
   - fast screen:
     - optional cheap `CLIP-S` triage only
   - paper-facing:
     - `IntroStyle` preferred
     - Distinct5 non-CLIP style classifier as fallback/support

2. `structure axis`
   - fast screen:
     - `LPIPS`
   - next preferred supplement:
     - DINO-style structure comparison

3. `artifact / visual axis`
   - qualitative visual audit
   - direct comparison to `Seedream`
   - artifact-sensitive diagnostics already in the repo

Reference note:

- [2026-06-08-eval-reliability-and-related-work-brief.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-eval-reliability-and-related-work-brief.md)
- [2026-06-08-introstyle-mainline-switch.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-introstyle-mainline-switch.md)
- [2026-06-08-introstyle-page1-baselines-smoke20.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-introstyle-page1-baselines-smoke20.md)
- [2026-06-08-introstyle-remote-modelscope-cache.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-introstyle-remote-modelscope-cache.md)
- [2026-06-08-introstyle-remote-page1-probe.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-introstyle-remote-page1-probe.md)
- [2026-06-09-introstyle-runtime-decoupling-fix.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-introstyle-runtime-decoupling-fix.md)
- [2026-06-09-bodydecoder-clean-introstyle-rerun.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-bodydecoder-clean-introstyle-rerun.md)
- [2026-06-09-local-finalists-introstyle-dino-full750.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-local-finalists-introstyle-dino-full750.md)
- [2026-06-09-midcycle-decision-full750.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-midcycle-decision-full750.md)
- [2026-06-09-local-vlm-full750-interim.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-local-vlm-full750-interim.md)
- [2026-06-09-local-vlm-method-summary.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-local-vlm-method-summary.md)
- [2026-06-09-status-snapshot-remote-live-local-cpu.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-status-snapshot-remote-live-local-cpu.md)
- [2026-06-09-theory-read-edgegated-vs-hold-families.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-theory-read-edgegated-vs-hold-families.md)
- [2026-06-09-qedgepattn-localreview-progress.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-qedgepattn-localreview-progress.md)
- [2026-06-09-next-mechanism-after-qedgepattn.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-next-mechanism-after-qedgepattn.md)
- [2026-06-09-dualpath-localreview-progress.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-dualpath-localreview-progress.md)
- [2026-06-09-dualpath-spatialtexture-trigger.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-dualpath-spatialtexture-trigger.md)
- [2026-06-09-qedge-vs-dualpath-first-dino-read.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-qedge-vs-dualpath-first-dino-read.md)
- [2026-06-09-qedge-vs-dualpath-vlm-direct-interim.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-qedge-vs-dualpath-vlm-direct-interim.md)
- [2026-06-09-dualpath-spatialtexture-early-read.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-dualpath-spatialtexture-early-read.md)
- [2026-06-09-current-round-read-spatialtexture.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-current-round-read-spatialtexture.md)
- [first_bestfew_dino_compare_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/first_bestfew_dino_compare_20260609.md)
- unified current board:
  - [current_mainline_evidence_board_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/current_mainline_evidence_board_20260609.csv)

## Current Style Read

The current `IntroStyle smoke20` page-1 shortlist now includes:

- `SaMAM` pixel
- `SaMST` pixel
- `SaMAM-latent`
- `SaMST-latent`
- `LBM-K`
- `LBM-Knee`
- `LBM-PS-v2`
- `Seedream`

Unified evidence table:

- [page1_shortlist_comparison.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/introstyle_page1/page1_shortlist_comparison.csv)
- [page1_shortlist_comparison.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/introstyle_page1/page1_shortlist_comparison.md)
- visual diagnosis:
  - [2026-06-08-introstyle-page1-visual-diagnosis.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-introstyle-page1-visual-diagnosis.md)

Current smoke read:

- `Seedream` is strongest on `IntroStyle target score` and `IntroStyle delta-IDT`.
- `LBM-Knee` remains the strongest current internal point on `IntroStyle delta-IDT`.
- `SaMST e15` beats `SaMAM-2250` on smoke `IntroStyle delta-IDT`, but still has a much worse specificity margin.
- `Lat SaMAM` is effectively at the `IDT` floor on `IntroStyle delta-IDT`.
- `Lat SaMST` falls below the `IDT` floor on `IntroStyle delta-IDT`, which keeps it as a negative latent baseline.

Important interpretation:

- `raw CLIP-S` and `IntroStyle` do not induce the same ordering.
- current mechanism writing should not describe `CLIP-S` as the supervisory target of the active LBM line.
- the actual training-side supervision remains:
  - transport / endpoint objective
  - kinetic / structure regularization
  - terminal SWD family losses
- formal remote eval should now prefer the integrated `IntroStyle` sidecar from `run_evaluation.py`, resolved through the reviewed `ModelScope` cache path
- `LBM-PS-v2` is still the strongest style-ceiling point on `CLIP delta-IDT`, but `LBM-Knee` is cleaner on the current smoke `IntroStyle` axis.
- this means the current main question is no longer just `how to raise CLIP style`, but also:
  - `how to make style more target-specific under IntroStyle / non-CLIP reads without losing the geometry anchor`
- the current visual diagnosis strengthens that reading:
  - `LBM-Knee` is geometry-strong but pale / under-committed
  - `LBM-PS-v2` is stronger on painterly energy but drifts toward generic foggy stylization
  - `SaMST` is target-style-stronger than `SaMAM`, but it pays for that with heavy structure damage
  - latent baselines are now visually documented as either near-no-op (`Lat SaMAM`) or collapse (`Lat SaMST`)
  - the expanded multi-source packet suggests these are not one-image accidents
  - the current failure modes repeat across:
    - figure scenes
    - landscapes
    - portraits
    - already-stylized ukiyo-e sources

Operational state for full `IntroStyle` remote closure:

- the reviewed remote `3060 WSL` cannot reach `huggingface.co` from WSL
- however, `ModelScope` download for `stabilityai/stable-diffusion-2-1-base` has now been proven on the remote surface
- this removes the old backbone-cache blocker
- the original runtime blocker from `StableDiffusionPipeline -> peft -> EncoderDecoderCache` has now been removed by the local runtime decoupling patch
- remaining work is now:
  - keep the remote single-lane discipline
  - sync the new `IntroStyle` runtime file before formal packet eval
  - rerun paper-facing packets under the new sidecar path
- a ready-to-run remote page-1 probe launcher now exists:
  - [launch_remote_introstyle_page1_probe.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_introstyle_page1_probe.py)

## Current Frontier

Current paper-facing promoted low-LPIPS frontier:

- `AnisoStokesQueue e13`
  - transfer: `0.7102169 / 0.4603146`
  - all-pairs: `0.7303320 / 0.4559407`
  - evidence:
    - [2026-06-07-inmortal-xpred-kmanifold-pattn-anisostokes-queue.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-07-inmortal-xpred-kmanifold-pattn-anisostokes-queue.md)

Current raw-style frontier:

- `Pattn + Stokes002 e13`
  - transfer: `0.7306720 / 0.6182822`
  - all-pairs: `0.7372152 / 0.6069087`
  - caution:
    - this is no longer the preferred style headline under the newer `IntroStyle + DINO + VLM` evidence stack

Current geometry anchor:

- `Hold4Mid e8`
  - transfer: `0.6679105 / 0.2877402`
  - all-pairs: `0.7013853 / 0.2877823`
  - evidence:
    - [2026-06-08-inmortal-anisostokes-queue-clamphold4mid-from-e13.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-inmortal-anisostokes-queue-clamphold4mid-from-e13.md)
  - non-CLIP audit:
    - [2026-06-08-hold-family-audit-nonclip-v5.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-hold-family-audit-nonclip-v5.md)

Current full750 decision:

- `Seedream` remains the external style ceiling.
- `LBM-Knee` remains the strongest current internal balanced point.
- `LBM-PS-v2` is now downgraded under the full750 `IntroStyle + DINO + VLM` read.
- `Hold4TwoStage` remains the highest-value open hold-family local heavy-review target.
- current local VLM method-level summary also now strongly supports this:
  - `Seedream` is still winning overwhelmingly:
    - about `709 / 713`
- `LBM-Knee` remains the strongest internal method
- `LBM-PS-v2` is clearly below `LBM-Knee` on all VLM axes
- expanded local external-baseline VLM now also shows:
  - `LBM-PS-v2` is still behind both `SaMST e15` and `SaMAM-2250`
 - current compressed decision read:
   - [2026-06-10-current-decision-read-stylesig-vs-fourway-vlm.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-current-decision-read-stylesig-vs-fourway-vlm.md)
 - current first eval read for stylesig:
   - [2026-06-10-stylesig-first-fast-eval-read.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-stylesig-first-fast-eval-read.md)
 - current full phase summary:
   - [2026-06-10-full-phase-experiment-summary.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-full-phase-experiment-summary.md)
 - current complete experiment catalog:
   - [2026-06-10-complete-experiment-catalog.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-complete-experiment-catalog.md)
  - so the next remote branch must target visual style specificity, not just a cleaner CLIP/LPIPS tradeoff

Immediate next remote mechanism:

- `dualpath_spatialtexture + Sinkhorn proximal routing`
- rationale:
  - current dual-path spatial branch already added capacity
  - local VLM says that was not enough
  - the next branch now attacks diffuse late-style routing itself

## Round 1 Status

Round 1 is mostly complete.

Completed mechanism families with closed reads:

- `K_spatial`
  - stable, not enough style lift
- `K_manifold`
  - better than `K_spatial`, useful transport-side control, still not a headline line alone
- `XPred + StructOT`
  - positive control, but below promoted frontier
- `XPred + EndpointTeacher`
  - positive control, but below promoted frontier
- `XPred + Queue`
  - positive control, but below promoted frontier
- `XPred + Kmanifold`
  - first strong style-ceiling signal, LPIPS still too weak
- `XPred + Kmanifold + Pattn`
  - first strong proximal family that improved the frontier on both axes
- `Pattn + late weak Stokes`
  - effective frontier refinement in the style-heavy family
- `AnisoStokesQueue from Pattn e13`
  - current promoted low-LPIPS frontier
- `Clamp reseed`
  - first successful recovery-family control against proximal takeover
- `Clamp release reseed`
  - improved over fixed clamp on LPIPS at nearly unchanged style
- `Hold4Wide`
  - slight positive incremental improvement over the first release-family `e3`
- `Hold4Mid`
  - strong positive geometry/content anchor

Completed negative or near-negative closures:

- `P_highpass`
  - clean negative
- `Kmanifold + Phighpass`
  - strong negative
- `P_mod`
  - better than `P_highpass`, still below `P_attn`
- `Trust` and `Trust reseed`
  - negative for preserving the parent `e13` basin
- `Wide release`
  - negative
- `Late wide release`
  - near-tie negative

Recently closed:

- `Hold4SlowMid`
  - near-tie negative closure
  - best retained point:
    - `e12 = 0.6673 / 0.2898`
    - all-pairs `= 0.7009 / 0.2898`
  - interpretation:
    - slower single-stage release is not enough
    - this closes the last coherent same-family single-stage smoothing question

Newly clarified:

- `Hold4Mid + spatial_carrier_gate body+decoder` was rerun under a clean single-lane eval contract
- clean `CLIP/LPIPS` closure remains weak:
  - `e8 = 0.6881 / 0.5177`
  - `e12 = 0.6881 / 0.5171`
- this keeps the family as a likely negative or near-negative rescue branch unless the upcoming `IntroStyle` rerun says otherwise
- the previous checkpoint/model mismatch was not a model-family issue:
  - it was caused by a stale remote shadow file `src/utils/config_schema.py`
  - this import-shadow bug has now been identified and repaired

## What Is Proven Effective

The following claims now have real supporting evidence:

1. `XPred + Kmanifold` is a real style-ceiling family.
   - It is much stronger than the earlier direct velocity-style packets on style lift.

2. `Cross-attention proximal (P_attn)` is the first proximal family that actually improved the frontier.
   - `P_highpass` and `P_mod` did not do this cleanly.

3. Late weak structural regularization can refine an already good high-style family.
   - `Pattn + late weak Stokes` is a real positive family.

4. Hard proximal clamping is an effective recovery mechanism once proximal takeover appears.
   - `Clamp reseed` and `Clamp release reseed` are both real positives.

5. An explicit early hold is useful.
   - `Hold4Wide` slightly beat the earlier release-family `e3` point.

6. `Hold + mid release` creates a genuinely strong geometry anchor.
   - `Hold4Mid e8 = 0.6679 / 0.2877` is the current best ultra-low-LPIPS operating point in the mechanism family.
   - non-CLIP audit now also shows that this family is not style-dead; it exceeds `LBM-Knee` on Distinct5 style-classifier target accuracy.

## What Is Not Yet Effective

The following ideas are not currently paying off enough:

1. High-pass residual proximal as a main solution.
   - too weak, often outright negative

2. Trust penalty as the recovery-family answer.
   - fails to preserve the parent low-LPIPS basin

3. A single-stage wider or slower release as the full answer.
   - `Wide`, `LateWide`, and likely `SlowMid` do not reopen enough style to justify replacing the current promoted point

4. More epochs alone for the hold-based geometry family.
   - the `Hold4Mid` / `Hold4SlowMid` family is flat and stable, but style remains capped
   - however, CLIP appears to understate their style signal, so this family should not be dismissed only from raw CLIP-S

## Archive Policy

Any unusually extreme point, especially very low-LPIPS outliers, should be archived as a dedicated reproducibility zip and committed.

Any point that influences theory, paper wording, or next-round mechanism choice must be classified as a `paper-facing audit point`.

`paper-facing audit points` must save generated images so they can be used in:

- non-CLIP style probes
- direct Seedream comparisons
- later qualitative figure generation

Immediate archive target:

- `Hold4Mid e8`
  - transfer: `0.6679105 / 0.2877402`
  - all-pairs: `0.7013853 / 0.2877823`
  - archive note:
    - [2026-06-08-hold4mid-e8-geometry-archive.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-hold4mid-e8-geometry-archive.md)

Expected archive contents:

- config JSON
- checkpoint
- `summary.json`
- `metrics.csv`
- `clip_lpips_curve.csv`
- training CSV
- closure note / README

## Active Lane

Current active remote line:

- `Knee e13 + Spatial Carrier-Gate Body+Decoder + Quantile Edge-Gated Structure Leash + DualPathSpatialTexture`
  - config:
    - [inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_seed42_b8a2.json)
  - note:
    - [2026-06-09-dualpath-spatialtexture-trigger.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-dualpath-spatialtexture-trigger.md)

Current read:

- the earlier plain `Knee + CarrierGate Injection` packet has now effectively closed as a negative or near-negative local heavy-review line
- the stronger `body+decoder spatial carrier` follow-up from `LBM-Knee e13` also closed as `near-negative / do not promote`
- the `edge-gated` line is now effectively a closed theory-positive but non-promotable branch
- the `qedgegated + pattn` line is now also locally closed as:
  - `positive over Knee`
  - but still `not promotable`
- the previous `dualpath_texture` line has now drained and been evaluated enough to justify the prepared follow-up
- the current live line is therefore the next explicit rescue step:
  - keep the more selective `quantile edge-gated` structure leash
  - upgrade the late branch from `dualpath_texture` to `dualpath_spatialtexture`
- current live state:
  - remote training for this line has completed
  - completed run:
    - `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_seed42_b8a2`
  - current checked remote GPU task:
    - `IntroStyle bestfew probe` for the same line
  - current checked remote GPU band:
    - about `5.4 GiB / 12 GiB`
  - matching post-train fresh-eval watcher has already landed the early curve
  - current checked progress:
    - first-health passed
    - early fresh-eval curve has now landed through multiple checkpoints:
      - `epoch_0001, 0002, 0003, 0004, 0005, 0006, 0007, 0008, 0009, 0010, 0012`
    - current read:
      - style stays in a narrow `0.6916 to 0.6929` band
      - `LPIPS` rises from about `0.401` toward `0.440`
    - current remote bestfew non-CLIP step:
      - handoff CSV exists
      - IntroStyle manifest exists
      - remote IntroStyle bestfew probe has now landed
      - current bestfew `IntroStyle` read:
        - `epoch_0001 target = 0.11198, margin = -0.05031`
        - `epoch_0012 target = 0.10755, margin = -0.04673`
    - local blind `VLM` and corrected `DINO` evidence from the drained `dualpath_texture` family remain the decision baseline while this new line warms up

## Immediate Audit Actions

1. Keep `Knee e13 + Spatial Carrier-Gate Body+Decoder + Quantile Edge-Gated Structure Leash + DualPathTexture` as the active remote performance/theory lane until the current train process fully drains.

2. Enforce the current remote/local split:
   - remote:
     - train
     - produce `fresh_localreview`
     - produce compact handoff CSV for the best few retained points
   - local:
     - full `VLM`
     - broader comparative plotting
     - theory interpretation
     - next-step direction control
   - current active-line local handoff roots:
     - [dualpath_bestfew_localreview_20260609](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_bestfew_localreview_20260609)
     - [qedgegated_pattn_bestfew_localreview_20260609](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedgegated_pattn_bestfew_localreview_20260609)

3. For every surprising low-LPIPS point, save images and run:
   - `IntroStyle`, when available
   - otherwise the Distinct5 non-CLIP style classifier
   - visual comparison against `Seedream`, `SaMST`, and `SaMAM`

   Current interpretation rule:
   - do not trust a `CLIP` win alone when `IntroStyle delta-IDT` and `style margin` disagree materially
   - especially audit points that look like:
     - high `CLIP`, weak `IntroStyle`
     - or low `LPIPS`, near-`IDT` `IntroStyle`

4. Add DINO-style structure comparison to the same selected frontier points:
   - `AnisoStokesQueue e13`
   - `Pattn + Stokes002 e13`
   - `Hold4Mid e8`
   - `Hold4TwoStage best`, once local review is finalized
   - `QEdgePattn best`
   - `DualPath best`, once image-backed local review is stable enough to pin the family point

5. Treat the hold family scientifically as:
   - `geometry anchor`
   - plus a classifier-supported hidden style family that CLIP may be under-reading

6. Treat the current live `dualpath` line scientifically as:
   - `target-specific style reopening from the best balanced basin with a stronger late branch split`
   - with `e01` and `e09` interpreted separately as:
     - safer all-around point
     - style-leaning point
   - and promote the prepared `dualpath_spatialtexture` follow-up only if the current live lane remains trapped in the same low-style conservative basin.

## Next Round Plan

After `Hold4SlowMid` is formally closed, next round should stop spending GPU time on single-stage clamp schedules and move to a different family.

Recommended next-round direction:

1. Treat `Hold4Mid` as a geometry-control anchor, not as a headline frontier candidate.
   - but do not treat it as style-dead anymore; use non-CLIP evidence before discarding the family

2. Reopen style with a different late mechanism instead of a single-stage release schedule.
   - current active candidate:
     - a new two-stage late schedule in code
     - shape:
       - early `hold`
       - mid controlled band
       - explicit late re-opening window after geometry has stabilized
   - paired with:
      - `IntroStyle`-based style auditing
      - direct visual comparison to `Seedream`, `SaMST`, and `SaMAM`
   - current hypothesis:
      - the family may already have real style movement that CLIP misses
      - the next goal is therefore not just “more style”, but “more visually obvious / style-specific style”

3. If the two-stage schedule still fails, the next move should not be another schedule micro-tweak.
   - after that point, switch to:
     - a new late style-recovery head or branch
     - or a new evaluation-guided diagnosis packet that explains what visual style component is still missing relative to `Seedream`

   The current diagnosis packet points to the missing component more concretely:
   - not just `more style`
   - but:
     - more explicit target-style texture hierarchy
     - more target-specific spatial statistics
     - without crossing into `SaMST`-style structure damage
     - and without collapsing into `LBM-PS-v2`-style generic painterly fog
   - and this must be checked on multiple source families immediately, not only on one iconic source image

4. Keep the current paper-facing headline anchored to:
   - `AnisoStokesQueue e13` for low-LPIPS frontier
   - `Pattn + Stokes002 e13` for raw style frontier
   - `Hold4Mid e8` for geometry/content anchor

5. Before the next paper-facing style claim update, expand `IntroStyle` from `smoke20` to a larger held-out bank on at least:
   - `Seedream`
   - `LBM-Knee`
   - `LBM-PS-v2`
   - `SaMST e15`
   - `SaMAM-2250`
   - `Lat SaMAM`
   - `Lat SaMST`

## Operational Notes

- Formal remote execution must remain single-lane.
- Stale old runs and eval jobs keep reappearing on the `3060`; always audit and kill them before trusting VRAM readings.
- If a line already has all `summary.json` files but is missing `clip_lpips_curve.csv`, finalize closure with:
  - `rerun_full_eval_for_run.py --skip-existing --output-subdir full_eval`
- Do not wait on watcher state alone when artifacts already prove completion.
- If a point needs non-CLIP style audit or visual comparison, it must keep generated images; `metrics.csv + summary.json` alone are not enough.
