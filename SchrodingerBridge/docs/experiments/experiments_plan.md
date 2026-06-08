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
    - `CLIP-S + LPIPS`
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
     - `CLIP-S`
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

Current geometry anchor:

- `Hold4Mid e8`
  - transfer: `0.6679105 / 0.2877402`
  - all-pairs: `0.7013853 / 0.2877823`
  - evidence:
    - [2026-06-08-inmortal-anisostokes-queue-clamphold4mid-from-e13.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-inmortal-anisostokes-queue-clamphold4mid-from-e13.md)

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

Current active lane:

- `Hold4TwoStage`
  - config:
    - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4twostage_reseed_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4twostage_reseed_from_e13_seed42_b8a2.json)
  - note:
    - [2026-06-08-inmortal-anisostokes-queue-clamphold4twostage-from-e13.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-inmortal-anisostokes-queue-clamphold4twostage-from-e13.md)

Current read:

- this is the first real post-`Hold4Mid` structural schedule change, not another one-stage smoothing tweak
- it should answer the key science question:
  - can the family keep the geometry anchor and still recover style if reopening is delayed until after a middle stabilization band?

## Immediate Audit Actions

1. Keep `Hold4TwoStage` as the active performance/theory lane.

2. For every surprising low-LPIPS point, save images and run:
   - `IntroStyle`, when available
   - otherwise the Distinct5 non-CLIP style classifier
   - visual comparison against `Seedream`

3. Add DINO-style structure comparison to the same selected frontier points:
   - `AnisoStokesQueue e13`
   - `Pattn + Stokes002 e13`
   - `Hold4Mid e8`
   - `Hold4TwoStage best`, once available

4. Treat the hold family scientifically as:
   - `geometry anchor`
   - not `style frontier`

## Next Round Plan

After `Hold4SlowMid` is formally closed, next round should stop spending GPU time on single-stage clamp schedules and move to a different family.

Recommended next-round direction:

1. Treat `Hold4Mid` as a geometry-control anchor, not as a headline frontier candidate.

2. Reopen style with a different late mechanism instead of a single-stage release schedule.
   - current active candidate:
     - a new two-stage late schedule in code
     - shape:
       - early `hold`
       - mid controlled band
       - explicit late re-opening window after geometry has stabilized
   - paired with:
     - `IntroStyle`-based style auditing
     - direct visual comparison to `Seedream`

3. If the two-stage schedule still fails, the next move should not be another schedule micro-tweak.
   - after that point, switch to:
     - a new late style-recovery head or branch
     - or a new evaluation-guided diagnosis packet that explains what visual style component is still missing relative to `Seedream`

4. Keep the current paper-facing headline anchored to:
   - `AnisoStokesQueue e13` for low-LPIPS frontier
   - `Pattn + Stokes002 e13` for raw style frontier
   - `Hold4Mid e8` for geometry/content anchor

## Operational Notes

- Formal remote execution must remain single-lane.
- Stale old runs and eval jobs keep reappearing on the `3060`; always audit and kill them before trusting VRAM readings.
- If a line already has all `summary.json` files but is missing `clip_lpips_curve.csv`, finalize closure with:
  - `rerun_full_eval_for_run.py --skip-existing --output-subdir full_eval`
- Do not wait on watcher state alone when artifacts already prove completion.
- If a point needs non-CLIP style audit or visual comparison, it must keep generated images; `metrics.csv + summary.json` alone are not enough.
