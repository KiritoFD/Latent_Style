# Flow-Loss Metric Ablation Protocol

Date: 2026-06-03

Purpose:

- close the largest current paper-risk gap;
- test whether the local flow residual choice materially changes the
  Distinct5-512 frontier;
- decide whether the latent-metric story can remain broad or must be narrowed
  to endpoint-side `W1` language.

## Primary claim under test

Current paper-safe state:

- endpoint-side `W1` / SA-SWD is supported;
- the broader `MSE vs Huber vs L1` local flow-residual thesis is not.

This experiment exists to change that status one way or the other.

## Required controls

1. dataset:
   - `Distinct5-512`
   - strict `5x5` all-pairs
   - `750` full outputs
   - `600` transfer-only outputs

2. hardware:
   - remote `RTX 3060`
   - formal run target remains the established `9.0G-10.8G` VRAM band

3. base family:
   - one fixed current content-preserving LBM family
   - preferred starting point: `H` or `F`
   - do not mix base families inside the same ablation block

4. only changed variable:
   - `bridge.loss_type`
   - variants:
     - `mse`
     - `huber`
     - `l1`

5. matched training protocol:
   - same seed set
   - same batch size
   - same epoch budget
   - same evaluation cadence
   - same optimizer and scheduler settings

6. minimum seed count:
   - at least `3` seeds per loss variant before writing paper-level conclusions

## Required outputs

For every seed and every loss variant, record:

- `clip_style`
- `content_lpips`
- `delta_idt_full`
- `delta_idt_transfer`
- `aggregate_artfid`
- `train_wall`
- exact config path
- exact summary path

Preferred secondary diagnostics:

- `EC`
- `MUSIQ`
- `MANIQA`
- one matched visual panel for obvious artifact or blur differences

## Stop rule

The experiment block is not reviewer-ready until all of the following are true:

1. all three loss variants have completed the matched budget;
2. each variant has at least `3` seeds;
3. each seed has a strict full-eval bundle;
4. median and spread can be reported for the primary metrics.

Do not stop early on one attractive seed.

## What counts as positive evidence

Positive evidence for the broader latent-metric story requires a consistent,
matched advantage from `huber` or `l1` over `mse`, such as:

- better `LPIPS` at similar or better `clip_style`, or
- better `delta_idt` at similar `LPIPS`, or
- visibly fewer severe artifacts at comparable quantitative metrics.

One lucky seed is not enough.

## What forces claim shrinkage

The paper must shrink the broader latent-metric claim if any of the following
holds:

1. `mse`, `huber`, and `l1` are effectively tied within spread;
2. `huber`/`l1` only help one metric while clearly regressing the rest of the
   frontier;
3. results are unstable enough that median behavior is ambiguous.

If that happens, keep the paper centered on:

- OT-coupled endpoint construction,
- `W1`-style terminal matching,
- and `idt`-anchored evaluation.
