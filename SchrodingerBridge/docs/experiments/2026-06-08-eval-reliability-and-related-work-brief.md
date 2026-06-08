# Evaluation Reliability And Related-Work Brief

Date: 2026-06-08

Purpose:

- stabilize the next week of `Distinct5-512` work around a more reliable scientific loop
- stop treating `CLIP-S` as the sole style judge
- connect evaluation design directly back to mechanism design for `LBM / inmortal`

## Main conclusion

`CLIP-S` should be demoted to a fast-screen signal.

For this project, paper-facing interpretation should use three axes instead of a single style score:

1. `target-style movement`
2. `structure / content preservation`
3. `artifact / visual fidelity`

That means:

- `CLIP-S` remains useful for cheap early pruning
- but final claims should be anchored by:
  - non-CLIP style evidence
  - structure evidence
  - direct visual or pairwise audit evidence

## Why CLIP-S is not enough

Project-relevant failure modes:

1. `no-op / existing-art bias`
   - unchanged artistic inputs can already score highly against target-style prototypes
   - this is already visible in the project IDT floor and should be treated as a core evaluator pathology, not an edge case

2. `global shortcut bias`
   - a high score may come from palette, composition, or broad semantics rather than real brushstroke or texture transfer

3. `human mismatch`
   - CLIP-style similarity does not guarantee that humans would judge the output as a convincing member of the target artistic family

4. `artifact-friendly optimization`
   - style-directed CLIP optimization can reward visually bad or locally broken images

5. `local blindness`
   - global embeddings miss local rhythm, brush hierarchy, and repeated artifact structure

6. `circularity risk`
   - if dataset separation, training intuition, and evaluation are all too CLIP-centered, reviewer skepticism becomes justified

## Recommended evaluation stack

### 1. Style axis

Primary recommendation:

- `target-style image classifier`

Project fit:

- Distinct5 is already a style-class task
- the existing ConvNeXt classifier is a much more task-aligned non-CLIP style axis than raw CLIP cosine

Current assets:

- [distinct5_convnext_style_classifier.pt](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_convnext_style_classifier.pt)
- [2026-06-08-distinct5-nonclip-style-probe.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-08-distinct5-nonclip-style-probe.md)

Use:

- `target accuracy`
- `target probability`
- `target-source margin`
- confusion matrix by target style

### 2. Structure axis

Primary recommendation:

- `DINOv2-based structure comparison`

Why:

- DINO-style self-supervised visual features are more suitable than CLIP for judging structure/layout retention
- use this alongside LPIPS, not as a replacement for LPIPS

Immediate project action:

- run the existing DINO structure tool on the same selected points used in the non-CLIP style probe

### 3. Artifact / visual axis

Primary recommendation:

- `pairwise visual audit against Seedream and key internal points`

What to judge explicitly:

1. style specificity
2. semantic drift
3. layout stability
4. brushstroke hierarchy
5. artifact load / repetition / gloss

Existing project evidence that this matters:

- the blind pairwise files already show Seedream is often the stronger external visual reference even when some scalar metrics favor LBM

## What the current hold-family points mean scientifically

`Hold4Mid e8`:

- not a style-ceiling point
- a geometry anchor
- current evidence strongly suggests it over-constrains style while preserving structure exceptionally well

`Hold4SlowMid`:

- closes the last coherent single-stage smoothing question
- does not beat `Hold4Mid`
- therefore the bottleneck is not “release is a bit too fast”
- the bottleneck is more structural:
  - style reopening must happen differently, not just later on the same one-stage schedule

## Related-work ideas that most directly matter next

### Highest priority

1. `two-stage / staged style reopening`
   - use geometry stabilization early
   - reopen style only after a middle basin is formed
   - this is now directly aligned with project evidence, not just literature intuition

2. `multi-level style representation`
   - stop assuming one global style control is enough
   - low / mid / high style components should likely enter different parts of the model or different phases of training

3. `query-preserving style injection`
   - style should modify the source in a controlled way rather than rewriting source geometry wholesale
   - this points to late style injection on top of a preserved geometry carrier

### Secondary priority

4. `pattern / rhythm-aware diagnostics`
   - especially useful for explaining why a low-LPIPS point may still fail visually

5. `artifact-band / spectral diagnostics`
   - useful for distinguishing “strong style” from “strong ugly texture”

## Immediate project actions

1. Every surprising or paper-facing point must be classified as either:
   - `fast-screen only`
   - `paper-facing audit point`

2. `paper-facing audit points` must save generated images.
   - this is now a hard requirement for:
     - non-CLIP style probe
     - visual comparison
     - later qualitative figure selection

3. The next completed selected points should be audited on:
   - non-CLIP style classifier
   - DINO structure metric
   - direct visual comparison to Seedream

4. Mechanism search should now prefer:
   - true staged reopening
   - or new late style-recovery branches
   over more one-stage schedule tuning

## Bottom line

The project has already learned something real:

- `P_attn + late weak structural repair` is the best style-heavy family
- `Hold4Mid` is the best geometry anchor
- `Hold4SlowMid` says single-stage smoothing is exhausted

So the next week should not be “more schedule fiddling”.

It should be:

- `better evaluation`
- `staged reopening mechanisms`
- `explicit visual diagnosis against Seedream`
