# Theory Reset: 2026-05-16

## Goal

Reset the project around the smallest set of claims that are both empirically supported and useful for the next iteration.

The immediate target is:

- keep training under roughly `1 min / epoch`
- beat or at least match SaMST on `clip_style`
- avoid adding new modules unless both theory and experiments justify them

## What Existing Evidence Actually Supports

### 1. The endpoint style term is non-negotiable

`D1_no_terminal_swd` drops from `clip_style = 0.7129` to `0.6654`.

Interpretation:

- terminal SWD is the only term that consistently pulls the endpoint toward the target style distribution
- any “fast” variant that weakens or replaces this term is not a real candidate

### 2. Kinetic regularization is a style-content tradeoff knob, not a correctness term

From destructive ablation:

- `D0_full_correct_7ep`: `clip_style = 0.7129`, `LPIPS = 0.4528`
- `D2_no_kinetic`: `clip_style = 0.7225`, `LPIPS = 0.6325`

Interpretation:

- reducing kinetic pressure increases style strength
- removing it completely destroys content preservation
- therefore the correct direction is not “new regularizers first”, but “lower kinetic gradually from the D0 baseline”

### 3. Most “new modules” did not improve the metric we care about

From `theory_switch_validation`:

- entropy gate and Sinkhorn routing improve EC/LPIPS slightly
- but they do not produce a better `clip_style` than the baseline
- contextual color variants are negative overall

Interpretation:

- for the current objective, color-loss branches are empirically disfavored
- entropy gating is not a first-line tool for exceeding SaMST style
- routing changes are also secondary unless style rises, which it did not

### 4. SWD cost strength is necessary, but micro-HF emphasis is wrong

From destructive ablation:

- `D10_micro_hf_swd_trap` underperforms D0 on both style and content

Interpretation:

- style transfer needs multi-scale endpoint matching
- forcing SWD toward tiny high-frequency detail overfits texture and hurts the actual goal

### 5. The speed target is already compatible with the baseline family

Documented timing:

- D0-family training is about `290-311 s / 7 epochs`
- this is roughly `41-44 s / epoch`

Interpretation:

- we do not need radical architectural compression to satisfy the `1 min / epoch` constraint
- the speed problem is therefore a secondary optimization, not a blocker

## Simplified Theoretical Position

We model the endpoint optimization as:

`min_theta  lambda_swd * SWD(z_1, Z_style) + lambda_kin * E ||v_theta||^2`

where:

- `lambda_swd` controls how strongly the endpoint is matched to the style distribution
- `lambda_kin` limits how far the latent can move away from content

Empirically:

- `lambda_kin = 0` overshoots style at the cost of unacceptable drift
- `lambda_kin = 1` is stable but leaves some style on the table

So the most defensible next step is a continuation path:

`lambda_kin: 1.0 -> 0.5`

while keeping the rest of the D0/K1 recipe clean.

## Speed Theory

Reducing SWD projection count from `64` to `32` is the safest speed optimization:

- estimator expectation is unchanged
- gradient variance increases by about `sqrt(2)`
- wall-clock cost of projection/sort-heavy SWD work drops materially

This is theoretically cleaner than deleting scales or adding compensating heuristic losses.

## Modules To De-Prioritize

These are not necessarily deleted from code yet, but they are removed from the active theoretical path:

- strong color loss
- contextual color transport variants
- micro high-frequency SWD emphasis
- entropy-gated kinetic as a primary route to better style
- Sinkhorn semantic routing as a primary route to better style

## Active Hypothesis

The best next experiment is:

1. resume from `S-add__K-1_C-0_W-20_Col-0/epoch_0007.pt`
2. keep the D0/K1 architecture unchanged
3. lower `w_kinetic` from `1.0` to `0.5`
4. reduce SWD projections from `64` to `32`

Expected effect:

- style should move upward relative to D0/K1
- LPIPS should degrade less severely than the `w_kinetic = 0` case
- training time should remain safely below the one-minute target
