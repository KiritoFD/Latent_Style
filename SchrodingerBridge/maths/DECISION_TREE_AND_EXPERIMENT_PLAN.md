# Mathematical Decision Tree and Experiment Plan

Reviewed on `2026-05-16`.

This document translates the current mathematical reading into an operational decision tree.

The goal is not to be short. The goal is to make every branch mathematically defensible and tied to actual repository evidence.

## 1. Optimization Target

The active reduced model is:

`min_theta lambda_swd * SWD(z_1, Z_style) + lambda_kin * E||v_theta||^2`

with additional architecture-level control coming from:

- semantic cross-attention routing sharpness
- skip-path retention versus overwrite
- delivered residual amplitude

The practical project objective is:

1. push `clip_style` above the current D0 / K1 baseline family
2. aim at or above `SaMST strict = 0.7194`
3. keep training near or below `1 min / epoch`
4. avoid entering the known collapse regimes

## 2. What Counts as a Good Regime

The repository already gives us four anchor regimes.

### 2.1 Safe baseline regime

- `D0_full_correct_7ep`: `style 0.7014`, `content 0.8022`, `LPIPS 0.4593`
- `K1_r00_balanced_default`: `style 0.716126`, `content 0.798365`, `LPIPS 0.460504`

Interpretation:

- strong enough content retention
- style still below SaMST in the strict baseline comparison
- good reference regime for future branches

### 2.2 Style frontier regime

- `residual_1p25`: `style 0.721854`, `content 0.763490`
- `anchor_hybrid_all`: `style 0.7186`, `content 0.6433`, `LPIPS 0.6876`

Interpretation:

- the architecture can reach or nearly reach the desired style range
- but content is already under visible stress

### 2.3 Collapse regime

- `D2_no_kinetic`: `style 0.7159`, `content 0.6624`, `LPIPS 0.6375`
- `anchor_skip_only`: `style 0.7363`, `content 0.5947`, `LPIPS 0.8528`

Interpretation:

- style can rise, but the route is unusable
- this is not a candidate endpoint, only a warning region

### 2.4 Over-constrained regime

- `high_tension` and `orthogonal_phase` families
- `K2` family as a whole relative to `K1`

Interpretation:

- content is safer
- style is too low
- regularization pressure dominates endpoint style pressure

## 3. Primary Variables and Their Mathematical Meaning

### 3.1 `lambda_kin` / `w_kinetic`

Meaning:

- controls the motion budget
- higher value discourages latent displacement
- lower value permits stronger style movement

Evidence:

- removing kinetic raises style but destroys content

So `w_kinetic` is the main first-order style-content knob.

### 3.2 `lambda_swd` / `terminal_swd_weight`

Meaning:

- controls how strongly the endpoint is forced toward target style statistics

Evidence:

- no terminal SWD causes large style failure
- increasing SWD from `15` to `30` helps in `experiments_root`
- increasing further to `45` does not keep helping

So `terminal_swd_weight` is a useful but saturating endpoint-pressure knob.

### 3.3 residual amplitude

Meaning:

- controls the delivered magnitude of the learned update

Evidence:

- `residual_1p25` and `residual_1p5` increase style
- `residual_2p0` overshoots and degrades both style and content

So residual amplitude is a local high-sensitivity lever.

### 3.4 routing sharpness / regularization

Meaning:

- controls whether cross-attention paints style sharply or diffusely

Evidence:

- Sinkhorn and entropy-gated variants slightly reduce style but improve content and LPIPS

So routing regularization is a second-line repair lever, not a first-line style lever.

### 3.5 step size and step count

Meaning:

- numerical integration controls

Evidence:

- both sweeps are essentially flat

So they are not first-line optimization variables in the present regime.

## 4. Mathematical Decision Tree

```text
Start
|
|-- A. Are we on the K1 / D0-style lineage?
|      |
|      |-- No:
|      |     Move back to K1 / trusted D0-family config.
|      |     Reason: K2 and high-tension families are systematically too conservative.
|      |
|      |-- Yes:
|            Continue.
|
|-- B. Can we recover baseline behavior?
|      |
|      |-- No:
|      |     Do not run theory experiments yet.
|      |     Fix config lineage, checkpoint path, logging, dataloader instability, or evaluation mismatch.
|      |
|      |-- Yes:
|            Continue.
|
|-- C. Is style still below the target frontier?
|      |
|      |-- No:
|      |     Continue to content repair branch.
|      |
|      |-- Yes:
|            First lower kinetic or slightly raise residual amplitude.
|
|-- D. After style-raising move, did content collapse?
|      |
|      |-- No:
|      |     Consider modest endpoint SWD increase next.
|      |
|      |-- Yes:
|      |     Undo the last aggressive move.
|      |     Then use smoother routing or more content-preserving pressure.
|
|-- E. Is style still below SaMST after the safe frontier move?
|      |
|      |-- Yes:
|      |     Increase endpoint pressure before adding new modules.
|      |
|      |-- No:
|            Lock the regime and only then test speed-preserving simplifications.
|
|-- F. Is training speed above the practical budget?
|      |
|      |-- No:
|      |     Keep the mathematically cleaner setup.
|      |
|      |-- Yes:
|            Reduce SWD projections first, not model semantics.
```

## 5. Branch-by-Branch Interpretation

### Branch A: Choose the correct family first

Decision:

- prefer `K1` over `K2` if the explicit objective is maximum achievable style

Mathematical reason:

`K2` behaves like a more averaged, more regularized transport regime.

Data reason:

- K1 best-per-experiment mean style: `0.710957`
- K2 best-per-experiment mean style: `0.706426`

while K2 preserves content better.

Conclusion:

`K2` is useful for safety, not for breaking the style frontier.

### Branch B: Baseline recovery is logically prior to theory

Decision:

- if baseline is not reproducible, stop experimental branching

Mathematical reason:

all later comparisons assume a stable local reference point in objective space.

If the baseline itself is not stable, then observed deltas cannot be attributed to the intended variable.

Practical triggers for this branch:

- wrong config lineage
- broken checkpoint continuation
- dataloader instability
- evaluation mismatch

### Branch C: The first style-raising move

Decision:

- lower `w_kinetic` first
- or make a modest residual-amplitude increase

Mathematical reason:

both levers directly increase effective displacement budget.

`w_kinetic` acts on the objective:

`style pressure / motion penalty ~ lambda_swd / lambda_kin`

Residual amplitude acts on the realized endpoint displacement:

`z_1(a) = z_0 + a * Delta_theta`

Data reason:

- `D2_no_kinetic` proves the style direction
- residual sweep proves amplitude sensitivity

Conclusion:

these are the cleanest first-line frontier levers.

### Branch D: Detect collapse versus progress

Decision:

- if style rises but content falls toward `D2` or `anchor_skip_only`, back off immediately

Mathematical reason:

the system has crossed from "useful displacement" into "overwritten structure".

Observable signs:

- LPIPS moving toward `0.60+`
- content falling toward `0.66` or below
- behavior approaching `D2_no_kinetic`

Corrective actions:

1. raise kinetic back up
2. reduce residual amplitude
3. only after that, consider routing smoothing

### Branch E: Raise endpoint pressure only after motion is in the right regime

Decision:

- increase `terminal_swd_weight` after style has responded to motion-budget tuning

Mathematical reason:

SWD pressure is effective only when the model has enough motion capacity to reach better style regions.

If the model is still motion-starved, more endpoint pressure can mostly create distortion.

Data reason:

- `15 -> 30` helps
- `30 -> 45` does not keep helping

Conclusion:

`terminal_swd_weight` should be a second move, not the first move.

### Branch F: Speed optimization comes last

Decision:

- only optimize speed after the quality regime is right

Mathematical reason:

step count and step size are already shown to be weak levers, so the cleanest speed lever is estimator cost reduction, not semantic redesign.

Best current speed lever:

- reduce SWD projections `64 -> 32`

Reason:

- same estimator target
- higher variance, but cleaner interpretation than changing semantics

## 6. Concrete Decision Thresholds

These thresholds are not universal laws. They are regime markers derived from current evidence.

### 6.1 Good zone

- style at or above baseline family
- content around `0.79 - 0.81`
- LPIPS around `0.45 - 0.47`

### 6.2 Promising frontier zone

- style near `0.718 - 0.722`
- content still clearly above the D2 regime
- LPIPS not exploding toward skip-collapse values

### 6.3 Warning zone

- content near `0.70`
- LPIPS near `0.55 - 0.60`

Interpretation:

style may be rising, but we are close to entering collapse.

### 6.4 Collapse zone

- content near `0.66` or lower
- LPIPS near `0.64+`
- behavior resembling `D2_no_kinetic` or worse

Action:

- stop the branch
- do not attempt to rescue it by piling on new losses first

## 7. Ordered Experiment Matrix

The order below follows the mathematics of the decision tree.

### Stage 0: Re-anchor

Use:

- trusted K1 / D0-family config
- trusted checkpoint lineage

Purpose:

- recover a stable local reference point

### Stage 1: Motion-budget search

Test in this order:

1. baseline reproduction
2. `w_kinetic` slightly lower than baseline
3. `w_kinetic` moderately lower than baseline
4. modest residual-amplitude increase

Reason:

- these are the cleanest ways to move toward the style frontier

### Stage 2: Endpoint-pressure search

Only after Stage 1 finds a non-collapsed style gain:

1. baseline SWD pressure
2. moderately increased SWD pressure

Reason:

- endpoint pressure should refine a useful displacement regime, not try to create one from nothing

### Stage 3: Safety repair

Only if style is high enough but content is slipping:

1. Sinkhorn routing
2. entropy-gated kinetic

Reason:

- these are content-repair mechanisms
- their measured effect is to trade a little style for better preservation

### Stage 4: Speed-preserving simplification

Only after the quality regime is chosen:

1. reduce SWD projections
2. test whether quality remains within tolerance

Reason:

- this is the least theory-disruptive speed move

## 8. What the Decision Tree Forbids

The current evidence is strong enough to rule out several unprincipled moves.

### 8.1 Forbidden as first moves

- increasing step count
- tuning step size in tiny increments
- switching from K1 to K2 to chase higher style
- reviving strong color loss
- using skip-heavy routes to win style metrics

### 8.2 Forbidden interpretation

Do not interpret these as success:

- style increase with LPIPS collapse
- style increase caused by skip overwrite
- one-off gains from a regime already known to be unstable

## 9. Final Mathematical Summary

The repository evidence supports the following compact decision law:

1. If style is too low, first increase effective displacement budget.
2. If displacement budget increase causes collapse, reduce amplitude before adding complexity.
3. If displacement is healthy but style still trails, increase endpoint distribution pressure.
4. If style is good enough but content degrades, use routing-smoothing or entropy-gated repair.
5. Only after the quality regime is solved should speed-oriented simplifications be tested.

In symbols, the project should currently prioritize:

`first: displacement control`

`second: endpoint pressure`

`third: repair regularization`

`fourth: estimator-cost reduction`

That order is the most mathematically faithful reading of the current code and all available experiment data.
