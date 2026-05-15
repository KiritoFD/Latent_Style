# Experiment 002: Velocity Field t-Dependence Analysis

## Goal
Test the t-independence assumption of Proposition 2 (kinetic as path-energy surrogate).
Specifically: is E[||v_θ(z, t, s)||^2] approximately constant across t ∈ [0,1]?

## Method
- Model: D0 (full control), epoch 7 checkpoint
- Data: 5 batches × 64 samples = 320 content latents
- Two scenarios:
  - **A. Fixed input**: v_θ(z_0, t, s) — same content, varying t
  - **B. Bridge states**: v_θ(z_t, t, s) — z_t follows bridge process z_t = (1-t)z_0 + t·z_1 + σ√(t(1-t))ε
- t values: 0.0, 0.1, ..., 1.0

## Results

### Scenario A: Fixed input z_0
Mean ||v||² ≈ 1017, nearly constant across t (variance = 0.03 out of mean 1017 = 0.003%)
→ The time conditioning produces negligible change for fixed input.

| t    | Mean ||v||²       | CV     |
|------|-------------------|--------|
| 0.0  | 1017.17 ± 68.16   | 0.067  |
| 0.5  | 1016.87 ± 68.16   | 0.067  |
| 1.0  | 1016.64 ± 68.16   | 0.067  |

### Scenario B: Bridge states z_t
Mean ||v||² varies significantly with t (variance = 19250, CV ∼ 28× mean)
→ The velocity norm depends strongly on the bridge state z_t, not on t itself.

| t    | Mean ||v||²       | CV     | Behavior |
|------|-------------------|--------|----------|
| 0.0  | 1017.17 ± 68.16   | 0.067  | Content → high velocity needed |
| 0.2  | 777.40 ± 48.59    | 0.063  | Moving toward target |
| 0.5  | 584.48 ± 33.01    | 0.057  | Midpoint: lower velocity |
| 0.8  | 592.51 ± 53.15    | 0.090  | Near target |
| 1.0  | 727.63 ± 54.77    | 0.075  | Endpoint: velocity rises again |

## Analysis

### Key finding
**The t-independence assumption is false for the bridge scenario.** However, the variation is caused by the changing INPUT (z_t), not by the time conditioning per se. The velocity field learns a smooth mapping from (z_t, t) that happens to produce lower norms when z_t is closer to the target distribution.

### Implications for Proposition 2

The continuous action is:
A(v) = ∫₀¹ E[||v(z_t, t, s)||²] dt

From the data, when we integrate over [0,1] with the bridge, the average is:
- Area ≈ trapezoidal rule: ≈ 698 (averaged over all t)

The training kinetic loss L_kin is evaluated at t=1 (single-step prediction):
- L_kin (t=1) = E[||v(z_0, 1, s)||²] ≈ 728

This is about 4% higher than the path average of 698.
→ L_kin at t=1 is a slight overestimate of the path energy, not an unbiased estimator.

If kinetic loss were computed from OT bridge velocity:
z_1 = content + v_θ(content, 1, s), not target_style
So the velocity at t=1 is not the bridge velocity but rather a displacement predictor.

Actually, looking at the code: L_kin = E[||v_θ(content, t=1, s)||²]. The content is fed as input at t=1. This gives the high value (1017 in Scenario A at t=1, which is just v_θ(z_0, 1, s)).

Wait, but in the training, the model predicts pred_velocity = model(content, t=1, s), and L_kin = ||pred_velocity||². The pred_endpoint = content + pred_velocity. So the kinetic loss at t=1 has content (not z_1) as input, giving 1017.

The contradiction is that Scenario A t=1 gives 1017, while Scenario B t=1 gives 728. The difference is the input: in Scenario A, input is z_0 (content); in Scenario B at t=1, input is z_1 ≈ target_style.

During training, the model sees content as input at t=1 (OMF mode), so the relevant L_kin value is ~1017.

### Revised conclusion for Proposition 2
- L_kin is NOT a direct path-energy surrogate since it's computed at a single point (t=1)
- However, under a bounded-velocity assumption, L_kin at t=1 provides an upper-bound-like control
- The discrete action A_Δt measured during inference DOES converge to the continuous action A(v)
- Proposition 2 should be weakened: the kinetic term is a one-step displacement regularizer, with a known gap to the full path energy

## Open Questions
1. Does L_kin at t=1 correlate with path straightness? (hypothesis: higher L_kin → more curved trajectory)
2. Does the ablation "D2: no kinetic" show curved trajectories?
3. Can we compute the full path action for D0 vs D2 to directly test the relationship?
