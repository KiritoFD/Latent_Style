# Full Evidence Analysis

Reviewed on `2026-05-16`.

This document is intentionally long. The goal is not elegance. The goal is to faithfully reflect the code and the existing experiment data, especially the CSV summaries already in the repository.

## 1. Scope

This analysis is based on the following code and data sources:

### Code

- `src/model.py`
- `src/lancet_backbone.py`
- `src/losses.py`
- `src/ot_cost.py`
- `src/trainer.py`

### Main CSV / report sources

- `summary_all_tested_metrics_with_ablations.csv`
- `ablation_destructive_7epoch/combined_summary/combined_750_with_destructive_ablations.csv`
- `weight_sweep_40_all_epochs.csv`
- `S-add__K-1_C-0_W-20_Col-0/step_size_sweep_epoch7/step_size_sweep_summary.csv`
- `S-add__K-1_C-0_W-20_Col-0/residual_scale_sweep_epoch7/residual_scale_sweep_summary.csv`
- `exp/configs/experiments_root/full_eval/batch_summary.csv`
- `exp/configs/high_tension_phase_space_sweep/full_eval/batch_summary.csv`
- `exp/configs/orthogonal_phase_space_sweep_60/full_eval/batch_summary.csv`
- `theory_switch_validation/theory_switch_validation_report.md`
- per-run `theory_switch_validation/*/full_eval/batch_summary.csv`
- destructive ablation training logs in `ablation_destructive_7epoch/*/logs/training_*.csv`
- `training_times_documentation.md`

## 2. Code-Faithful Structural Reading

### 2.1 Body cross-attention is the central style painting mechanism

In `lancet_backbone.py`, the body blocks are `SemanticCrossAttn`.

Each block computes:

- content queries from normalized content features
- style keys and values from normalized style maps
- a routing matrix by either `softmax` or `sinkhorn`
- a painted style feature
- a gated residual update

Mathematically:

`A = softmax(q k^T / tau)` or `A = Sinkhorn(q k^T / tau)`

`P = A v`

`x_out = x + g_global * g_local(x) * (1 + gamma) * P`

So the body is not a passive conditioner. It is a spatially routed style painting operator.

### 2.2 Skip routing is a second style-content control axis

After the body, the decoder fuses skip features. When skip routing is active, the network can decide how much clean structure to carry through the skip path.

This means there are two separate control axes:

1. how aggressively the body paints style
2. how much the skip path preserves structure

That separation explains why some experiments get very high style at catastrophic LPIPS: they are not only changing endpoint loss; they are changing where content can bypass style distortion.

### 2.3 Losses

The OMF family contains:

- terminal SWD
- kinetic regularization
- optional low-frequency anchor
- optional color branch
- optional patch NCE
- optional cycle
- optional repulsive term

This matters because the data repeatedly shows that not all "reasonable" terms are equally useful.

## 3. Global Outcome Table

From `summary_all_tested_metrics_with_ablations.csv`, the top `clip_style` rows are:

- `StyleID strict`: `0.7597`, but `clip_content = 0.5519`, `LPIPS = 0.7497`
- `SaMST strict`: `0.7194`, `clip_content = 0.8193`, `LPIPS = 0.4664`
- `D2_no_kinetic`: `0.7159`, `clip_content = 0.6624`, `LPIPS = 0.6375`
- `S2WAT strict`: `0.7139`
- `AdaIN v32k`: `0.7130`
- `D0_full_correct_7ep`: `0.7014`, `clip_content = 0.8022`, `LPIPS = 0.4593`

This already says something important:

- chasing raw `clip_style` alone is easy
- beating SaMST on `clip_style` while preserving content is the real problem

So any theory that celebrates a style increase without looking at LPIPS/content is incomplete.

## 4. Destructive Ablation: What Is Actually Necessary

### 4.1 Core numbers

From `combined_750_with_destructive_ablations.csv`:

- `D0_full_correct_7ep`: `style 0.7014`, `content 0.8022`, `LPIPS 0.4593`
- `D1_no_terminal_swd`: `style 0.6708`, `content 0.8989`, `LPIPS 0.3490`
- `D2_no_kinetic`: `style 0.7159`, `content 0.6624`, `LPIPS 0.6375`
- `D3_no_swd_no_kinetic`: `style 0.6884`, `content 0.8527`, `LPIPS 0.3938`
- `D4_conv_body_no_global_attn`: `style 0.7022`
- `D5_disable_skip_routing`: `style 0.6951`
- `D6_disable_spatial_prior`: `style 0.7022`
- `D7_no_residual_path`: `style 0.7013`
- `D8_strong_color_loss`: `style 0.6923`, `content 0.6629`, `LPIPS 0.5675`
- `D9_l2_ot_cost`: `style 0.7016`
- `D10_micro_hf_swd_trap`: `style 0.6989`, `content 0.7772`, `LPIPS 0.4863`
- `D11_single_terminal_step`: `style 0.7012`

### 4.2 Immediate conclusions

#### Terminal SWD is necessary

`D1_no_terminal_swd` drops style from `0.7014` to `0.6708`.

Mathematically this is expected. Without terminal SWD, the network can minimize content-preserving pressures while never being forced to land near the target style distribution.

#### Kinetic is a tradeoff knob, not a style source

`D2_no_kinetic` increases style to `0.7159`, but LPIPS collapses to `0.6375`.

The right reading is:

- kinetic regularization suppresses displacement magnitude
- reducing it reveals latent style capacity
- removing it entirely creates uncontrolled drift

So kinetic is not "good" or "bad". It is the main knob on the style-content frontier.

#### Color branch is actively harmful in the tested form

`D8_strong_color_loss` is worse than D0 in both style and content preservation.

This means the tested color term is not aligning with the desired style metric. It is likely constraining the wrong subspace too strongly.

#### Micro high-frequency SWD is a trap

`D10_micro_hf_swd_trap` lowers style and also hurts content.

This suggests style transfer quality is not just high-frequency texture matching. For this model, balanced multi-scale endpoint statistics matter more than over-emphasizing tiny patches.

#### Several architecture removals barely move the metric

`D4`, `D6`, `D7`, `D9`, `D11` are all close to `D0`.

That means those toggles are not the main limiting factor for current style performance.

### 4.3 Cross-attention statistics from joined training logs

Joining destructive ablation final logs with final evaluation gives:

- `semantic_attn_mean` is effectively constant and not informative here
- `semantic_k_abs` has a strong positive correlation with final `clip_style`
- `plan_entropy` is moderately negatively correlated with final `clip_style`

Observed correlation inside the destructive ablations:

- `corr(clip_style, semantic_k_abs) ~= +0.88`
- `corr(clip_style, plan_entropy) ~= -0.34`
- `corr(clip_style, clip_content) ~= -0.62`

Interpretation:

- stronger style-key activation tends to coincide with stronger style output
- more diffuse transport plans tend to soften style
- style gains are consistently paid for by content loss

This is exactly the kind of relation we should expect from cross-attention painting. A sharper routing plan concentrates style injection; a higher-entropy plan averages it out.

## 5. Historical `experiments_root`: Endpoint Pressure vs Collapse

From `exp/configs/experiments_root/full_eval/batch_summary.csv`, the best style rows are:

- `06_anchor_skip_only`: `style 0.7363`, `content 0.5947`, `LPIPS 0.8528`
- `07_anchor_hybrid_all`: `style 0.7186`, `content 0.6433`, `LPIPS 0.6876`
- `02_omf_swd_30`: `style 0.7042`, `content 0.7077`, `LPIPS 0.5947`
- `03_omf_swd_45`: `style 0.7029`, `content 0.7017`, `LPIPS 0.5935`
- `01_omf_swd_15`: `style 0.6938`, `content 0.7344`, `LPIPS 0.5590`
- `04_anchor_kin_only`: `style 0.6904`, `content 0.7730`, `LPIPS 0.5174`

### 5.1 What this means

#### Skip-only anchor is a style leak, not a usable model

`06_anchor_skip_only` wins on raw style and loses catastrophically on content.

Mathematical interpretation:

if style is injected primarily through the skip route, the network can shortcut structural preservation and directly overwrite content geometry.

#### Hybrid anchor nearly reaches SaMST style, but still collapses too much

`07_anchor_hybrid_all` gets to `0.7186`, essentially at the SaMST frontier, but LPIPS is far too large.

This is valuable evidence because it proves the architecture can reach that style region. The problem is not absolute style capacity; the problem is stabilizing it.

#### SWD strength helps up to a point

Within the plain OMF variants:

- `swd=15`: best `0.6938`
- `swd=30`: best `0.7042`
- `swd=45`: best `0.7029`

So increasing SWD from `15` to `30` helps, but going to `45` does not keep helping. That is a classic diminishing-return or overshoot pattern.

Mathematical explanation:

the effective optimization is not linear in `terminal_swd_weight`. Once endpoint pressure is already strong, extra weight can mostly amplify distortion directions instead of producing cleaner style alignment.

## 6. `weight_sweep_40`: K-family Is More Important Than Recipe Details

### 6.1 Best-per-experiment statistics

When we take the best epoch of each experiment in `weight_sweep_40_all_epochs.csv`:

- `K1` family mean best style: `0.710957`
- `K2` family mean best style: `0.706426`

At the same time:

- `K1` mean best content: `0.799625`
- `K2` mean best content: `0.836888`

and:

- `K1` mean best LPIPS: `0.462262`
- `K2` mean best LPIPS: `0.420191`

### 6.2 Interpretation

`K2` is systematically more conservative:

- lower style
- higher content
- lower LPIPS

`K1` is systematically more expressive:

- higher style
- lower content
- higher LPIPS

This suggests the `K` family is not a minor hyperparameter. It is a regime switch.

Mathematical explanation:

if `K` effectively increases the number of style-conditioned updates or the amount of path regularization, then the model moves closer to an averaged transport regime. That improves stability but suppresses peak style amplitude.

### 6.3 Recipe variation inside a fixed K family is second-order

Within `K1`, the best runs span roughly:

- `0.7085` to `0.7161` style

That is meaningful, but much smaller than the systematic `K1` vs `K2` gap.

So:

- sampler / weight recipe matters
- but the coarse dynamical regime matters more

### 6.4 Best K1 evidence

The best K1 row is still:

- `K1_r00_balanced_default`: `style 0.716126`

This matters because it says the original balanced default recipe is already very competitive. We do not need an exotic sampler to get close to SaMST.

## 7. Step Size Sweep: Horizon Is Not the Main Lever

From `step_size_sweep_summary.csv`:

- `step_1p5`: `style 0.716197`
- `step_2p0`: `style 0.716126`
- `step_1p25`: `style 0.716120`
- `base_epoch7`: `style 0.716114`

These are almost identical.

Interpretation:

the local direction learned by the model is much more important than modest changes in inference horizon around this operating point.

Mathematical explanation:

if the field is already close to self-calibrated in magnitude, then changing the inference scalar by `1.25` vs `1.5` vs `2.0` does not move the endpoint into a new regime. The model is direction-limited rather than horizon-limited in this neighborhood.

## 8. Step Count Sweep: More Integration Steps Do Not Materially Help

From `review_additional_experiments/review_additional_experiments/step_count_sweep`:

- `steps_01`: `style 0.715977`, `content 0.808622`, `LPIPS 0.451390`
- `steps_04`: `style 0.716029`, `content 0.808607`, `LPIPS 0.451416`
- `steps_08`: `style 0.715928`, `content 0.808500`, `LPIPS 0.451408`
- `steps_12`: `style 0.716167`, `content 0.808688`, `LPIPS 0.451406`
- `steps_16`: `style 0.716105`, `content 0.808645`, `LPIPS 0.451392`

This is another near-flat sweep.

### 8.1 Interpretation

Increasing the number of integration steps from `1` to `16` does not change the result in any meaningful way.

That means the model is not currently bottlenecked by coarse numerical integration.

### 8.2 Mathematical explanation

If repeated application of the learned residual field gave progressively better alignment, then step count should produce a monotone or at least structured gain curve.

Instead, the curve is flat. The most likely reading is:

- the learned update is already behaving like a one-shot endpoint corrector
- repeated subdivision of the same field does not unlock a better manifold
- therefore style quality is limited by the learned field itself, not by insufficient Euler resolution

This supports the same conclusion as the step-size sweep:

the next gains should come from better endpoint pressure and better amplitude control, not from simply using more inference steps.

## 9. Residual Scale Sweep: Amplitude Matters More Than Horizon

From `residual_scale_sweep_summary.csv`:

- `residual_1p25`: `style 0.721854`, `content 0.763490`
- `residual_1p5`: `style 0.720807`, `content 0.721171`
- `base_epoch7`: `style 0.716114`, `content 0.808575`
- `residual_2p0`: `style 0.706930`, `content 0.655791`

This is one of the strongest insights in the repository.

### 8.1 Interpretation

Increasing residual scale slightly above baseline improves style immediately.

But pushing it too far causes overshoot:

- content falls sharply
- style eventually also falls

This tells us the field direction is not the only issue. The delivered amplitude of the learned residual is itself under-tuned near baseline and over-tuned past a threshold.

### 8.2 Mathematical explanation

Let the learned endpoint update be

`z_1(a) = z_0 + a * Delta_theta`

where `a` is the residual scale.

Then style score is not monotone in `a`. Instead it appears to be locally concave:

- for small increases in `a`, the endpoint reaches a better style region
- after the optimum, the endpoint overshoots and exits the useful style-content manifold

That is exactly what the residual sweep shows.

This also explains why `step_size` is almost flat while `residual_scale` matters: the model is sensitive to the amplitude of the learned update branch itself, not to small external changes in horizon around the same branch.

## 10. Theory Switch Validation: What Cross-Attention Variants Really Do

From `theory_switch_validation_report.md` and the per-run batch summaries:

Using best style epoch per run:

- `T0_k2_baseline`: `style 0.703216`, `content 0.859817`, `LPIPS 0.397394`
- `T1_sinkhorn_routing`: `d_style -0.003864`, `d_content +0.007326`, `d_lpips -0.010369`
- `T2_entropy_gate_2p5`: `d_style -0.001561`, `d_content +0.005538`, `d_lpips -0.007025`
- `T3_entropy_gate_5p0`: `d_style -0.001588`, `d_content +0.006331`, `d_lpips -0.009715`
- `T4_sinkhorn_entropy`: `d_style -0.003917`, `d_content +0.011286`, `d_lpips -0.012170`
- `T5_color_soft_w2`: `d_style +0.002219`, `d_content -0.034347`, `d_lpips +0.033941`
- `T6_color_gumbel_w2`: `d_style +0.002093`, `d_content -0.026158`, `d_lpips +0.026706`
- `T7_all_switches_mild`: near-neutral style, slightly worse LPIPS than baseline

### 9.1 Interpretation

#### Sinkhorn and entropy gate are content-preserving regularizers

They improve content and LPIPS, but they slightly reduce style.

That is not a failure. It means they are behaving exactly like transport-smoothing or uncertainty-penalizing mechanisms should behave.

Mathematical explanation:

- Sinkhorn routing spreads mass more evenly
- entropy-gated kinetic penalizes uncertain regions more strongly
- both reduce effective style concentration

So these switches are valid if the goal is a better Pareto point. They are not first-line tools if the goal is to maximize `clip_style`.

#### Color variants add style in the wrong way

The color variants slightly increase style, but they degrade content and LPIPS.

That suggests the CLIP style gain they create is too entangled with global palette drift rather than controlled semantic style transfer.

#### Cross-attention insight

These results are actually a cross-attention story:

- softer or more constrained routing gives safer outputs
- freer routing gives stronger style but worse preservation

So the body cross-attention is already expressing the main project tradeoff.

## 11. High-Tension and Orthogonal Sweeps: Over-constraint Regimes

### 10.1 High-tension sweep

Best rows from `high_tension_phase_space_sweep`:

- `g2_swd_nuke`: best `style 0.655134`, `content 0.846339`, `LPIPS 0.378047`
- `g1_high_tension_base`: best `style 0.639189`, `content 0.879374`, `LPIPS 0.334654`

### 10.2 Orthogonal phase sweep

Best rows from `orthogonal_phase_space_sweep_60`:

- `g3_gravity_black_hole`: `style 0.667729`, `content 0.803358`, `LPIPS 0.436491`
- `g1_absolute_release`: `style 0.662420`
- `g6_structure_amnesty`: `style 0.657839`

### 10.3 Shared statistical pattern

Inside these sweeps:

- style is strongly positively correlated with LPIPS
- style is strongly negatively correlated with content

This is not surprising, but it is useful. These sweeps mostly move along the same frontier instead of discovering a new one.

Mathematical interpretation:

these phase-space variants are mostly redistributing regularization pressure rather than changing the underlying endpoint objective. They do not create a new mechanism for style alignment, so they stay in a low-style safe regime.

## 12. Training Speed Is Not the Main Bottleneck

From `training_times_documentation.md`:

- D0 family: about `290 - 311 s / 7 epochs`
- roughly `41 - 44 s / epoch`

That already satisfies the target of about one minute per epoch.

So the mathematically correct optimization problem is not:

"How do we make training fast enough?"

It is:

"How do we spend the already acceptable speed budget to increase style without entering collapse?"

This is a very different question.

## 13. Evidence Tiers

Not all statements in this document are equally strong. The repository now supports three evidence tiers.

### 13.1 Strong evidence

These are supported by direct ablations or repeated sweep structure:

- terminal SWD is necessary
- removing kinetic raises style but collapses content
- K1 is more style-seeking than K2
- step size is not an important lever near the current baseline
- step count is not an important lever near the current baseline
- residual amplitude is an important lever
- strong color loss is harmful in the tested form

### 13.2 Medium evidence

These are well supported but still regime-dependent:

- Sinkhorn and entropy-gated variants are better viewed as safety regularizers than style maximizers
- `semantic_k_abs` tracks style intensity better than `semantic_attn_mean`
- high-tension and orthogonal sweeps mostly move along the same tradeoff frontier

### 13.3 Weak or provisional evidence

These are useful working hypotheses, but should not be treated as settled:

- the exact functional form of style gain versus residual amplitude
- the exact causal meaning of `plan_entropy`
- the degree to which `semantic_k_abs` is a universal predictor outside the current ablations

## 14. Statistical Cautions

Some correlations should not be over-read.

### 12.1 Small-sample danger

The destructive ablation tables are high quality but small in sample size. Correlations like `semantic_k_abs -> clip_style` are informative, but they are not universal laws yet.

### 12.2 Mixed-method danger

The global summary mixes:

- our model
- baselines like StyleID, AdaIN, SaMST, S2WAT

So global correlations across that file can be misleading if interpreted causally.

The safer analyses are:

- within destructive ablations
- within weight sweeps
- within experiments_root
- within theory-switch runs

That is what this document prioritizes.

## 15. Main Insights

### Insight 1

The current model already has enough style capacity to touch or slightly exceed SaMST-level style, but every known way of doing so currently damages content too much.

Evidence:

- `residual_1p25` reaches `0.721854`
- `anchor_skip_only` reaches `0.7363`
- `anchor_hybrid_all` reaches `0.7186`

### Insight 2

The style bottleneck is not a lack of expressive style routing. It is lack of control over where and how much that style routing acts.

That is why:

- skip-heavy variants overshoot badly
- entropy / Sinkhorn variants are safer but slightly weaker

### Insight 3

The most informative frontier is not "new branch vs old branch". It is the continuum between D0 and D2.

The strongest evidence still says:

- D0 = stable but style-limited
- D2 = style-strong but content-collapsed

The real win should live between them.

### Insight 4

Residual amplitude is a stronger immediate lever than local step-size.

That means future experiments should treat update amplitude and endpoint pressure as the main optimization axes, not just number of integration steps.

### Insight 5

K-family regime matters more than sampler recipe details.

`K1` is consistently more style-seeking.
`K2` is consistently more content-preserving.

So if the explicit goal is to exceed SaMST on style, starting from `K1` is more logical than starting from `K2`.

## 16. Mathematical Reading of the Frontier

The best reduced model remains:

`min_theta lambda_swd * SWD(z_1, Z_style) + lambda_kin * E||v||^2`

But now we can refine its interpretation using the code:

### 14.1 `lambda_swd`

This controls endpoint distribution pressure.

Empirical reading:

- too low: style ceiling stays low
- moderate increase helps
- too high eventually saturates or distorts

### 14.2 `lambda_kin`

This controls motion budget.

Empirical reading:

- high kinetic preserves structure
- low kinetic exposes style capacity
- zero kinetic is too destructive

### 14.3 cross-attention routing sharpness

This controls style concentration in space.

Empirical reading:

- sharper / less regularized routing supports higher style
- smoother / more entropic routing preserves content better

### 14.4 skip-path power

This controls whether structure is preserved or bypassed.

Empirical reading:

- too much skip freedom creates a direct style overwrite channel
- too little skip freedom makes the model rigid and content-safe but style-limited

So the actual operative model is:

`style gain = endpoint pressure x routing sharpness x delivered residual amplitude`

`content preservation = kinetic pressure x skip retention x routing smoothness`

That is not a formal theorem. But it is the most faithful mathematical summary of the code plus data we currently have.

## 17. Immediate Research Implications

The evidence now rules out several tempting but low-yield next moves.

### 17.1 What not to do first

- do not spend GPU budget on larger step counts
- do not spend GPU budget on tiny step-size adjustments
- do not assume K2 can out-style K1 by recipe tuning alone
- do not revive strong color modules as the first rescue path

### 17.2 What the current evidence says to do first

1. stay on the `K1` side of the family
2. treat residual amplitude and kinetic pressure as primary levers
3. treat endpoint SWD pressure as the next lever after amplitude/kinetic
4. use routing-smoothing ideas only when style is already high enough and content needs repair

## 18. Actionable Theory Conclusion

The next serious experiment family should not start from scratch and should not begin with new modules.

It should start from:

- the `K1` baseline lineage
- the trusted D0-style configuration family
- controlled movement in motion budget and endpoint pressure

because the repository evidence already says:

1. the model can reach SaMST-level style
2. the failure mode is overshoot, not under-capacity
3. the overshoot enters through amplitude, skip leakage, and weak content control

That is the correct theoretical position to build on.
