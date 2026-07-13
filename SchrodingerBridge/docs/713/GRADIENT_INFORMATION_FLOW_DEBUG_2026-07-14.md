# Gradient and Information-Flow Debug

Date: 2026-07-14

This note records the mechanism diagnosis after adding probes inside the model, not just reading eval metrics. The main probe output is:

- `docs/model_probe/target_hf_subband_gradinfo_actual.json`
- `docs/model_probe/target_hf_subband_affine_delta_gradinfo_actual.json`
- `docs/model_probe/target_hf_subband_wct_direction_gradinfo_actual.json`

## Direct Answers

### Is the training target correct?

Yes, for the current method hypothesis. It is not reconstructing the source image:

```text
LL       = 0.7 * content_LL + 0.3 * AdaIN(content_LL -> style_LL)
LH/HL/HH = target_style bands
```

The target image is therefore already strong as supervision, especially in HF bands. The bottleneck is not that the loss asks for the wrong endpoint. The bottleneck is that the target image is weak as a condition input to the velocity predictor.

### Which route is weak?

The weak route is:

```text
target_style image -> target-HF condition route -> HF velocity prediction
```

On `target_hf_subband_ft6/epoch_0006.pt`, under the actual FM-HF objective, gradients into target style as condition are tiny compared with gradients into target style as training target:

| band | condition grad/tensor | target grad/tensor | condition / target |
|---|---:|---:|---:|
| LH | 4.48e-6 | 1.77e-4 | 2.53% |
| HL | 2.29e-6 | 1.74e-4 | 1.32% |
| HH | 1.71e-6 | 3.46e-4 | 0.49% |

The route is clean but narrow. Single-band intervention is nearly diagonal and has essentially no LL leakage:

| target condition band | output LH | output HL | output HH |
|---|---:|---:|---:|
| LH | 0.075624 | ~0 | ~0 |
| HL | ~0 | 0.097229 | ~0 |
| HH | ~0 | ~0 | 0.119432 |

Values are output delta/base.

### Is strength the whole problem?

No. The new direction-alignment probe compares:

```text
condition_delta = v(style_latent=target) - v(style_latent=content)
desired_delta   = target_velocity - v(style_latent=content)
```

For the current best subband route:

| band | delta/desired | cos(delta, desired) | projection | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|
| LH | 0.0166 | 0.0538 | 0.0009 | 0.9983 | 0.0016 |
| HL | 0.0206 | 0.0448 | 0.0010 | 0.9988 | 0.0016 |
| HH | 0.0191 | 0.0316 | 0.0006 | 0.9994 | 0.0009 |

So the target-specific condition delta is both small and mostly orthogonal to the immediate target correction. This explains why simple residual scaling and route widening spend content budget without improving the final frontier.

## Architecture Attempts From This Diagnosis

Both attempts below were trained for the same 6-epoch recipe from `brk_a_ll03_10ep` and evaluated with the same AdaIN 1.5 protocol. Both implementations/configs were removed after failing; probe/eval outputs are kept.

### 1. Affine subband delta

Hypothesis:

```text
old: h_styled = norm(h) * (1 + gamma(z_target_hf))
new: h_styled = norm(h) * (1 + gamma(z_target_hf)) + beta(z_target_hf)
```

This tested whether the subband route was too multiplicative/scale-only.

Mechanism result:

| metric | baseline | affine-delta |
|---|---:|---:|
| LH condition/target grad ratio | 2.53% | 4.08% |
| HL condition/target grad ratio | 1.32% | 3.02% |
| HH condition/target grad ratio | 0.49% | 1.15% |
| LH condition delta/base | 0.0756 | 0.1747 |
| HL condition delta/base | 0.0972 | 0.2363 |
| HH condition delta/base | 0.1194 | 0.1431 |

But direction was still mostly orthogonal:

| band | cos(delta, desired) | orthogonal fraction |
|---|---:|---:|
| LH | 0.0691 | 0.9974 |
| HL | 0.0652 | 0.9974 |
| HH | 0.0435 | 0.9988 |

Full eval:

| run | DINO-S | DINO-C | CLIP-S | LPIPS | off DINO-S |
|---|---:|---:|---:|---:|---:|
| baseline subband | 0.488624 | 0.798123 | 0.720880 | 0.296553 | 0.403917 |
| affine-delta | 0.482449 | 0.790343 | 0.717787 | 0.298913 | 0.398861 |

Verdict: failed. It widened the route but widened a mostly off-direction perturbation.

### 2. WCT-stat direction residual

Hypothesis:

```text
delta_hf = WCT(current_HF_band -> target_HF_band) - current_HF_band
v_hf     = v_hf + tanh(gate_band) * delta_hf
```

This tested a coordinate-free, content/current-placement-preserving direction prior. It uses target HF statistics, not target spatial coordinates. Gate was zero-initialized and learned to about `0.0345`.

Mechanism result:

| metric | baseline | WCT-direction |
|---|---:|---:|
| LH condition/target grad ratio | 2.53% | 2.60% |
| HL condition/target grad ratio | 1.32% | 1.64% |
| HH condition/target grad ratio | 0.49% | 0.85% |
| LH cos(delta, desired) | 0.0538 | 0.1099 |
| HL cos(delta, desired) | 0.0448 | 0.1252 |
| HH cos(delta, desired) | 0.0316 | 0.0935 |

This was a better mechanism signal than affine-delta: direction alignment improved without simply making the perturbation huge.

Full eval:

| run | DINO-S | DINO-C | CLIP-S | LPIPS | off DINO-S |
|---|---:|---:|---:|---:|---:|
| baseline subband | 0.488624 | 0.798123 | 0.720880 | 0.296553 | 0.403917 |
| WCT-direction | 0.486511 | 0.793320 | 0.719448 | 0.297849 | 0.402438 |

Verdict: failed. Local direction alignment improved, but the final transport/image frontier still worsened.

## Current Theory

The problem is not a single scalar bottleneck. Three facts must be true at the same time:

1. The target-HF route must be strong enough to affect HF velocity.
2. The target-HF delta must point in a useful direction.
3. The injected direction must preserve the learned ODE transport geometry over the full trajectory.

The failed attempts isolate these:

| Attempt | What improved | What failed |
|---|---|---|
| affine-delta | condition strength | direction and final metrics |
| direct dir-aux | residual/desired cosine | final metrics |
| WCT-direction | condition-direction cosine | final metrics |

So a probe improvement is necessary but not sufficient. The current subband residual remains best because it is small, continuous through the trajectory, and compatible with the learned HF heads, even though its image-specific condition component is weak.

## Practical Conclusions

Keep:

- `target_hf_subband_ft6/epoch_0006.pt` as the primary architecture probe.
- `target_hf_subband_texture_ft6/epoch_0006.pt` as the conservative alternate.
- The upgraded `tools/probe_gradient_information_flow.py`; it now reports condition strength and condition-direction alignment.

Avoid as-is:

- route widening by affine shift;
- analytic WCT/AdaIN direction residual inside the velocity field;
- direct residual-direction auxiliary loss;
- residual scalar amplification;
- raw target-HF spatial maps;
- cross-orientation code mixing;
- target-current global/pool-stat codes.

Next useful architecture must change the target-HF route while preserving the pretrained transport field, not merely add a larger residual or a locally aligned analytic correction.
