# Gradient and Information-Flow Debug

Date: 2026-07-14

This note records the mechanism diagnosis after adding probes inside the model, not just reading eval metrics. The main probe output is:

- `docs/model_probe/target_hf_subband_gradinfo_actual.json`
- `docs/model_probe/target_hf_subband_route_competition.json`
- `docs/model_probe/target_hf_subband_affine_delta_gradinfo_actual.json`
- `docs/model_probe/target_hf_subband_wct_direction_gradinfo_actual.json`
- `docs/model_probe/target_hf_subband_memdrop_route_competition.json`
- `docs/model_probe/target_hf_subband_memdrop_gradinfo.json`

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

### Does style memory block target-HF?

Not in the simple "bad shortcut" sense. A route-competition probe decomposes the same checkpoint into:

```text
backbone only      = no style-memory cross-attention, no target-HF residual
style-memory only  = style-memory cross-attention, no target-HF residual
target-HF only     = target-HF residual, no style-memory cross-attention
full               = both routes active
```

On `target_hf_subband_ft6/epoch_0006.pt`, both style memory and target-HF are weakly helpful, but both have small useful projection and large orthogonal energy:

| transition | mean delta/desired | mean cos(delta, desired) | mean projection | mean orthogonal fraction | mean MSE improvement |
|---|---:|---:|---:|---:|---:|
| backbone -> style memory | 0.1438 | 0.1599 | 0.0258 | 0.9840 | 0.0263 |
| backbone -> target-HF | 0.1403 | 0.1498 | 0.0286 | 0.9832 | 0.0306 |
| style memory -> full target-HF marginal | 0.1616 | 0.1555 | 0.0345 | 0.9824 | 0.0322 |
| target-HF -> full style-memory marginal | 0.1676 | 0.1610 | 0.0306 | 0.9835 | 0.0279 |
| backbone -> full | 0.2339 | 0.2273 | 0.0611 | 0.9681 | 0.0576 |

The full route is better than either route alone, so deleting style memory is wrong. But the gradient competition is real: under FM-HF, disabling style memory raises target-HF subband gradient norm from `8.18e-2` to `1.38e-1`, while disabling target-HF raises head-HF gradient norm from `5.74e-1` to `3.06`. The model can route error through the generic memory/main HF heads rather than making the image-specific target-HF condition more predictive.

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

### 3. Training-only style-memory dropout

Hypothesis:

```text
During training only:
  with probability 0.25, replace style-memory tokens by learned null-memory tokens
  keep target-HF image conditioning active
During inference:
  use the normal full route
```

This tested the route-competition diagnosis: style memory is useful as a coarse prior, but sometimes letting it dominate may prevent target-HF from learning image-specific residual style.

Mechanism result was mixed:

| metric | baseline | memory-dropout |
|---|---:|---:|
| mean cos(style-memory, desired) | 0.1599 | 0.0463 |
| mean cos(target-HF, desired) | 0.1498 | 0.1561 |
| mean cos(target-HF \| memory, desired) | 0.1555 | 0.1586 |
| full route MSE improvement | 0.0576 | 0.0361 |
| residual cos(residual, desired) | 0.1575 | 0.1606 |
| residual MSE improvement | 0.0319 | 0.0334 |

The target-HF residual became only slightly more aligned, while the learned style-memory route became much less target-aligned and the full route got worse.

Full eval:

| run | DINO-S | DINO-C | CLIP-S | LPIPS | off DINO-S |
|---|---:|---:|---:|---:|---:|
| baseline subband | 0.488624 | 0.798123 | 0.720880 | 0.296553 | 0.403917 |
| memory-dropout | 0.486414 | 0.791995 | 0.719449 | 0.298218 | 0.402734 |

Verdict: failed. The implementation and config were removed; probe/eval outputs are kept. Training-only memory dropout is too blunt: it weakens the useful coarse prior more than it improves the target-HF image-specific route.

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
| memory-dropout | slight target-HF direction | style-memory prior and final metrics |

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
- training-only style-memory dropout as a blunt route regularizer.

Next useful architecture must change the target-HF route while preserving the pretrained transport field, not merely add a larger residual or a locally aligned analytic correction.
