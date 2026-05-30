# Tokenizer + LANCET Representation Model

Date: 2026-05-30

This note fixes the conceptual boundary for the tokenizer line. It is meant to
be read before adding tokenizer code or launching another run.

## 1. Module Responsibilities

The model has two different jobs.

```text
style_id -> Tokenizer -> style_code / style_tokens -> LANCET -> latent edit
```

Tokenizer answers: what is this style?

LANCET answers: how do I draw the content latent using that style control?

The tokenizer is not an image translator, adapter, or teacher model. It must
produce an executable style representation from information that is available at
training and evaluation time. In the current benchmark this conditioning input is
`target_style_id`. Per-sample target image/latent evidence is not available in
standard inference, so it must not be used by the main tokenizer path.

## 2. Data Boundary

Allowed mainline tokenizer inputs:

- `target_style_id`
- learned vocabulary parameters tied to style ids
- optional global style statistics only if the same statistics are fixed and
  available at inference as part of the style definition

Disallowed mainline tokenizer inputs:

- current batch `target_style` latent
- generated images from Seedream or other outside models
- per-sample reference images unless the evaluation protocol is explicitly
  changed to reference-guided transfer

The reason is not philosophical. It is a graph and protocol issue. If tokenizer
training uses `target_style` latent but full evaluation calls only
`target_style_id`, then training optimizes a different conditional distribution
from deployment. That can improve training loss while making the reported
benchmark meaningless.

## 3. Training Graphs

Tokenizer-only training still includes LANCET:

```text
style_id -> Tokenizer(theta_T) -> style_code
content, style_code -> LANCET(theta_L frozen) -> endpoint
endpoint, target style distribution -> OMF/SWD loss
```

`theta_L` has `requires_grad=False`, but gradients pass through frozen LANCET
into `theta_T`. This is the only way the tokenizer learns a representation that
the current LANCET can actually execute.

Backbone-only training freezes `theta_T` and updates the consumer:

```text
style_id -> Tokenizer(theta_T frozen) -> style_code
content, style_code -> LANCET(theta_L) -> endpoint
endpoint, target style distribution -> OMF/SWD loss
```

Joint training is allowed only after each side has a measured role. Otherwise a
loss improvement cannot be attributed to representation or actuation.

## 4. Mathematical View

Let each style be represented by a control object `c_s`:

```text
c_s = T_phi(s)
z_1 = F_theta(z_0, c_s)
```

The tokenizer should make the metric geometry of `c_s` useful:

- nearby codes should correspond to styles that LANCET can render similarly;
- code dimensions or atoms should have measurable usage;
- the representation should avoid collapse to one average style direction;
- the representation must remain executable by the current LANCET consumer.

The frozen-tokenizer probe asks whether `F_theta` can learn a stronger actuator
for a fixed representation. The frozen-LANCET probe asks whether `T_phi` can find
better style controls inside the current actuator landscape. These are different
questions and should not be conflated.

## 5. Valid Tokenizer Hypotheses

### Direct Code Control

`style_id -> style_code`

This is a control probe. It measures the frozen single-code interface with no
field bottleneck. It is not a general tokenizer upper bound.

### Factorized Fields

`style_id -> identity / texture / geometry -> style_code`

This makes fields inspectable but can lose rank or collapse through projection.
It is useful as a diagnostic, not yet proven as the best representation.

### Sparse Concept Atoms

```text
style_id -> logits over K shared atoms
style_code = sum_k softmax(logits_s / tau)_k * atom_k
```

This tests whether styles are better represented as combinations of shared
learned concepts rather than independent continuous codes. It uses no target
latent and can be evaluated with the normal style-id protocol.

### Distributional Style Code

```text
style_id -> mu_s, logvar_s
style_code_train = mu_s + sigma_s * epsilon
style_code_eval = mu_s
```

This tests whether uncertainty helps avoid tokenizer collapse. It needs explicit
entropy/KL diagnostics and should not be mixed with target-image evidence.

### Multi-Token Injection

Multi-token style representation is only valid after LANCET is extended to
consume those tokens at defined layers. Producing five tokens and then reducing
them to one CLS-style vector is still a single-code tokenizer.

## 6. Current Experimental Reading

Current best tokenizer-line result is around `0.7126 / 0.4453`, below the
documented `t01` style endpoint but with better LPIPS. Tokenizer-only enlargement
did not break the style ceiling, while backbone-only continuation recovered some
style. This means the next tokenizer probe must test representation geometry,
not just parameter count.

The immediate valid next probe is sparse concept atoms under frozen LANCET. Its
result should be compared to direct code, factorized concat, and big concat under
the same checkpoint and evaluation protocol.

## 7. Acceptance Checks Before Training

A tokenizer experiment is invalid unless:

- no tokenizer path reads per-sample `target_style` latent;
- `freeze_mode=tokenizer_only` leaves only tokenizer parameters trainable;
- LANCET still participates in the forward/backward graph;
- tokenizer gradients are non-zero in a batch smoke;
- debug logs report code norm and representation usage, such as atom entropy or
  field cosines.

Only after these checks pass should the run go to remote 3060 training.
