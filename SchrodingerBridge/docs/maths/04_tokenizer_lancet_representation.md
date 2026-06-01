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

There is only one style-side representation module in the main benchmark:
`Tokenizer`. LANCET may internally contain U-Net encoder/decoder blocks, but
those are part of the renderer/actuator, not a second style encoder. A separate
target-evidence encoder is a different reference-guided protocol and is not part
of this benchmark.

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

The target latent remains valid on the right side of the objective: it is the
sample set used by SWD/OMF and related distribution losses. It is invalid only
when it becomes conditioning input to the tokenizer or LANCET forward path in the
main style-id benchmark.

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

Tokenizer-pretrain followed by fresh LANCET training is also valid if the
checkpoint loading boundary is explicit:

```text
Stage A: freeze LANCET, briefly warm up Tokenizer through frozen LANCET.
Stage B: initialize LANCET from scratch, load only style_tokenizer.*, freeze or
nearly freeze Tokenizer, and train LANCET to consume that fixed vocabulary.
```

This tests representation portability. If the fixed tokenizer transfers to a
fresh LANCET, it is closer to a real style vocabulary. If it only works with the
old LANCET that shaped it, then the tokenizer has learned an actuator-specific
coordinate system rather than a stable style representation.

Stage A should not be treated as a full optimizer race. Its role is to produce a
good initialization for the vocabulary: finite gradients, non-collapsed code
statistics, and a style-code region the renderer can execute. The performance
claim belongs to Stage B or later alternating cycles.

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

### Prototype Plus Shared Residual Atoms

```text
style_id -> prototype_code
style_id -> logits over K shared atoms
style_code = prototype_code + gain * sum_k softmax(logits_s / tau)_k * atom_k
```

This is the nested form of the concept-atom hypothesis. The prototype preserves
the per-style full-rank executable control expected by the current single-code
LANCET consumer. The shared atom residual tests whether styles also benefit from
a reusable vocabulary. This is a safer probe than pure atoms because it can
degenerate to direct code if the shared vocabulary is not useful.

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

The newer 512-line evidence changes the priority. The confirmed base is
`0.790531 / 0.300558` from the spectral-stat tokenizer plus full-model
adaptation. Attempts to improve this by merely reweighting weak matrix cells
failed on quick/n6 (`0.781003 / 0.396649` vs base quick repeat
`0.798453 / 0.330614`). This is important: weak-cell sampling is not a
representation. It increases gradient frequency for hard cases but does not
give LANCET a new conditional control variable, so the shared vector field can
leave the good basin.

The immediate valid tokenizer question is therefore no longer "can a larger
style embedding fit the five style IDs?" The right question is:

```text
Can the tokenizer expose a compact style metric plus a bounded execution budget
that tells the frozen/mostly frozen LANCET how far to move for each
source-target condition?
```

This keeps the style representation scientific rather than procedural:

- the style metric remains target-domain evidence, such as spectral/color
  statistics and inter-style distances;
- the execution budget is a small conditional gate, not a free spatial actuator;
- the budget should reduce LPIPS in content-sensitive cells without lowering the
  global style level;
- if the budget cannot beat the spectral-stat base under the OR gate, it should
  be rejected before full evaluation.

Sparse concept atoms remain a valid probe, but only as a vocabulary geometry
test. They are not sufficient by themselves unless LANCET has a reason to use
different atoms differently across source conditions.

The first budget evidence is consistent with this. A target-only
`tokenbudget_gradfix` run looked good on quick/n6 (`0.798214 / 0.299028`) but
did not beat the selected full base after all-5x5 evaluation (`0.790876 /
0.306589`). Two safety-budget variants were clearly negative on quick/n6
(`0.783885 / 0.352117` and `0.784784 / 0.357282`). Therefore a budget table or
metric decoder indexed only by target style is still an overgrown style-strength
knob. It lacks the missing variable: how risky this target style is for this
source/content geometry.

The next representation object should be:

```text
target_style_metric = tokenizer(target_style_id)
execution_budget = small_bounded_head(target_style_metric, source_style_or_content_stats)
style_code = renderer_style_code(target_style_metric)
renderer consumes style_code plus [low_gain, high_gain]
```

This separates two questions:

- what is the target style as a metric object?
- how much of that style should be executed on this source image?

This is also the cleanest way to absorb the SaMAM observation: SaMAM's LPIPS can
keep improving after style saturates, which implies the missing factor is not
raw style capacity but execution restraint.

## 7. Acceptance Checks Before Training

A tokenizer experiment is invalid unless:

- no tokenizer path reads per-sample `target_style` latent;
- `freeze_mode=tokenizer_only` leaves only tokenizer parameters trainable;
- LANCET still participates in the forward/backward graph;
- tokenizer gradients are non-zero in a batch smoke;
- debug logs report code norm and representation usage, such as atom entropy or
  field cosines.

Only after these checks pass should the run go to remote 3060 training.
