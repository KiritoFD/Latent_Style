# Tokenizer Code Geometry vs Executed Output Geometry

Date: 2026-06-03

Scope: theory-owner memo for the current tokenizer story only. This note defines
the code-geometry / output-geometry distinction in the present LBM/LANCET
setting, explains why code-space separability alone is insufficient, and names
the minimum next probe needed to close the gap without over-claiming.

## 1. Precise distinction in the current LBM/LANCET setting

Let

\[
T_\phi: s \mapsto c_s
\]

denote the tokenizer map from target style/domain id `s` to a compact control
code `c_s`.

Let

\[
v_\theta(z,t,c_s)
\]

denote the content-conditioned LANCET executor, and let its induced endpoint or
residual output be written abstractly as

\[
R_\theta(z_0,s) \quad \text{or} \quad \Phi_\theta(z_0,s).
\]

Then there are two distinct geometries:

### Code geometry

This is the geometry of the control codes themselves:

\[
\mathcal{G}_{\text{code}} = \{c_s\}_s
\]

Typical questions:

- are different styles linearly separable in code space?
- are code directions well spread or collapsed?
- do atoms / prototypes / factor fields yield distinct controls?

### Executed output geometry

This is the geometry of the style-conditioned outputs after execution through
the renderer:

\[
\mathcal{G}_{\text{exec}} = \{R_\theta(z_0,s)\}_{z_0,s}
\]

or, at endpoint level,

\[
\{\Phi_\theta(z_0,s)\}_{z_0,s}.
\]

Typical questions:

- do different styles induce distinguishable residual directions after content
  conditioning?
- does target-style information survive execution?
- do outputs move toward the requested style beyond no-op?

The core distinction is:

> code geometry is a property of the tokenizer alone; executed output geometry
> is a property of the tokenizer composed with the content-conditioned executor.

## 2. Why code-space separability alone is insufficient evidence

A tokenizer can have clean code geometry while still failing end-to-end.

This can happen because the map

\[
c_s \mapsto v_\theta(z,t,c_s) \mapsto R_\theta(z_0,s)
\]

is content-conditioned and nonlinear. Even if `c_s` are well separated:

- the executor can attenuate those differences
- content dependence can dominate style dependence
- different codes can collapse into similar residuals after execution
- style distinctions can survive in some content regions but not others

So the implication

\[
\text{separable codes} \Rightarrow \text{separable executed style control}
\]

does **not** currently hold by theory or by existing evidence.

In the current project, this matters because:

1. tokenizer-side experiments operate on code objects,
2. the actual paper claim depends on generated outputs,
3. Distinct5 `idt` / `delta_idt` results are measured after execution, not in
   code space.

Therefore code-space separability alone is insufficient for any of the
following stronger claims:

- "the tokenizer already represents style correctly"
- "the renderer is definitely the only bottleneck"
- "larger tokenizer capacity is irrelevant"
- "the next tokenizer factorization is identified"

At most, code-space separability can support a bounded internal statement:

> the tokenizer emits distinguishable controls before execution.

That is weaker than:

> those controls remain distinguishable and effective after execution.

## 3. Minimum next probe to close the theory gap

The minimum next probe is a **code-to-executed-output alignment probe**.

### Probe definition

For a fixed evaluation set of source latents/images:

1. compute tokenizer codes `c_s` for each style
2. measure pairwise code-space geometry:
   - cosine matrix, Euclidean distances, effective rank, cluster separation
3. run the same styles through the frozen current executor
4. measure executed-output geometry on:
   - generated latent residuals `R_\theta(z_0,s)`
   - optionally integrated endpoints `\Phi_\theta(z_0,s)`
5. compare whether style-space distinctions in `\mathcal{G}_{code}` survive in
   `\mathcal{G}_{exec}`

### Minimum readout

The minimum readout should contain:

- code-space separability
- residual/output separability
- alignment between code-distance and output-distance
- relation to no-op-adjusted style gain (`delta_idt`) if available

### Why this is the minimum useful probe

This is the smallest probe that can answer the central open question:

> are tokenizer-side distinctions preserved, damped, or destroyed by the
> current LANCET execution path?

It is more informative than:

- code-only inspection, which cannot test execution survival
- metric-only endpoint scores, which cannot localize failure to code vs
  executor

It is also narrower than a full architecture redesign, so it closes the theory
gap without over-claiming.

## 4. Safe conclusion boundary

Before this probe is run, the safest tokenizer statement is:

> current tokenizer theory should be framed at the level of executed style
> control, not code geometry alone.

After this probe:

- if code geometry is strong but executed geometry collapses, execution becomes
  the primary suspect
- if both are weak, tokenizer-side weakness becomes more plausible
- if both are strong and `delta_idt` still fails, the problem shifts to metric
  interpretation or downstream trade-off constraints

## 5. One-line takeaway

The open theory gap in the current tokenizer story is not whether style codes
can be made separable in principle, but whether that separability survives the
content-conditioned executor strongly enough to appear as real target-style gain
in outputs.
