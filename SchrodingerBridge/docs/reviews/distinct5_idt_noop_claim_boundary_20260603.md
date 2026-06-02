# Distinct5 `idt` / No-Op Claim Boundary

Date: 2026-06-03

Scope: theory-owner / claim-boundary memo only. This note formalizes the
current Distinct5 phenomenon that an unchanged-image control (`idt` / no-op)
can equal or exceed stylizers under the current metric bundle, without
over-claiming what that means.

## 1. Core phenomenon

On Distinct5-512, the unchanged-image control can be surprisingly strong under
the current evaluation bundle:

- `idt` already attains substantial absolute `CLIP-S`
- some stylizers fail to exceed the `idt` floor in `delta_idt`
- some stylizers exceed the floor only by paying heavy LPIPS / ArtFID damage

The phenomenon is therefore not merely "raw style can be gamed upward." It is
stronger and more specific:

> under the current Distinct5 protocol, a model may alter the image, incur
> perceptual displacement, remain inside the broad art manifold, and still fail
> to beat leaving the source image unchanged.

## 2. What this does prove

This currently proves a **bounded protocol-level fact**:

1. Absolute `CLIP-S` is not self-sufficient on this split.
2. The unchanged-image prior is large enough that `idt` must be reported
   explicitly.
3. `delta_idt` is necessary to distinguish raw target-style affinity from
   no-op-adjusted style gain.
4. A stylizer can be nontrivially changing the image while still failing the
   stricter question:
   - "did it move toward the requested target style beyond what no-op already
     gives?"

This is enough to justify:

- `idt` as a diagnostic control on Distinct5
- transfer-only filtering as a companion view
- caution around reading raw `CLIP-S` as stylization success

## 3. What this does not prove

It does **not** prove any of the following:

- not that the affected stylizer is "bad" in a universal sense
- not that all prior AST evaluation is invalid
- not that `CLIP-S` is useless everywhere
- not that `ArtFID`, `LPIPS`, or other metrics should be discarded
- not that Distinct5 is a universal art benchmark
- not that no-op always beats stylization on art-to-art transfer

The safest reading is regime-specific:

> Distinct5 exposes a concrete boundary case where the current style-affinity
> metric family is heavily entangled with an unchanged-image prior.

## 4. Open mathematical hypotheses

The current evidence is empirical. The following mathematical hypotheses remain
open:

### H1. Prior-overlap hypothesis

The style classes in Distinct5 remain close enough in the embedding geometry
used by `CLIP-S` that unchanged source images already occupy a large fraction
of the target-style affinity range.

Open status:

- supported empirically
- not yet reduced to a formal geometry statement

### H2. Metric-decomposition hypothesis

Observed `CLIP-S` may be decomposed into at least two entangled parts:

- a broad art-manifold prior term
- a task-specific target-style movement term

Open status:

- `delta_idt` is a practical diagnostic proxy
- no formal decomposition theorem is established

### H3. Damage-without-gain hypothesis

There exists a region of stylizer outputs where LPIPS increases and visible
change occurs, yet the target-style gain beyond no-op remains near zero or
negative.

Open status:

- strongly supported by current Distinct5 curves
- not yet characterized as a general property of style-transfer metrics

### H4. Representation-execution ambiguity hypothesis

When a stylizer fails to beat `idt`, the failure may come from either:

- weak target-style representation, or
- failure of the renderer/executor to realize an adequate representation

Open status:

- not identifiable from `delta_idt` alone
- requires separate representation and execution probes

## 5. Constraint on tokenizer / representation claims

This phenomenon sharply limits what we may infer about tokenizer quality.

### What is allowed

- If a model fails to beat `idt`, we may say the **end-to-end executed control**
  failed to realize sufficient target-style movement under the current metric
  regime.
- We may say Distinct5 is useful for stress-testing whether target-style
  control survives execution.
- We may say tokenizer-side gains must be interpreted through executed outputs,
  not code geometry alone.

### What is not allowed

- We may not infer from `idt` failure alone that the tokenizer representation
  has collapsed.
- We may not infer that larger tokenizer capacity is useless in general.
- We may not infer that the renderer is solely to blame without additional
  probes.
- We may not use the no-op phenomenon by itself as proof that the next
  tokenizer must be a specific carrier/gate factorization.

The most disciplined conclusion is:

> Distinct5 `idt` failure is an end-to-end execution diagnostic, not a pure
> tokenizer-identifiability result.

## 6. Safe wording boundary

Safe:

- "On Distinct5-512, raw `CLIP-S` must be interpreted together with an
  unchanged-image control."
- "A stylizer may incur perceptual change yet still fail to exceed the no-op
  style-affinity floor."
- "This constrains tokenizer claims: executed style gain, not code geometry
  alone, is the relevant object on this split."

Unsafe:

- "No-op beating a stylizer proves the tokenizer is bad."
- "Distinct5 disproves prior AST evaluation."
- "`CLIP-S` is invalid."
- "This phenomenon alone identifies the correct next representation design."

## 7. One-sentence takeaway

The Distinct5 `idt` / no-op phenomenon is best treated as a **bounded
evaluation and execution diagnostic**: it shows that under the current metric
regime, end-to-end stylization can fail to produce target-style gain beyond
no-op, but it does not by itself localize that failure to tokenizer geometry or
justify universal metric-level conclusions.
