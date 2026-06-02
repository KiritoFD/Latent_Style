# Tokenizer / Representation Theory Queue

Date: 2026-06-03

Scope: tokenizer / style representation only. This memo separates what is
already supported by current evidence from what remains hypothesis.

## 1. Most credible mathematical object

### Supported

The tokenizer is best treated as a map

\[
T_\phi: s \mapsto c_s
\]

from a target style/domain identifier `s` to a compact control code `c_s`
consumed by the LANCET executor.

In the current codebase, this object is not a target-image encoder and not an
independent renderer. It is a **style-side control generator** whose output can
be composed from:

- direct per-style codes
- shared atoms with mixture weights
- class-local prototypes
- factorized identity / texture / geometry fields
- optional carrier-plus-residual composition

The safest object-level reading is:

> tokenizer quality is the quality of the control signal it supplies to the
> downstream executor, not the geometry of the code in isolation.

### Only hypothesis

That the correct next representation must factor into a target-style carrier
plus an explicit execution-risk gate is still a design hypothesis, not a proven
mathematical decomposition.

## 2. Weakest assumptions worth keeping

### Supported

1. **Single-code interface assumption**
   - A meaningful style-control object can be represented as one compact code
     delivered to the existing LANCET consumer.

2. **Execution dependence assumption**
   - A style code is only useful insofar as its distinction survives
     execution through a content-conditioned renderer.

3. **Non-capacity-only assumption**
   - Increasing representation capacity alone is not sufficient; measured
     behavior depends on routing, target selection, and execution.

### Only hypothesis

1. **Factor identifiability assumption**
   - Identity / texture / geometry are truly separable latent factors in the
     learned code, rather than merely useful engineering coordinates.

2. **Carrier separability assumption**
   - There exists a target-style component that can remain separable after
     execution without an LPIPS penalty if exposed explicitly.

3. **Risk-gate sufficiency assumption**
   - A content-risk gate would be enough to preserve content while unlocking
     stronger style execution.

## 3. Wrong formulations to avoid

Avoid all of the following:

- "The tokenizer already represents the target style correctly."
- "Tokenizer collapse has been proven."
- "Larger token capacity does not matter."
- "Because Distinct5 no-op is strong, the tokenizer is bad."
- "Representation is the only remaining bottleneck."
- "Carrier + gate is the mathematically correct next design."

Safe replacement language:

- "Current evidence suggests the bottleneck is not raw code capacity alone."
- "The representation must remain executable after passing through the
  content-conditioned renderer."
- "Current probes constrain tokenizer claims to end-to-end executed control,
  not code geometry in isolation."

## 4. Three highest-information probe experiments

### Probe 1: Code-to-output alignment probe

Question:

- do pairwise distances or directions between tokenizer codes survive in the
  generated latent residuals and outputs?

Why it matters:

- this is the cleanest probe of whether representation structure is preserved
  or destroyed by execution.

Supported outcome meaning:

- if code geometry is strong but output geometry collapses, execution is the
  bottleneck.
- if both collapse, the representation itself is weak.

### Probe 2: Frozen-tokenizer / fresh-executor versus frozen-executor / fresh-tokenizer

Question:

- where does the marginal gain come from when only one side is allowed to
  adapt?

Why it matters:

- this is the most informative identifiability probe for separating
  representation-side weakness from executor-side weakness.

Supported outcome meaning:

- strong gains from fresh executor imply code was usable but under-executed.
- strong gains from fresh tokenizer imply current control object is the limit.

### Probe 3: Target-style gain versus code-separability probe under `idt`

Question:

- do runs with more separable tokenizer codes also achieve larger
  no-op-adjusted style gain (`delta_idt`) after execution?

Why it matters:

- this ties representation quality to the stricter Distinct5 end-to-end target:
  not raw style, but style gain beyond no-op.

Supported outcome meaning:

- if code separability rises without `delta_idt` gain, representation-only
  claims must be weakened.
- if both rise together, tokenizer-side claims become more credible.

## 5. Practical queue rule

### Supported now

- treat tokenizer theory as an **executed representation** problem
- treat code geometry alone as insufficient evidence
- keep claims at the end-to-end control level

### Still hypothesis

- any specific next-factorization story
- any theorem-like statement about the true latent factor structure of style
- any claim that current results localize failure solely to tokenizer or solely
  to executor

## 6. One-sentence working conclusion

The most credible current tokenizer theory is that `T_\phi(s)` defines an
executed style-control object whose value is determined by how much of its
target-style distinction survives the content-conditioned renderer; stronger
claims about factorization or optimal representation remain hypotheses until the
three probes above are run.
