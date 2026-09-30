# Tokenizer Code-to-Execution Alignment Protocol

Date: 2026-06-03

Purpose:

- close the current theory gap between tokenizer code geometry and executed
  output geometry;
- turn the `Darwin` memo into one concrete paper-facing probe packet;
- produce evidence that can narrow tokenizer claims without relying on
  code-space anecdotes alone.

## Core question

The open question is not whether tokenizer codes can be made distinct in
principle. It is whether those distinctions survive execution strongly enough
 to appear as real target-style gain after the content-conditioned LANCET field
 acts on them.

In short:

> does code-space separation survive as executed output separation?

## Scope

Primary benchmark:

- `Distinct5-512`

Primary protocol:

- current style-id setup only
- no target-image conditioning at inference
- evaluate on the same source set used by current Distinct5 paper-facing runs

## Probe design

For one fixed checkpoint family and one fixed evaluation set:

1. compute tokenizer controls `c_s` for each target style;
2. measure code-space geometry:
   - pairwise cosine matrix
   - pairwise Euclidean distances
   - effective rank / spectrum concentration
3. execute the same styles through the current LANCET renderer on fixed source
   latents;
4. measure executed-output geometry on:
   - generated latent residuals `R_\theta(z_0,s)`
   - optionally integrated endpoints `\Phi_\theta(z_0,s)`
5. compare:
   - code-space distances vs residual-space distances
   - code-space separability vs style-wise `delta_idt`
   - residual-space separability vs style-wise `delta_idt`

## Minimum outputs

Required durable artifacts:

- one CSV for code geometry
- one CSV for executed residual / endpoint geometry
- one joined CSV with code-to-output alignment statistics
- one short README packet with:
  - evaluated checkpoint
  - dataset/scope
  - whether source-content dominance was removed or controlled
  - paper-safe conclusion boundary
- at least one vector figure showing code separation vs executed separation

## Minimum readouts

The packet should report at least:

1. code-space separability:
   - mean off-diagonal cosine
   - effective rank
2. executed residual separability:
   - same metrics in residual space
3. code-to-output alignment:
   - correlation between code distances and residual distances
4. execution relevance:
   - relation between executed residual separation and `delta_idt`

## Acceptance logic

This probe is useful even if it is negative.

Interpretation rules:

- strong code geometry + weak executed geometry:
  - execution is the primary suspect
- weak code geometry + weak executed geometry:
  - tokenizer-side weakness remains plausible
- strong code geometry + strong executed geometry + weak `delta_idt`:
  - metric or trade-off constraints remain the next suspect

## Claim boundary

Before this packet lands, the paper should only say:

- code geometry alone is insufficient;
- current evidence is about executed style control, not tokenizer separability
  alone.

After this packet lands, the paper may sharpen the tokenizer story only to the
extent directly supported by matched code/output evidence.

## Preferred execution owner

Formal run owner:

- remote `3060` via `Linnaeus`

Local runs:

- smoke checks only
