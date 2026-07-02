# Distinct5 `idt` Control and the Raw-CLIP-S Failure Mode

Updated: 2026-06-03

This note sharpens a paper-safe claim:

`SaMAM < idt` on Distinct5-512 does not by itself prove that SaMAM is a bad
model. It shows that, in art-to-art transfer, absolute `CLIP-style` can become
hard to interpret unless it is anchored by an unchanged-image control.

## 1. Empirical progression across our three datasets

The same baseline behaves very differently once we compare it to an explicit
`idt` / no-op reference.

| Dataset | Scope | No-op CLIP-S | SaMAM best style delta over no-op | SaMAM best LPIPS delta over no-op | Reading |
| --- | --- | ---: | ---: | ---: | --- |
| Legacy256 / overfit50 | full | 0.661913 | +0.034954 | +0.031910 | Real but expensive style gain |
| Legacy256 / overfit50 | transfer | 0.616694 | +0.057198 | +0.050283 | Real transfer-only gain |
| WikiArt512 five-style | full | 0.781528 | +0.009716 | +0.003561 | Almost all raw CLIP-S is prior |
| WikiArt512 five-style | transfer | 0.773026 | +0.011563 | +0.004330 | Same effect survives off-diagonal filtering |
| Distinct5-512 | full | 0.680123 | -0.096777 | -0.099026 | Worse than unchanged source |
| Distinct5-512 | transfer | 0.639921 | -0.082738 | -0.087669 | Worse than unchanged source even off-diagonal |

The important point is structural:

1. `Legacy256` shows that SaMAM can exceed `idt`.
2. `WikiArt512` shows the gain collapsing toward zero.
3. `Distinct5-512` pushes the same metric into the negative region.

So the correct narrative is not "SaMAM is always degenerate." The stronger and
more defensible conclusion is:

> raw absolute `CLIP-style` is protocol-sensitive, and on some art-to-art
> splits it becomes dominated by the unchanged-image prior.

## 2. Distinct5 is not a trivial overlap split

Distinct5 is not a strange private benchmark or an adversarially hand-built
failure set. It was constructed from ordinary WikiArt categories by selecting
the five most separated classes under a CLIP-style screening pass:

- `Early_Renaissance`
- `Impressionism`
- `Minimalism`
- `Rococo`
- `Ukiyo_e`

Yet the unchanged source image still obtains high cross-style similarity.

- Transfer-only `idt` mean: `0.639921`
- Off-diagonal target averages:
  - `Early_Renaissance`: `0.651100`
  - `Impressionism`: `0.650076`
  - `Rococo`: `0.640409`
  - `Ukiyo_e`: `0.630033`
  - `Minimalism`: `0.627986`

Representative high-scoring off-diagonal `idt` pairs:

- `Early_Renaissance -> Rococo`: `0.692269`
- `Rococo -> Early_Renaissance`: `0.676847`
- `Ukiyo_e -> Minimalism`: `0.675195`
- `Ukiyo_e -> Impressionism`: `0.671572`

This matters because it makes two easy explanations much less plausible.
Distinct5 is not too close: the selected domains are deliberately far apart. It
is also not an exotic trap: the source images are ordinary WikiArt images from
standard art style categories. Even after separation, the art-domain prior
remains high enough that a copied source image already looks target-like to
CLIP. That is exactly why this split is useful. If a metric still certifies
unchanged-like outputs as success here, the problem is not that the dataset is
too weird; the problem is that the success condition is too weak.

## 3. Why Distinct5 is a stronger critique than "metric hacking"

The cleanest evidence is the full SaMAM training curve on Distinct5-512.

### Full 5x5 scope

- step `250`: `delta_idt = -0.132132`, `LPIPS = 0.600625`
- step `1000`: `delta_idt = -0.114213`, `LPIPS = 0.460542`
- step `2000`: `delta_idt = -0.096777`, `LPIPS = 0.362153`
- step `2250`: `delta_idt = -0.099026`, `LPIPS = 0.353820`

### Transfer-only scope

- step `250`: `delta_idt = -0.101439`, `LPIPS = 0.603106`
- step `1000`: `delta_idt = -0.091520`, `LPIPS = 0.463969`
- step `1250`: `delta_idt = -0.082738`, `LPIPS = 0.448703`
- step `2250`: `delta_idt = -0.087669`, `LPIPS = 0.360452`

The whole evaluated curve stays below `idt` while paying substantial perceptual
change. That is stronger than saying "the metric can be gamed upward." It shows
the reverse failure mode:

> a model can alter the image, incur LPIPS, remain in the broad art manifold,
> and still fail to move toward the requested target style once compared
> against an unchanged control.

This is the point that should sting. A respected modern baseline is not merely
over-scoring some trivial no-op. It is spending distortion budget and reaching
lower ArtFID while still failing the most basic directional test: did the image
move toward the requested target style beyond just leaving it alone?

## 4. What the literature already supports

### `ArtFID` is valuable, but it is not a target-style gain metric

`ArtFID` was introduced because NST lacked standardized quantitative evaluation
and because plain FID was misaligned with style-transfer perception. Its stated
goal is a style-transfer-specific metric with better human correlation.

That helps our argument, but it does not solve the Distinct5 issue by itself.
`ArtFID` is still an absolute metric over generated outputs versus target
references, combined with a content term. It does not ask whether the model
improved target-style similarity relative to leaving the source untouched.

The Distinct5 reproduced SaMAM packet makes that limitation concrete. Under the
same target-wise ArtFID protocol used in the paper table:

- `idt`: `ArtFID = 216.5` full, `323.7` transfer-only
- `SaMAM 2250`: `ArtFID = 146.1` full, `148.2` transfer-only

So `SaMAM < idt` in no-op-adjusted `CLIP-style` is not evidence that the
reproduction is simply broken. The reproduced model does find a lower-ArtFID
region. What it fails to do is convert that source-structure-preserving,
art-domain-plausible movement into positive target-style gain beyond the
unchanged image. On this split, lower `ArtFID` can therefore coexist with
negative `\Delta_{\mathrm{idt}}`.

This is also why the interpretation must stay narrow. The issue is not that
blind structure preservation is always useless; it is that structure
preservation by itself becomes an absurd success criterion when a copied source
image is already a strong art-domain sample and already receives substantial
target-style similarity from `CLIP-style`.

The paper should therefore avoid the weak claim "SaMAM is bad" and make the
stronger protocol claim: on a normal WikiArt split where target styles are
deliberately far apart, a respected modern baseline can improve the absolute
ArtFID target-domain score while still failing an explicit target-style movement
test. That is a robustness failure of the metric/protocol combination on
separated art-to-art transfer, not a license to dismiss the whole AST
literature.

### Prior AST evaluation work already warns that protocols are unstable

Two references are especially aligned with our finding:

- `Improving Style Transfer with Calibrated Metrics` (WACV 2020):
  style transfer metrics need calibration to human preference, and much of the
  observed variability can come from the chosen styles rather than the method.
- `A Comprehensive Evaluation of Arbitrary Image Style Transfer Methods`
  (TVCG 2024): AST evaluation is still plagued by inconsistent objective and
  subjective protocols, making cross-paper comparison unreliable.

These works do not propose an `idt` baseline. But they do support the broader
claim that AST evaluation remains under-specified.

### Recent AST papers still mainly report absolute metrics

Recent direct AST baselines and related systems continue to report absolute
metrics such as `LPIPS`, `FID`, `ArtFID`, or CLIP-family similarities:

- `SaMST` (ACCV 2024)
- `Mamba-ST` (WACV 2025)
- `SaMAM` (CVPR 2025)
- `HSI` (CVPR 2025)

That is not a flaw by itself. It means the field still lacks a standard control
for the specific question:

> did the model move toward the requested target style beyond what an unchanged
> source image already provides?

## 5. Paper-safe statement

The strongest version of the claim is:

> Distinct5-512 reveals a failure mode of raw absolute `CLIP-style` in
> art-to-art evaluation. On this split, an unchanged source image already
> attains substantial target-style similarity, so a trained model can incur
> nonzero perceptual change yet still fail to exceed the unchanged control.
> Therefore, on separated art-to-art splits such as Distinct5-512, raw
> `CLIP-style` is materially safer to interpret when reported together with an
> explicit `idt` baseline, transfer-only filtering, and no-op-adjusted style
> gain.

What we should not claim:

- not "all prior AST evaluation is invalid"
- not "SaMAM is universally worse than no-op"
- not "`ArtFID` is useless"

What we can claim:

- Distinct5 exposes a concrete evaluation boundary case
- `idt` is a useful diagnostic control for art-to-art transfer, especially on
  separated style splits
- our current paper uses that control to reinterpret raw style scores
- on an ordinary separated WikiArt split, the common absolute-score protocol
  can certify the wrong success condition for cross-style transfer

## 6. Immediate paper implications

1. Keep the main Distinct5 story centered on `delta_idt`, not on raw `CLIP-S`.
2. Frame the result as a protocol-level diagnosis, not a model-only attack.
3. Cite the AST evaluation literature explicitly in the evaluation section.
4. Keep `Legacy256` in reserve as the balancing counterexample showing that this
   is a regime change, not a universal condemnation of the baseline.
