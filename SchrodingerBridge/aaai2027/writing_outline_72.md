# AAAI 2027 Paper Outline After docs/72 Rewrite

## Core Claim

FC-SB shows that real style transfer can be affordable when evaluation is calibrated by IDT and the latent bridge is frequency-conditioned. The paper should not claim to beat diffusion-prior methods in raw style strength. The claim is high-fidelity, low-cost, correct-direction transfer.

## Section-Level Story

### Title
Signal the new main contribution: affordable real style transfer, frequency-conditioned bridge, OT/SB geometry.

### Abstract
State five facts in order:
1. Raw art-to-art style similarity can reward unchanged images.
2. IDT calibration defines real transfer as gain above the unchanged-image floor.
3. FC-SB uses Haar frequency routing, terminal subband matching, and cheap ODE integration.
4. T11 reaches 0.7213 CLIP-S / 0.2868 LPIPS with 903K parameters and about 30 minutes of training.
5. The scientific contribution is a practical high-fidelity frontier plus a diagnosis of the DWT-route tradeoff.

### Introduction
Paragraph 1: Define the evaluation problem. Style score alone is insufficient because the source is already art.

Paragraph 2: Explain IDT calibration and why it changes the interpretation of related methods.

Paragraph 3: Introduce FC-SB. Source image plus style ID only; Haar DWT separates frequency roles; terminal matching injects style.

Paragraph 4: State the empirical frontier. 4F.1 is style ceiling, 4I.7b is remote balance, T11 is main high-fidelity local point.

Contribution list: IDT evaluation, FC-SB method, main-text theory, aligned 12-baseline comparison.

### Related Work
Paragraph 1: Separate exemplar-guided transfer and large-prior editing from the style-ID-only contract.

Paragraph 2: Position compact baselines including SaMST, Mamba-ST, and SaMAM; explicitly remove invalid SaMAM 0.7222.

Paragraph 3: Tie FC-SB to flow matching / OT / SB as geometry, not a full stochastic SB solver.

Paragraph 4: Motivate calibrated metrics and multi-metric evaluation.

### Method
Problem Contract: source latent plus style ID only; no target exemplar or per-image optimization.

Frequency-Conditioned Bridge: Haar DWT produces LL/LH/HL/HH; LL protects structure; high-frequency bands carry texture and style.

DWT Route and EOTA: LL bypass preserves content, high-frequency cross-attention injects style, terminal-only style matching preserves alpha semantics.

Theory: four narrow claims.
Haar energy preservation justifies subband control.
Terminal-only injection explains EOTA.
Heun order explains solver result.
Stochastic DWT routing explains p=0.8 train/inference alignment.

### Experiments
Protocol: Distinct5-WikiArt, 750 pairs, HF CLIP ViT-B/32, LPIPS-Alex, IDT=0.6933.

Main comparison: T11 is not the highest CLIP method; it is the best high-fidelity local point, beats Seedream on LPIPS by 0.1899 and slightly on CLIP, and is much faster than CUT/SaMAM-style training.

FC-SB frontier: 4F.1 style ceiling, 4I.7b balance, 4J.1 route start, T11 local main point, T10 content extreme.

Ablations: multilevel DWT, LL role, EOTA, Heun, stochastic route. Negative results are included because they prove the current tradeoff.

### Discussion
Main lesson: calibrated evaluation plus frequency routing makes affordable real transfer possible.

SaMAM correction: old 0.7222 is invalid and must not be used.

Limitation: DWT route protects LL, but CLIP-S also rewards low-frequency style; this creates the 1:8 tradeoff.

Future: independent global style carrier, low-frequency target alignment, human preference study, larger style sets.

### Conclusion
Repeat the precise claim: FC-SB is an affordable high-fidelity real-transfer frontier, not a diffusion-style raw CLIP maximizer.
