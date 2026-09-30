# AAAI 2027 Paper Update and Experiment Plan

Updated: 2026-06-02

## External writing and plotting guidance

Installed local skills:

- `ml-paper-writing` from `Orchestra-Research/AI-Research-SKILLs`
- `academic-plotting` from `Orchestra-Research/AI-Research-SKILLs`

Pulled writing prompt reference:

- `Leey21/awesome-ai-research-writing` README into `%TEMP%/ai_research_refs/awesome-ai-research-writing-README.md`

Usage decisions:

- Use the paper-writing guidance for claim discipline: keep one core story, separate completed results from pending baselines, and avoid citation hallucination.
- Use the plotting guidance for data-chart sizing, color, caption, and vector export.
- Do not use Gemini or generated bitmap diagrams. Architecture diagrams should be TikZ or deterministic SVG/PDF; numerical figures should be matplotlib PDF plus PNG fallback.

## Current paper status

Primary draft:

```text
SchrodingerBridge/aaai_submission/paper_aaai2026.tex
```

Target venue:

```text
AAAI 2027
```

The current TeX style remains `aaai2026` because that is the available author-kit style in the repository. Update the style only when an official AAAI 2027 author kit is available.

## 2026-06-02 rewrite pass

Updated:

```text
SchrodingerBridge/aaai_submission/paper_aaai2026.tex
```

Rewrite decisions:

- Experiments are now organized by evidence family:
  1. historical strict-750 main protocol;
  2. WikiArt512 / SaMAM convergence reference;
  3. Distinct5-512 stress benchmark;
  4. tokenizer and representation ablations;
  5. mechanism/efficiency diagnostics.
- The historical strict-750 table was restored as the main comparison instead of mixing SaMAM-512 and SaMST historical rows in one table.
- WikiArt512 SaMAM is now framed as a convergence reference: LBM reaches the SaMAM best-style region, but SaMAM teaches the LPIPS-after-style-saturation gap.
- Distinct5-512 wording remains bounded: LBM dominates the evaluated SaMAM curve up to 2000 steps, not all Distinct5 baselines.
- Method now includes a tokenizer subsection:
  - tokenizer answers "what target style control is requested";
  - LANCET answers "how that control is executed as a latent vector field";
  - main protocol tokenizer cannot read per-sample target latent/reference evidence;
  - mature current formula is prototype/direct code plus shared atom residual, with content-guided routing treated as execution modulation rather than target evidence.
- Discussion and conclusion now emphasize the current bottleneck: target-style representation must survive the content-conditioned renderer; simply increasing tokenizer size is not sufficient.

## Result separation policy

Do not mix these result families in one row group:

1. Historical strict-750 protocol:
   - domains: photo, Hayao, Monet, Van Gogh, Cezanne
   - includes historical SaMST, S2WAT, StyleID, AdaIN comparisons
   - current LBM headline: `CLIP-S=0.716`, `LPIPS=0.451`, `EC=0.393`

2. Historical WikiArt512 / SaMAM convergence:
   - SaMAM-512 reaches best style at 5k and best LPIPS at 10k
   - LBM historical high point is around `CLIP-S=0.7905`, `LPIPS=0.3006`
   - use this as a convergence/reference discussion, not as the Distinct5 main table

3. Distinct5-512 stress benchmark:
   - domains: Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e
   - train: 1000/class
   - test: 30/class
   - eval: 5x5 all pairs, 750 outputs
   - this is the clean current stress-test table

## Distinct5-512 current claim

Completed same-protocol comparison up to SaMAM 2000:

| Method | CLIP-S | LPIPS | EC | Train |
|---|---:|---:|---:|---:|
| SaMAM 2000 | 0.5833 | 0.3622 | 0.3721 | 6.8h |
| LBM-F e1 | 0.6969 | 0.3186 | 0.4748 | 1.2m |
| LBM-H e1 | 0.6974 | 0.3213 | 0.4733 | 1.2m |
| LBM-H e2 | 0.6994 | 0.3484 | 0.4557 | 2.3m |
| LBM-K e1 | 0.7010 | 0.3623 | 0.4470 | 1.2m |

Current wording gate:

- Allowed: LBM dominates the evaluated SaMAM convergence curve up to 2000 steps on Distinct5-512.
- Not allowed yet: LBM beats all Distinct5 baselines.
- Reason: SaMST-512 Distinct5 is prepared but not completed; SaMAM 2250+ may still lower LPIPS.

## New paper assets

Added vector chart:

```text
SchrodingerBridge/aaai_submission/figures/fig_distinct5_pareto.pdf
SchrodingerBridge/aaai_submission/figures/fig_distinct5_pareto.png
SchrodingerBridge/aaai_submission/scripts_gen_distinct5_pareto.py
```

The figure plots `CLIP-style` against `1-LPIPS`. It is generated from:

```text
SchrodingerBridge/docs/experiments/distinct5_512_20260602/tables/clip_style_vs_1lpips_points.csv
```

## Next experiments

Priority 0: validate the distance-metric story directly.

This is now a paper-critical block, not a side note. The current code and
paper already support a strong `W1` terminal-alignment claim, but the broader
`latent-space metric correction` story is not yet fully defended until the
following ablations exist under a controlled protocol.

### 0A. Flow-loss metric ablation

Goal:

- test whether `MSE` vs `Huber` vs `L1` in the flow residual materially changes
  artifact behavior, LPIPS, or training stability

Minimum setup:

- same Distinct5-512 data split
- same backbone / tokenizer family
- same queue family, preferably one content-preserving point (`F` or `H`)
- only switch `bridge.loss_type`

Required outputs:

- per-epoch strict-750 eval for the selected checkpoints
- `clip_style`, `content_lpips`, `EC`, `MUSIQ`, `MANIQA`, `ArtFID`
- a short visual panel showing whether high-frequency artifacts or blur change

Remote policy:

- formal runs on the remote 3060
- local only for smoke and config validation

### 0B. Time-to-parity curves

Goal:

- replace vague speedup prose with wall-clock convergence evidence

Minimum setup:

- LBM curve: use historical strict-750 and Distinct5 representative checkpoints
- SaMAM and SaMST curves: report wall time to each evaluated checkpoint
- plot `wall-clock time -> CLIP-S` and `wall-clock time -> LPIPS`
- Distinct5 should also include no-op-adjusted style gain where possible

Required outputs:

- CSV table of wall time, checkpoint id, CLIP-S, LPIPS, and evaluation scope
- vector PDF figure for the paper

### 0C. Path-stability probe

Goal:

- supply empirical support for the bounded kinetic / path-energy story

Minimum setup:

- sample `t ~ U(0,1)` along trained trajectories
- measure `||v_theta(z_0, t, s)||` statistics and variance
- compare at least one full model against one weakened kinetic variant

Required outputs:

- summary table for paper/supplement
- one compact plot of velocity magnitude variance over time
- explicit note on whether the bound is merely local/practical or globally
  violated in the sampled regime

Priority 1: finish Distinct5 baselines for paper claims.

- Continue SaMAM from 2000 to at least 2250/2500 if stable.
- Complete SaMST-512 Distinct5 under the same 750-output protocol.
- Update the Distinct5 table only after full all-pairs eval, not from partial target directions.

Priority 2: improve LBM Distinct5 Pareto frontier.

- Start from F/H/K family.
- Goal: keep F/H LPIPS band near `0.318-0.321` while absorbing part of K's style gain.
- Probe small grids only:
  - hard exploration probability: `0.05 / 0.10 / 0.15`
  - active atom top-k: `1 / 2 / 3`
  - K-router gain: reduce gain before increasing any loss
  - route temperature: test sparse but not deterministic mixtures
- Keep each probe at the same formal 3060 memory policy and evaluate all 5x5/750.

Priority 3: representation diagnostics tied to paper text.

- Measure per-style atom usage entropy and cross-style overlap.
- Measure generated-delta rank and direction clustering.
- Check whether style improvements are content-adaptive or shared color shifts.
- Add visual probes only after metric movement passes an OR gate: either CLIP-S improves materially or LPIPS improves materially.

Priority 4: tokenizer design experiments.

- Do not run "bigger tokenizer" as the next main experiment.
- Start from F/H/K family.
- Test target carrier plus bounded content-risk gate:
  - target carrier should remain source-invariant enough to be visible after execution;
  - content gate should reduce endpoint movement in LPIPS-sensitive cells;
  - report generated-delta rank/cosine and residual variance decomposition before promoting a full run.
- Promote a switch if it improves either CLIP-S or LPIPS materially without catastrophic regression in the other metric.

Priority 5: paper hygiene.

- Keep Distinct5 as a stress benchmark section until SaMST and longer SaMAM are complete.
- Keep historical and Distinct5 tables visually separated.
- Use vector PDF for charts and TikZ/SVG for diagrams.
- Update AAAI style only when the official 2027 kit exists.
- Where comparisons use unequal total training budgets, write them explicitly as
  `time-to-reported operating point` or `time-to-parity`, not as unconstrained
  raw speedup claims.
- After each paper rewrite or new paper-facing experiment block, re-run the
  three-lane reviewer protocol in `docs/reviews/aaai2027_review_protocol.md`
  and append the outcome to `docs/reviews/aaai2027_review_registry.csv`.
