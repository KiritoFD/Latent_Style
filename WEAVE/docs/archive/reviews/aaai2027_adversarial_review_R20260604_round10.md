# AAAI 2027 adversarial review round 10

Date: 2026-06-04

Scope:
- Current draft: `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- Current PDF: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Trigger: round-9 structural rewrite of abstract, introduction, contributions,
  primary Distinct5 section, and table/caption wording.

## Reviewer scores

Four independent reviewer roles were run after the round-9 rewrite:

| Reviewer role | Score | Main risk |
|---|---:|---|
| Area-chair / writing | 7.2 | Story is strong, but selected-run speed and SaMAM point estimates can overread. |
| Statistics / validity | 6.3 | Evidence chain lacks cluster-corrected uncertainty and closed SaMAM timing/alignment. |
| Method / math | 7.0 | LBM wording must remain endpoint-supervised, not OT/SB/causal overclaim. |
| Figures / layout | 7.4 | Page 1 works; Figure 3 and page 8 remain visually conservative/crowded. |

## Consensus

- The paper is now a real evaluation-contract paper rather than a project
  report.
- IDT is the clearest contribution and should stay the primary claim.
- LBM should be framed as a compact evidence point for low-displacement
  positive target movement, not as a universal stylizer or a transport theorem.
- SaMAM must remain `point/open` until Dalton closes a retained aligned packet.
- Training-time claims must be operating-point checkpoint times, not normalized
  time-to-parity.
- Figure 3 is honest but visually conservative; use it as audited evidence, not
  as a visual SOTA claim.

## Applied fixes after review

- Scoped the abstract opener to CLIP-S-based art-to-art evaluation.
- Rewrote the abstract cost sentence:
  retained LBM checkpoints are 1.2 recorded training minutes in selected runs,
  while baseline checkpoint times are not controlled time-to-parity.
- Replaced `Current SaMAM checkpoints` with `Current SaMAM point estimates`
  and later with `are consistent with the failure mode`.
- Replaced the intro's overly strong non-positive wording with:
  a non-positive observed mean provides no evidence of target movement under
  the CLIP-S criterion.
- Replaced `LANCET vector field` with `LBM time-conditioned residual field`.
- Added a method bridge explaining the three design requirements induced by
  IDT: target pressure, content-risk budgeting, and no-reference inference.
- Replaced stronger method wording:
  `plausible style endpoints` -> `candidate training endpoints`;
  `lower-variance target pressure` -> `feature-matched target pressure`;
  `target-style distributions` -> `selected target endpoint projection
  statistics`;
  `prices motion` -> `limits executed displacement`.
- Renamed the semantic projection equation label away from `swd_estimator`.
- Rewrote the Sinkhorn diagnostic as an offline diagnostic rather than the
  active training sampler.
- Changed the contribution from `proof point` to `evidence point`.
- Changed tokenizer evidence language from `supports` / `rule out` to
  `motivates` / `argues against`.
- Reworked Table 1:
  `Evidence` -> `Pkt.`, with `open`, `part.`, and `sel.` packet states.
- Updated Figure 1:
  panel (b) title now states `Targetwise ArtFID + train time`, so the bar
  labels no longer masquerade as pure ArtFID.
- Updated Figure 3 caption to state the aligned examples' interpretation:
  SaMAM edits without positive target movement; SaMST clears IDT with high
  damage; LBM-F/K show low-displacement positive movement.
- Follow-up local layout pass after round 10:
  aligned the Figure 4 / Table 5 minipages at the top of the page-8 block,
  enlarged the texture crop panel slightly, and shortened the Table 6 caption.

## Verification

- Build command: `cmd /c build_paper.bat`
- Output: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Page count: 11.
- Log scan: no unresolved references/citations, no fatal errors, no overfull
  boxes after the Table 1 packet-code fix.
- Rendered pages checked:
  - page 1: figure/caption synchronized with revised TeX;
  - page 6: Table 1 and Figure 3 fit without overflow;
  - page 8: still dense, but Figure 4 / Table 5 are top-aligned and no longer
    visually sag;
  - pages 10/11: references flow without stranded main text.

## Remaining gates

Do not run another four-reviewer pass until a real gate closes:

1. Dalton closes SaMAM/SaMST timing and aligned-packet evidence; or
2. an additional fixed-rule WikiArt stress split is integrated; or
3. Figure 1, Figure 2, Figure 3, or Table 1 is structurally redesigned; or
4. a mechanism claim is upgraded by matched ablation evidence.

Highest-value next work:

- Dalton: close SaMAM final/tuned aligned packet and same-scope timing.
- Dalton/Faraday: if compute allows, add one fixed-rule stress split with IDT
  and LBM-F/K packets.
- Main writing thread: only small polishing until new evidence arrives; avoid
  another review loop on the same claim surface.
