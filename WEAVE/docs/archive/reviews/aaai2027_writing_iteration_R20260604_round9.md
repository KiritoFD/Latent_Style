# AAAI 2027 writing iteration round 9

Date: 2026-06-04

Scope:
- Main draft: `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- Current PDF: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Baseline packet status:
  `SchrodingerBridge/docs/experiments/distinct5_512_20260602/baseline_packet_status_20260604/`

## Purpose

This pass keeps experiments delegated to Dalton and focuses only on writing.
The goal is to make the paper read as an aggressive but evidence-bound
evaluation-contract paper:

1. CLIP-S-based art-to-art Style-ID transfer must first beat IDT.
2. Positive movement is priced by LPIPS, targetwise ArtFID, and compute.
3. LBM is the compact low-displacement proof point, not the whole thesis.

## Paper changes

- Rewrote the abstract to state the null hypothesis first, then give the
  Distinct5 evidence and cost comparison.
- Made the first page more direct: IDT is now introduced as the first column of
  the leaderboard rather than a secondary calibration.
- Rewrote the introduction around a three-regime separation:
  art-domain plausibility without target obedience, target obedience purchased
  by heavy displacement, and low-damage positive target movement.
- Tightened contribution bullets to three claims:
  IDT falsification contract, LBM executable-control proof point, and costed
  WikiArt stress test.
- Removed an experiment-result digression from Related Work so the section
  explains why compact Style-ID systems need an unchanged-output control.
- Compressed defensive method wording around bridge/OT/path claims while
  preserving the formal boundary that LBM is endpoint-supervised deterministic
  latent editing, not stochastic-bridge or OT-map estimation.
- Renamed Table 1's awkward engineering columns:
  `row sign only` -> `Evidence`, and `ret-ckpt min` -> `Train min`.
- Rewrote the primary Distinct5 result section so each method maps to a single
  diagnostic role:
  SaMAM as below-IDT art-domain diagnostic improvement,
  SaMST as above-IDT high-damage movement,
  LBM-F/K as low-damage positive movement.
- Sharpened Discussion and Conclusion around the reporting rule:
  movement beyond IDT first; LPIPS, ArtFID, and compute are the price.

## Evidence updates

- Ran `SchrodingerBridge/tools/build_distinct5_baseline_packet_status.py`.
- Generated:
  - `packet_status.json`
  - `README.md`
  - `samst_e5_idt_aligned_rows.csv`
  - `samst_e15_idt_aligned_rows.csv`
- SaMST is now partially closed from existing artifacts:
  full/transfer metrics, targetwise ArtFID, training wall time, and 750/750
  IDT-aligned rows are available for e5/e15.
- SaMST is still not fully closed because same-scope inference `ms/img` is not
  bound into the packet.
- SaMAM remains open and must stay a point-estimate claim until Dalton closes
  the authoritative Distinct5 packet.

## Verification

- Build command: `cmd /c build_paper.bat`
- Output: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Page count: 11.
- Log scan: no unresolved citations/references, no fatal errors, no overfull
  boxes.
- Remaining warnings: XeLaTeX font substitution warnings and underfull boxes.
- Rendered page checks:
  - page 1: abstract plus IDT figure remains readable;
  - page 10/11: references flow naturally; no stranded main-text figure/table.

## Review policy

No four-reviewer pass was run in this round. The changes are substantial
writing improvements, but they do not upgrade the claim surface:

- SaMAM remains point-estimate wording.
- SaMST remains partially closed.
- No additional WikiArt stress split was integrated.
- No mechanism claim was upgraded.

Trigger the next four-reviewer pass only when one of the gates in
`SchrodingerBridge/docs/reviews/aaai2027_next_review_gate_20260604.md` closes,
or if Figure 1 / Table 1 / Figure 3 is structurally replaced.
