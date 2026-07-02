# AAAI 2027 Writing Iteration Round 18 Non-Review Cleanup

Date: 2026-06-04

## Scope

This was not a four-reviewer pass. The round only aligned the active manuscript
with the current SaMAM-2250 evidence boundary, strengthened the SaMST e5/e15
plateau wording, and fixed a layout regression that left a mostly blank
floating page.

## Changes Made

- Paper:
  - kept SaMAM Distinct5 evidence at `2250` only;
  - strengthened the SaMST paragraph with the closed e5/e15 packet:
    transfer CLIP-S/LPIPS differ by less than `0.004/0.002`, and e15 lowers
    targetwise ArtFID from `465.7` to `444.5` while remaining high-damage;
  - changed the efficiency paragraph from a CLIP-S/LPIPS-only e5 check to a
    limited e5/e15 plateau and high-damage check;
  - shortened the conclusion rule to remove an orphan word on the next page;
  - removed the Discussion `FloatBarrier`, eliminating the large blank page
    after the artifact/ablation float block.
- Claim ledger:
  - updated Claim 4 so SaMST e5/e15 evidence is internally consistent;
  - kept the forbidden claims bounded: SaMST is not claimed fully converged on
    all metrics, and SaMAM beyond 2250 remains excluded.
- Dalton:
  - sent a sidecar boundary update: do not use SaMAM 2500/3000/3k, `394.8`, or
    `10.2h` in the active manuscript path unless a clean independent rerun
    closes a new packet.
  - reviewed Dalton's returned baseline packet update and aligned
    `baseline_packet_status_20260604/README.md` plus `packet_status.json`:
    SaMAM `3000/3250` are now explicitly audit-only/excluded, while the active
    manuscript row remains `SaMAM 2250`.

## Verification

- Rebuilt `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf` with
  `build_paper.bat`.
- PDF remains `11` pages.
- Text check found no stale `SaMAM 3k`, `SaMAM 3000`, `SaMAM 2500`, `10.2h`,
  `394.8`, `no-op-adjusted`, `Euler steps`, `z_K`, `TODO`, `PLACEHOLDER`,
  `FIXME`, or `TBD`.
- Log check found no undefined citations/references, LaTeX errors, fatal
  errors, or overfull boxes.
- Rendered and inspected pages 9 and 10 after the layout fix. The previous
  blank-page failure is gone.
- Loaded `packet_status.json` successfully and asserted:
  `valid_samam_max_step == 2250`, the manuscript row matches Table 1, and
  post-2250 SaMAM short jobs are disabled for the active manuscript path.
- Full-page layout scan over all `11` rendered pages found no blank or
  half-empty page regression.
- Rendered and inspected pages `2-5`, which had not been visually checked in
  the previous pass. No method-figure, formula, or related-work layout defect
  required a paper edit.
- Faraday returned a stale review based on the superseded `SaMAM 3k` framing.
  It was not applied. The active manuscript boundary remains `SaMAM 2250`.

## Current Gate

Do not start another four-reviewer loop from this cleanup alone. Trigger the
next adversarial pass only after a clean SaMAM rerun supersedes the 2250
boundary, a Dalton/Faraday packet changes Table 1/Figure 1, a new fixed-rule
stress split closes, or the paper receives a substantial structural rewrite.

## Immediate Next Step

Continue writing polish only where it changes claim clarity or layout quality.
Performance experiments remain sidecar work and must return full_eval plus
targetwise ArtFID before entering the manuscript.
