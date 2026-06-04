# AAAI 2027 Writing Iteration Round 15

Date: 2026-06-04

## Scope

Main-thread work stayed on writing and layout. Remote experiment sidecars remain delegated to Dalton/Faraday and were not used to change paper claims in this round.

## Changes Made

- Rewrote the abstract from a metric list into the paper's core claim:
  - art-to-art Style-ID transfer is missing an unchanged-output control;
  - IDT turns target-style score into signed movement beyond the unchanged-artwork floor;
  - LBM is reported as an IDT-calibrated, costed operating point rather than another raw CLIP-S claim.
- Reframed the introduction around a deployed Style-ID contract:
  - source + style id only;
  - unchanged artwork as operational floor;
  - LBM as compact executable-control evidence.
- Tightened the contribution bullets:
  - IDT falsification contract;
  - executable-control tokenizer formulation;
  - costed WikiArt stress test.
- Rewrote the tokenizer/method prose:
  - tokenizer is a command interface, not the result;
  - representation quality is judged after rendering;
  - capacity alone is no longer presented as a sufficient representation solution.
- Compressed defensive table notes and primary Distinct5 interpretation:
  - SaMAM 3k is described as positive target movement that leaves the low-ArtFID region;
  - SaMST is described as stronger target movement at high LPIPS/ArtFID;
  - LBM-F/K are framed as low-damage positive-movement operating points.
- Compressed late ablation/discussion prose:
  - moved from experiment-log narration toward mechanism interpretation;
  - kept causal boundaries explicit.

## Verification

- Built `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf` successfully.
- PDF remains 11 pages.
- Log check found no overfull boxes, undefined references, undefined citations, LaTeX errors, fatal errors, emergency stops, or missing glyphs.
- Remaining warnings are font substitution and underfull boxes only.
- Rendered and inspected pages 1, 4, 6, 8, 9, 10, and 11.

## Current Writing Assessment

The paper now has a clearer top-level argument:

1. Raw CLIP-S is not an execution claim in art-to-art Style-ID transfer.
2. IDT is the necessary unchanged-output floor.
3. Baselines that clear IDT can still be expensive in damage, ArtFID, or training time.
4. LBM is a compact renderer that demonstrates positive movement in a low-damage region.

The largest remaining writing risk is not the main claim; it is evidence packaging. The paper still depends on careful wording around selected operating points, unavailable same-scope inference timings for some baselines, and non-causal coupled tokenizer sweeps.

## Experiment Handoff List

Do not block writing on these, but they are the next evidence upgrades if Dalton/Faraday returns stable packets:

- SaMAM convergence packet:
  - final transfer-only CLIP-S, LPIPS, targetwise ArtFID, and train wall;
  - curve point labels aligned to the same Distinct5 test set;
  - inference timing only if captured under the same output packet.
- SaMST e5/e15 stability:
  - confirm whether e5 and e15 remain within the currently reported small CLIP-S/LPIPS gap;
  - if yes, keep e15 as the main row and use e5 only as a convergence/stability note.
- Additional fixed-rule WikiArt stress split:
  - only useful if selected before output inspection;
  - report IDT first, then methods, with transfer-only as the primary scope.
- LBM performance frontier:
  - only promote if full_eval + targetwise ArtFID closes;
  - keep OR retention rule: style gain, LPIPS gain, or targetwise ArtFID gain can justify a candidate.

