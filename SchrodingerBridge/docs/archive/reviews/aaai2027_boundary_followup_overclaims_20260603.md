# AAAI 2027 Boundary Follow-Up Overclaims

Date: 2026-06-03  
Scope: follow-up after `aaai2027_boundary_alignment_pass_20260603.md`  
Question: after the current boundary pass, what concrete overclaims still remain before Gate B and Gate C close?

## Short answer

The manuscript is materially safer than before. Most of the earlier hard overclaims have been removed. What remains is narrow and concentrated in two places: one residual efficiency overreach and one contribution-positioning risk for SA-SWD.

## Remaining concrete overclaims

### 1. Distinct5 timing comparison still slightly outruns Gate C

Location:

- Distinct5 subsection in `paper_aaai2026.tex`
- specifically the sentence:
  - `under that limited reading, the reproduced LBM points also have lower selected-checkpoint training time than the reproduced SaMAM points`

Why this still counts as an overclaim:

- even with the new caveat, it is still a comparative training-time statement before normalized `time-to-parity` exists;
- a skeptical reviewer can still read this as a fairness conclusion rather than pure bookkeeping.

Practical rewrite boundary:

- safest pre-Gate-C version is to report the timing columns as recorded operating-point context only;
- do not explicitly conclude `lower training time than SaMAM` until the normalized artifact lands.

### 2. SA-SWD still sits in contribution space before Gate B closes

Location:

- contribution bullet describing `semantic-aligned terminal matching design (SA-SWD)`

Why this is still borderline:

- the wording is much safer now, but it still places SA-SWD in the contribution list before the matched semantic-vs-random control closes;
- if Gate B comes back weak or null, a skeptical reviewer can still say the paper advertised a contribution whose distinctive semantic part was not yet isolated.

Practical rewrite boundary:

- before Gate B closes, the bullet is safest if read strictly as:
  - `the terminal-matching design used in the current mainline`
- not as a closed novelty claim about semantic-axis necessity.

## What no longer counts as a concrete overclaim

- the repaired endpoint trio is now within boundary if it is described only as a **negative endpoint-only closure**;
- the Distinct5 frontier claim is now mostly safe as long as it stays pinned to:
  - `metric-stress benchmark`
  - `currently reproduced points`
  - `idt` / no-op interpretation;
- the theory/checklist wording is now close to safe because it says `partial empirical support`.

## Reviewer-ready takeaway

Before Gate B and Gate C close, the manuscript should still avoid exactly two things:

1. any sentence that reads like a fair comparative training-speed win;
2. any sentence that lets SA-SWD read as fully isolated novelty rather than current mainline design.

If those two boundaries are held, the remaining weakness is mainly missing evidence, not active overclaim.
