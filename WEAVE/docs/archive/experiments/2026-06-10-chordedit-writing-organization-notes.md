# ChordEdit Writing and Figure-Organization Notes

This note records the concrete paper-writing patterns extracted from
`F:\CVPR-chordedit.pdf` and how they should be applied to the active AAAI-27
draft.

## What ChordEdit does well

1. Page 1 is a claim surface, not a summary dump.
   - The hero figure already states the method promise.
   - The caption explains what the figure proves, not only what it contains.

2. The introduction advances in three moves only.
   - task and why it matters
   - failure mode of existing practice
   - method claim plus evidence preview

3. Related work is compressed into boundary setting.
   - It does not try to be exhaustive in the main paper.
   - It explains what contract prior work solves and why that is different.

4. The method section starts from a contract.
   - The paper first says what is visible at inference and what is only
     training-side supervision.
   - The mathematics then refine that contract instead of replacing it.

5. The experiments section is organized by proof function.
   - setup
   - main comparison
   - qualitative confirmation
   - ablation or refinement evidence
   - efficiency / implementation details

6. Qualitative figures are comparison arguments.
   - Methods are grouped into meaningful families.
   - The reader can tell why each column exists before reading the caption.

## What was applied to the AAAI draft

- Page-1 Figure 1 was rewritten as a claim surface around
  `failure zone -> frontier -> artifact cost`.
- The introduction was compressed to
  `IDT failure -> style-ID inference contract -> LBM frontier`.
- Related work was shortened to contract boundaries and evaluation boundaries.
- The method section now foregrounds the style-ID inference contract and the
  training-side role of OT pairing and SA-SWD.
- The experiments section now follows
  `experimental setup -> main result -> artifact diagnostics -> qualitative read -> retest and cost`.
- The main qualitative figure now groups columns into controls, high-style
  baselines, frontier rows, and reference-only columns.

## What still needs watching

1. Figure 1 remains the strongest page.
   - Later pages should keep that same directness and should not fall back into
     ledger-style prose.

2. Method presentation should stay compact.
   - If a compact definition aid becomes unreadable, prefer a sharper prose
     contract over a tiny main-text table.

3. Main-text figures should justify their space.
   - If a figure does not advance the core frontier story, move it to the
     supplement.

4. Historical support must stay secondary.
   - ChordEdit never lets side evidence compete with the main narrative.
   - The AAAI draft should keep Distinct5-WikiArt as the only real battleground
     in the main paper.

## Current rule of thumb for future revisions

When revising a section, ask:

- Does this paragraph sharpen the contract, the failure mode, or the frontier?
- Does this figure prove a new claim, or only repeat an earlier one?
- Would ChordEdit keep this material in the main paper, or push it to the appendix?

If the answer is no, compress it or remove it.
