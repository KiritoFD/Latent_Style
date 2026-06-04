# AAAI 2027 writing iteration round 11

Date: 2026-06-04

Scope:
- Main draft: `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- Current PDF: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Experiment handoff:
  `SchrodingerBridge/docs/experiments/dalton_aaai2027_experiment_backlog_20260604.md`

## Purpose

This was a post-review cleanup pass, not a new adversarial review. Round 10 had
already consumed the four reviewer agents. This round only handled local
layout/writing residue and made Dalton's evidence-closure task explicit.

## Local paper changes

- Improved the dense historical-evidence page:
  - forced Figure 4 and Table 5 minipages to top-align;
  - enlarged the texture-crop panel from `0.90\linewidth` to
    `0.96\linewidth`;
  - shortened the Table 6 caption.
- Follow-up Figure 2 cleanup:
  - simplified the framework figure title to an inference-contract framing;
  - labeled the three bands as style request, executed latent motion, and
    training-only target pressure;
  - shortened the vector-field and terminal-pressure labels;
  - replaced the caption with a direct statement that inference uses only the
    source image and style id, while endpoint queues, projection matching, and
    velocity budgeting are training-only supervision.
- No new claims were added.
- No SaMAM/SaMST wording was upgraded.
- No new reviewer pass was run.

## Dalton handoff

Sent Dalton a sidecar task to close or explicitly mark missing the Distinct5
SaMAM/SaMST evidence packet. The sidecar is instructed not to edit the main
paper and not to start a long training run before resolving existing artifact
status.

Immediate required Dalton answer:

- visible SaMAM Distinct5 checkpoints and exact missing fields;
- SaMAM full/transfer CLIP-S, LPIPS, targetwise ArtFID, timing, and
  IDT-aligned rows if recoverable;
- SaMST e5/e15 same-scope inference `ms/img` if recoverable;
- whether any paper wording can be upgraded, with valid outcomes restricted to
  no upgrade, SaMST timing closed, SaMAM closed below IDT, SaMAM closed above
  IDT, or additional split packet closed.

## Verification

- Build command: `cmd /c build_paper.bat`
- PDF page count: 11.
- Log scan: no unresolved references/citations, no fatal errors, no overfull
  boxes.
- Rendered page 4 checked after the Figure 2 cleanup.
- Rendered page 8 checked after the local layout pass; it remains dense but is
  visually more balanced than round 10.

## Next gate

Do not run another four-reviewer pass until at least one real gate closes:

1. Dalton returns a closed or explicitly unrecoverable SaMAM/SaMST packet.
2. A new fixed-rule WikiArt split is integrated.
3. Figure 1/2/3 or Table 1 is structurally redesigned.
4. A mechanism claim is upgraded by matched ablation evidence.
