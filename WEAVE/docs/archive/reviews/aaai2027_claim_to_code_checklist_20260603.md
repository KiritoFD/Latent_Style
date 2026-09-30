# AAAI 2027 Claim-to-Code Checklist

Date: 2026-06-03

Purpose: map the current manuscript claims to code truth, missing evidence, and
the safest wording boundary. This is a drafting checklist, not a prose rewrite.

## Source anchors

- paper: `aaai_submission/paper_aaai2026.tex`
- objective code: `src/losses.py`
- model code: `src/model.py`
- tokenizer code: `src/style_tokenizer.py`
- metric-audit docs:
  - `docs/experiments/2026-06-03-flow-loss-metric-ablation/README.md`
  - `docs/experiments/2026-06-03-flow-loss-metric-ablation/repaired_endpoint_metric_ablation_packet_20260603.md`
  - `docs/experiments/2026-06-03-distinct5-idt-evaluation-note.md`

| current paper claim | code truth | experiment evidence needed | safe wording boundary |
| --- | --- | --- | --- |
| `LBM is a latent flow-matching transport model` | Shared design is transport-style, but the active Distinct5 tokenizer family resolves through `objective_mode=omf` in `src/losses.py`, not a clean random-time FM path. | None for the bounded transport-design claim; a separate non-OMF family is needed if the paper wants direct empirical closure of the random-time FM story. | Safe: "structured latent transport design" or "OT-coupled latent transport framework." Narrow: explicitly separate historical bridge-FM lineage from current OMF Distinct5 lineage. Unsafe: write current Distinct5 mainline as if it were already the direct random-time FM empirical result family. |
| `Tokenizer experiments show representation, not capacity, is the bottleneck` | `src/style_tokenizer.py` supports direct codes, atoms, prototypes, carrier-residual variants, but execution still happens through the same content-conditioned LANCET consumer in `src/model.py`. Current results show code changes alone do not automatically solve executed style transfer. | Existing tokenizer tables plus current representation probes are enough for a bounded diagnosis. Stronger claims about the exact next representation factorization need future tests. | Safe: "capacity alone is insufficient; execution through the renderer is the measured bottleneck." Narrow: "carrier + risk gate" only as next-stage hypothesis. Unsafe: claim the final correct tokenizer factorization is already established. |
| `The paper closes a broad latent metric thesis against MSE` | Current code truth is narrower. The resolved three-arm packet in `docs/experiments/2026-06-03-flow-loss-metric-ablation/README.md` has `objective_mode=omf`, `w_flow=0.0`, and only switches `loss_type`; that trio is near-null for the intended metric thesis. Also, not all latent L2 terms are problematic: `src/losses.py` uses L2 naturally for kinetic energy. | A repaired endpoint-objective ablation packet where the compared kernel is active on endpoint matching; current memo already specifies the valid design. | Safe: "pointwise Euclidean endpoint reconstruction is high-risk for style alignment in compressed latents." Narrow: "the unresolved local transport penalty remains open." Unsafe: "latent-space MSE is broadly wrong" or "the current mse/huber/l1 trio proves the metric thesis." |
| `SA-SWD is the mechanism that fixes endpoint style matching` | `src/losses.py` and `src/ot_cost.py` support terminal distribution matching, and destructive ablations already support that terminal matching matters. But necessity of semantic axis selection over random SWD is not yet closed. | A matched semantic-vs-random SWD ablation. | Safe: "terminal distribution matching is an important style driver; SA-SWD is the proposed semantic-aligned version." Narrow: keep semantic-axis superiority as pending. Unsafe: say semantic SA-SWD is already proven necessary or decisively superior. |
| `Distinct5 shows raw CLIP-S is misleading without no-op control` | This is well aligned with the current metric docs. `docs/experiments/2026-06-03-distinct5-idt-evaluation-note.md` and the paper's Distinct5 table already interpret `delta_idt`, transfer-only filtering, and `ArtFID` as a broader artifact diagnostic rather than direct style gain. | Existing evidence is sufficient for the bounded Distinct5 protocol claim. | Safe: "on separated art-to-art transfer, raw CLIP-style should be read together with `idt` and no-op-adjusted style gain." Narrow: keep it as a protocol-level diagnosis on this regime. Unsafe: generalize to "all AST evaluation is invalid" or "all prior baselines are degenerate." |
| `LBM has a strong efficiency advantage` | Current evidence is operating-point wall-clock observation under reproduced protocols; it is not a universal time-to-quality theorem. The paper already partially narrows this in later sections. | None for operating-point wording. A normalized same-scope time-to-parity study is needed for stronger claims. | Safe: "measured operating-point wall-clock advantage" and "practical inference profile under the present protocol." Narrow: compare only reproduced operating points with stated hardware/provenance. Unsafe: universal speedup rhetoric or theorem-like efficiency claims. |
| `Kinetic regularization controls the path in a theoretically grounded way` | `src/losses.py` uses L2 on velocity as energy regularization, and destructive ablations support that removing kinetic hurts content. But the stronger path-energy theorem remains local and assumption-bound. | Path-statistics probe if the paper wants stronger empirical support for the theorem boundary. | Safe: "kinetic regularization discourages excessive latent displacement and shapes the style-content tradeoff." Narrow: theorem should be framed as local design-grounding. Unsafe: present endpoint kinetic as exact global trajectory control. |

## Fast drafting rule

Before keeping any mechanism sentence, check:

1. does the claim describe the resolved code path actually used by the reported
   result family?
2. is the compared loss term active on that path?
3. is the wording still valid if the reader opens the anchored code or memo?
