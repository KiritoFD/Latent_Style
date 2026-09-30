# WikiArt Stress-Split Selection Audit

Updated: 2026-06-04

## Purpose

This note records the dataset-construction evidence behind the paper's
Distinct5-512 stress-case wording and the follow-up fixed-rule WikiArt stress
splits. It is meant to prevent a reviewer-facing overclaim:

- Distinct5-512 is a useful separated art-to-art stress case.
- It is not yet a universal benchmark result.
- Additional fixed-rule splits are selected and materialized, but no LBM or
  baseline performance claim is allowed on them until IDT, full evaluation, and
  targetwise ArtFID packets are complete.

## Paper-Facing Distinct5-512 Split

The current paper-facing split uses standard WikiArt categories:

| style | train | test |
|---|---:|---:|
| Early_Renaissance | 1000 | 30 |
| Impressionism | 1000 | 30 |
| Minimalism | 1000 | 30 |
| Rococo | 1000 | 30 |
| Ukiyo_e | 1000 | 30 |

Audited dataset and IDT artifacts:

- Dataset identity / local + remote paths:
  `SchrodingerBridge/docs/experiments/distinct5_512_20260602/dataset_audit.md`
- IDT/no-op full evaluation:
  `SchrodingerBridge/docs/experiments/idt_eval_20260602/distinct5_512/idt_5x5/`
- IDT metrics:
  `.../idt_5x5/metrics.csv`
- IDT summary:
  `.../idt_5x5/summary.json`
- IDT targetwise ArtFID:
  `.../idt_5x5/aggregate_targetwise_artfid.csv`
- Cross-dataset IDT interpretation note:
  `SchrodingerBridge/docs/experiments/2026-06-03-distinct5-idt-evaluation-note.md`

Current IDT values used by the paper:

| scope | CLIP-S | LPIPS | ArtFID |
|---|---:|---:|---:|
| full 5x5 | 0.680123 | 0.000000 | 216.5 |
| transfer-only | 0.639921 | 0.000000 | 323.7 |

Important boundary: the repository currently contains the Distinct5 dataset,
IDT outputs, IDT per-image metrics, and dataset identity hashes. It does not
currently contain a complete original pre-training ranked CLIP-prototype list
for the exact Distinct5 choice. The manuscript should therefore describe
Distinct5 as a fixed stress case built from standard WikiArt classes and should
avoid implying that the full original ranking artifact is already present in
the repo.

## Follow-Up Fixed-Rule Stress Splits

To address the possible "custom split" reviewer attack, the follow-up selector
was implemented and retained:

- Selector:
  `SchrodingerBridge/tools/select_wikiart_stress_splits.py`
- Selection artifact:
  `SchrodingerBridge/docs/experiments/wikiart_stress_splits_20260603/selected_splits.json`
- Materialized datasets:
  `Dataset/wikiart_stress_splits_512/wikiart_stress1`
  `Dataset/wikiart_stress_splits_512/wikiart_stress2`
  `Dataset/wikiart_stress_splits_512/wikiart_stress3`

Selection rule:

1. Start from WikiArt class directories with at least 1000 training images plus
   30 held-out test images.
2. Exclude the five current Distinct5 classes so follow-up splits are disjoint
   from the paper-facing stress split.
3. Sample up to 96 prototype images per eligible class with seed `20260603`.
4. Encode sampled images with CLIP ViT-B/32 from the local cache.
5. Average normalized CLIP image features to form one class prototype.
6. Define pairwise class distance as `1 - cosine(prototype_i, prototype_j)`.
7. Exhaustively score 5-class combinations by mean pairwise distance, using
   minimum pairwise distance as the tie-breaker.
8. Select three disjoint 5-style splits greedily.

Follow-up selected splits:

| split | styles | mean pairwise CLIP distance | min pairwise CLIP distance |
|---|---|---:|---:|
| wikiart_stress1 | Color_Field_Painting, High_Renaissance, Mannerism_Late_Renaissance, Pop_Art, Realism | 0.167321 | 0.015848 |
| wikiart_stress2 | Abstract_Expressionism, Baroque, Cubism, Northern_Renaissance, Post_Impressionism | 0.138306 | 0.039864 |
| wikiart_stress3 | Art_Nouveau_Modern, Expressionism, Naive_Art_Primitivism, Romanticism, Symbolism | 0.053411 | 0.025931 |

Each materialized split has:

- `1000` train images per style.
- `30` held-out test images per style.
- A `manifest.json` containing the exact source-to-target copy records and the
  selection record.

## Paper-Safe Wording

Allowed:

- "Distinct5-512 is a CLIP-separated WikiArt stress case."
- "The split is built from standard WikiArt categories and evaluated with an
  explicit IDT/no-op reference."
- "The current Distinct5 result is a stress-case diagnostic, not proof of
  universal benchmark dominance."
- "Additional fixed-rule stress splits have been selected and materialized for
  follow-up validation."

Not allowed until more evidence lands:

- "Distinct5 proves the result generalizes across WikiArt."
- "The full original Distinct5 ranking artifact is retained."
- "The follow-up stress splits support the paper's performance claim."
- "Distinct5 IDT deltas are statistically significant."

## Next Evidence Required

For each follow-up split, the minimum paper-safe packet is:

1. IDT/no-op output, full/transfer summary, and targetwise ArtFID.
2. LBM output, full/transfer summary, and targetwise ArtFID.
3. Baseline output where feasible, especially SaMAM or SaMST, with the same
   full/transfer and ArtFID scope.
4. If per-image metrics align by source/target/image, paired bootstrap over
   `method - IDT` transfer CLIP-S.

