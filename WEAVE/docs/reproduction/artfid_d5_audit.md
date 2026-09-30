# D5 ArtFID Audit

Date: 2026-07-15

## Protocol

The audited implementation reports the mean over five target styles of
`(1 + art-domain FID) * (1 + source LPIPS)`. StyleAligned and Z-STAR use their recorded Random20 source manifest;
WEAVE uses the current Distinct5 test manifest. Both contain 30 images per each
of the same five styles.

## Results

| Method | ArtFID | FID term | LPIPS term | Count |
|---|---:|---:|---:|---:|
| IDT | 216.51 | 215.51 | 0.000 | 750 |
| WEAVE, oriented-HF epoch 4 | 295.27 | 230.52 | 0.283 | 750 |
| SaMam | 297.32 | 231.87 | 0.283 | 750 |
| Seedream 4.5 | 310.97 | 217.37 | 0.422 | API |
| Z-STAR | 332.91 | 251.99 | 0.321 | 750 |
| StyleAligned | 368.63 | 202.67 | 0.822 | 750 |
| TGT, deterministic reference 0 | 518.12 | 290.75 | 0.783 | 750 |

Machine-readable figure data are in
`aaai2027_v4/fig_data/artfid_d5_audit.{csv,json}`. The current WEAVE images were
generated directly from
`runs/submission/hf_oriented_internal_early_stop/epoch_0004.pt`; the generated
packet has 750 files and reproduces the paper CLIP-S/LPIPS point before ArtFID
is computed.

## Interpretation

WEAVE has the lowest ArtFID among the listed nontrivial transfer outputs, but
IDT remains much lower. This is expected from the formula: IDT receives an
exact zero content term, while any real edit multiplies the FID term by
`1 + LPIPS`. StyleAligned is the clearest example. It has the best raw FID term
in this subset, yet its high source distance makes its composite ArtFID worst.
Seedream also has a lower FID term than WEAVE but a higher composite score.

The defensible claim is therefore local: on this art-to-art board, ArtFID is an
artifact/plausibility diagnostic whose content factor strongly favors no-op or
near-no-op outputs. It cannot by itself establish target-direction style gain.
This does not imply that ArtFID is invalid on every artistic transfer protocol;
it is why the paper reports it alongside IDT/TGT, DINO-S, CLIP-S, LPIPS, and
DINO-C rather than treating it as the primary ranking.

## TGT exemplar sensitivity

The deterministic TGT row above uses the first reference image in each target
style. To test whether its high ArtFID is an artifact of those five images, we
sampled 30 additional TGT sets with seed `20260716`. Each replicate independently
selects one of the 30 held-out references per target style and reuses that image
for all 150 sources requesting the style, exactly matching the TGT definition.
All replicates use the same 750-pair identity manifest as the D5 audit.

| Quantity | Mean | Std. | Min | Median | 2.5%-97.5% | Max |
|---|---:|---:|---:|---:|---:|---:|
| ArtFID | 545.70 | 56.08 | 471.00 | 542.72 | 472.07-671.13 | 697.95 |
| Raw FID | 305.26 | 31.81 | 262.00 | 303.64 | 263.83-376.54 | 386.45 |
| Source LPIPS | 0.7870 | 0.0169 | 0.7580 | 0.7851 | 0.7590-0.8239 | 0.8261 |

All 30 random TGT sets have higher ArtFID than WEAVE, SaMam, Seedream 4.5,
Z-STAR, and StyleAligned. This supports TGT as a stable opposite failure
reference for the methods emphasized in the paper. It is not a mathematical
upper bound: TGT exceeds AdaIN in 27/30 samples, SaMST in 23/30, and WCT in only
2/30. WCT's still higher composite score is consistent with an output that can
move even farther from both source content and the target distribution than an
unrelated target exemplar.

Machine-readable records, including the selected reference filenames for every
replicate, are in `results/tgt_artfid_random_stability.{json,csv}`. The evaluator
is `tools/compute_tgt_artfid_stability.py`.

For the figure, IDT is also shown with uncertainty rather than as an isolated
scalar. A fixed-seed bootstrap over its five target-style ArtFID components
(10,000 replicates) gives a 95% interval of 209.93--223.09 around the board
estimate of 216.51. This interval measures variation across the benchmark's
target styles; unlike the TGT interval, it does not represent stochastic IDT
inference.

## Reproduction

```powershell
python tools/compute_artfid_simple.py --dataset D5-512 --method weave_oriented_e4 --batch-size 16 --device cuda
python tools/compute_artfid_simple.py --dataset D5-512 --method stylealigned --batch-size 16 --device cuda
python tools/compute_artfid_simple.py --dataset D5-512 --method zstar --batch-size 16 --device cuda
```

The tool fails visibly when a generated filename cannot be matched to its
source image. The audited runs matched 750/750 files for all three methods.
