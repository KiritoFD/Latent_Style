# No-op Reference and CLIP-style Metric Trap

Updated: 2026-06-02

This note records a critical evaluation finding: on art-to-art style transfer
sets, an unchanged source image can already obtain a high `clip_style`. Therefore
absolute `clip_style` and low LPIPS are not sufficient evidence of real style
transfer. Every art-to-art benchmark should report a no-op reference and, when
possible, a no-op-adjusted style gain.

## Files

Aggregate data:

- `docs/experiments/metric_hacking_noop_20260602/noop_full_transfer_summary.csv`

New no-op evaluations:

- 256 no-op summary:
  `docs/experiments/metric_hacking_noop_20260602/legacy256_no_op_identity_5x5/summary.json`
- 256 no-op copied images:
  `docs/experiments/metric_hacking_noop_20260602/legacy256_no_op_identity_5x5/images`
- 512 no-op summary:
  `docs/experiments/metric_hacking_noop_20260602/legacy512_no_op_identity_5x5/summary.json`
- 512 no-op copied images:
  `docs/experiments/metric_hacking_noop_20260602/legacy512_no_op_identity_5x5/images`
- Distinct5 no-op summary:
  `docs/experiments/distinct5_512_20260602/no_op_identity_5x5_summary.json`
- Distinct5 full/transfer points:
  `docs/experiments/distinct5_512_20260602/tables/clip_style_vs_1lpips_full_transfer_points.csv`
- Distinct5 aggregate ArtFID diagnostic:
  `docs/experiments/distinct5_512_20260602/artfid_metric_hacking/distinct5_aggregate_artfid_keypoints.csv`
- Distinct5 visual alignment audit:
  `docs/experiments/distinct5_512_20260602/visual_metric_alignment_20260602/README.zh.md`

Source image roots used for no-op:

- 256 legacy / overfit50:
  `G:\GitHub\Latent_Style\style_data\overfit50`
- WikiArt512 five-style:
  `F:\wikiart_images_512_ema_test`
- Distinct5-512:
  remote WSL source ` /mnt/i/wikiart_distinct5_samam_512_classview/test `

SaMAM result roots:

- SaMAM 256 initial:
  `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samam_wsl_mamba_256_formal_750_eval`
- SaMAM 256 continue:
  `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samam_wsl_mamba_256_continue17k_to25k\formal_eval_750`
- SaMAM 512:
  `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samam_wsl_mamba_512_scratch_clean_silent_b1_20k\formal_eval_750`
- SaMAM Distinct5:
  `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag\eval_curve`

LANCET Distinct5 points:

- `docs/experiments/distinct5_512_20260602/tables/clip_style_vs_1lpips_full_transfer_points.csv`
- `docs/experiments/distinct5_512_20260602/lancet_runs.md`

## Protocol

For each dataset, no-op is built by copying the source image unchanged to every
target style slot, using the same filename convention as generated outputs.

The evaluation has two scopes:

- `full`: all 5x5 / 750 images, including identity diagonal.
- `transfer`: off-diagonal only, 600 images where source style differs from
  target style.

The key quantity is:

```text
no_op_adjusted_style = clip_style(model) - clip_style(no-op)
```

This does not solve perceptual evaluation, but it prevents the most obvious
failure mode: declaring victory because CLIP already considers the unchanged
source image close to the target art domain.

## Headline Results

### Legacy 256 / overfit50

Styles: `Hayao`, `cezanne`, `monet`, `photo`, `vangogh`.

| scope | method | point | clip_style | LPIPS | style gain over no-op |
|---|---|---|---:|---:|---:|
| full | No-op | unchanged | 0.661913 | 0.000000 | 0.000000 |
| full | SaMAM | best style, 14k | 0.696867 | 0.436278 | +0.034954 |
| full | SaMAM | best LPIPS, 25k | 0.693823 | 0.393958 | +0.031910 |
| transfer | No-op | unchanged | 0.616694 | 0.000000 | 0.000000 |
| transfer | SaMAM | best style, 15k | 0.673892 | 0.445060 | +0.057198 |
| transfer | SaMAM | best LPIPS, 25k | 0.666977 | 0.402174 | +0.050283 |

Interpretation: SaMAM-256 does beat no-op in CLIP-style, especially
transfer-only. However, the gain is bought with very large LPIPS. This is a
real style-signal increase by the metric, but it is not an efficient
style/content tradeoff.

### WikiArt512 five-style

Styles: `Realism`, `Impressionism`, `Post_Impressionism`, `Expressionism`,
`Symbolism`.

| scope | method | point | clip_style | LPIPS | style gain over no-op |
|---|---|---|---:|---:|---:|
| full | No-op | unchanged | 0.781528 | 0.000000 | 0.000000 |
| full | SaMAM | best style, 5k | 0.791244 | 0.283292 | +0.009716 |
| full | SaMAM | best LPIPS, 10k | 0.785089 | 0.164336 | +0.003561 |
| transfer | No-op | unchanged | 0.773026 | 0.000000 | 0.000000 |
| transfer | SaMAM | best style, 5k | 0.784589 | 0.283310 | +0.011563 |
| transfer | SaMAM | best LPIPS, 10k | 0.777356 | 0.164393 | +0.004330 |

Interpretation: this is the clearest metric trap. SaMAM-512 looks strong in
absolute `clip_style`, but the unchanged image already scores around `0.78`.
The apparent style gain above no-op is only about `+0.01` at the style-optimal
checkpoint and about `+0.004` at the LPIPS-optimal checkpoint. Most of the
headline `clip_style` is therefore dataset/CLIP prior, not generated style
change.

### Distinct5-512

Styles: `Early_Renaissance`, `Impressionism`, `Minimalism`, `Rococo`,
`Ukiyo_e`.

| scope | method | point | clip_style | LPIPS | style gain over no-op |
|---|---|---|---:|---:|---:|
| full | No-op | unchanged | 0.680123 | 0.000000 | 0.000000 |
| full | SaMAM | best style, 2000 | 0.583346 | 0.362153 | -0.096777 |
| full | SaMAM | best LPIPS, 2250 | 0.581097 | 0.353820 | -0.099026 |
| full | LANCET | best style, K e1 | 0.700995 | 0.362294 | +0.020872 |
| full | LANCET | best LPIPS, F e1 | 0.696915 | 0.318645 | +0.016792 |
| transfer | No-op | unchanged | 0.639921 | 0.000000 | 0.000000 |
| transfer | SaMAM | best style, 1250 | 0.557183 | 0.448703 | -0.082738 |
| transfer | SaMAM | best LPIPS, 2250 | 0.552252 | 0.360452 | -0.087669 |
| transfer | LANCET | best style, K e1 | 0.671167 | 0.372281 | +0.031246 |
| transfer | LANCET | best LPIPS, F e1 | 0.664360 | 0.324528 | +0.024440 |

Interpretation: on Distinct5-512, SaMAM is below no-op under both full and
transfer-only CLIP-style. Its nonzero LPIPS is evidence that the reproduced
checkpoints do change the image, but those changes do not become positive
target-style movement. LANCET remains above no-op in style, including
transfer-only, so it is the only tested model here that shows no-op-adjusted
target style movement.

## Aggregate ArtFID Diagnostic

The table below is a diagnostic aggregate ArtFID/FID run, not the official
pairwise per-target ArtFID protocol. It pools generated images and target
reference images across the active target set for each scope. The goal is to
separate broad art-domain preservation from target-style transfer.

Artifacts:

- CSV:
  `docs/experiments/distinct5_512_20260602/artfid_metric_hacking/distinct5_aggregate_artfid_keypoints.csv`
- JSON:
  `docs/experiments/distinct5_512_20260602/artfid_metric_hacking/distinct5_aggregate_artfid_keypoints.json`
- Remote execution log:
  `docs/experiments/distinct5_512_20260602/artfid_metric_hacking/aggregate_artfid.log`

| scope | method | point | count | clip_style | eval LPIPS | aggregate ArtFID | ArtFID-FID | ArtFID-LPIPS |
|---|---|---|---:|---:|---:|---:|---:|---:|
| full | No-op | unchanged | 750 | 0.680123 | 0.000000 | 1.001258 | 0.001258 | 0.000000 |
| transfer | No-op | unchanged | 600 | 0.639921 | 0.000000 | 1.001099 | 0.001099 | 0.000000 |
| full | SaMAM | step 2000 | 750 | 0.583346 | 0.362153 | 152.212344 | 118.749864 | 0.271086 |
| transfer | SaMAM | step 2000 | 600 | 0.554579 | 0.369102 | 154.258947 | 119.854077 | 0.276407 |
| full | SaMAM | step 2250 | 750 | 0.581097 | 0.353820 | 146.070968 | 114.061974 | 0.269498 |
| transfer | SaMAM | step 2250 | 600 | 0.552252 | 0.360452 | 148.205852 | 115.222829 | 0.275187 |
| full | LANCET-F | epoch 1 | 750 | 0.696915 | 0.318645 | 122.632547 | 92.617272 | 0.309935 |
| transfer | LANCET-F | epoch 1 | 600 | 0.664360 | 0.324528 | 126.825714 | 95.074200 | 0.320081 |
| full | LANCET-K | epoch 1 | 750 | 0.700995 | 0.362294 | 157.168750 | 112.523611 | 0.384459 |
| transfer | LANCET-K | epoch 1 | 600 | 0.671167 | 0.372281 | 161.957657 | 114.797481 | 0.398629 |

Interpretation: aggregate ArtFID is not being arbitrarily fooled here. The
unchanged no-op images obtain near-perfect aggregate ArtFID because they are
already real art-domain images and have zero content distance from themselves.
That is a valid answer to the question "does this output remain in the broad
art image distribution?", but it is not the same question as "did this output
move toward the specified target style?" SaMAM and LANCET correctly incur large
aggregate FID once they alter the image distribution, even when LANCET improves
target `clip_style` over no-op. For paper tables, ArtFID should therefore be
reported as an art-domain/content diagnostic with a no-op reference, not as a
standalone target-style winner criterion.

## Conclusion

The no-op reference changes the story:

1. `clip_style` has a high art-domain prior. On some art-to-art sets, the source
   image is already close to many target style prototypes.
2. LPIPS alone rewards no-op perfectly. A model can look good on LPIPS by doing
   little, but this is not style transfer.
3. Absolute CLIP-style alone is also unsafe. WikiArt512 shows that SaMAM's
   absolute `clip_style ~= 0.79` mostly comes from no-op baseline
   `clip_style ~= 0.78`.
4. Future tables should report at least:
   `clip_style`, `content_lpips`, no-op `clip_style`, and
   `clip_style - no_op_clip_style`.
5. For art-to-art transfer, transfer-only metrics are mandatory. Full all-5x5
   can hide diagonal identity effects, although in this audit the stronger
   conclusion survives transfer-only filtering.
6. Aggregate ArtFID/FID exposes the art-domain prior explicitly: no-op can look
   nearly perfect because the source images are already drawn from the same
   broad art domain as the target references. This is not sufficient evidence
   of target-style transfer.

This does not mean CLIP-style should be discarded. It means CLIP-style must be
anchored by no-op and interpreted as an incremental gain, not as an absolute
proof of stylization.
