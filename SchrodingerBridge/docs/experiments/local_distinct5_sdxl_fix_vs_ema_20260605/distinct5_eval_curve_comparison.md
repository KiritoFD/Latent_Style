# Distinct5 Eval Curve Comparison

## Baseline

- label: `LBM-K e1 (EMA latent)`
- summary: `G:\GitHub\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote\full_eval\epoch_0001\summary.json`
- full clip-style: `0.700995`
- full content LPIPS: `0.362294`
- transfer clip-style: `0.671167`
- transfer content LPIPS: `0.372281`
- full ArtFID: ``
- transfer ArtFID: ``

## Curve

| epoch | full clip-style | delta | full LPIPS | delta | transfer clip-style | delta | transfer LPIPS | delta | full ArtFID | delta | transfer ArtFID | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.694594 | -0.006401 | 0.368943 | 0.006649 | 0.665137 | -0.006030 | 0.375133 | 0.002852 |  |  |  |  |
| 2 | 0.695685 | -0.005310 | 0.347303 | -0.014991 | 0.665297 | -0.005870 | 0.353529 | -0.018752 |  |  |  |  |
| 3 | 0.696674 | -0.004321 | 0.367220 | 0.004927 | 0.667725 | -0.003442 | 0.377341 | 0.005060 |  |  |  |  |
| 4 | 0.695947 | -0.005048 | 0.339183 | -0.023111 | 0.664908 | -0.006259 | 0.347322 | -0.024958 |  |  |  |  |
| 5 | 0.697256 | -0.003739 | 0.357732 | -0.004562 | 0.666348 | -0.004819 | 0.368286 | -0.003995 |  |  |  |  |
| 6 | 0.695439 | -0.005556 | 0.368292 | 0.005999 | 0.665538 | -0.005629 | 0.378342 | 0.006061 |  |  |  |  |
| 7 | 0.693905 | -0.007090 | 0.373225 | 0.010931 | 0.663967 | -0.007200 | 0.381955 | 0.009674 |  |  |  |  |
| 8 | 0.694299 | -0.006696 | 0.378291 | 0.015997 | 0.664678 | -0.006489 | 0.387758 | 0.015477 |  |  |  |  |
