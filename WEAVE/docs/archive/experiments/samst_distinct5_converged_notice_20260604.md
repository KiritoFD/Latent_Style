## SaMST Distinct5 Done

SaMST Distinct5 is converged and no SaMST job is currently using the 3060.

- Conservative endpoint:
  `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602\eval_epoch15\epoch_0015`
- Convergence midpoint:
  `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603\eval_bundle\eval_epoch5\epoch_0005`
- Direct comparison packet:
  `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603\eval_bundle\compare_e5_vs_e15`

Training wall time:

- `e5`: `6958s` (`1:55:58.5`)
- `e15`: `20835s` (`5:47:15.4`)

Transfer-only metrics:

- `e5`: `CLIP-S 0.6989188100`, `LPIPS 0.6334999498`, `targetwise ArtFID 465.6860418255`
- `e15`: `CLIP-S 0.6957412316`, `LPIPS 0.6319495817`, `targetwise ArtFID 444.4870406091`

Full metrics:

- `e5`: `CLIP-S 0.7275811868`, `LPIPS 0.6270693954`, `targetwise ArtFID 432.0511083215`
- `e15`: `CLIP-S 0.7247245136`, `LPIPS 0.6255497488`, `targetwise ArtFID 395.7117071285`

Interpretation:

- `e5 -> e15` changes in CLIP-S / LPIPS are small enough to treat the curve as plateaued.
- `e15` remains the safer manuscript endpoint because targetwise ArtFID is lower.
- Main-model optimization and longer-training work can proceed now; SaMST no longer blocks the GPU queue.
