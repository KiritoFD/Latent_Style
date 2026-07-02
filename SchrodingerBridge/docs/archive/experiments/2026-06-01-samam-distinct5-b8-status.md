# SaMAM Distinct-5 512 b8 Status

Date: 2026-06-01

Run:

```text
/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_mamba_b8_seg250_remote_wsl_20260601_1935
```

Purpose: match LANCET's formal VRAM range by running SaMAM at batch size 8 and
save/evaluate every 250 steps.

Observed status:

- VRAM was in the intended range, about 9.5G.
- The run did not reach the first 250-step checkpoint.
- At approximately step 64, training metrics in `segmented.log` became NaN:

```text
id_loss1=nan, id_loss2=nan, loss_style=nan, loss_content=nan
```

Action:

- The b8 segmented task was stopped to avoid wasting the remote 3060.
- No 250-step metric should be reported from this run.
- This is an invalid baseline curve, not evidence of SaMAM convergence.

Next baseline choice should be explicit: either run the previously stable SaMAM
setting as the convergence baseline, or keep b8 as a stress test and change only
the minimal stability variable while clearly labeling it.

