# Target-HF subband residual time-window probe

This probe evaluated the already-trained `target_hf_subband_ft6/epoch_0006.pt`
checkpoint without changing weights. The additive target-HF subband residual was
multiplied by a hard time window during inference:

`v(t) = v_base(t) + w(t) * delta_hf(t)`

Both windows used normalization by `1 / window_width`, so the approximate
integrated residual energy matches the original full-trajectory residual.

| variant | active window | all DINO-S | off DINO-S | DINO-C | CLIP-S | LPIPS | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| subband-only baseline | full | **0.488624** | **0.403917** | **0.798123** | **0.720880** | **0.296553** | keep |
| late normalized | [0.5, 1.0] | 0.486637 | 0.402562 | 0.793614 | 0.719335 | 0.297480 | reject |
| early normalized | [0.0, 0.5] | 0.486602 | 0.402543 | 0.793654 | 0.719377 | 0.297480 | reject |

## Reading

Temporal localization does not improve the residual route. Early-only and
late-only normalized windows are almost identical and both underperform the
full-trajectory residual on style and content metrics. The residual is therefore
not merely a late texture endpoint correction or an early structure transport
term; it works as a small continuous correction across the ODE path. The next
architecture should change residual parameterization or conditioning, not just
when the residual is applied.
