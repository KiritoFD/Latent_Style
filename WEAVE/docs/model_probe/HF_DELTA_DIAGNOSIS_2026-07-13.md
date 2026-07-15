# HF Delta Probe Diagnosis

Date: 2026-07-13

## Diagnosis

Baseline `brk_a_ll03_10ep` had the wrong style information path for image-specific transfer:

- Training target uses `LL = 0.7 * content_LL + 0.3 * AdaIN(content_LL -> style_LL)` and `LH/HL/HH = target_style` bands.
- The model condition path mostly reads `target_style_id -> style_memory -> cross-attn`.
- `target_style` latent affects target construction, but baseline does not read it as a model condition.
- `HH` was in the target/loss construction but had no output head when `enable_hh_head=false`.

The first target-latent token fusion fixed the missing path, but it was too global. It injected target latent into all `style_tokens`, so the easiest learned route was LL control:

| run | epoch | target latent LL | LH | HL | DINO-S | DINO-C | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| target token fusion | 3 | 0.7012 | 0.1712 | 0.1454 | 0.480885 | 0.788606 | 0.323213 |
| target token fusion | 15 | 1.0632 | 0.3078 | 0.2587 | 0.481694 | 0.782969 | 0.339134 |

Conclusion: global token fusion connects the path but over-controls LL and does not raise the useful style-learning ceiling.

## Fix

Added an HF-only target latent residual path:

- `target_latent_hf_head_fusion_enabled`
- Extracts only target-style `LH/HL/HH` DWT bands.
- Adds residual velocity only to `LH/HL/HH`; LL head remains unchanged.
- Enables `HH` head so the HH target has a supervised output channel.

Implemented in:

- `src/model.py`: `StyleOnlyVelocityDelta`, HF target encoder, HF-only residual heads, debug probes.
- `src/config_schema.py`: config fields for HF target latent fusion.
- `configs/exp_probe_target_hf_delta_ft15.json`
- `configs/exp_probe_target_hf_delta_strong_ft6.json`

## Results

Remote GPU: `administrator@100.115.18.62`, repo `I:/Github/Latent_Style/SchrodingerBridge`.
Evaluation: 750 D5 pairs, canonical DINOv2-small, main style metric is DINO-S.

| config/checkpoint | endpoint AdaIN | DINO-S | DINO-C | DINO-structure | CLIP-S | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| `target_hf_delta_ft15/epoch_0006.pt` | 1.0 | 0.482656 | 0.791748 | 0.024586 | 0.717485 | 0.295013 |
| `target_hf_delta_ft15/epoch_0015.pt` | 1.0 | 0.482480 | 0.791313 | 0.024650 | 0.718030 | 0.299062 |
| `target_hf_delta_ft15/epoch_0006.pt` | 1.5 | 0.484984 | 0.796570 | 0.024582 | 0.717586 | 0.292892 |
| `target_hf_delta_ft15/epoch_0006.pt` | 2.0 | 0.484675 | 0.830479 | 0.023864 | 0.702580 | 0.267950 |
| `target_hf_delta_strong_ft6/epoch_0006.pt` | 1.5 | **0.487036** | **0.799077** | 0.024591 | 0.717586 | 0.295459 |

Best current point:

- Checkpoint: `I:/Github/Latent_Style/SchrodingerBridge/exp/model_probe/target_hf_delta_strong_ft6/epoch_0006.pt`
- Override: `configs/eval_adain_15.json`
- DINO-S: `0.487036`
- This exceeds the previous delivered max-DINO point (`0.4859`) while keeping DINO-C much higher.

## Interpretation

Training longer was not the core blocker. The useful change was increasing the model's correct-frequency capacity:

- LL must not receive target-latent shortcuts.
- HF and HH need an explicit supervised output path.
- Stronger HF residual capacity improves DINO-S without the content collapse seen in global target-token fusion.

CFG note: previous CFG results were not clean CFG. They also enabled style delta heads, DWT route, HH head, and larger cross-attn gates, so content preservation cannot be attributed to CFG alone.
