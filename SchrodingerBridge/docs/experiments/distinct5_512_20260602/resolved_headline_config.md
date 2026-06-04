# Distinct5-512 Headline LBM Configuration Disclosure

Updated: 2026-06-04

Purpose: record the resolved configuration values behind the paper-facing
LBM-F/H/K Distinct5 rows, after JSON `_base` inheritance. This note prevents
the manuscript from overstating optional OT / flow-residual components that are
not active in the reported headline rows.

## Source Configs

| row | config |
|---|---|
| LBM-F | `SchrodingerBridge/configs/distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3.json` |
| LBM-H | `SchrodingerBridge/configs/distinct5_512_ema_variant_h_hard_explore_queue_e3.json` |
| LBM-K | `SchrodingerBridge/configs/distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3.json` |

## Shared Resolved Objective

All three rows inherit the same active objective values:

| field | value |
|---|---:|
| `objective_mode` | `omf` |
| `w_flow` | `0.0` |
| `terminal_swd_weight` | `20.0` |
| `w_kinetic` | `1.0` |
| `kinetic_mode` | `endpoint` |
| `terminal_num_steps` | `4` |
| `swd_patch_sizes` | `[3, 5, 7, 15]` |
| `swd_num_projections` | `64` |

Interpretation: the Distinct5 headline rows should be described as the
pairing-cache / terminal-SWD / kinetic OMF setting. They should not be described
as active flow endpoint-residual experiments or as online minibatch Sinkhorn
optimization at every training step.

## Shared Training Setup

| field | value |
|---|---:|
| `batch_size` | `80` |
| `num_epochs` | `3` |
| `learning_rate` | `0.0002` |
| `use_amp` | `true` |
| `amp_dtype` | `bf16` |
| `channels_last` | `true` |
| `use_gradient_checkpointing` | `true` |

The paper-facing LBM-F/K operating points use epoch 1.

## Row Differences

### LBM-F

- `pairing_cache_path`: prototype-pairing top-8 cache.
- `pairing_cache_sample_mode`: `rank_biased`.
- `pairing_cache_rank_schedule`: `easy_to_hard`.
- `pairing_cache_min_topk`: `2`.
- `pairing_cache_curriculum_epochs`: `3`.
- `pairing_cache_rank_power`: `1.5`.
- tokenizer: global VQ with `64` atoms and residual gain `0.25`.
- style spatial mode: `vq_content_guided`.

### LBM-H

LBM-H inherits LBM-F and changes the queue schedule:

- `pairing_cache_active_topk`: `2`.
- `pairing_cache_rank_schedule`: `fixed`.
- `pairing_cache_curriculum_epochs`: `0`.
- `pairing_cache_explore_prob`: `0.15`.
- `pairing_cache_explore_topk`: `8`.

### LBM-K

LBM-K inherits LBM-H and adds content-adaptive VQ atom routing:

- `tokenizer_content_adaptive`: `true`.
- `tokenizer_content_hidden_dim`: `64`.
- `tokenizer_content_gain`: `0.5`.
- `tokenizer_content_stopgrad`: `true`.

## Verification Command

The values above were inspected with the project config loader:

```powershell
cd G:\GitHub\Latent_Style\SchrodingerBridge
$env:PYTHONPATH='src'
@'
from config_schema import load_config
for p in [
    'configs/distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3.json',
    'configs/distinct5_512_ema_variant_h_hard_explore_queue_e3.json',
    'configs/distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3.json',
]:
    cfg = load_config(p)
    print(p, cfg['bridge'], cfg['training'], cfg['data'], cfg['model'])
'@ | py -3 -
```
