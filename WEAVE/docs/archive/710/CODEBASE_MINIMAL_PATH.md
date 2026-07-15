# WEAVE Minimal Code Path

## Active Source Files

The current 710/712 pipeline has one model contract: `weave`.

| File | Responsibility |
|---|---|
| `src/model.py` | WEAVE network, velocity heads, ODE solver, endpoint alignment |
| `src/flow.py` | Training interpolation and band-weighted flow-matching loss |
| `src/blocks.py` | Residual and cross-attention block |
| `src/wavelet.py` | Haar DWT/IDWT and subband schedules |
| `src/style.py` | DINO patch/style-memory projection |
| `src/trainer.py` | Training loop, checkpointing, logging |
| `src/config_schema.py` | Typed experiment configuration |
| `src/run.py` | CLI entry point |
| `src/style_families.py` | Checkpoint/config validation helpers |

The old spatial bridge, its loss, masking experiment, and spatial-only probes were removed. There is no second model implementation or runtime branch.

## Effective 710 Baseline

The resolved `configs/710_b0_weave_d5.json` path currently uses:

- Three learned velocity heads: LL, LH, and HL. HH has no learned head.
- Band weights `0.3 / 1.0 / 1.0` for LL/LH/HL.
- Plain band-wise flow matching only. SWD, edge, Gram, moment, and content losses do not enter `src/flow.py`.
- Target DINO patches as style conditioning.
- No DWT-only attention route (`cross_attn_dwt_route=false`).
- No subband time schedule (`subband_time_schedule_enabled=false`).
- Bridge noise `sigma=0.02`.
- `spatial_fiber` endpoint alignment with scale `1.0` and extrapolation `0.1`.
- Endpoint alignment is not restricted to the final solver step (`endpoint_adain_only_last_step=false`).

These facts should be treated as the implementation baseline. Paper claims and ablation labels must be checked against this list.

## Refactor Verification

The minimal-source retrain used the same five-epoch, batch-24 protocol and 750-image evaluation as the clean baseline. It was evaluated on the remote 3060 after the source-module consolidation.

| Run | CLIP-S | LPIPS | DINO-S | DINO-C |
|---|---:|---:|---:|---:|
| Old baseline (`t1_asg_5ep_noasg`) | 0.7261 | 0.3354 | 0.4843 | 0.7692 |
| Clean baseline | 0.7272 | 0.3431 | 0.4829 | 0.7552 |
| Minimal-source retrain | 0.7266 | 0.3343 | 0.4813 | 0.7573 |

The code consolidation does not cause a CLIP-S or LPIPS regression. DINO-S and DINO-C remain within the variation seen between the historical clean runs, but DINO-C is the sensitive metric to monitor in the endpoint-alignment and solver controls below.

## Required Ablations Before More Deletion

Run every row from one checkpoint lineage and evaluate the same 750 images with CLIP-S, LPIPS, DINO-S, and DINO-C.

| Priority | Comparison | Decision enabled |
|---|---|---|
| P0 | Endpoint alignment off vs current per-step vs final-step-only | Delete per-step WCT/AdaIN branches or retain one endpoint path |
| P0 | `sigma=0` vs `sigma=0.02` | Remove stochastic bridge code if noise does not improve the four metrics |
| P0 | LL weight `0 / 0.1 / 0.3 / 1.0` | Fix one LL policy and delete generic band-weight plumbing |
| P1 | DINO patch conditioning vs style-ID memory only | Decide whether DINO cache/projection and cross-attention are necessary |
| P1 | Cross-attention off vs current | Remove attention diagnostics/routing if the residual backbone is sufficient |
| P1 | Three heads vs enabling HH head | Confirm whether passive HH is intentional before deleting the dormant HH-head option |
| P1 | Uniform time weighting vs the 712 subband schedule | Keep or delete all gamma schedule code |
| P2 | Structure-aligned target off vs on | Delete target projection branch if it remains ineffective |

## Code-Only Probes

These do not need a full training run first:

1. Verify `style_global`, `cls_proj`, head FiLM, attention entropy maps, and guidance maps have zero influence on model outputs under the resolved baseline.
2. Load the current checkpoint after pruning each dead state-dict prefix and compare outputs bit-for-bit.
3. Trace one train step and one inference pass to list executed branches in `model.py` and `trainer.py`.
4. After the P0/P1 decisions, split `trainer.py` and `config_schema.py`; doing this earlier would only reorganize unresolved options.

## Acceptance Rule

A component is removable when either:

- It is unreachable under the resolved baseline and output-equivalence is verified, or
- Its controlled ablation changes none of the four metrics beyond run-to-run variation while increasing code or runtime complexity.
