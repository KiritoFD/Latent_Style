# Round 1 Theory Coverage Matrix

Date: 2026-06-11

Purpose:

- map the concrete scheme list in [tokenizer.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/tokenizer.md) and [attn.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/attn.md) onto the actual round-1 family switches
- make it auditable which ideas are:
  - implemented as switches
  - already exercised in formal round-1 runs
  - still represented only as a policy / approximation / later-stage item

## Tokenizer Coverage

| Theory item | Round-1 switch / infra | Code status | Experiment status | Notes |
|---|---|---|---|---|
| `global_code + spatial_map` dual-field output | `StructuredStyleOutput`, `StyleMaps`, `tokenizer_family=*` | implemented | active round-1 base contract | legacy path remains backward-compatible |
| Scheme A: semantic codebook router | `tok_a_dino_dict` | implemented | `planned` | universal keys + style-specific values + DINO-masked SWD |
| Scheme B: dual-stream / exemplar matcher | `tok_b_cross_image` | implemented | `planned` | content DINO query over style-bank patches |
| Scheme C: residual semantic adapter | `tok_c_residual_adapter` | implemented | `planned` | retains global code, routes high-frequency residual map only |
| Prompt-token style prior / VLM-style prior | `tok_d_vlm_prompt` | implemented | `planned` | closed-set wrapper through learned prompt tokens |
| Closed-set `style_id` surface with open-set-compatible internals | `tok_b_cross_image`, `tok_d_vlm_prompt` | implemented | `planned` | round-1 benchmark surface remains `style_id` |
| DINO sidecar loading and style-bank caches | dataset `dino_cache_*`, runtime conditioning | implemented | active infra | smoke-tested and used by tokenizer-family path |
| Tokenizer-only training before backbone reopening | `freeze_mode=style_branch` | implemented | active policy | current DINO families are queued under this rule |
| Dedicated tokenizer warm-start packet | `prepare_round1_tokenizer_warmstart_config.py`, `launch_remote_round1_tokenizer_warmstart.py` | implemented | configs prepared for `tok_a/b/c/d`, not yet launched | pragmatic teacher/distill-based warm-start path for tokenizer families |
| Reconstruction-style tokenizer pretrain packet | `prepare_round1_tokenizer_reconstruction_pretrain_config.py`, `launch_remote_round1_tokenizer_reconstruction_pretrain.py` | implemented | configs prepared for `tok_a/b/c/d`, not yet launched | identity-only reconstruction-flavored packet using existing trainer/objective path |
| Full phased curriculum (`phase1/2/3`) | represented as round-1 operating policy, not separate launcher families | partial | pretrain packet exists, but full curriculum not automated end-to-end | current round-1 still runs family-by-family from the common parent |

## Backbone / Solver Coverage

| Theory item | Round-1 switch / infra | Code status | Experiment status | Notes |
|---|---|---|---|---|
| SA-Mod self-attention | `attn_sa_mod` | implemented | `rejected` | formal convergence done; not promotable |
| GW-OT attention | `attn_gw_ot` | implemented | `recalibration_needed` | formal curve exists, but lane still needs better VRAM/useful frontier behavior |
| Gated SPADE attention | `attn_gated_spade` | implemented | `recalibration_needed` | under-band / stalled formal behavior |
| PnP / self-injection attention | `attn_pnp_selfinject` | implemented | `recalibration_needed` | real curve exists; segmented non-concurrent train/eval path built |
| Tangent RK solver | `solver_tangent_rk` | implemented | `reviewing` | formal training closed through `epoch_0032`; waiting on deep review / stage-close packet |
| Predictor-corrector solver | `solver_pc` | implemented | `reviewing` | training closed through `epoch_0036`; no new Pareto point after the long bounded tail |
| UNSB / cycle solver | `solver_unsb_cycle` | implemented | `recalibration_needed` | first direct launch was under-band at about `5223 MiB`; next retry needs a higher effective batch |
| DINO-masked semantic SWD | `semantic_supervision_family=dino_masked_swd` | implemented | active for tokenizer families | loaded through runtime conditioning sidecars |
| Remote segmented train/eval alternation | `run_remote_round1_family_segmented.py` | implemented | used on `attn_pnp_selfinject` | avoids concurrent train+eval VRAM spikes |
| Remote all-ckpt fast-eval authority | remote fast-eval watcher + local sync watcher | implemented | active | current authority path for convergence reads |

## High-Cost Theory Mapping

| Higher-cost theory in docs | Round-1 representation | Current read |
|---|---|---|
| Exact DINO-driven semantic routing | direct in `tok_a/tok_b/tok_c/tok_d` families | implemented |
| Null-space / tangent projection solver | approximated by `solver_tangent_rk` family contract | partial approximation, not explicit full Jacobian projection |
| Multi-step geometric solver | `solver_tangent_rk`, `solver_pc`, `solver_unsb_cycle` | implemented |
| Pure endogenous self-attention structure anchoring | `attn_pnp_selfinject` and `attn_sa_mod` | implemented as round-1 families |
| Full separate curriculum runner | policy only | not yet launched as a dedicated packet |

## Current Queue Read

| Slot | Family | Status | Reason |
|---|---|---|---|
| Active formal lane | `none` | `idle` | `solver_pc` training is closed and `solver_unsb_cycle` still needs recalibration before the next formal launch |
| Reviewing solver family | `solver_pc`, `solver_tangent_rk` | `reviewing` | both solver-family training phases are now closed; deep review still pending |
| Next queue candidate | `solver_unsb_cycle` | `recalibration_needed` | first launch was under-band; next retry should raise the effective batch before formal relaunch |
| Auto handoff | `watch_launch_round1_queue_when_idle.py` | armed | once manifest has zero `running` families, invoke round-1 queue automatically |

## Gaps That Still Matter

- A dedicated tokenizer warm-start packet now exists, and reconstruction-flavored identity-only tokenizer pretrain packets are already prepared for `tok_a_dino_dict`, `tok_b_cross_image`, `tok_c_residual_adapter`, and `tok_d_vlm_prompt`, but the stronger fully custom self-supervised reconstruction trainer described in `tokenizer.md` is still not implemented as a separate trainer.
- The more radical full Jacobian-based null-space projection described in `attn.md` is only represented approximately by the current tangent solver family, not as a separate explicit implementation.
- Round-1 is already auditable family-by-family, but not yet fully factorized into an explicit `phase1/phase2/phase3` launcher stack.
