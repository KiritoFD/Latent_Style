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
| UNSB / cycle solver | `solver_unsb_cycle` | implemented | `running` | formal lane remains active; `epoch_0009` reopened the Pareto frontier and `epoch_0010` is the first post-frontier follow-up read |
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
| Active formal lane | `solver_unsb_cycle` | `running` | `epoch_0009` became a new Pareto point, so the solver patience clock reset; `epoch_0010` is weaker but not enough to close the line |
| Reviewing solver family | `solver_pc`, `solver_tangent_rk` | `reviewing` | both earlier solver-family training phases are now closed; deep review still pending |
| Next queue candidate | `defer until unsb closure` | `planned` | tokenizer families remain tail items; exact next lane should be resolved only after the current solver family is formally closed and the DINO-last rule is re-applied |
| Auto handoff | `watch_launch_round1_queue_when_idle.py` | armed | once manifest has zero `running` families, invoke the queue automatically, but do not bypass the DINO-last and stage-summary policy |

## Current UNSB Read

- latest settled point:
  - `epoch_0011`
  - transfer `0.6888 / 0.4889`
  - all-pairs `0.7103 / 0.4783`
- current family-best points:
  - best transfer `CLIP-S`:
    - `epoch_0001`
    - `0.7057 / 0.5669`
  - best transfer `LPIPS`:
    - `epoch_0009`
    - `0.6996 / 0.4421`
  - best all-pairs `CLIP-S`:
    - `epoch_0009`
    - `0.7245 / 0.4311`
- interpretation:
  - `epoch_0009` is the first genuinely new late-stage Pareto point after the earlier `epoch_0003` frontier
  - `epoch_0010-0011` both soften from `epoch_0009`, so the frontier reactivation is real but not yet stable
  - the current question is no longer "is UNSB near closure"
  - it is now "does the post-epoch_0009 tail stabilize near the new frontier or collapse back toward the earlier mid-curve regime"
  - `epoch_0011` also raises a secondary efficiency question because its fast-eval wall time is much higher than the surrounding checkpoints

## Current Shortlist Surface

- canonical fast shortlist handoff now exists for `solver_unsb_cycle`:
  - [full_eval_fast_snapshot_bestfew_handoff.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_solver_unsb_cycle_fast_local/full_eval_fast_snapshot_bestfew_handoff.csv)
- current canonical picks are:
  - `epoch_0001` for best transfer `CLIP-S`
  - `epoch_0009` for best transfer `LPIPS`
  - `epoch_0009` for best all-pairs `CLIP-S`
  - `epoch_0009` for best structure-preserving point inside the current fast contract
- implication:
  - once local heavy review budget is reopened, `epoch_0001` and `epoch_0009` are the first mandatory UNSB checkpoints for `IntroStyle / DINO / VLM`

## Gaps That Still Matter

- A dedicated tokenizer warm-start packet now exists, and reconstruction-flavored identity-only tokenizer pretrain packets are already prepared for `tok_a_dino_dict`, `tok_b_cross_image`, `tok_c_residual_adapter`, and `tok_d_vlm_prompt`, but the stronger fully custom self-supervised reconstruction trainer described in `tokenizer.md` is still not implemented as a separate trainer.
- The more radical full Jacobian-based null-space projection described in `attn.md` is only represented approximately by the current tangent solver family, not as a separate explicit implementation.
- Round-1 is already auditable family-by-family, but not yet fully factorized into an explicit `phase1/phase2/phase3` launcher stack.
- The queue policy is now intentionally asymmetric: non-DINO solver/backbone closure work stays ahead of DINO-heavy tokenizer launches, even though those tokenizer families are already implemented and smoke-passing.
