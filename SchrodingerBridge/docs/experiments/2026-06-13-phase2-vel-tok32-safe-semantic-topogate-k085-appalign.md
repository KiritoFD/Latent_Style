# Phase 2: vel_tok32_safe_semantic_topogate_k085_appalign

Date: 2026-06-13

## Role

- current preferred structure-side mainline packet
- inherits the promoted `topogate epoch_0003` parent while keeping the same `velocity + pure_latent_spatial + semantic_topology_gate` family
- adds only a conservative tokenizer-guided output appearance head

## Why This Exists

- the active `vel_tok32_safe_semantic_topogate_k085` packet already recovered the old all-pairs safe shelf on `epoch_0001`
- but transfer style still trails the formal shelf slightly even though LPIPS is much cleaner than the formal lane
- that pattern raises a narrower question than another structure-family swap:
  - is the remaining shortfall partly low-order appearance mismatch
  - such as brightness / contrast / exposure statistics
  - rather than missing structure routing
- this packet tests that question directly without changing:
  - the tokenizer family
  - the solver family
  - the topology-gated structure mechanism

## Config

- preferred relaunch config:
  - [phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1.json)
- first launch config:
  - [phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b16a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b16a1.json)
- comparison parent:
  - `vel_tok32_safe_semantic_topogate_k085 epoch_0003`
  - transfer `0.675388 / 0.375598`
  - all-pairs `0.702936 / 0.371762`
- active sibling reference:
  - [2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085.md)

## Deltas

- keep:
  - `tokenizer_family = pure_latent_spatial`
  - `transport_prediction_mode = velocity`
  - `solver_family = euler_legacy`
  - `semantic_self_topology_gate = true`
  - `semantic_self_topology_blend = 1.0`
  - `proximal_mode = crossattn_texture`
  - `batch_size = 16`
  - `accumulation_steps = 1`
- appearance head:
  - `output_appearance_alignment_mode = tokenizer_latent_affine`
  - `output_appearance_hidden_dim = 96`
  - `output_appearance_log_scale_span = ln(1.2)`
  - `output_appearance_shift_span = 0.2`
  - `output_appearance_blend = 0.75`
  - `output_appearance_use_spatial_stats = true`
  - `output_appearance_use_gate_mask_stats = true`

## Intended Read

- success:
  - transfer style rises while LPIPS stays near the active topology-gate band
  - all-pairs shelf recovery is retained
  - new appearance metrics show nontrivial but still small corrections
  - tokenizer observability remains live rather than collapsing
- failure:
  - no measurable style lift relative to the active sibling
  - LPIPS drifts upward enough to erase the safe-band gain
  - the head learns large unstable corrections instead of small low-order alignment

## Queue Position

- this packet is now the active preferred `structure_reentry` packet
- it was promoted after `vel_tok32_safe_semantic_topogate_k085` closed on runtime guard while preserving `epoch_0003` as the promoted best point
- it still stays ahead of `safe_pnp_selfinject`
- reason:
  - it is the cleanest attribution test after the first structure-side breakout
  - and it costs much less architectural disturbance than changing the attention family
- guide-aligned close rule:
  - if this line is still style-limited when the close gate is reached, stop the structure lane and auto-launch the queued `i2sb_diagnostic_only` packet
  - the next preferred read should be:
    - [2026-06-13-phase2-i2sb-tok32-safe-semantic-topogate-sigma0p02-residual-tfloor005.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-i2sb-tok32-safe-semantic-topogate-sigma0p02-residual-tfloor005.md)
    - then the existing eval-only `solver_pc` check
  - [2026-06-13-phase2-vel-tok32-safe-pnp-selfinject.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-pnp-selfinject.md) remains queued, but no longer as the automatic next hop behind `appalign`

## Launch Read

- launch status:
  - first `b16a1` launch resumed from `vel_tok32_safe_semantic_topogate_k085 epoch_0003`
  - but it hit runtime guard before the first settled authority point:
    - observed guard event: `11449 MiB > 11000 MiB`
  - preferred relaunch now switches to `b12a1`
  - the active remote mainline is now `b12a1`
  - latest settled authority point is now `epoch_0003`
  - `epoch_0004` has already been checkpointed and is currently in full eval
  - current board state should be read as `eval_in_progress_or_pending`

## Settled Reads

- `epoch_0001`
  - transfer `0.672604 / 0.336357`
  - all-pairs `0.703506 / 0.332992`
  - identity `0.827117 / 0.319531`
  - read:
    - first clean all-pairs shelf recovery
    - transfer still short of the formal shelf by about `0.00133`

- `epoch_0002`
  - transfer `0.671361 / 0.314290`
  - all-pairs `0.703097 / 0.311966`
  - identity `0.830038 / 0.302673`
  - `style - IDT`:
    - transfer `+0.031439`
    - all-pairs `+0.022972`
  - runtime observability from checkpoint summary:
    - transfer-side `tok_eff=3.7, gate=0.626, mask=0.649, topo_ent=0.973, app_on=1.0, app_s=1.000, app_d=0.000`
  - read:
    - all-pairs remains above the safe shelf
    - LPIPS improves materially again versus both `epoch_0001` and the promoted `topogate epoch_0003`
    - transfer style slips further below the formal shelf
    - so the lane is still active as a structure-clean Pareto frontier, but it has not yet converted into a full transfer recovery

- `epoch_0003`
  - transfer `0.671810 / 0.314716`
  - all-pairs `0.703130 / 0.312658`
  - identity `0.828410 / 0.304422`
  - `style - IDT`:
    - transfer `+0.031888`
    - all-pairs `+0.023005`
  - runtime observability from checkpoint summary:
    - transfer-side `tok_eff=4.1, gate=0.594, mask=0.634, topo_ent=0.873, app_on=1.0, app_s=1.000, app_d=0.000`
  - read:
    - all-pairs still stays above the safe shelf
    - LPIPS remains near the `0.31` floor instead of rebounding
    - transfer style recovers a little versus `epoch_0002`, but is still below the formal shelf
    - so this point extends the same structure-clean Pareto frontier instead of creating a real style breakout

- interpretation:
  - `appalign` is no longer just a one-point anomaly; it already has three settled Pareto points
  - the current behavior is "keep structure and LPIPS pinned near the floor while style saturates below the transfer gate"
  - the remote eval script is now writing checkpoint-level `runtime_observability`, so later settled checkpoints can be judged on both board metrics and tokenizer / appearance activity
  - the guide read agrees with the current evidence: this lane should be treated as style-limited, not tokenizer-capacity-limited
  - current close-gate blocker is still simple:
    - latest settled epoch is only `3`
    - the tail is not yet flat
    - so the watcher should keep this lane alive until `epoch_0004` settles
## Parent Refresh

- Source packet: `vel_tok32_safe_semantic_topogate_k085`
- Selection policy: `latest`
- Selected parent epoch: `epoch_0003`
- Selected parent checkpoint: `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b16a1/epoch_0003.pt`
- Selected parent metrics: transfer `0.675388 / 0.375598`, all-pairs `0.702936 / 0.371762`
