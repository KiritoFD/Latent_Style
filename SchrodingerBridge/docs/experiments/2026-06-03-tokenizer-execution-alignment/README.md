# Tokenizer Execution Alignment Packet

Date: 2026-06-03

Purpose:

- convert the tokenizer code-vs-execution theory gap into one formal paper-side
  packet;
- measure whether style-code separability survives execution strongly enough to
  predict real no-op-adjusted style gain;
- close the next mechanism gate after the negative Gate B SA-SWD result.

## Scope

Primary benchmark:

- `Distinct5-512`

Primary run owner:

- remote `3060` via `Linnaeus`

## Selected checkpoint

Current formal target:

- family:
  - `distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote`
- checkpoint:
  - `epoch_0001`

Why this point:

- it stays inside the currently reviewed `H` mainline family;
- it is the lower-LPIPS balance point within the reviewed `H` pair;
- it keeps this probe tied to the same family already used in the Gate B
  closure work, instead of switching mechanism stories midstream.

## Required inputs

1. checkpoint:
   - remote workspace checkpoint for family
     `distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote`
   - exact path must be verified by `Linnaeus` at launch time because the local
     repository currently retains the evaluated `full_eval` metrics but not the
     `.pt` payload itself
2. training latent root:
   - Distinct5-512 latent train root used by the `H` family
3. evaluated checkpoint metrics:
   - `exp/distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote/full_eval/epoch_0001/metrics.csv`
4. unchanged-image reference metrics:
   - `docs/experiments/idt_eval_20260602/distinct5_512/idt_5x5/metrics.csv`

## Required outputs

Runtime output directory:

- `exp/aaai2027_tokenizer_execution_alignment_h_e1`

Durable paper-side artifacts:

- `delta_probe_content_paths.csv`
- `target_style_metrics.csv`
- `tokenizer_execution_alignment.csv`
- `tokenizer_execution_alignment_pairs.csv`
- `fig_code_vs_executed_pair_l2.pdf`
- `fig_stylewise_code_exec_delta_idt.pdf`
- `summary.json`

## Intended reading

The packet is useful even if it is negative.

Interpretation boundary:

- strong code geometry + weak executed geometry:
  - execution remains the first suspect
- weak code geometry + weak executed geometry:
  - tokenizer weakness remains plausible
- strong code geometry + strong executed geometry + weak `delta_idt`:
  - downstream trade-off or metric effects remain the next suspect

This packet does **not** by itself prove a new tokenizer design. Its job is to
localize whether the present bottleneck is code-space weakness or execution-side
attenuation.

Minimum provenance requirement for paper use:

- the landed packet must preserve the exact content-file list, delta-probe
  arguments, resolved checkpoint path, and runtime metadata alongside the
  geometry/eval outputs.

## Current prelaunch state

Heartbeat-confirmed state as of `2026-06-03`:

- remote clean worktree synced by `Linnaeus`:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge`
- synced branch / commit:
  - `codex/tokenizer-clean-c3058eab`
  - `ebb194669`
- launch status:
  - not started
- blocker:
  - the selected remote checkpoint payload for
    `distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote`
    `epoch_0001` is currently missing
- roots searched by `Linnaeus`:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge`
  - `I:\Github\Latent_Style_TokenizerClean`
  - `I:\Github\Latent_Style`
- confirmed surviving same-family evidence:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote\config.json`
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote\logs\training_20260602_235921.csv`
- confirmed latent train root from the surviving family config:
  - `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train`
- candidate inventory note:
  - `payload_candidate_inventory_20260603.md`
- successor packet:
  - `../2026-06-03-tokenizer-execution-alignment-l-family/README.md`
- policy:
  - do not substitute another checkpoint silently
  - recover or prove absence first, then decide the next action explicitly

## Next update rule

Once the remote run lands:

1. update this packet with the exact remote/runtime paths;
2. add the concrete correlation and figure results;
3. hand the landed packet to `Carver` and `Darwin` before any manuscript
   escalation.
