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

## Next update rule

Once the remote run lands:

1. update this packet with the exact remote/runtime paths;
2. add the concrete correlation and figure results;
3. hand the landed packet to `Carver` and `Darwin` before any manuscript
   escalation.
