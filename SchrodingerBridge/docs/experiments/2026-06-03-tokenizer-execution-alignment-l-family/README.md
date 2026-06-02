# Tokenizer Execution Alignment Successor Packet (`L` family)

Date: 2026-06-03

Purpose:

- replace the now-blocked same-family `H e1` tokenizer execution-alignment
  packet with one explicit payload-backed successor packet;
- keep the code-to-execution probe moving on remote `3060`;
- preserve paper safety by treating this as a **new mechanism-family probe**,
  not as a fallback for the original `H` packet.

## Why a successor packet exists

The original packet at:

- `docs/experiments/2026-06-03-tokenizer-execution-alignment/README.md`

is now blocked as a same-family packet because:

- `H e1` payload is missing on remote;
- same-family adjacent fallback `H e2` is also unavailable;
- `F` is likewise unavailable as a payload-backed paper-facing fallback.

See:

- `docs/experiments/2026-06-03-tokenizer-execution-alignment/payload_candidate_inventory_20260603.md`
- `docs/reviews/tokenizer_probe_checkpoint_reselection_policy_20260603.md`
- `docs/reviews/aaai2027_tokenizer_probe_successor_family_reread_20260603.md`

## Successor selection

Selected successor family:

- `distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote`

Selected point:

- `epoch_0001`

Why `L e1`:

- it is payload-backed on remote;
- it has landed `full_eval` summaries/metrics;
- adversarial reread ranked it as the least unsafe paper-facing successor among
  the currently available `K/J/L/M` payload-backed families.

Path-truth audit confirmed by remote owner:

- artifact root:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote`
- checkpoint:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote\epoch_0001.pt`
- config:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote\config.json`
- eval metrics:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote\full_eval\epoch_0001\metrics.csv`

Current runtime state:

- launch status:
  - started
- clean code root:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge`
- runtime output dir:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_execution_alignment_l_e1`
- execution mode:
  - split-root launch
  - clean worktree code plus old artifact-bearing checkpoint / config / metrics

## Claim boundary

This packet is **not** allowed to inherit the old `H`-family continuity claim.

Safe interpretation:

- this is a new `L`-family execution-alignment probe launched after the
  original same-family `H` packet failed operationally.

Unsafe interpretation:

- presenting `L e1` as if it were a routine fallback for `H e1`
- claiming that the `L`-family probe preserves the original reviewed
  `H`-mechanism story unchanged

## Required inputs

1. checkpoint:
   - remote workspace payload for family
     `distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote`
   - target point:
     - `epoch_0001`
2. training latent root:
   - the Distinct5-512 latent train root actually used by the `L` family
3. evaluated checkpoint metrics:
   - remote `full_eval/epoch_0001/metrics.csv` for the `L` family
4. unchanged-image reference metrics:
   - `docs/experiments/idt_eval_20260602/distinct5_512/idt_5x5/metrics.csv`

## Required outputs

Runtime output directory:

- `exp/aaai2027_tokenizer_execution_alignment_l_e1`

Durable artifacts:

- `delta_probe_content_paths.csv`
- `target_style_metrics.csv`
- `tokenizer_execution_alignment.csv`
- `tokenizer_execution_alignment_pairs.csv`
- `fig_code_vs_executed_pair_l2.pdf`
- `fig_stylewise_code_exec_delta_idt.pdf`
- `summary.json`

## Next update rule

Once the remote run lands:

1. add exact remote checkpoint / config / metrics paths;
2. record the measured code/output correlations and figures;
3. hand the landed packet to `Carver` and `Darwin` before any manuscript use;
4. keep all manuscript wording explicit that this evidence comes from the
   payload-backed `L`-family successor packet, not the blocked `H` packet.
