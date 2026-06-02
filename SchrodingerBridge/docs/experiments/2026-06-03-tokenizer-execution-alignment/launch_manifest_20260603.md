# Tokenizer Execution Alignment Launch Manifest

Date: 2026-06-03

## Formal run owner

- `Linnaeus`

## Script

- `tools/probe_style_representation.py`

## Selected checkpoint family

- `distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote`

## Selected point

- `epoch_0001`

## Input contract

- checkpoint:
  - remote workspace checkpoint for family
    `distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote`
  - exact `.pt` path must be resolved by `Linnaeus` during remote launch
    because this local repository currently retains the evaluated
    `full_eval/epoch_0001/metrics.csv` but not the checkpoint payload
- latent root:
  - Distinct5-512 latent train root used by the `H` family
- eval metrics:
  - `exp/distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote/full_eval/epoch_0001/metrics.csv`
- idt metrics:
  - `docs/experiments/idt_eval_20260602/distinct5_512/idt_5x5/metrics.csv`

## Required switches

- `--checkpoint`
- `--latent-root`
- `--delta-probe`
- `--eval-metrics-csv`
- `--idt-metrics-csv`

## Required output directory

- `exp/aaai2027_tokenizer_execution_alignment_h_e1`

## Expected durable outputs

- `style_latent_stats.csv`
- `style_latent_pairs.csv`
- `tokenizer_code_stats.csv`
- `tokenizer_code_pairs.csv`
- `generated_delta_stats.csv`
- `generated_delta_pairs.csv`
- `target_style_metrics.csv`
- `tokenizer_execution_alignment.csv`
- `tokenizer_execution_alignment_pairs.csv`
- `fig_code_vs_executed_pair_l2.pdf`
- `fig_stylewise_code_exec_delta_idt.pdf`
- `summary.json`
