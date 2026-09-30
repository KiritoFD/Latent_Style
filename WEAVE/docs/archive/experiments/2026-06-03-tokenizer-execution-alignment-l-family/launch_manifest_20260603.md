# Tokenizer Execution Alignment `L`-Family Launch Manifest

Date: 2026-06-03

## Formal run owner

- `Linnaeus`

## Script

- `tools/probe_style_representation.py`

## Selected family / point

- family:
  - `distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote`
- point:
  - `epoch_0001`

## Input contract

- checkpoint:
  - remote workspace payload for the selected `L` family `epoch_0001`
- latent root:
  - actual Distinct5-512 latent train root used by the `L` family config
- eval metrics:
  - remote `full_eval/epoch_0001/metrics.csv` for the selected `L` family
- idt metrics:
  - `docs/experiments/idt_eval_20260602/distinct5_512/idt_5x5/metrics.csv`

## Required switches

- `--checkpoint`
- `--latent-root`
- `--output-dir`
- `--delta-probe`
- `--eval-metrics-csv`
- `--idt-metrics-csv`

## Required output directory

- `exp/aaai2027_tokenizer_execution_alignment_l_e1`

## Expected durable outputs

- `style_latent_stats.csv`
- `style_latent_pairs.csv`
- `tokenizer_code_stats.csv`
- `tokenizer_code_pairs.csv`
- `generated_delta_stats.csv`
- `generated_delta_pairs.csv`
- `delta_probe_content_paths.csv`
- `target_style_metrics.csv`
- `tokenizer_execution_alignment.csv`
- `tokenizer_execution_alignment_pairs.csv`
- `fig_code_vs_executed_pair_l2.pdf`
- `fig_stylewise_code_exec_delta_idt.pdf`
- `summary.json`
