# Related Works

Baseline methods, reproduced outputs, and evaluation infrastructure for style transfer comparison.

## Current Ledgers

These files are the current source of truth for the reproduction state:

- `docs/REPRO_DATA_INDEX.md`: human-readable inventory of reproduced outputs, reusable legacy image folders, timing/status rows, and strict protocol-750 coverage.
- `results/repro_data_inventory.csv`: machine-readable run/output inventory.
- `results/repro_data_files.csv`: machine-readable CSV/JSON/Markdown/HTML data-file ledger.
- `results/metrics_summary/`: standalone metric-summary folder for quick reading and paper/result aggregation.
- `results/json_archive/`: top-level aggregate JSON archive.
- `run_511/complete_750/summary_all_tested_metrics.csv`: source copy of all tested strict protocol-750 metrics.
- `run_511/docs/BASELINE_RUN_PLAN.md`: current baseline completion/next-run plan.
- `run_511/docs/ADVANCED_METRICS_TOOLCHAIN.md`: advanced metric scripts, dependencies, and interpretation notes.

Refresh after new runs:

```bat
python Related_Works\scripts\collect_repro_inventory.py
```

## Directory Structure

```text
Related_Works/
  repos/                  # Baseline method repositories
  run_511/                # Protocol-750 launchers, metrics, summaries, and results
    complete_750/         # Aggregated strict 750-image eval results
    outputs/              # Training/inference outputs and smoke runs
    repos/                # Working copies used by run_511 wrappers
    eval/                 # Evaluation scripts
    launchers/            # .bat and .py train/infer launchers
    summaries/            # Report builders and timing scripts
    docs/                 # run_511 status, timing, and metric docs
  baseline_pipeline/      # Older baseline pipeline infrastructure and migrated ckpts
  runs/                   # Legacy experiment run history and reusable generated images
  summary/                # Aggregated historical experiment summaries
  scripts/                # Utility scripts, including the reproduction inventory collector
  results/                # CSV/JSON result files and generated inventories
  docs/                   # Top-level documentation
```

## Completed Evaluations (Protocol-750)

See `run_511/complete_750/` for strict 750-image eval suites. The canonical generated tables are:

- `run_511/complete_750/summary_complete_750.md`
- `run_511/complete_750/summary_complete_750.csv`
- `run_511/complete_750/summary_all_tested_metrics.md`
- `run_511/complete_750/summary_all_tested_metrics.csv`

Current strict rows include `Ours epoch_0007`, `SaMST strict`, `StyleID strict`, `AdaIN v32k`, `AdaIN vgg19`, and `AdaIN bad`.

## Advanced Metrics

The advanced anti-artifact stack is documented in `run_511/docs/ADVANCED_METRICS_TOOLCHAIN.md`.

The main interpretation so far: SaMST is structurally strong, so SSIM/Edge/CLIP-content favor it, but the artifact pack starts exposing its micro-grain weakness through NR-IQA, high-frequency patch KID, and FFT-shape diagnostics. Plain KID is kept for completeness but is not sufficient to catch this failure mode.
