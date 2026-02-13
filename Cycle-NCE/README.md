# Cycle-NCE (Latent AdaCUT)

Lightweight latent-space style transfer training codebase.

This repository is organized around:
- `src/run.py`: training entrypoint
- `src/trainer.py`: train loop, checkpointing, full evaluation hooks
- `src/model.py`: Latent AdaCUT model
- `src/losses.py`: style/content objective terms
- `src/utils/run_evaluation.py`: offline full evaluation
- `src/utils/inference.py`: inference helper

## Repository Layout

```text
.
|- src/
|  |- run.py
|  |- trainer.py
|  |- model.py
|  |- losses.py
|  |- config.json
|  |- overfit50.json
|  `- utils/
|- scripts/
|  `- style_ablation.sh
|- docs/
|  `- STYLE_ABLATION_PLAN.md
`- run_full_eval_epochs.sh
```

## Quick Start

Run from `src/`:

```bash
python run.py --config config.json
```

Optional resume:

```bash
python run.py --config config.json --resume ../your_exp/epoch_0012.pt
```

## Full Evaluation

Single checkpoint:

```bash
python utils/run_evaluation.py \
  --checkpoint ../your_exp/epoch_0012.pt \
  --output ../your_exp/full_eval/epoch_0012
```

Batch evaluate selected epochs:

```bash
bash run_full_eval_epochs.sh "50 100 150 200"
```

## Style Ablation

Use:

```bash
bash scripts/style_ablation.sh
```

`scripts/style_ablation.sh` is intentionally fixed to run full ablation (`mode=all`).
For any custom selection/tuning, use the Python entry.

Or run the Python entry directly:

```bash
python scripts/style_ablation.py --mode all
```

This script:
- generates ablation configs from a base config
- covers style injection methods in `model` and style/content loss weights in `loss`
- runs training for each variant
- collects final and best `clip_style` metrics into summary CSV/Markdown

See full plan: `docs/STYLE_ABLATION_PLAN.md`.
Code map: `docs/CODEBASE_GUIDE.md`.

## Experiment Outputs

Each run directory (from `checkpoint.save_dir`) typically contains:
- checkpoints: `epoch_XXXX.pt`
- training logs: `logs/training_*.csv`
- full eval outputs: `full_eval/epoch_XXXX/summary.json`
- optional interval inference outputs: `inference/epoch_XXXX/...`

## Notes

- Prefer WSL + CUDA environment for reproducible speed.
- Keep config files explicit; do not rely on hidden defaults for ablation runs.
- Track metrics from `summary.json`:
  - `analysis.style_transfer_ability.clip_style`
  - `analysis.photo_to_art_performance.clip_style`
