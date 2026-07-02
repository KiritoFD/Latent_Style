# Quick Start

## Training

```bash
cd SchrodingerBridge
PYTHONPATH=src python src/run.py --config config.json
```

For experiment configs with `_base` inheritance:
```bash
PYTHONPATH=src python src/run.py --config configs/exp_sanity.json
```

## Evaluation

```bash
python run_evaluation.py <checkpoint_dir>
```

Runs inference on all `.pt` files, computes CLIP style/content + LPIPS, saves to `<checkpoint_dir>/full_eval/batch_summary.csv`.

## Config Inheritance

Configs support `_base` for deep-merge inheritance:

```json
{
  "_base": "../config.json",
  "bridge": { "terminal_swd_weight": 2.0 },
  "training": { "num_epochs": 8, "save_interval": 1 },
  "checkpoint": { "save_dir": "./exp/my_experiment" }
}
```

Only specify keys that differ from the base config.

## Key Config Parameters

### Model (`model.*`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `style_strength_default` | 1.0 | Style intensity at inference [0, 1] |
| `skip_routing_mode` | "none" | Skip connection mode |
| `output_moment_match` | false | AdaIN output alignment |
| `batch_size` | 64 | Training batch size |

### Bridge (`bridge.*`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_kinetic` | 1.5 | Kinetic energy weight |
| `w_curvature` | 0.0 | Curvature regularization weight |
| `terminal_swd_weight` | 0.15 | Terminal SWD loss weight |
| `terminal_num_steps` | 4 | Integration steps for terminal SWD |

### Training (`training.*`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 80 | Training epochs |
| `save_interval` | 10 | Checkpoint save frequency |
| `learning_rate` | 0.0002 | Adam learning rate |
| `use_amp` | true | Mixed precision |
