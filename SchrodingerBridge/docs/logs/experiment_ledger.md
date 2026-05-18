# Experiment Ledger

Format: Each experiment block logs hypothesis, config delta, results, and verdict.

---

## Experiment 000: armored_breakthrough_8ep_sinkhorn_baseline

**Status**: Planned
**Date**: 2026-05-19
**Config**: `exp/armored_breakthrough_proper/config_sinkhorn_baseline.json`

### Hypothesis
Sinkhorn alone (no SWD increase) should reproduce baseline Style (~0.703) with better LPIPS (~0.44-0.46). This is the control: measure the LPIPS improvement from Sinkhorn alone.

### Config Delta from Baseline (config.json)
```json
{
  "model": {
    "semantic_attn_routing_mode": "sinkhorn"
  },
  "training": {
    "num_epochs": 8,
    "save_interval": 8
  },
  "checkpoint": {
    "save_dir": "./exp/armored_breakthrough_proper/sinkhorn_baseline"
  }
}
```

### Predictions
- clip_style: 0.700-0.708 (slight regression from softmax)
- content_lpips: 0.44-0.46 (improvement from softmax)

### Results
(TBD)

### Verdict
(TBD)

---

## Experiment 001: armored_breakthrough_8ep_sinkhorn_sw60

**Status**: Planned  
**Date**: 2026-05-19  
**Config**: `exp/armored_breakthrough_proper/config_sw60.json`

### Hypothesis (see docs/maths/03_predictive_models.md)
Sinkhorn routing protects LPIPS while terminal SWD weight at 0.60 pushes Style past 0.72.

### Config Delta from Baseline (config.json)
```json
{
  "model": {
    "semantic_attn_routing_mode": "sinkhorn"
  },
  "bridge": {
    "terminal_swd_weight": 0.60,
    "w_kinetic": 1.0
  },
  "training": {
    "num_epochs": 8,
    "save_interval": 8
  },
  "checkpoint": {
    "save_dir": "./exp/armored_breakthrough_proper/sw60"
  }
}
```

### Predictions
- clip_style: 0.718-0.725
- content_lpips: 0.45-0.49

### Results
(TBD)

### Verdict
(TBD)

---

## Experiment 002: armored_breakthrough_8ep_sinkhorn_sw80

**Status**: Planned  
**Date**: 2026-05-19  
**Config**: `exp/armored_breakthrough_proper/config_sw80.json`

### Hypothesis
Pushing SWD to 0.80 with kinetic at 0.5 should break the 0.72 barrier at the cost of slightly higher LPIPS.

### Config Delta from Baseline
```json
{
  "model": {
    "semantic_attn_routing_mode": "sinkhorn"
  },
  "bridge": {
    "terminal_swd_weight": 0.80,
    "w_kinetic": 0.5
  },
  "training": {
    "num_epochs": 8,
    "save_interval": 8
  },
  "checkpoint": {
    "save_dir": "./exp/armored_breakthrough_proper/sw80"
  }
}
```

### Predictions
- clip_style: 0.725-0.735
- content_lpips: 0.48-0.54

### Results
(TBD)

### Verdict
(TBD)

---

## Experiment 003: armored_breakthrough_8ep_sinkhorn_sw30

**Status**: Planned  
**Date**: 2026-05-19  
**Config**: `exp/armored_breakthrough_proper/config_sw30.json`

### Hypothesis
Moderate SWD increase with Sinkhorn to find the "sweet spot" — the minimal SWD weight that pushes past 0.72.

### Config Delta
```json
{
  "model": {
    "semantic_attn_routing_mode": "sinkhorn"
  },
  "bridge": {
    "terminal_swd_weight": 0.30,
    "w_kinetic": 1.0
  },
  "training": {
    "num_epochs": 8,
    "save_interval": 8
  },
  "checkpoint": {
    "save_dir": "./exp/armored_breakthrough_proper/sw30"
  }
}
```

### Predictions
- clip_style: 0.710-0.718
- content_lpips: 0.44-0.47

### Results
(TBD)

### Verdict
(TBD)
