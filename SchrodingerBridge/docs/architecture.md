# Architecture

## Core Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/losses.py` | 436 | OTFlowMatchingObjective — flow matching + kinetic + terminal SWD |
| `src/ot_cost.py` | 76 | SWDTransportCost — full-band random projection SWD |
| `src/trainer.py` | 614 | SBTrainer — training loop, CSV logging, checkpoints |
| `src/model.py` | ~350 | TimeConditionedLANCETBridge — bridge wrapper over backbone |
| `src/lancet_backbone.py` | ~1580 | LatentAdaCUT — U-Net backbone with style modulation |
| `src/run.py` | ~170 | Entry point |
| `src/config_loader.py` | ~44 | Config loading with `_base` inheritance |
| `src/inference_config.py` | ~98 | Inference/eval config resolution |
| `src/inference_config.json` | ~20 | Default inference parameters |

## Loss Components

The training loss is:
```
L = L_flow + w_kinetic * L_kinetic + w_curvature * L_curvature + terminal_swd_weight * L_terminal_swd
```

### L_flow (Flow Matching)
MSE between predicted and target velocity. The velocity field is trained to match the optimal transport path between content and style latents.

### L_kinetic (Kinetic Energy)
Isotropic kinetic regularizer: `mean(velocity^2)`. Prevents the velocity field from exploding. Modes: `endpoint`, `path`, `time_gated`.

### L_curvature (Curvature Regularization)
Penalizes velocity difference at t vs t+dt to encourage straight flow trajectories. Default weight: 0.0 (disabled).

### L_terminal_swd (Terminal SWD)
SWD between the integrated endpoint and OT-matched target. Computed via `model.integrate()` with configurable steps. The main style transfer driver.

## Data Flow

```
config.json
  → config_loader.load_config()
    → trainer.SBTrainer
      → model.build_model_from_config() → TimeConditionedLANCETBridge
        → lancet_backbone.LatentAdaCUT (parent class)
      → losses.OTFlowMatchingObjective
        → ot_cost.SWDTransportCost (transport_cost)
```

## Training Loop

1. Load content batch + style batch
2. OT matching: pair content/style via Sinkhorn or Hungarian
3. Sample time t, construct bridge state x_t
4. Forward pass: predict velocity
5. Compute loss: flow + kinetic + curvature + terminal SWD
6. Backward + optimizer step
7. Log to CSV, save checkpoint at intervals
