# SchrodingerBridge

This directory is a root-level sibling of `Cycle-NCE`.

It keeps the LANCET / AdaCUT backbone, but changes the training objective and
the model role:

- the backbone is treated as a time-conditioned vector field
  `v_theta(z_t, t, style_id)`
- training uses SWD-guided OT coupling plus stochastic bridge matching
- inference integrates an ODE trajectory from `t=0` to `t=1`

## Mathematical Position

This implementation is a practical SWD + Schrodinger-Bridge style model built
on top of the LANCET latent backbone.

- It is mathematically cleaner than the old residual-plus-heuristic-loss setup.
- It uses SWD as the coupling geometry, Sinkhorn as the entropic transport
  solver, and a stochastic Brownian-bridge interpolation for training states.
- It also keeps an explicit terminal SWD regularizer, so SWD participates both
  in coupling and in endpoint supervision.
- `style_strength` is interpreted as the integration horizon, not as an
  embedding hack.
- `identity_endpoint` is disabled by default so same-style batches also follow
  the OT coupling instead of a hand-written identity shortcut.

## Layout

- `src/lancet_backbone.py`: copied LANCET backbone from `Cycle-NCE`
- `src/model.py`: time-conditioned vector field wrapper on top of LANCET
- `src/ot_cost.py`: SWD-based OT cost oracle with migrated projection/CDF and
  micro/macro feature handling
- `src/losses.py`: Sinkhorn coupling, stochastic SB bridge loss, terminal SWD
- `src/trainer.py`: training loop, logging, checkpoints
- `src/dataset.py`: latent dataset loader reused from the existing project
- `src/utils/`: evaluation and inference utilities kept compatible with the old
  tooling
- `run.py`: root wrapper that dispatches into the `src/` package

## SWD Oracle

`ot_cost.py` reuses the old SWD optimization ideas, but in a new role:

- SWD defines the geometry of the OT coupling
- SWD also appears as a terminal bridge regularizer
- pairwise cost is computed in float32 for stable Sinkhorn / Hungarian matching
- projection banks are cached
- CDF evaluation uses chunking
- micro and macro branches are kept separate
- optional high-frequency features are preserved

## Train

```powershell
cd F:\GitHub\Latent_Style\SchrodingerBridge
python run.py --config config.json
```

## Evaluate

```powershell
cd F:\GitHub\Latent_Style\SchrodingerBridge
python run_evaluation.py --checkpoint .\artifacts\epoch_0020.pt --output_dir .\artifacts\full_eval\epoch_0020
```
