# Stylesig First Fast-Eval Read

Date: 2026-06-10

This note records the first actual quality read for the active `stylesig` line.

## Status

The active run is no longer train-only.

Remote `full_eval/` outputs have now landed at least through:

- `epoch_0001`
- `epoch_0002`
- `epoch_0003`
- `epoch_0004`
- `epoch_0005`

The dedicated `fast-eval watcher` is also live, but these first landed evals are
already enough to form an initial mechanism read.

## First three points

### `epoch_0001`

- transfer `clip_style = 0.7046372694`
- transfer `content_lpips = 0.4513353770`
- full `clip_style = 0.7258133266`
- full `content_lpips = 0.4469408680`
- eval wall `97.98s`

### `epoch_0002`

- transfer `clip_style = 0.7057233430`
- transfer `content_lpips = 0.4690293399`
- full `clip_style = 0.7247472544`
- full `content_lpips = 0.4648580081`
- eval wall `86.87s`

### `epoch_0003`

- transfer `clip_style = 0.7070692201`
- transfer `content_lpips = 0.4780182931`
- full `clip_style = 0.7247771766`
- full `content_lpips = 0.4737420767`
- eval wall `87.61s`

## Initial read

The current read is:

- style is rising monotonically across the first five points
- but `LPIPS` is also degrading monotonically across the same five points

So the first visible `stylesig` trajectory looks more like:

- `style-up / structure-down`

than:

- `new target-specific ceiling unlock`

This does **not** fully close the family yet.

But it does mean the first evidence is not a clean rescue.

By `epoch_0005`, the branch has reached:

- transfer `clip_style = 0.7094051092`
- transfer `content_lpips = 0.4915431555`

So the current path is improving style in the same direction that it is paying
more structure cost.

## Implication

For now, `stylesig` should be treated as:

- a real mechanism branch with live eval evidence
- but still an unproven one

It has crossed from:

- `train-only candidate`

to:

- `first-eval candidate`

But it has not yet crossed into:

- `promotable quality branch`
