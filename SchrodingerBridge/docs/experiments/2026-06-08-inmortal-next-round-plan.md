# inmortal Next Round Plan

Date: 2026-06-08

Current closure after the first complete mechanism round on `Distinct5-512`:

- style-best successor point:
  - `XPred + Kmanifold + Pattn + Stokes002`
  - transfer `0.7307 / 0.6183`
- balanced-best successor point:
  - `XPred + Kmanifold + Pattn + late Stokes`
  - transfer `0.7274 / 0.6033`
- low-LPIPS-best successor point:
  - `XPred + Kmanifold + Pattn + AnisoStokes + Queue`
  - `e13`
  - transfer `0.7102 / 0.4603`
  - all-pairs `0.7303 / 0.4559`

What the round established:

1. the current style ceiling is not the bottleneck anymore
   - several successor families exceed the earlier compact-LBM band on style
2. the main unsolved problem is now Pareto shape
   - style-heavy families still sit around `0.60+` LPIPS
   - structure-heavy families can drive LPIPS down sharply, but give back style
3. the strongest evidence split is now:
   - `late Stokes` for balanced frontier
   - `AnisoStokesQueue e13` for low-LPIPS anchor

Interpretation:

- the model family is no longer trapped in the original trivial no-op regime
- instead, it is split between:
  - a style-restoration branch
  - and a structure-preservation branch
- next-round work should therefore target mechanism fusion, not another broad family sweep

## Next-round objective

Keep the `AnisoStokesQueue e13` LPIPS gain as much as possible while recovering style toward the `0.72+` transfer band.

Desired target for the next round:

- transfer `clip_style >= 0.72`
- with transfer `content_lpips` materially below the current balanced frontier
- ideally moving toward `<= 0.50`

## Priority order

1. style-rescue on top of the `e13` low-LPIPS anchor
2. structure-penalty routing refinements that keep the `e13` content benefit but reduce its style choke
3. only after those, any further queue or teacher combinations

## Candidate mechanism directions

### 1. Proximal style rescue over the low-LPIPS anchor

Keep the transport-side `AnisoStokesQueue` regime, but add a small explicit style-restoration proximal path that is:

- high-frequency constrained
- terminal-loss bound
- discouraged from changing the low-frequency base endpoint

Reason:

- current evidence suggests the transport side can already preserve content extremely well
- the missing piece is restoring style after that conservative transport

### 2. Edge-gated structure pressure instead of uniform structure pressure

Refine the current `anisotropic_plus_stokes` pressure so the strongest structure penalty is only applied where content edges are actually strong.

Reason:

- the current low-LPIPS anchor likely over-regularizes style away in flat regions
- edge-gating is the cleanest mechanism-level way to keep content benefit without paying the same global style tax

### 3. Base-vs-final endpoint decision audit

For the next promoted family, explicitly inspect whether the LPIPS gain is already in `z_base`, or only appears in `z_final`.

Reason:

- if the gain is mostly in `z_base`, the next round should keep working transport-side
- if it is mostly in `z_final`, the next round should become a proximal-restoration problem

## What not to do next

- do not reopen broad single-mechanism sweeps
- do not spend more GPU time on plain queue-only or plain teacher-only followups
- do not treat further style-only gains above `0.73` as the main target unless LPIPS also improves

## First concrete next experiment

Start from the current low-LPIPS anchor family and add a constrained style-rescue proximal branch.

This is the highest-information next move because it directly tests whether the remaining gap is:

- style lost in transport, or
- style recoverable after transport
