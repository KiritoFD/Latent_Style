# DualPath SpatialTexture Early Read

Date: 2026-06-09

This note records the first-health and earliest visible train-side read for:

- `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_seed42_b8a2`

It exists so the project does not later rely on vague memory such as:

- `it started fine`
- `VRAM looked okay`
- `early loss seemed normal`

## Launch state

Previous line:

- `dualpath_texture`

Observed state before handoff:

- remote `src/run.py` for `dualpath_texture` had exited
- GPU was still non-idle because of residual remote load / watcher activity
- but the formal train lane itself was no longer active

Current new live line:

- `dualpath_spatialtexture`

## First-health

First-health passed.

Observed launch band:

- prelaunch remote GPU:
  - about `4525 MiB`
- health read after launch:
  - about `8014 MiB`
- later checked live band during the same early stage:
  - about `5799 MiB`
- latest checked live band after longer runtime:
  - about `5112 MiB`

This is comfortably below the formal hard cap:

- `< 11.0 GiB`

## Earliest visible train-side read

Visible startup contract succeeded:

- latent dataset loaded correctly
- pairing cache loaded correctly
- tokenizer initialized
- partial resume succeeded
- freeze mode entered as expected

Visible early train lines:

- first shown batch:
  - `flow approx 0.8803`
  - `kin approx 0.0880`
  - `loss approx 8.0544`
  - `tswd approx 6.9688`
- around step `20`:
  - `flow approx 0.8331`
  - `kin approx 0.0914`
  - `loss approx 8.3200`
  - `tswd approx 7.2500`
- around step `39`:
  - `flow approx 0.8196`
  - `kin approx 0.0922`
  - `loss approx 8.1864`
  - `tswd approx 7.1562`

## Current fresh-eval curve

Current available fresh-eval rows:

- `epoch_0001`
  - transfer `0.69294 / 0.40146`
  - all-pairs `0.71828 / 0.39782`
- `epoch_0002`
  - transfer `0.69222 / 0.41932`
  - all-pairs `0.71638 / 0.41556`
- `epoch_0003`
  - transfer `0.69196 / 0.42805`
  - all-pairs `0.71553 / 0.42417`
- `epoch_0004`
  - transfer `0.69242 / 0.43269`
  - all-pairs `0.71571 / 0.42872`
- `epoch_0005`
  - transfer `0.69163 / 0.43316`
  - all-pairs `0.71510 / 0.42904`
- `epoch_0006`
  - transfer `0.69192 / 0.43618`
  - all-pairs `0.71529 / 0.43199`
- `epoch_0007`
  - transfer `0.69245 / 0.43765`
  - all-pairs `0.71553 / 0.43339`
- `epoch_0008`
  - transfer `0.69222 / 0.43667`
  - all-pairs `0.71537 / 0.43233`
- `epoch_0009`
  - transfer `0.69233 / 0.43768`
  - all-pairs `0.71545 / 0.43333`
- `epoch_0010`
  - transfer `0.69250 / 0.43895`
  - all-pairs `0.71548 / 0.43460`
- `epoch_0012`
  - transfer `0.69229 / 0.43960`
  - all-pairs `0.71533 / 0.43523`

## Immediate interpretation

This line is no longer just machine-safe; it now has a real early curve.

Current curve read is still conservative:

- style stays in a very narrow band around `0.6916 to 0.6929`
- `LPIPS` steadily worsens from about `0.401` toward `0.440`
- all-pairs `CLIP-style` also stays in a narrow band around `0.715 to 0.718`
- the additional middle points still do not show a hidden late style reopening

So the current early evidence says:

- `dualpath_spatialtexture` has not yet shown a meaningful style reopening over the earlier `dualpath_texture` family
- and the visible trajectory still looks like the same low-style / low-LPIPS basin, just traced across more checkpoints

## Bestfew IntroStyle read

The first remote `IntroStyle` bestfew probe has now landed:

- [2026-06-09-dualpath-spatial-bestfew-introstyle-read.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-09-dualpath-spatial-bestfew-introstyle-read.md)

Current bestfew `IntroStyle` read:

- `epoch_0001`
  - target score `0.11198`
  - margin `-0.05031`
- `epoch_0012`
  - target score `0.10755`
  - margin `-0.04673`

This reinforces the current curve read:

- the branch is not yet showing a target-specific style breakthrough

## Current role in the program

Use this line as the active remote mechanism probe while local evidence continues accumulating on:

- `QEdgePattn e01`
- `DualPath e01`
- `Seedream`

The intended next decision is:

- not `is the line alive`
- but `does spatialtexture reopen target-specific style without giving up the corrected geometry/cleanliness compromise`
