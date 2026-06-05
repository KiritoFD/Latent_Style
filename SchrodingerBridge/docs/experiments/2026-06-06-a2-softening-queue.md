# A2 Softening Queue

Date: 2026-06-06

Purpose:

- move `A2` from a plan-only mention into an auditable queued packet
- keep the next mainline-improvement lanes explicit while the remote `3060`
  is still occupied by the bounded latent `SaMam` side quest
- preserve the narrow reviewer-safe claim boundary for the sweep

## Queue order

The current intended order after the remote GPU lane frees is:

1. `A1` executor-side promotion
2. `A2a` conservative softening
3. `A2b` balanced softening
4. `A2c` more aggressive softening

This order is deliberate:

- `A1` is still the strongest current improvement hint
- `A2a` stays closest to the current `H` surface
- `A2b` increases routing softness without also pushing endpoint pressure down
  too far
- `A2c` is the largest allowed move before the sweep should stop and pick a
  keep/drop decision

## Config packet

`A2a`

- config:
  - [mainline_h_softterm18_sem010_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/mainline_h_softterm18_sem010_seed42_b44.json)
- remote log:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_mainline_h_softterm18_sem010_seed42_b44/remote_train.log`
- read:
  - `terminal_swd_weight = 18.0`
  - `semantic_attn_temperature = 0.10`
  - `w_kinetic = 1.0`

`A2b`

- config:
  - [mainline_h_softterm18_sem012_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/mainline_h_softterm18_sem012_seed42_b44.json)
- remote log:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_mainline_h_softterm18_sem012_seed42_b44/remote_train.log`
- read:
  - `terminal_swd_weight = 18.0`
  - `semantic_attn_temperature = 0.12`
  - `w_kinetic = 1.0`

`A2c`

- config:
  - [mainline_h_softterm16_sem012_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/mainline_h_softterm16_sem012_seed42_b44.json)
- remote log:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_mainline_h_softterm16_sem012_seed42_b44/remote_train.log`
- read:
  - `terminal_swd_weight = 16.0`
  - `semantic_attn_temperature = 0.12`
  - `w_kinetic = 1.0`

## Preflight status

All three configs already pass local launcher dry-run through:

- [launch_remote_aaai2027_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_aaai2027_packet.py)

Dry-run outputs already confirm:

- stable task names
- stable remote log paths
- stable remote launcher paths
- the current hard prelaunch gate:
  - `max_prelaunch_memory_mib = 1500`

## Keep / stop rule

Promote at most one `A2` variant.

Keep if:

1. `content_lpips` improves without losing almost all adjusted style movement
2. or the outputs are visibly cleaner while `delta_idt` stays materially
   positive

Stop the sweep once either happens:

1. one `A2` point is clearly promotable
2. all three points are neutral or worse than the current `H` references

## Current blocker

The queue is still waiting on the single-lane remote `3060` contract:

- the active latent `SaMam` run has not reached its first retained
  `5000-step` checkpoint yet
- automatic handoff to `A1` remains the first transition
- `A2` must not overlap either the latent side quest or `A1`

## Closure

All three queued `A2` arms are now fully landed with `full_eval` summaries for
`epoch_0001 .. epoch_0003`.

Transfer-side readout:

- `A2a softterm18 sem010`
  - best observed transfer packet:
    - `epoch_0001`
    - `clip_style = 0.6656`
    - `content_lpips = 0.3363`
- `A2b softterm18 sem012`
  - best observed transfer packet:
    - `epoch_0001`
    - `clip_style = 0.6651`
    - `content_lpips = 0.3385`
- `A2c softterm16 sem012`
  - best observed transfer packet:
    - `epoch_0001`
    - `clip_style = 0.6645`
    - `content_lpips = 0.3281`

Comparison target:

- paper-facing `H e1`
  - `clip_style = 0.6653`
  - `content_lpips = 0.3281`

Interpretation:

- none of the three softening arms creates a clear new frontier over `H e1`
- `A2c` matches `H e1` on `content_lpips` but does not improve style movement
- `A2a` and `A2b` are slightly worse on content preservation without gaining a
  stronger style packet

Current closure:

- the `A2` sweep is a **neutral-to-negative closure**
- mild endpoint / routing softening does not presently justify replacing the
  reviewed `H` packet
