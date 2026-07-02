# attn_gated_spade Decision

- Decision date:
  - `2026-06-11`
- Current status:
  - `reviewing`
- Current formal standing:
  - the old `batch=19` retrial is no longer the authority read
  - the canonical authority surface is now the segmented `batch=20` continuation through `epoch_0030`

## Recalibration Read

- failed fresh attempts:
  - `batch=19`
    - launch-time live read could drift to `under_band`
    - therefore not safe enough as the canonical reopening batch
  - `batch=20` with the old from-parent fresh restart:
    - the 30-second health window fired before the model finished fully loading
    - this was a launch-shape problem, not a clean formal verdict
- corrective decision:
  - keep `batch=20`
  - but reopen from the existing `epoch_0022` checkpoint through segmented continuation
  - this preserves the existing eval authority surface and avoids overwriting old epoch names

## Segmented Continuation Read

- bounded continuation from `epoch_0022` through `epoch_0026` is now settled
- formal health gate:
  - passed in-band at about `9580 MiB / 12288 MiB`
- new settled points:
  - `epoch_0023`
    - transfer `0.6882 / 0.4289`
    - full `0.7135 / 0.4236`
  - `epoch_0024`
    - transfer `0.6896 / 0.4287`
    - full `0.7150 / 0.4232`
  - `epoch_0025`
    - transfer `0.6884 / 0.4276`
    - full `0.7139 / 0.4225`
  - `epoch_0026`
    - transfer `0.6887 / 0.4265`
    - full `0.7142 / 0.4215`
- current family frontier:
  - best transfer `CLIP-S` still `epoch_0001`
  - best transfer `LPIPS` still `epoch_0022`
  - best all-pairs `CLIP-S` still `epoch_0011`
  - new Pareto extension:
    - `epoch_0026`
    - improved all-pairs LPIPS while staying on the frontier set
- decision:
  - `batch=20` is the current formal-safe `attn_gated_spade` batch
  - this family is reopened as the active logical non-DINO lane
  - continue bounded segmented continuation from `epoch_0026`

## Convergence Closure Read

- bounded continuation from `epoch_0026` through `epoch_0030` is now settled
- latest settled points:
  - `epoch_0027`
    - transfer `0.6890 / 0.4350`
    - full `0.7138 / 0.4292`
  - `epoch_0028`
    - transfer `0.6906 / 0.4299`
    - full `0.7158 / 0.4241`
  - `epoch_0029`
    - transfer `0.6878 / 0.4346`
    - full `0.7129 / 0.4292`
  - `epoch_0030`
    - transfer `0.6889 / 0.4282`
    - full `0.7141 / 0.4230`
- convergence rule read:
  - `last_pareto_epoch = epoch_0026`
  - `since_last_pareto = 4`
  - `best_in_newest_2 = false`
  - `tail_flat = true`
  - `patience = 4`
  - `converged = true`
- final family read:
  - best transfer `CLIP-S` remains `epoch_0001`
  - best transfer `LPIPS` remains `epoch_0022`
  - best all-pairs `CLIP-S` remains `epoch_0011`
  - best all-pairs LPIPS frontier extension remains `epoch_0026`
- decision:
  - close `attn_gated_spade` for round-1 training
  - keep the family in `reviewing` for stage-close deep evaluation
  - do not allocate another formal remote train segment to this family unless deep review contradicts the settled fast curve
