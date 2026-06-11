# attn_gated_spade Decision

- Decision date:
  - `2026-06-10`
- Current status:
  - `recalibration_needed`
- Current formal standing:
  - the old `batch=19` retrial is no longer the authority read
  - the family has now been re-opened successfully under a fresh bounded continuation path

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
