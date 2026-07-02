# Distinct5-512 LBM Same-Cost Calibration

Date: 2026-06-05

Scope: calibrate which `LBM` point should be described as the retained
`Distinct5-512` frontier, and which point should be described as the page-1
matched-budget packet.

## Verdict

The currently used page-1 `LBM` same-cost row is **not** the best retained LBM
frontier point.

- The page-1 same-cost point is:
  - `step_000350`
  - train wall `111.80 s = 1.8633 min`
  - transfer `CLIP-S = 0.6629465200`
  - transfer `LPIPS = 0.3376801931`
  - transfer `1-LPIPS = 0.6623198069`
  - transfer `delta_idt = +0.0230256947`
  - evidence:
    - [distinct5_same_cost_20260605.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/timing/distinct5_same_cost_20260605.csv)
    - [step_000350 summary](/G:/GitHub/Latent_Style/SchrodingerBridge/exp/local_wsl_distinct5_512_ema_k_b16_step2min_v350/full_eval/step_000350/summary.json)
- The strongest retained `LBM` transfer-style point is still:
  - `K e1`
  - train wall `1.2077 min`
  - transfer `CLIP-S = 0.6711669415`
  - transfer `LPIPS = 0.3722808782`
- The strongest retained `LBM` transfer content-preserving point is still:
  - `F e1`
  - train wall `1.2161 min`
  - transfer `CLIP-S = 0.6643604031`
  - transfer `LPIPS = 0.3245282069`
- The cleanest balanced retained point remains:
  - `H e1`
  - train wall `1.2207 min`
  - transfer `CLIP-S = 0.6652551527`
  - transfer `LPIPS = 0.3281051474`

## Why The Distinction Matters

`step_000350` was selected because it closes the **matched roughly two-minute**
packet against the re-run `SaMAM step 16` and `SaMST step 40/style` packets.
It should therefore be described as the **page-1 same-cost packet**, not as the
best retained LBM operating point.

The retained `LBM` frontier already appears earlier, near `1.21-1.22 min`.

## Direct Pareto Read

Using the retained transfer-only `Distinct5-512` table
[clip_style_vs_1lpips_full_transfer_points.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/distinct5_512_20260602/tables/clip_style_vs_1lpips_full_transfer_points.csv),
the non-dominated retained `LBM` points are:

- `K e1` at `1.2077 min`
- `F e1` at `1.2161 min`
- `L e1` at `1.2172 min`
- `H e1` at `1.2207 min`
- `M e1` at `1.2214 min`
- `J e1` at `1.2229 min`
- `H e2` at `2.2656 min`

Relative to the page-1 same-cost `step_000350` row, the retained `F e1` and
`H e1` points both dominate it on the plotted page-1 axes:

- `F e1`: higher transfer `CLIP-S` and higher `1-LPIPS`
- `H e1`: higher transfer `CLIP-S` and higher `1-LPIPS`

This is acceptable only if the manuscript explicitly says:

- page 1 uses the matched-budget packet
- the stronger retained `LBM` frontier already appears near `1.2 min`
- the same-cost packet is intentionally conservative because it is aligned to
  the re-run baseline budget rather than chosen as the single best `LBM` point

## Paper-Safe Reading

Use the following separation consistently:

- `LBM frontier`: retained reviewed operating points around `1.2 min`
- `LBM same-cost packet`: the explicit `step_000350` row at `1.8633 min`

Do not collapse those two claims into one sentence.
