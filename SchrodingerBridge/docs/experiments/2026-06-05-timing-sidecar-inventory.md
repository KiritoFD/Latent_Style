# Timing Sidecar Inventory

Date: 2026-06-05

Scope: timing-sidecar audit only. No paper-body edits. No SDXL line. This note
reviews the currently indexed timing materials for:

- `Distinct5-512`
- historical `legacy256_overfit50` timing carried in old strict-750 materials
- user-mentioned `DisDict 512` only insofar as repo-local artifacts exist

## Verdict

1. `Distinct5-512` is the only mature timing sidecar surface in the repo.
   The authoritative audit layer is:
   - `SchrodingerBridge/docs/experiments/2026-06-04-distinct5_same_cost_inventory.csv`
   - `SchrodingerBridge/docs/experiments/distinct5_512_20260602/baseline_packet_status_20260604/packet_status.json`
2. There is no first-class `Distinct5-256` surface in the repo.
   Current dataset roots are:
   - `Dataset/distinct5_512`
   - `Dataset/legacy256_overfit50`
   - `Dataset/wikiart512_5style`
   If a timing row says `Distinct5-256`, it is almost certainly a relabel of
   `legacy256_overfit50` and should not stay as-is.
3. There is no repo-local `DisDict` / `DisDict 512` timing material.
   Any `DisDict 512` row should be deleted unless external artifacts are added.

## Usable Now

### A. Main-table-safe Distinct5-512 train-wall rows

Use these as recorded operating-point cost, with evaluation excluded:

- `LBM F e1`: train `1.2161 min`
- `LBM H e1`: train `1.2207 min`
- `LBM H e2`: train `2.2656 min`
- `LBM K e1`: train `1.2077 min`
- `SaMST e5`: train `115.9750 min`
- `SaMST e15`: train `347.2567 min`
- `SaMAM 2250`: train `458.5503 min`

Status caveats:

- `SaMAM 3000`: audit-only closed, not active-manuscript safe
- `K-longer e4`: partial-eval only, not closed

### B. Distinct5-512 inference / eval fields that are actually usable

- `SaMST e5` is the only closed Distinct5 baseline row with packet-bound pure
  generation timing:
  - eval `0.3703 min`
  - inference `430.761 ms/img`
- `SaMST e15` has train wall and eval wall, but no same-run-root inference log
  bound to the current packet root.
- `LBM` Distinct5 rows and `SaMAM` Distinct5 rows do not form a same-grade
  shared pure-generation timing table today.

### C. Historical / legacy256 timing rows that are truly closed

Only `LBM` is fully evidence-closed as a historical timing row:

- train `309.902 s`
- infer-750 `85.414 s`
- `0.113885 s/img`

This is the exact logic already reflected in the current timing snippet.

## Downgrade To Appendix Or Audit Only

- historical `SaMST` train time:
  - only extrapolated from a `67.687 s` 1-epoch probe
- historical `S2WAT` train time:
  - only estimated as about `5.3 s * 2000 = 10600 s`
  - strict infer-750 is not separately measured
- historical `StyleID` infer-750:
  - `603.316 s` recorded run is not a fair full-750 packet
  - `3016.335 s` is an estimate
- `Distinct5-512 SaMAM 3000`:
  - closed as audit history, excluded from active manuscript path
- `Distinct5-512 SaMST e15` inference `168 ms/img`:
  - external note exists, but it is not packet-bound to the retained run root

## Delete If A Closed Table Is Needed

- any `DisDict 512` timing row
- any `Distinct5-256` timing row that is really `legacy256_overfit50`
- any historical `S2WAT` / `StyleID` / `SaMST-train` row from a closed timing
  table unless it is explicitly labeled `estimated`, `mixed-protocol`, or
  `unfair`
- any shared `Distinct5` inference column spanning `LBM`, `SaMAM 2250`, and
  `SaMST e15`

## Recommended Table Split

### Main paper

Use one compact table for `Distinct5-512` recorded operating-point cost:

- columns: `Method | Recorded point | Train to point`
- representative rows:
  - `LBM-F e1`
  - `LBM-K e1`
  - `SaMAM 2250`
  - `SaMST e15`

Why this is the safest split:

- these rows are manuscript-facing and already reflected in the current snippet
- train wall is the only field closed enough across all retained rows
- adding a shared inference column would overstate closure

### Appendix

Appendix Table A: `Distinct5-512 timing audit`

- columns:
  - `Method`
  - `Point`
  - `Train min`
  - `Eval min`
  - `Infer ms/img`
  - `Transfer ArtFID`
  - `Status`
  - `Evidence note`
- source from:
  - `2026-06-04-distinct5_same_cost_inventory.csv`
  - `baseline_packet_status_20260604/packet_status.json`

Appendix Table B: `Historical strict-750 / legacy256 timing evidence quality`

- columns:
  - `Method`
  - `Train to point`
  - `Infer-750`
  - `ms/img`
  - `Evidence grade`
  - `Reason`
- evidence grades should be explicit:
  - `actual`
  - `estimated`
  - `mixed-protocol`
  - `unfair packet`

## Closest Existing Paper-Facing Assets

Best existing TeX snippet:

- `SchrodingerBridge/aaai_submission/snippets/timing_tables_leibniz.tex`

Current paper hook points:

- `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
  - `\input{snippets/timing_tables_leibniz}`
  - `\DistinctFiveTimingTableLeibniz`
  - `\HistoricalTimingTableLeibniz`

Closest existing figure-side timing packet:

- `SchrodingerBridge/docs/experiments/2026-06-03-time-to-parity/distinct5_time_to_parity_points.csv`
- `SchrodingerBridge/aaai_submission/scripts_gen_distinct5_time_context.py`

Important limitation of the figure-side packet:

- it is useful for the Distinct5 time-context figure
- it is not the best canonical audit table source because packet-closure status
  and targetwise ArtFID gaps live more clearly in the same-cost inventory and
  packet-status files
