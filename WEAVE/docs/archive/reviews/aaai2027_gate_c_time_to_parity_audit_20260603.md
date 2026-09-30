# AAAI 2027 Gate C Time-to-Parity Audit

Date: 2026-06-03  
Lane: `adversarial_review`  
Scope: audit of `docs/experiments/2026-06-03-time-to-parity/`

## Files inspected

- `G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-03-time-to-parity/README.md`
- `G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-03-time-to-parity/distinct5_time_to_parity_points.csv`
- `G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-03-time-to-parity/figures/distinct5_time_to_clip_style.pdf`
- `G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-03-time-to-parity/figures/distinct5_time_to_lpips.pdf`
- `G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-03-time-to-parity/figures/distinct5_time_to_delta_idt.pdf`

## Short verdict

This packet is **good enough to replace the most vulnerable mixed-anecdote speed wording with a bounded same-scope Distinct5 timing-context artifact**, but it is **not** strong enough to support a true normalized `time-to-parity` or fair training-speed-win claim.

In reviewer terms:

- **yes** for narrow replacement of loose speed rhetoric;
- **no** for strong parity closure.

## What the packet gets right

### 1. The protocol boundary is explicit

The README now does the important reviewer-safe work:

- same dataset family is enforced: `Distinct5-512`
- plotted scope is explicit: `full 5x5 / 750`
- clock meaning is explicit: cumulative training wall time, eval excluded
- abnormal runs are quarantined out of timing claims

This is already much safer than the old state where timing language was floating on top of mixed operating-point anecdotes.

### 2. The CSV is structurally usable

The CSV includes the right fields for review:

- `timing_mode`
- `includes_eval`
- `eval_scope`
- `timing_quality_flag`
- `evidence_path`

This makes the artifact auditable rather than purely presentational.

### 3. The figures are visually coherent and honest about shape

The three figures communicate the intended bounded story clearly:

- `clip_style vs wall time`
- `content_lpips vs wall time`
- `delta_idt vs wall time`

They also visually distinguish:

- `LBM` as operating points
- `SaMAM` as a partial curve
- `SaMST` as a single operating point
- `idt` as a reference line

That distinction is exactly what keeps the figures reviewer-safe.

## Is this sufficient to narrow or replace current paper speed rhetoric?

### Yes, for a narrow replacement

This packet is sufficient to support wording of the following kind:

- on `Distinct5-512`, we report a same-scope timing-context artifact rather than mixed cross-protocol timing anecdotes;
- under this protocol, reviewed `LBM` operating points appear in the low-minute regime, while the indexed `SaMAM` partial curve remains far slower and stays below the `idt` floor over the currently recorded range;
- the currently available `SaMST` evidence is a single Distinct5 operating point rather than a matched curve.

That is a real improvement over the previous manuscript state.

### No, for a full replacement of Gate C with strong parity language

This packet is not sufficient to justify language of the following kind:

- `LBM trains faster than competing methods`
- `LBM reaches parity sooner`
- `LBM has a fair comparative time-to-quality advantage`

Those stronger claims are still ahead of the evidence in this packet.

## Highest remaining vulnerability

The highest remaining reviewer vulnerability is:

- **timing-mode asymmetry across methods**

Concretely:

1. `LBM` is represented by selected `operating_point_record` rows;
2. `SaMAM` is represented by a `full_curve_partial`;
3. `SaMST` is represented by one `operating_point_record` only.

That means this artifact still does **not** instantiate a common thresholded parity test.

The reviewer attack line is straightforward:

> this is a same-scope timing-context figure, but not yet a true normalized time-to-parity comparison, because the compared families do not contribute equivalent trajectory evidence.

This is the line most likely to survive rebuttal unless the paper stays narrow.

## Practical paper-safe use

The safest use of this packet is:

- keep the historical strict-750 table as operating-point bookkeeping;
- use this Distinct5 packet as the primary same-scope timing figure for any remaining efficiency prose;
- explicitly call it a `timing-context` or `time-to-operating-region` artifact, not a closed parity theorem.

## Reviewer recommendation

My recommendation is:

- treat this packet as sufficient to **narrow and partially replace** current paper speed rhetoric;
- do **not** claim Gate C is fully closed if `closed` means fair normalized time-to-parity across methods;
- if a stronger efficiency claim is desired later, the next missing piece is a common threshold rule with matched curve evidence, especially for `SaMST`.

## Bottom line

This artifact successfully removes the worst speed-claim vulnerability from the paper, because it replaces mixed anecdotes with one explicit same-scope timing packet.

It does **not** yet earn a broad comparative speed claim. The remaining weakness is not presentation; it is the still-unequal timing evidence geometry across `LBM`, `SaMAM`, and `SaMST`.
