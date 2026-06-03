# AAAI 2027 Experiment Audit

Date: 2026-06-03  
Round: `R20260603N`  
Lane: `experiment_audit`

Checkpoint label:

- `current_paper_after_agent_cleanup_before_next_path_stability_integration`

- `overall_status`: `weak_reject`
- `claim_safety_band`: `narrow_only`
- `evidence_closure_band`: `partial`
- `blocking_issue`: `The largest remaining provenance gap is still the Distinct5 same-family H-packet for kinetic/path stability. The paper's bounded path-energy story currently rests on historical destructive ablations plus protocol-prepared Distinct5 follow-up, but the required H-family base + k025 + k000 + probe artifact chain has not yet landed cleanly in the reviewed ledger. The interrupted base run is explicitly archived and the clean rerun is logged as live/running context, but neither state is admissible as closed mechanism evidence.`
- `next_action_1`: `Finish one clean Distinct5 H-family base run under the current logging contract and retain the full evidence chain: resolved config, remote_train.log, epoch_0001..0003.pt, and full_eval/epoch_0001..0003/summary.json.`
- `next_action_2`: `Immediately run the matched H-family k025 and k000 arms, then execute tools/probe_path_stability.py and promote the kinetic/path claim only if the packet satisfies the protocol accept rule on retained probe outputs.`
- `support_score`: `1`
- `fairness_score`: `1`
- `artifact_path_score`: `2`
- `closure_value_score`: `1`

## Directly supported

1. Historical strict-750 frontier wording is directly supported if kept narrow. The manuscript's claim that LBM is a cleaner content-preserving operating point than the reproduced SaMST / S2WAT / StyleID / AdaIN-family baselines is supported by the retained historical table surface and the cited strict-750 artifact roots.
2. The Distinct5 no-op-aware evaluation diagnosis is directly supported. The current paper claim that raw CLIP-S can overstate success on separated art-to-art transfer is backed by explicit `idt` rows, full-scope and transfer-only deltas, and the retained Distinct5 frontier artifact set.
3. The current Distinct5 frontier claim is directly supported at the paper-safe level already used in the tex: among the currently reproduced positive-`Delta_idt` points, LBM-F and LBM-K define the strongest measured `CLIP-S / LPIPS` trade-off, while SaMAM remains below the `idt` floor and SaMST reaches higher raw style only in a severe LPIPS / ArtFID regime.
4. Endpoint-only pointwise supervision is directly supported as a negative closure. The repaired endpoint-metric packet in the master log supports the paper's current narrow statement that the evidence favors the OT + `W1`-style mainline over pure endpoint-only pointwise supervision.
5. SA-SWD is directly supported only in the current narrow form already used by the paper: retained as the mainline terminal design, but not as a positively closed semantic-axis-superiority claim.
6. Operating-point cost accounting is directly supported only as bounded timing context. The manuscript now presents wall-clock records as operating-point measurements rather than normalized time-to-parity proof, which matches the documented evidence surface.

## Indirectly supported

1. The tokenizer-as-executable-representation story is indirectly supported beyond the tested local boundary. The current ablation table plus the landed `L`-family successor and localization packets support the direction of the argument, but they do not restore same-family `H` continuity or prove the renderer-side bottleneck in a broader mechanism sense.
2. The kinetic/path-energy stabilizer story is indirectly supported, not directly closed, on the current Distinct5 surface. Historical destructive ablations and theorem-grounded wording point in the right direction, but the protocol itself states that Distinct5 promotion requires a matched H-family probe packet that is not yet landed.
3. The fairness of timing comparisons is only indirectly supported. The paper is careful to call the reported numbers operating-point records, but the comparison surface still mixes different training trajectories and should not be read as a normalized parity claim.

## Unsupported

1. Any wording that implies the Distinct5 H-family kinetic/path-stability mechanism is already empirically closed.
2. Any reading that upgrades SA-SWD from a retained design choice to a cleanly demonstrated semantic-axis win on the current Distinct5 packet.
3. Any broad tokenizer-generalization claim that goes beyond the tested Distinct5 variants and the currently landed `L`-family-local packets.
4. Any efficiency rhetoric stronger than bounded operating-point context, especially anything that sounds like normalized time-to-parity or universally fair training-cost superiority.

## Blocked by provenance

1. The highest-value blocked claim is the paper's same-family Distinct5 mechanism story for kinetic/path regularization. The protocol, launch-status note, and master log all agree that the necessary packet is prepared and partially exercised, but not yet landed as reviewer-safe evidence.
2. The interrupted base run cannot be used as quality evidence. It is useful provenance because it proves the packet can launch and retain artifacts, but it does not satisfy the retained-checkpoint-plus-full-eval contract.
3. The clean rerun improves operational confidence but does not yet change the closure status inside the paper. Until its checkpoints and summaries are retained and paired with `k025`, `k000`, and the probe outputs, it remains pre-closure state.

## Minimum next experiment

The minimum next experiment that closes the largest remaining gap is still the full Distinct5 same-family H-family path-stability packet:

- clean `base` completion;
- matched `k025` completion;
- matched `k000` completion;
- one retained `tools/probe_path_stability.py` packet judged by the protocol accept/reject rule.

This is higher value than launching new tokenizer, timing, or metric experiments because it closes the only remaining mechanism gap that is both paper-central and still unblocked.

## Promotion Gate

Broadening the current mechanism claims is justified only if all of the following evidence exists in the retained artifact path:

1. A clean same-family Distinct5 `H` packet with three completed arms: `base`, `k025`, and `k000`.
2. For each arm, the retained chain must include:
   - resolved config,
   - `remote_train.log`,
   - `epoch_0001..0003.pt`,
   - `full_eval/epoch_0001..0003/summary.json`.
3. A completed retained probe packet from `tools/probe_path_stability.py` containing at least:
   - `summary.json`,
   - `per_time_stats.csv`,
   - `run_summary.csv`,
   - `fig_velocity_over_time.pdf`.
4. The retained probe must satisfy at least one protocol promotion condition under matched evaluation:
   - weakening or removing kinetic clearly raises transfer-direction velocity magnitude or path-length ratio; or
   - the full model retains lower path-energy statistics at similar or better endpoint movement.
5. The result must be same-family with the paper's reviewed Distinct5 mechanism surface. `L`-family-local tokenizer packets, interrupted runs, or runtime-only traces are not sufficient substitutes.

If any one of those items is missing, the paper should keep the current narrow wording: historical destructive-ablation support is admissible, but Distinct5 same-family kinetic/path-energy closure is not yet promotable.
