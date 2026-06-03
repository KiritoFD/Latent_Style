# AAAI 2027 Experiment Audit

Date: 2026-06-03  
Round: `R20260603M`  
Lane: `experiment_audit`

Checkpoint label:

- `current_paper_after_agent_cleanup_and_partial_path_stability_launch`

- `overall_status`: `weak_reject`
- `claim_safety_band`: `narrow_only`
- `evidence_closure_band`: `partial`
- `blocking_issue`: `The paper's bounded kinetic/path-energy story still lacks a landed same-family Distinct5 packet: the partial remote base-arm launch proves the path-stability packet can start and retain artifacts on the 3060, but without a clean remote_train.log, per-epoch full_eval summaries, and matched weakened-kinetic controls it remains operational evidence, not mechanism closure.`
- `next_action_1`: `Rerun the Distinct5 H-family base arm under the logging contract until it lands one clean save dir with remote_train.log, epoch_0001..0003 checkpoints, and full_eval/epoch_0001..0003/summary.json.`
- `next_action_2`: `Immediately follow the clean base rerun with matched k025 and k000 runs, then execute tools/probe_path_stability.py and judge paper promotion strictly by the protocol accept/reject rule.`
- `support_score`: `1`
- `fairness_score`: `1`
- `artifact_path_score`: `1`
- `closure_value_score`: `1`

## Directly supported now

1. The Distinct5 no-op-aware frontier claim remains directly supported: LBM stays above the `idt` floor and defines the strongest measured `CLIP-S / LPIPS` trade-off among the currently reproduced positive-`Delta_idt` points.
2. The historical strict-750 LBM operating-point claim remains directly supported as a narrow quality-frontier result.
3. Endpoint-only pointwise supervision is directly supported as a negative closure, not as an open mechanism question.
4. SA-SWD semantic-vs-random is directly supported only as a negative closure: it can stay as the current mainline design choice, not as a positive semantic-axis superiority result.
5. Tokenizer localization is directly usable only as `L`-family-local evidence.
6. New since `R20260603L`: the path-stability packet is directly supported as operationally runnable on the remote 3060 at formal batch 44. The `base` arm produced retained runtime artifacts (`epoch_0001.pt`, two training CSVs, `numeric_debug.jsonl`), and the master ledger now records that interrupted state explicitly.

## Indirectly supported now

1. The paper's bounded kinetic/path-energy wording is still only indirectly supported on the current Distinct5 surface: historical destructive ablations and local theorem-grounding point in the right direction, but the matched same-family `H` packet is not landed.
2. The partial launch strengthens confidence that the remaining blocker is no longer remote asset sync, UTF-8 BOM handling, or the Windows data-root rewrite; it is clean packet completion and retention.
3. Path-stability remains the highest-value unblocked mechanism lane, ahead of new timing or tokenizer work.

## Unsupported now

1. Any statement that the Distinct5 `H`-family path-stability / weakened-kinetic claim is empirically closed.
2. Any use of the partial `base` runtime as quality evidence.
3. Any stronger theory or path-energy rhetoric that reads as if the current same-family packet already confirms the claim.
4. Any decision to promote `H`-family tokenizer continuity, broader efficiency rhetoric, or fresh endpoint/semantic reruns ahead of the path packet.

## Does the partial launch close anything?

It closes only operational readiness:

- remote sync problems, BOM issues, and remote data-root mismatch are no longer the main uncertainty;
- the formal `base` arm can start, hold VRAM, write retained artifacts, and survive into the ledger.

It does not close the paper blocker:

- no clean `remote_train.log`;
- no current live process;
- no retained `full_eval` summaries;
- no matched `k025` or `k000` controls;
- no `tools/probe_path_stability.py` readout.

## Minimum next experiment for promotion

The minimum promotable experiment is still the full matched path packet, not the current partial `base` launch. The minimum next rerun is a clean logged rerun of `base`, because without one stable `base` packet the controls and probe cannot become reviewer-safe evidence.

## Deprioritize until this lands

Yes. Keep formal remote budget away from:

- extra endpoint-only reruns;
- extra semantic-vs-random reruns;
- new speed-claim experiments;
- `H`-family tokenizer recovery unless path-stability becomes blocked again for an external reason.
