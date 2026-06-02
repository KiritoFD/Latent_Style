# AAAI 2027 Experiment Logging Contract

Date: 2026-06-03

Purpose:

- unify how formal experiments are logged;
- make remote 3060 runs auditable without re-reading chat history;
- ensure every paper-facing result has enough provenance for review,
  continuation, and rerun.

This contract applies to:

- all new paper-facing `SchrodingerBridge` runs;
- all claim-closing ablations;
- any run whose numbers may enter the manuscript, a figure, or a review memo.

## 1. Required artifact set

Every formal run must produce or be paired with all of the following.

### Runtime directory

Under `SchrodingerBridge/exp/`:

- one run directory
- one stable run id
- one immutable resolved config snapshot

Preferred pattern:

- `exp/aaai2027_<family>_<base>_<seed>_b<batch>`

Examples:

- `exp/aaai2027_endpoint_metric_h_omf_flow_huber_seed42_b44`
- `exp/aaai2027_saswd_axis_h_base_seed42_b44_saswd_semantic`

### In-run files

At minimum:

- `config.json`
- `remote_train.log` or equivalent primary train log
- `logs/` if the trainer already writes structured epoch logs
- `full_eval/epoch_xxxx/summary.json` for each retained eval point

If the run is generate-only or timing-only:

- one retained `summary.json`
- one retained wall-time file or explicit timing entry in the summary

### Paper-side packet

Under `docs/experiments/YYYY-MM-DD-<block>/`:

- `README.md`
- config registry or manifest
- launch note
- completion or anomaly note if needed

### Master ledger

One row in:

- `docs/experiments/aaai2027_master_experiment_log.csv`

Minimum row expectations:

- date
- family
- dataset
- scope
- method
- variant or point
- checkpoint or step
- train wall
- hardware
- status
- keep decision
- evidence path
- note
- review required
- claim safety band

## 2. Required provenance fields

The experiment packet must record:

1. run id
2. resolved config path
3. code branch and commit when launched
4. hardware and environment scope
5. batch size and important throughput knobs
6. exact eval scope
7. whether reported time includes evaluation

For remote 3060 runs, also record:

1. remote workspace root
2. remote log path
3. first successful heartbeat
4. completion heartbeat or failure heartbeat

## 3. Status vocabulary

Use these status bands consistently in the master ledger and README notes:

- `planned`
- `running`
- `completed`
- `blocked_inference`
- `review_pending`
- `retired`

Use these evidence-intent bands in notes:

- `formal`
  - normal paper-facing quality and timing evidence
- `quality_only`
  - metrics may be usable, runtime may not
- `timing_only`
  - throughput measurement, not a quality comparison
- `smoke_only`
  - sanity check only, never a paper claim

## 4. Abnormal-run policy

If a run suffers from runtime anomaly, cache corruption, or host interference:

1. do not silently discard it;
2. preserve the log path;
3. record the anomaly in the packet README and master ledger;
4. mark whether the result is:
   - unusable,
   - quality-only,
   - or fully invalidated.

Examples already in scope:

- invalidated near-null `mse/huber/l1` flow-loss trio after config audit
- random-axis SA-SWD run whose runtime behavior is not admissible as fair speed
  evidence

## 5. Review handoff rule

No new experiment is paper-facing just because the run finished.

After a formal result lands:

1. update the packet README;
2. update `aaai2027_master_experiment_log.csv`;
3. hand it to the reviewer lane through `docs/reviews/`;
4. only then allow manuscript escalation.

## 6. What should stop happening

Do not keep doing the following:

- leaving a useful run only in `exp/...` with no packet README
- reporting numbers from chat only
- mixing smoke timings with formal timings
- creating top-level one-off naming families with no ledger row
- citing a local probe directory directly in the paper

## 7. Immediate operational rule

For the current AAAI 2027 push:

- all new formal runs should be remote-3060-first unless they are explicit local
  smoke checks;
- all remote formal runs should use this logging contract;
- `Linnaeus` is the execution owner responsible for keeping the runtime side of
  this contract intact.
