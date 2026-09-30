# attn_gw_ot Closure

- Status: recalibration needed
- Current closure path:
  - the current remote lane was stopped on `2026-06-10`
  - reason:
    - repeated runtime samples stayed below the requested `9.0 GiB` floor
    - a concurrent stale remote training/eval lane was also found on the same `3060`, so the run was not single-lane formal evidence
  - what remains useful:
    - the deferred local fast-eval launcher is intentionally kept alive
    - after `SaMAM` finishes, existing `attn_gw_ot` checkpoints can still be evaluated locally for directional signal
  - what is explicitly disabled:
    - the `attn_gw_ot` runtime watcher was stopped
    - the `attn_gw_ot` deferred stage-close packet was stopped
  - next formal action:
    - if the later fast curve looks promising, relaunch `attn_gw_ot` in-band as a fresh single-lane formal run before any promote/reject decision
