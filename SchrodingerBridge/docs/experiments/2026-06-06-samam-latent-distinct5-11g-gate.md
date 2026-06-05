# SaMAM Latent Distinct5-512 11G Gate

Date: 2026-06-06

Scope:

- method: latent `SaMAM`
- dataset: `Distinct5-512`
- lane: `same-cost`
- machine: remote `RTX 3060 WSL`
- hard runtime cap: `< 11.0 GiB`

## Why this note exists

This note closes the first formal low-VRAM launch audit for latent `SaMAM` on
`Distinct5-512`.

The question was narrow:

- can latent `SaMAM` stay launch-stable on the paper machine under the hard
  `< 11.0 GiB` runtime cap while still following the reviewed same-cost
  protocol?

The answer from the current packet is **no**.

## Attempt 1: reviewed 32-bit low-VRAM launch

Run root:

- `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_distinct5_512_samecost_20260606_034059`

Protocol:

- `batch-size=1`
- `gradient_checkpointing=1`
- `checkpoint-every-n-steps=50`
- `limit_val_batches=0`
- `num_sanity_val_steps=0`
- `precision=32-true`

Observed result:

- the host-owned launcher path worked correctly
- the lane passed process/log creation
- the `30s` first-health gate observed about `12065 MiB`
- this exceeded the hard runtime cap and the lane was stopped immediately

Interpretation:

- the execution surface is valid
- the training packet itself is not admissible under the remote `3060` cap

## Attempt 2: AMP fallback

Run root:

- `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_distinct5_512_samecost_20260606_034359`

Change:

- replaced `precision=32-true` with `precision=16-mixed`

Observed result:

- launch stayed under the runtime cap only because training crashed early
- the run failed with:
  - `RuntimeError: Expected D.scalar_type() == at::ScalarType::Float`
- the failure happens in `mamba_ssm selective_scan`

Interpretation:

- AMP is not a valid low-VRAM rescue path for the current latent `SaMAM`
  implementation on this environment

## Attempt 3: identity-branch checkpointing

Run root:

- `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_distinct5_512_samecost_20260606_034632`

Change relative to Attempt 1:

- kept `precision=32-true`
- added `identity-gradient-checkpointing=1`

Observed result:

- the lane again passed process/log creation
- the `30s` first-health gate observed about `12067 MiB`
- this again exceeded the hard runtime cap and the lane was stopped immediately

Interpretation:

- checkpointing the identity branch alone is not enough to make the packet
  formal-machine-safe

## Closure

Current closure:

- latent `SaMAM` same-cost on `Distinct5-512` is **not launch-stable** on the
  reviewed `3060` under the formal `< 11.0 GiB` runtime policy
- therefore it should be treated as a **structural failure under the paper
  machine contract**, not as a missing evaluation artifact

What this does **not** mean:

- it does not prove latent `SaMAM` is universally impossible
- it does not provide a paper-facing quality row
- it does not justify adding a row to `distinct5_same_cost_inventory.csv`

Immediate consequence:

- keep this result in the main experiment log as negative operational evidence
- do not spend more `Distinct5` budget on latent `SaMAM` same-cost until a new
  low-VRAM mechanism is identified
- move the next baseline slot to latent `SaMST` or another baseline whose
  protocol still has a realistic chance to close under the same machine rules
