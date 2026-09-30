# Remote h1 Checkpoint-vs-Init Style Response Audit

This folder compares one pulled trained checkpoint against the same config at random initialization:

- config: `docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json`
- checkpoint: `docs/experiments/2026-06-18-remote-h1-e18-diagnosis/epoch_0018.pt`
- run family: remote `h1_linear_fm`

The goal is narrower than full metric evaluation:

> When results are very close, did training leave the executed model path unchanged, or did it reshape the style-response contract in a specific way?

## Command

```powershell
py -3.12 tools/probe_checkpoint_style_response.py `
  --config docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json `
  --checkpoint docs/experiments/2026-06-18-remote-h1-e18-diagnosis/epoch_0018.pt `
  --output-dir docs/experiments/2026-06-19-checkpoint-style-response-audit/remote_h1_epoch18 `
  --device cpu `
  --batch-size 2 `
  --latent-size 32 `
  --style-id 0 --style-id 1 --style-id 2 --style-id 3 --style-id 4
```

Generated artifacts:

- `summary.json`
- `comparison_metrics.csv`
- init/checkpoint `conditioning`, `topology`, `path_anatomy`, and `styleid` CSVs

## Key result

`summary.json["overall_reading"]` is:

```text
matched_target_suppressed_styleid_amplified_body_dead
```

That label is intentional. It means:

1. training did **not** leave the style path globally unchanged
2. matched-target / topology sensitivity was strongly suppressed
3. no-reference `style_id` sensitivity was strongly amplified
4. body-level no-reference style actuation still stayed dead

## Core numbers

### Matched-target / topology branch

From `comparison_metrics.csv`:

- `matched_target_spatial_forward_delta`
  - init: `0.0272638574`
  - checkpoint: `0.0006785966`
  - ratio: `0.0249x`
  - transition: `trained_suppression`
- `matched_target_both_forward_delta`
  - init: `0.0292907786`
  - checkpoint: `0.0007176386`
  - ratio: `0.0245x`
  - transition: `trained_suppression`
- `topology_gate1_blend_effect_delta`
  - init: `0.0292829946`
  - checkpoint: `0.0008277031`
  - ratio: `0.0283x`
  - transition: `trained_suppression`

Interpretation:

- the trained checkpoint almost erased the strong random-init matched-target spatial lever
- the trained checkpoint also nearly erased the topology-blend lever

### No-reference style-id branch

- `styleid_max_forward_pair_delta`
  - init: `0.0107860174`
  - checkpoint: `0.2062852979`
  - ratio: `19.13x`
  - transition: `trained_amplification`
- `styleid_mean_forward_pair_delta`
  - init: `0.0086832735`
  - checkpoint: `0.1107746411`
  - ratio: `12.76x`
  - transition: `trained_amplification`
- `styleid_max_body_pair_delta`
  - init: `0.0`
  - checkpoint: `0.0`
  - transition: `persistent_noop`

Interpretation:

- training did wake up a much stronger no-reference `style_id -> decoder` response
- but that response still does not reach `h_body`

## Anatomy read

The init vs checkpoint anatomy summaries in `summary.json` say:

- init:
  - `anatomy_code_first_live_stage = first_hires_block_gate1_a_vs_b_mean_abs`
  - `anatomy_code_body_dead_spatial_body_live = true`
- checkpoint:
  - `anatomy_code_first_live_stage = adapted_code_a_vs_b_mean_abs`
  - `anatomy_code_body_dead_spatial_body_live = true`

And the checkpoint `styleid.best_forward_pair` shows:

- `style_map_a_vs_b_mean_abs = 0.0`
- `h_body_a_vs_b_mean_abs = 0.0`
- `h_fused_a_vs_b_mean_abs = 0.0951578543`
- `h_dec_post_mod_a_vs_b_mean_abs = 0.3387754858`
- `delta_a_vs_b_mean_abs = 0.2062852979`

So the trained checkpoint did not learn a body-level no-reference spatial carrier. It learned a stronger late decoder-only style route.

## What this rules out

This artifact rules out the simplest failure story:

> "The close results mean the model never really changed."

That story is too weak because:

1. the matched-target/topology branch changed a lot, in the direction of suppression
2. the no-reference style-id branch changed a lot, in the direction of amplification
3. the stable failure is more specific: `body dead, decoder live`

## What this now suggests

The stronger diagnosis is:

1. training can collapse the matched-target spatial/topology lever that looked live at init
2. training can simultaneously amplify a late decoder-only `style_id` lever
3. if benchmark quality remains weak, the bottleneck is no longer "did the code path change at all?"
4. the bottleneck is "did training move style actuation into the same body-level path that no-reference evaluation actually needs?"

That is why the next paradigm changes should favor:

- matched-target instance style distilled into the plain no-reference path
- body-level no-reference carriers
- feature-level transfer

over:

- more tiny blend sweeps
- more decoder-only style scalars
- more OT variants that still depend on the unmatched train/eval contract
