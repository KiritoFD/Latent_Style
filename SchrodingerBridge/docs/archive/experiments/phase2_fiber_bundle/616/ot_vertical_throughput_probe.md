# OT Vertical Throughput Probe

This packet is the matched throughput scan that follows the current `phase616_ot_vertical_scratch_b8a2_e24` foundation run. It does not change OT, solver, tokenizer, or dataset semantics. It only changes:

- `batch_size`
- `accumulation_steps`
- `virtual_length_multiplier`

## Probe Order

1. `phase616_ot_vertical_throughput_b12a2_vlen100_step80`
2. `phase616_ot_vertical_throughput_b16a2_vlen100_step80`
3. `phase616_ot_vertical_throughput_b16a1_vlen100_step80`
4. `phase616_ot_vertical_throughput_b16a1_vlen125_step80`

The current live control lane stays unchanged at `b8 a2 vlen1.0`. This packet starts only after the foundation lane closes or reaches a stable enough checkpoint boundary to fork from the same mechanism contract.

## What Stays Fixed

These probes are not allowed to change:

- OT mechanism
- target projection mechanism
- solver family
- tokenizer family
- dataset split
- eval contract

They only change throughput knobs so the next formal lane can target the `9.0-11.0 GiB` band without contaminating the OT conclusion.

## Runtime Artifacts

- shell launcher:
  [run_phase616_ot_vertical_throughput_probe.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_ot_vertical_throughput_probe.sh)
- training CSV per run:
  `exp/.../logs/training_*.csv`
- GPU sampler outputs per run:
  `docs/experiments/phase2_fiber_bundle/616/logs/ot_vertical_throughput_probe/*.gpu_metrics.csv`
  `docs/experiments/phase2_fiber_bundle/616/logs/ot_vertical_throughput_probe/*.gpu_summary.json`

## Decision Rule

- Prefer the first config whose observed VRAM settles inside `9.0-11.0 GiB`.
- Hard reject any config that repeatedly exceeds `11.2 GiB`.
- If multiple configs stay in-band, prefer the one with lower `avg_optimizer_step_time_sec`.
- If a probe raises VRAM but lowers GPU util or worsens step time materially, reject it even if it lands inside the target band.
- These probes are `diagnostic-only`; do not compare their style/LPIPS outputs against mechanism experiments.
