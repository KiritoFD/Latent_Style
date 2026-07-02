# Phase 1 Diagnostic Launcher

This pass adds a single entry script: [run_phase1_diagnostic_probes.py](/G:/GitHub/Latent_Style/SchrodingerBridge/run_phase1_diagnostic_probes.py).

The script is intentionally faithful to the current root code rather than to older branches:

- `semantic_attn_routing_mode` now supports `softmax`, `sinkhorn`, and `gumbel_hard` in the current `src/lancet_backbone.py`.
- The requested hard-Monge probe is implemented as semantic `gumbel_hard` routing with `semantic_gumbel_tau`.
- The launcher defaults to the D0 baseline config and, when present, injects resume checkpoint `S-add__K-1_C-0_W-20_Col-0/epoch_0007.pt`.
- The launcher also forces `training.num_workers=0` unless `--unsafe-workers` is passed, because a previous Windows fast-run attempt failed under worker pressure.
- The launcher default action is now `launch`, not `plan`: it generates configs, starts training, then automatically evaluates the newest checkpoint for each completed run.

Default behavior:

```powershell
python .\SchrodingerBridge\run_phase1_diagnostic_probes.py
python .\SchrodingerBridge\run_phase1_diagnostic_probes.py --probe-group manifold_resistance
python .\SchrodingerBridge\run_phase1_diagnostic_probes.py --action plan
```

Artifacts written by the launcher:

- `exp/phase1_diagnostic_probes/configs/*.json`
- `exp/phase1_diagnostic_probes/phase1_manifest.json`
- `exp/phase1_diagnostic_probes/phase1_manifest.csv`
- `exp/phase1_diagnostic_probes/evaluation_summary.json`
- `exp/phase1_diagnostic_probes/evaluation_summary.csv`
- `exp/phase1_diagnostic_probes/runs/<experiment_id>/`

The generated plan covers four probe families:

- `ot_coupling_plan`
- `manifold_resistance`
- `terminal_measure_pressure`
- `bypass_and_residual_dynamics`
