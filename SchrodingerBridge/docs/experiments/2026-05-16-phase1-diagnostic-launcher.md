# Phase 1 Diagnostic Launcher

This pass adds a single entry script: [run_phase1_diagnostic_probes.py](/G:/GitHub/Latent_Style/SchrodingerBridge/run_phase1_diagnostic_probes.py).

The script is intentionally faithful to the current root code rather than to older branches:

- `semantic_attn_routing_mode` is executable only for `softmax` and `sinkhorn` in the current `src/lancet_backbone.py`.
- The requested hard-Monge / `gumbel_hard` semantic routing probe is preserved in the manifest as an unsupported placeholder instead of being silently run as softmax.
- The launcher defaults to the D0 baseline config and, when present, injects resume checkpoint `S-add__K-1_C-0_W-20_Col-0/epoch_0007.pt`.
- The launcher also forces `training.num_workers=0` unless `--unsafe-workers` is passed, because a previous Windows fast-run attempt failed under worker pressure.

Default behavior:

```powershell
python .\SchrodingerBridge\run_phase1_diagnostic_probes.py --action plan
python .\SchrodingerBridge\run_phase1_diagnostic_probes.py --action launch --probe-group manifold_resistance
```

Artifacts written by the launcher:

- `exp/phase1_diagnostic_probes/configs/*.json`
- `exp/phase1_diagnostic_probes/phase1_manifest.json`
- `exp/phase1_diagnostic_probes/phase1_manifest.csv`
- `exp/phase1_diagnostic_probes/runs/<experiment_id>/`

The generated plan covers four probe families:

- `ot_coupling_plan`
- `manifold_resistance`
- `terminal_measure_pressure`
- `bypass_and_residual_dynamics`
