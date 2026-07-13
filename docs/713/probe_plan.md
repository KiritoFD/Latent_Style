# Probe Plan

## P0: Current T11 Path Strength Probe

Purpose: quantify learned style path versus endpoint statistical alignment on the actual `aaai_v4`-style checkpoint.

Inputs:

- Config: `SchrodingerBridge/exp/710_infra_t11_distinct5_5ep/config.json`
- Checkpoint: `SchrodingerBridge/exp/710_infra_t11_distinct5_5ep/epoch_0005.pt`
- Dataset: config-provided latent cache paths.

Measurements:

- Per-block debug values: `cross_attn_delta_abs`, `ca_input_std`, `ca_output_std`, `style_gate_value`, entropy.
- Per-subband flow delta: `ODE_no_endpoint - content`.
- Per-subband endpoint delta: `full_output - ODE_no_endpoint`.
- Per-subband transfer ratio toward target style statistics.
- Style-swap sensitivity for fixed content across target style IDs.
- Injection-mode swap: no endpoint, configured endpoint, per-subband AdaIN, per-subband WCT, spatial-fiber AdaIN, spatial-fiber WCT.
- Path separation: fix endpoint `style_latent` and swap learned `style_id`; disable endpoint and compare target/source/shifted `style_id`.
- Route controls: endpoint disabled with DWT-routed cross-attn, full cross-attn, and cross-attn off.
- Time sweep: fixed-content style-swap sensitivity at `t=0.1/0.5/0.9`.

Output:

- JSON and Markdown summary under `docs/713/probe_outputs/`.

Current strongest run:

- `docs/713/probe_outputs/t11_ep5_style_path_n32_gpu_pathsep.md`
- Local GPU: RTX 4070 Laptop, n=32.
- Pairing cache was absent locally, so the dataset fell back to non-cache sampling. This is acceptable for mechanism diagnosis, but final DINO-S ranking must use the official eval protocol.

## P1: DINO-S Candidate Filter

Purpose: only send plausible variants to expensive image evaluation.

Candidates should pass at least one latent criterion:

- Endpoint delta improves LH/HL style-transfer ratio without large LL movement.
- Style-swap sensitivity increases in LH/HL but not LL.
- Content latent L2 does not jump more than 10% over baseline probe mode.

## P2: Remote Train / Local Eval Pipeline

Use remote RTX 3060 over:

`ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`

Remote workspace:

`I:/Github/Latent_Style/SchrodingerBridge`

Pipeline:

1. Patch or generate one config per candidate.
2. Launch training remotely in detached PowerShell/WSL shell.
3. Poll logs and checkpoint existence.
4. Run local or remote full eval.
5. Rank by DINO-S first, then LPIPS and DINO-C.

## P3: Minimal Model Change Candidates

Default order:

1. Config-only activation of existing high-frequency style-conditioned velocity heads, with LL kept zero-init and HF nonzero-init.
2. Config-only activation of Q-side style AdaLN for DWT-routed cross-attn.
3. New gated high-frequency endpoint residual if P0 shows endpoint is the only strong path and learned flow remains style-blind.

Current P0 result favors candidate 1. Endpoint scale changes alone are not favored because stronger LH/HL/HH scales reduced latent transfer ratios in both n=16 and n=32 probes.

Rejected for now:

- Global style AdaLN as first move: style-swap sensitivity already leaks strongly into LL.
- Full cross-attention route as first move: no-endpoint full cross-attn raises content L2 without producing high-frequency transfer in the probe.
- Stronger endpoint alpha sweep as first move: latent transfer ratios got worse while content L2 increased.
