# Repaired Endpoint-Metric Launch Manifest

Date: 2026-06-03

This manifest converts the repaired endpoint-metric packet into a remote-launch
contract for the standing 3060 owner.

## Packet intent

Primary question:

> On the current Distinct5 H base, what happens when target-style endpoint
> supervision is enforced through an actually active pointwise latent penalty?

This packet is intentionally narrower than the earlier invalidated
`mse / huber / l1` block. It is not about velocity-regression kernels. It is
about endpoint-matching kernels on the active OMF path.

## Shared controls

- base family:
  - `configs/distinct5_512_ema_variant_h_hard_explore_queue_e3.json`
- resolved bridge controls:
  - `objective_mode = omf`
  - `w_flow = 1.0`
  - `terminal_swd_weight = 0.0`
  - `terminal_swd_aux_weight = 0.0`
  - `w_kinetic = 1.0`
  - `kinetic_mode = endpoint`
- data root:
  - `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train`
- test split:
  - `/mnt/i/wikiart_distinct5_samam_512_classview/test`
- formal batch:
  - `44`
- seed:
  - `42`
- epoch budget:
  - inherited `3` epochs
- eval bundle:
  - per-epoch strict full eval for `epoch_0001`, `epoch_0002`, `epoch_0003`

## Arms

### 1. MSE

- config:
  - `configs/aaai2027/endpoint_metric_h_omf_flow_mse_seed42.json`
- task name:
  - `SB_EndpointMetric_H_OMF_MSE_S42`
- output dir:
  - `exp/aaai2027_endpoint_metric_h_omf_flow_mse_seed42_b44`
- train log:
  - `exp/aaai2027_endpoint_metric_h_omf_flow_mse_seed42_b44/remote_train.log`

### 2. Huber

- config:
  - `configs/aaai2027/endpoint_metric_h_omf_flow_huber_seed42.json`
- task name:
  - `SB_EndpointMetric_H_OMF_HUBER_S42`
- output dir:
  - `exp/aaai2027_endpoint_metric_h_omf_flow_huber_seed42_b44`
- train log:
  - `exp/aaai2027_endpoint_metric_h_omf_flow_huber_seed42_b44/remote_train.log`

### 3. L1

- config:
  - `configs/aaai2027/endpoint_metric_h_omf_flow_l1_seed42.json`
- task name:
  - `SB_EndpointMetric_H_OMF_L1_S42`
- output dir:
  - `exp/aaai2027_endpoint_metric_h_omf_flow_l1_seed42_b44`
- train log:
  - `exp/aaai2027_endpoint_metric_h_omf_flow_l1_seed42_b44/remote_train.log`

## Success gate

Each arm counts as completed only if all of the following exist:

1. `remote_train.log` ends with training completion
2. `full_eval/epoch_0001/summary.json`
3. `full_eval/epoch_0002/summary.json`
4. `full_eval/epoch_0003/summary.json`
5. one resolved-config audit note showing:
   - `objective_mode = omf`
   - `w_flow = 1.0`
   - `terminal_swd_weight = 0.0`

## Interpretation contract

Paper-safe interpretation after the run:

- if the three arms separate, that is valid evidence about **endpoint
  pointwise-matching kernels**
- if the three arms overlap, that is valid parity evidence for this isolated
  endpoint penalty
- neither result may be described as a closed theorem about velocity-regression
  kernels or all latent-space local losses

## Comparison baseline

Use the already-reviewed H-base Distinct5 mainline as the external control for
paper discussion:

- `distinct5_512_ema_variant_h_hard_explore_queue_e3`
- representative reviewed point:
  - `H_balanced`, `epoch_0002`
- note:
  - this control still uses `terminal_swd_weight = 20.0` and `w_flow = 0.0`
  - therefore it is a different object and should be described as the current
    W1-style mainline, not as another arm inside the repaired packet
