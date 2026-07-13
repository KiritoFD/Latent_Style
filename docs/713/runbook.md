# Runbook

## Local Probe

```powershell
python SchrodingerBridge/tools/probe_713_style_path.py `
  --config SchrodingerBridge/exp/710_infra_t11_distinct5_5ep/config.json `
  --checkpoint SchrodingerBridge/exp/710_infra_t11_distinct5_5ep/epoch_0005.pt `
  --output docs/713/probe_outputs/t11_ep5_style_path.json `
  --num-samples 16
```

## Remote Access

```powershell
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command `"Get-ChildItem I:\Github\Latent_Style\SchrodingerBridge | Select-Object -First 5`""
```

## Ranking Rule

Primary style metric: DINO-S.

Keep a candidate only if:

- DINO-S improves or stays within noise while another important metric improves.
- LPIPS and DINO-C do not indicate unacceptable content loss.
- Visual examples do not show obvious structure leakage or whitening.

