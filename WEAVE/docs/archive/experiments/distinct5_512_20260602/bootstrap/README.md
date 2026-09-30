# Distinct5-512 Paired IDT Bootstrap

Date: 2026-06-04

This packet computes paired bootstrap intervals for transfer-only CLIP-S gain
over the identical-image reference (`idt`). The pairing key is:

`src_style, tgt_style, src_image`

Only off-diagonal transfer rows are used (`src_style != tgt_style`), giving
`n=600` paired rows per method. Each listed method has exact row alignment with
the retained IDT metrics (`missing_from_method=0`, `extra_in_method=0`).

## Command

```powershell
py -3 SchrodingerBridge/tools/compute_distinct5_idt_bootstrap.py
```

## Result

| method | transfer CLIP-S | IDT CLIP-S | delta vs IDT | 95% bootstrap CI | LPIPS |
|---|---:|---:|---:|---:|---:|
| LBM-F e1 | 0.664360 | 0.639922 | +0.024438 | [0.020961, 0.027965] | 0.324528 |
| LBM-H e1 | 0.665255 | 0.639922 | +0.025333 | [0.021611, 0.029100] | 0.328105 |
| LBM-H e2 | 0.668395 | 0.639922 | +0.028472 | [0.024631, 0.032375] | 0.356105 |
| LBM-K e1 | 0.671167 | 0.639922 | +0.031244 | [0.027273, 0.035242] | 0.372281 |
| SaMST e5 | 0.698919 | 0.639922 | +0.058996 | [0.051759, 0.066506] | 0.633500 |
| SaMST e15 | 0.695741 | 0.639922 | +0.055819 | [0.049400, 0.062480] | 0.631950 |

All bootstrap draws in this packet had positive mean delta (`P(delta>0)=1.0`).

## Boundary

SaMAM is intentionally not included in this bootstrap table yet. The currently
retained local evidence for the Distinct5 SaMAM checkpoints is aggregate
full/transfer metrics plus targetwise ArtFID for selected checkpoints, not a
complete IDT-aligned per-image `metrics.csv` packet. SaMAM therefore remains a
measured point-estimate row in the paper until Dalton or a rerun lands the
complete per-image packet.

The main interpretation is narrow:

- LBM's positive transfer CLIP-S deltas over IDT are statistically stable for
  the retained paired rows.
- SaMST also clears IDT, but in a high-LPIPS and high-ArtFID operating region.
- SaMAM's current paper-safe claim remains `transfer CLIP-S below IDT` at the
  measured checkpoints, not a bootstrap significance claim.

## Files

- `paired_idt_transfer_bootstrap.csv`: generated bootstrap table.
- `SchrodingerBridge/tools/compute_distinct5_idt_bootstrap.py`: reproduction script.
