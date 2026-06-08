# Distinct5 Extended IDT Bootstrap

Date: 2026-06-08

Script:

- [compute_distinct5_idt_bootstrap.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/compute_distinct5_idt_bootstrap.py)

Output CSV:

- [distinct5_idt_bootstrap_extended.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_idt_bootstrap_extended.csv)

Scope:

- row-resampled transfer-only CLIP-S delta over the Distinct5 IDT baseline
- pairing key:
  - `src_style, tgt_style, src_image`
- off-diagonal transfer rows only
- `n = 600` paired rows per method

Extended paper-facing rows:

| point | transfer CLIP-S | delta vs IDT | 95% row-resampled interval | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `LBM-Knee e13` | `0.7102` | `+0.0702` | `[0.0635, 0.0769]` | `0.4603` |
| `LBM-PS-v2 e13` | `0.7300` | `+0.0901` | `[0.0826, 0.0976]` | `0.6069` |
| `Seedream-4.5` | `0.6937` | `+0.0538` | `[0.0484, 0.0592]` | `0.4440` |

Notes:

- all row-resampled draws remained positive for these three rows:
  - `P(delta > 0) = 1.0`
- this is still a row-resampled diagnostic, not a clustered source/style bootstrap
- it is appropriate for the narrow paper claim:
  - the promoted Distinct5 operating points remain stably above the IDT floor
