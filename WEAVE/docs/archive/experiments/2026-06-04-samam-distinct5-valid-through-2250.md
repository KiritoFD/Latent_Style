# SaMAM Distinct5 Validity Boundary

Date: 2026-06-04

Scope: SaMAM evidence used by the AAAI 2027 Distinct5-512 manuscript.

## Decision

Treat `SaMAM 2250` as the final validated SaMAM Distinct5 reproduction
checkpoint for the manuscript.

For manuscript tables, plots, and prose, handle the reproduced SaMAM Distinct5
run as if it stops at step 2250.

Outputs after 2250 are treated as reproduction-chain failures and are excluded
from manuscript evidence. They should not be described as positive-IDT results,
audit evidence, or candidate headline points.

## Manuscript Row

Transfer-only IDT floor: `0.639922`.

| point | transfer CLIP-S | transfer LPIPS | targetwise ArtFID | train min | manuscript status |
| --- | ---: | ---: | ---: | ---: | --- |
| SaMAM 2250 | 0.552252 | 0.360452 | 148.206 | 458.6 | final validated SaMAM row |

## Writing Policy

- Table 1 reports `SaMAM 2250`.
- Figure 1 plots the SaMAM curve only through 2250.
- Main text says the validated SaMAM Distinct5 reproduction remains below IDT.
- Main text does not discuss post-2250 outputs.
- Any future use of later SaMAM points requires a clean independent rerun and a
  closed metric packet.
