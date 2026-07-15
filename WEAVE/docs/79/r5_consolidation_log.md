# R5 Consolidation Log

## 2026-07-09 local merge into `results\R5-WikiArt`

Target directory:

`G:\GitHub\Latent_Style\SchrodingerBridge\results\R5-WikiArt`

The user confirmed that R5 should be consolidated under this directory. To avoid destructive changes, the first pass copied verified local Random5 packets from `results\R5-512` into `results\R5-WikiArt` and kept the source directories intact.

| Method | Source | Destination | Images | Status |
|---|---|---|---:|---|
| `stylealigned` | `G:\GitHub\Latent_Style\SchrodingerBridge\results\R5-512\stylealigned` | `G:\GitHub\Latent_Style\SchrodingerBridge\results\R5-WikiArt\stylealigned` | 750 | copied, clean |
| `zstar` | `G:\GitHub\Latent_Style\SchrodingerBridge\results\R5-512\zstar` | `G:\GitHub\Latent_Style\SchrodingerBridge\results\R5-WikiArt\zstar` | 750 | copied, clean |
| `styleshot` | `G:\GitHub\Latent_Style\SchrodingerBridge\results\R5-512\styleshot` | `G:\GitHub\Latent_Style\SchrodingerBridge\results\R5-WikiArt\styleshot` | 740 | copied, still incomplete |

Current direct/recursive image counts in `results\R5-WikiArt` after this merge:

| Method | Direct images | Recursive images | Note |
|---|---:|---:|---|
| `cut` | 750 | 750 | Existing packet; style-set audit still required. |
| `samam` | 750 | 750 | Existing packet; style-set audit still required. |
| `samst` | 750 | 750 | Existing packet; style-set audit still required. |
| `sdturbo` | 1123 | 1123 | Nonstandard count; likely mixed/20-style or partially repaired packet. |
| `seedream` | 0 | 724 | Nested/incomplete packet. |
| `stylealigned` | 750 | 750 | Copied from canonical `R5-512`. |
| `styleid` | 750 | 750 | Existing packet; style-set audit still required. |
| `styleshot` | 740 | 740 | Copied from canonical `R5-512`; incomplete. |
| `zstar` | 750 | 750 | Copied from canonical `R5-512`. |

## Remote search status

Remote host checked:

`ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`

Remote paths checked for direct clean R5/wikiarts20 image packets:

| Remote area | Result |
|---|---|
| `I:\Github\Latent_Style\SchrodingerBridge\results` | Missing. |
| `I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20` | Missing at this exact path. |
| `I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval` | Missing at this exact path. |
| `I:\results\aaai2027_v4_tables` | CSV/table artifacts only, no image packets. |
| `I:\results\tables` | CSV/table artifacts only, no image packets. |
| `I:\latent_style_remote_curated\by_dataset\wikiart512_5style.csv` | Summary/metric paths only; useful archaeology source, not a direct image-packet root. |

Important nuance: the remote curated CSV has many `wikiart512_5style` summary rows under old TokenizerClean/SchrodingerBridge experiment paths, but this pass did not identify a clean 750-image baseline packet there that can be copied directly into `results\R5-WikiArt`.

## Next R5 work

1. Parse filenames inside existing `results\R5-WikiArt\cut`, `samam`, `samst`, `styleid` to confirm whether they are true Random5 or D5-style legacy packets.
2. Locate or reconstruct missing R5 main-table packets for `identity`, `adain`, `wct`, `IP-Adapter`, and `ours/weave`.
3. Repair `styleshot` from 740 to 750 if the table row remains.
4. Decide whether `sdturbo` 1123 and `seedream` 724 are partial Random5 packets, mixed 20-style packets, or D5-style legacy packets.
