# Remote RAR Deep Provenance - 2026-06-05

Remote root:
`I:\Github\Latent_Style`

This pass uses local `UnRAR.exe` copied temporarily to the remote host to list RAR contents. It does not install tools or modify the remote environment. RAR entries are compared against expanded directories by same relative path and exact byte size.

## Tooling

- Local tool used: `C:\Program Files\WinRAR\UnRAR.exe`.
- Remote temporary copy: `C:\Users\Administrator\AppData\Local\Temp\codex_UnRAR.exe`.
- Listing script: `inspect_remote_rar_provenance_deep.ps1`.
- Deep provenance CSV: `manual_remote_rar_provenance_deep_20260605.csv`.
- Policy CSV: `manual_remote_rar_deep_provenance_policy_20260605.csv`.

## Results

| archive | size_mb | file_entries | same-size existing | missing | mismatch | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `experiments.rar` | 8091.026 | 343186 | 343177 | 0 | 9 | keep |
| `Cycle-NCE\Gate.rar` | 3384.032 | 72520 | 72376 | 144 | 0 | delete archive |
| `Cycle-NCE\Attn_48.part*.rar` | 1975.113 | 21176 | 21135 | 41 | 0 | delete archive |
| `Cycle-NCE\chess.part*.rar` | 1194.239 | 21171 | 21132 | 39 | 0 | delete archive |
| `Cycle-NCE\45.rar` | 507.452 | 6096 | 0 | 6096 | 0 | keep |

## Delete decision

`Gate.rar`, `Attn_48.part*.rar`, and `chess.part*.rar` are delete candidates because:

- UnRAR listing succeeds.
- Every nonweight file entry exists same-size in the expanded directory.
- The only unique archive payload is old checkpoint/tokenizer weights (`.pt` payloads).
- The expanded directories remain as nonweight evidence packages: configs, full_eval outputs, logs, summaries, images, and CSV/JSON evidence.

Deletion is archive-only: remove the RAR files/parts, not the expanded directories.

## Retain decision

`experiments.rar` is retained because 9 eval-cache payload files differ in size, including CLIP model/tokenizer/cache files. It is not a clean duplicate archive.

`Cycle-NCE\45.rar` is retained because its entries do not match an expanded current directory under the checked roots. It is the primary retained copy after the exact duplicate `Cycle-NCE\src\45.rar` was already deleted.

## Remaining gap

After deleting the three archive groups, remote archive provenance still has two retained gaps:

- `experiments.rar`: cache mismatch needs cross-cache/cache-payload audit.
- `Cycle-NCE\45.rar`: unique historical archive unless extracted or owner confirms disposability.
