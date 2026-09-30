# Faraday Split2 Prep Launch

Date: 2026-06-06

Scope:

- fixed-rule follow-up split:
  - `wikiart_stress2`
- objective:
  - close the remote prep surface so split2 can enter the same
    `prep -> F-family train/eval -> IDT/no-op` pipeline already used on split1

## Split identity

Split slug:

- `wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism`

Styles:

- `Abstract_Expressionism`
- `Baroque`
- `Cubism`
- `Northern_Renaissance`
- `Post_Impressionism`

## Sync issue and repair

The first direct launch path reused the original split-prep launcher, which
tries to tar-sync the whole local packet before remote prep.

Observed issue:

- `split2` local packet is larger than `split1`
  - about `1.531 GiB`
- repeated launch attempts timed out locally during the packet sync stage
- remote owner surface already retained:
  - `classview/`
- but the launcher never reached the remote scheduled-task start

Repair:

- [launch_remote_faraday_split_prep.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_faraday_split_prep.py)
  now supports:
  - `--skip-packet-sync`

Why this is safe for split2:

- the remote owner surface already contains the required `classview/train` and
  `classview/test` packet
- the missing part was only:
  - `latents_ema/train`
  - packed latent cache
  - prototype pairing cache

## Launch command

```powershell
python SchrodingerBridge\tools\experiments\launch_remote_faraday_split_prep.py `
  --split-slug wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism `
  --skip-packet-sync
```

## Remote contract

Task name:

- `faraday-prep-wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism`

Remote split root:

- `/mnt/i/wikiart_faraday_splits/wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism`

Remote prep log:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/_codex_tmp/wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism_prep.log`

## First-health snapshot

Observed at launch:

- remote GPU prelaunch:
  - `287 MiB`
- first-health GPU:
  - about `2183 MiB / 12288 MiB`
- active remote prep pid:
  - launcher pid file resolved to `562`

Log proof of healthy start:

- remote prep entered:
  - `run_faraday_split_prep.py`
- first active encode class:
  - `Abstract_Expressionism`
- observed warm throughput after stabilization:
  - roughly `7 images/s`

## Current status

Current status at note time:

- `running`

Closed already:

- remote `classview` packet exists
- launcher bug around oversized sync has a reviewed workaround

Still pending:

- `latents_ema/train/<style>/*.pt`
- `.latent_cache/manifest.json`
- `.latent_cache/prototype_pairing_top8.pt`

Once those land, the next formal GPU action should be:

- `split2 F-family` launch on the reviewed remote `3060`

## Closure

The split2 prep packet is now closed.

Final retained prep state:

- train latent root:
  - `/mnt/i/wikiart_faraday_splits/wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism/latents_ema/train`
- packed manifest:
  - `.latent_cache/manifest.json`
- pairing cache:
  - `.latent_cache/prototype_pairing_top8.pt`

Closed style counts after the repaired cache rebuild:

- `Abstract_Expressionism`: `1000`
- `Baroque`: `1000`
- `Cubism`: `1000`
- `Northern_Renaissance`: `1000`
- `Post_Impressionism`: `1000`

Important repair note:

- an earlier partial owner-side packet left `Post_Impressionism` incomplete on
  the remote surface
- after patching that classview folder and rerunning prep with
  `--skip-packet-sync --rebuild-cache`, the final packed cache and pairing cache
  were rebuilt against the full `1000/30` split contract

This split is therefore ready for the next formal lane:

- `split2 F-family`
