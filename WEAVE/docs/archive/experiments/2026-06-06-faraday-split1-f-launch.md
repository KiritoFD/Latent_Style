# Faraday Split1 F-Family Launch

Date: 2026-06-06

Scope:

- fixed-rule follow-up split:
  - `wikiart_stress1`
- method family:
  - `LBM-F`
- machine:
  - remote `RTX 3060`

## Precondition closure

This launch only happened after the split-prep packet closed successfully:

- [2026-06-06-faraday-split1-prep-launch.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-faraday-split1-prep-launch.md)

Verified remote prep outputs before training launch:

- split root:
  - `/mnt/i/wikiart_faraday_splits/wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism`
- train latents:
  - `latents_ema/train/<style>/*.pt`
  - `1000` files for each of the `5` styles
- packed latent cache:
  - `latents_ema/train/.latent_cache/manifest.json`
  - `latents_ema/train/.latent_cache/packed/*.pt`
- pairing cache:
  - `latents_ema/train/.latent_cache/prototype_pairing_top8.pt`
  - sidecar count:
    - `20000` routes

## Config bug found and repaired

The first launch attempt failed before remote start because the generated
`faraday_splits` configs used a broken `_base` path.

Bad path class:

- `../distinct5_512_ema_variant_e_latent_prototype_ot_queue.json`

Actual config root:

- `SchrodingerBridge/configs/distinct5_512_ema_variant_e_latent_prototype_ot_queue.json`

Repair:

- the generators now write:
  - `../../distinct5_512_ema_variant_e_latent_prototype_ot_queue.json`
- the three existing split configs were corrected in place

Files repaired:

- [prepare_wikiart_stress_split_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/prepare_wikiart_stress_split_packet.py)
- [build_wikiart_faraday_splits.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/build_wikiart_faraday_splits.py)
- [wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism_variant_f_b44_remote.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/faraday_splits/wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism_variant_f_b44_remote.json)
- [wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism_variant_f_b44_remote.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/faraday_splits/wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism_variant_f_b44_remote.json)
- [wikiart_stress3_Art_Nouveau_Modern__Expressionism__Naive_Art_Primitivism__Romanticism__Symbolism_variant_f_b44_remote.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/faraday_splits/wikiart_stress3_Art_Nouveau_Modern__Expressionism__Naive_Art_Primitivism__Romanticism__Symbolism_variant_f_b44_remote.json)

## Launch command

```powershell
python SchrodingerBridge\tools\experiments\launch_remote_aaai2027_packet.py `
  --config SchrodingerBridge/configs/aaai2027/faraday_splits/wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism_variant_f_b44_remote.json
```

## Remote contract

Task name:

- `SB-AAAI2027_wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism_variant`

Remote output root:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism_variant_f_b44_remote`

Remote train log:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism_variant_f_b44_remote/remote_train.log`

Launcher log:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/_codex_tmp/SB-AAAI2027_wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism_variant.launcher.log`

## First-health readout

Observed immediately after launch:

- remote GPU prelaunch:
  - `268 MiB`
- remote first-health / immediate recheck:
  - about `8590 MiB / 12288 MiB`
  - about `95%` GPU util
  - about `151.76 W`

The packet therefore sits in the expected formal memory band and remains safely
below the hard `< 11.0 GiB` contract.

## Log proof of healthy start

Confirmed from `remote_train.log`:

- latent manifest cache loaded successfully
- all `5` packed latent caches loaded
- pairing cache loaded with `20000` source-target routes
- tokenizer init succeeded
- model params:
  - `6,092,023`
- training entered:
  - `Epoch 1/3`

## Current status

Current status at launch note time:

- `running`

This note is a launch/status artifact only. It is **not** yet a paper-facing
result packet. Do not write any split1 performance claim until the run retains:

- `epoch_0001..0003.pt`
- `remote_train.log`
- `full_eval/epoch_0001..0003/summary.json`
- and any later split-level IDT comparison packet required for the stress-split
  claim
