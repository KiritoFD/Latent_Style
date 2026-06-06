# Faraday Split2 F-Family Launch

Date: 2026-06-06

Scope:

- fixed-rule follow-up split:
  - `wikiart_stress2`
- method family:
  - `LBM-F`
- machine:
  - remote `RTX 3060`

## Preconditions

This launch uses the now-closed split2 prep packet:

- [2026-06-06-faraday-split2-prep-launch.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-faraday-split2-prep-launch.md)

Verified remote prep contract before launch:

- `5` styles
- `1000` train latents per style
- packed latent manifest present
- prototype pairing cache present with:
  - `20000` source-target routes

## Launch command

```powershell
python SchrodingerBridge\tools\experiments\launch_remote_aaai2027_packet.py `
  --config SchrodingerBridge/configs/aaai2027/faraday_splits/wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism_variant_f_b44_remote.json
```

## Remote contract

Task name:

- `SB-AAAI2027_wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism_variant_f_`

Remote output root:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism_variant_f_b44_remote`

Remote train log:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism_variant_f_b44_remote/remote_train.log`

## First-health readout

Observed immediately after launch:

- remote GPU:
  - about `8683 MiB / 12288 MiB`
  - about `100%` GPU util
  - about `148.49 W`

This is inside the expected formal train band and below the hard `< 11.0 GiB`
contract.

## Log proof of healthy start

Confirmed in `remote_train.log`:

- latent manifest cache loaded successfully
- all `5` packed latent caches loaded
- pairing cache loaded with `20000` routes
- tokenizer init succeeded
- model params:
  - `6,092,023`
- training entered:
  - `Epoch 1/3`

## Current status

Current status at note time:

- `running`

This note is launch/status only. It is not yet a paper-facing result packet
until retained checkpoints and `full_eval/epoch_0001..0003/summary.json` land.
