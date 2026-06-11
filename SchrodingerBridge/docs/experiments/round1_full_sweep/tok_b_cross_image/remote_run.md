# tok_b_cross_image Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round1_tok_b_cross_image_seed42_b8a2`

- new-data retry on `2026-06-12`:
  - matching `wikiarts_5_full_notest` DINO cache is now available
  - first strict tokenizer-tail retry:
    - `batch=8`
    - entered a real formal lane
    - representative early live read about `9793MiB`
  - later read:
    - the same `epoch_0001` run drifted down to about `8279MiB`
    - and was killed by the strict under-band guard before the first retained checkpoint landed
- current conclusion:
  - keep `tok_b_cross_image` in `recalibration_needed`
  - next useful retry is `batch=9` under the same strict contract

- reconstruction-pretrain incident on `2026-06-12`:
  - active run:
    - `aaai2027_round1_tok_b_cross_image_reconpretrain_seed42_b8a2`
  - observed progress before failure:
    - reached late `epoch_0002`
    - train log reached about `step 1751/1889`
    - retained checkpoints now present:
      - `epoch_0001.pt`
      - `epoch_0002.pt`
  - interruption:
    - training process disappeared before `epoch_0002` completed
    - tail ends with a Python fatal block including `lost sys.stderr`
  - root-cause audit:
    - remote `I:` drive free space was measured as `0`
    - the authoritative remote workspace manifest on `I:` was found as a zero-length file
    - remote fast-eval could not create `full_eval_fast_snapshot`
    - a copied manifest on remote `C:` remained intact, confirming the failure is on the `I:` workspace write path rather than the family config itself
  - current decision:
    - downgrade the family back to `recalibration_needed`
    - do not relaunch tokenizer-tail training or fast-eval until remote disk space is freed and the workspace write path is healthy again
  - recovery work completed after the incident:
    - freed `34G` by deleting `eval_cache/modelscope/stabilityai/stable-diffusion-2-1-base`
    - restored the remote workspace manifest from the intact copy on remote `C:`
    - validated one retained fast-eval point:
      - `epoch_0001`
      - `transfer_clip_style = 0.6771`
      - `transfer_lpips = 0.7927`
      - `all_pairs_clip_style = 0.6785`
      - `all_pairs_lpips = 0.7922`
    - validated newest retained checkpoint failure:
      - `epoch_0002.pt` is only `64` bytes
      - remote load fails with `PytorchStreamReader failed reading zip archive`
  - revised decision:
    - keep `tok_b_cross_image` at `recalibration_needed`
    - the blocker is now `corrupted newest retained ckpt after disk-full incident`, not unresolved DINO-cache alignment

- clean restart on `2026-06-12`:
  - new active run:
    - `aaai2027_round1_tok_b_cross_image_reconpretrain_seed42_b8a2_r2`
  - launch contract:
    - matched `wikiarts_5_full_notest` DINO cache override
    - health wait extended to `240s` so DINO-cache load is not mistaken for under-band failure
  - current live read after restart:
    - first formal health sample about `10489 MiB / 12288 MiB`
    - `band_status=in_band`
    - `formal_status=formal_in_band`
    - observed train progress about `epoch 1/8`, `step 313/1889`
    - `loss=16.3870`
    - `tswd=0.1476`
  - fast-eval:
    - remote watcher had to be relaunched once with `--sync-remote-scripts --sync-manifest`
    - watcher is now alive again and waiting for the first retained checkpoint
  - current node:
    - the reconstruction-pretrain program is back to `running`
    - closure authority is still pending first valid `r2` checkpoint eval

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Family id: `tok_b_cross_image`
- Run name: `aaai2027_round1_tok_b_cross_image_seed42_b8a2`
- Remote run dir: `./exp/inmortal-exp/aaai2027_round1_tok_b_cross_image_seed42_b8a2`
- Config: [aaai2027_round1_tok_b_cross_image_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_tok_b_cross_image_seed42_b8a2.json)
- Manifest status: `running`
- Local fast root: [round1_tok_b_cross_image_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_fast_local)
- Local review root: [round1_tok_b_cross_image_localreview](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_localreview)
- Prelaunch switch smoke: `ok`
- Switch smoke artifact: [round1_tok_b_cross_image_switch_smoke_latest.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_tok_b_cross_image_switch_smoke_latest.json)
- Switch smoke row count: `1`
- Tokenizer warmstart config: [aaai2027_round1_tok_b_cross_image_warmstart_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/warmstart/aaai2027_round1_tok_b_cross_image_warmstart_seed42_b8a2.json)
- Tokenizer reconstruction-pretrain config: [aaai2027_round1_tok_b_cross_image_reconpretrain_seed42_b8a2.json](G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/pretrain/aaai2027_round1_tok_b_cross_image_reconpretrain_seed42_b8a2.json)
- Remote GPU live sample:
  - `10877 MiB / 12288 MiB`, `util=89%`
  - `band_status=in_band`
  - `formal_status=formal_in_band`
- Remote train progress:
  - `epoch 1/24`
  - `step 868/2361`
  - `loss=1.9812`
  - `tswd=0.2916`
<!-- ROUND1_AUTO_STATUS:END -->
