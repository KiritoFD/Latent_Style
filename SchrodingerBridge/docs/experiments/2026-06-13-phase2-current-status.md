# Phase 2 Current Status

Date: 2026-06-13

## Sources
- Queue manifest: [phase2_queue_manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_queue_manifest.csv)
- Validation snapshot: [phase2_queue_manifest_validation.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_queue_manifest_validation.json)
- State snapshot: [phase2_queue_state_snapshot.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_queue_state_snapshot.json)

## Formal Lane
- Preferred packet: `vel_tok32_safe_rescan_r2`
- Status: `closed_plateau`
- Run: `aaai2027_phase2_vel_tok32_safe_rescan_r2_seed42_b20a1`
- Config: [formal config](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json)
- Note: [formal note](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-rescan-r2.md)
- Live state: `settled_no_live_process`
- Remote GPU: n/a
- Current read: latest settled authority point is now epoch_0008 at transfer 0.672774/0.389067 and all-pairs 0.700669/0.384913; still in-band, but transfer style is short by 0.001160 and all-pairs style is short by 0.000997 against the formal recovery shelf

### Latest Settled Point
- Epoch: `epoch_0008`
- Transfer `CLIP-S / LPIPS`: `0.672774 / 0.389067`
- Transfer `style - IDT`: `+0.032852`
- All-pairs `CLIP-S / LPIPS`: `0.700669 / 0.384913`
- All-pairs `style - IDT`: `+0.020544`
- Identity `CLIP-S / LPIPS`: `0.812248 / 0.368297`
- Eval timing: wall `238.05s`, eval `34.01s`, generation `132.55s`, decode `58.61s`

### Recovery Gate
- Min settled epoch: `3`
- All-pairs target: style `>= 0.701666`, LPIPS `<= 0.381724`
- Transfer target: style `>= 0.673934`, LPIPS `<= 0.384340`
- Latest all-pairs read: style short by 0.000997, LPIPS over by 0.003189
- Latest transfer read: style short by 0.001160, LPIPS over by 0.004727

### Best Settled Points In This Run
- Best transfer epoch: `epoch_0002` with `0.675645 / 0.395898`
- Best transfer gate read: not eligible before settled epoch 3
- Best all-pairs epoch: `epoch_0002` with `0.702225 / 0.393204`
- Best all-pairs gate read: not eligible before settled epoch 3

## Next Packets
- Structure-side preferred packet: `vel_tok32_safe_semantic_topogate_k085_appalign`
- Structure config: [structure config](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1.json)
- Structure note: [structure note](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085-appalign.md)
- Structure read: epoch_0001 settled at transfer 0.672604/0.336357 and all-pairs 0.703506/0.332992; all-pairs safe-shelf recovery is already achieved, while transfer style still trails the formal shelf
- Structure live state: `eval_in_progress_or_pending`
- Structure GPU: 2444 / 12288 MiB
- Structure latest settled epoch: `epoch_0001`
- Structure latest settled `CLIP-S / LPIPS`: `0.672604 / 0.336357`, `0.703506 / 0.332992`
- Structure latest `style - IDT`: transfer `+0.032681`, all-pairs `+0.023381`
- Structure runtime observability: train_ep=2, tok_eff=3.6, gate=0.633, mask=0.696, topo_ent=1.013, app_on=1.0, app_s=1.000, app_d=0.000, bex=0, sigma=0.000
- I2SB diagnostic preferred packet: `i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_tfloor005`
- I2SB config: [I2SB config](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_tfloor005_seed42_b20a1.json)
- I2SB note: [I2SB note](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-i2sb-tok32-safe-semantic-topogate-sigma0p02-residual-tfloor005.md)
- I2SB read: early-time aligned exact-I2SB follow-on: keep the same safe parent and residual sigma0.02 Brownian bridge
- I2SB live state: `n/a`
- I2SB runtime observability: n/a

## Contract Read
- `true I2SB` is already implemented as exact-Brownian endpoint transport with `solver_i2sb`.
- `true tokenizer` is already implemented as `pure_latent_spatial` with a null legacy tokenizer shell and structured runtime path.
- The current formal lane remains on `velocity + pure_latent_spatial` because the exact-I2SB line has not returned to the documented `LPIPS < 0.40` band.

## Remote Host Read
- SSH ok: `True`
- WSL exec ok: `True`
- HCS failure: `False`
- Hypervisor launch type: `Auto`

## Local Watchers
- Active phase2 handoff watchers: `1`
- Structure watcher stdout: [phase2_structure_reentry_watch.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_structure_reentry_watch.stdout.log)
