# Phase 2 Current Status

Date: 2026-06-13

## Sources
- Queue manifest: [phase2_queue_manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_queue_manifest.csv)
- Validation snapshot: [phase2_queue_manifest_validation.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_queue_manifest_validation.json)
- State snapshot: [phase2_queue_state_snapshot.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_queue_state_snapshot.json)

## Formal Lane
- Preferred packet: `vel_tok32_safe_rescan_r2`
- Status: `running`
- Run: `aaai2027_phase2_vel_tok32_safe_rescan_r2_seed42_b20a1`
- Config: [formal config](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json)
- Note: [formal note](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-rescan-r2.md)
- Live state: `training_after_settled_eval`
- Remote GPU: 10167 / 12288 MiB
- Current read: latest settled authority point is now epoch_0007 at transfer 0.672700/0.384116 and all-pairs 0.700060/0.381072; the line is still formally in-band and still below the old safe shelf on style, so the family remains alive but has not yet produced a promotable shelf break

### Latest Settled Point
- Epoch: `epoch_0007`
- Transfer `CLIP-S / LPIPS`: `0.672700 / 0.384116`
- All-pairs `CLIP-S / LPIPS`: `0.700060 / 0.381072`
- Identity `CLIP-S / LPIPS`: `0.809499 / 0.368896`
- Eval timing: wall `225.78s`, eval `33.74s`, generation `120.97s`, decode `58.43s`

### Recovery Gate
- Min settled epoch: `3`
- All-pairs target: style `>= 0.701666`, LPIPS `<= 0.381724`
- Transfer target: style `>= 0.673934`, LPIPS `<= 0.384340`
- Latest all-pairs read: style short by 0.001606, LPIPS margin +0.000652
- Latest transfer read: style short by 0.001234, LPIPS margin +0.000224

### Best Settled Points In This Run
- Best transfer epoch: `epoch_0002` with `0.675645 / 0.395898`
- Best transfer gate read: not eligible before settled epoch 3
- Best all-pairs epoch: `epoch_0002` with `0.702225 / 0.393204`
- Best all-pairs gate read: not eligible before settled epoch 3

## Next Packets
- Structure-side preferred packet: `vel_tok32_safe_semantic_topogate_k085`
- Structure config: [structure config](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b20a1.json)
- Structure note: [structure note](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085.md)
- Structure read: preferred structure-side successor on the current safe tokenizer profile and the cleaner in-band epoch_0004 parent
- I2SB diagnostic preferred packet: `i2sb_tok32_safe_semantic_topogate_sigma0p02_residual`
- I2SB config: [I2SB config](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_seed42_b20a1.json)
- I2SB note: [I2SB note](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-i2sb-tok32-safe-semantic-topogate-sigma0p02-residual.md)
- I2SB read: preferred current exact-Brownian theory-check successor on the safe tokenizer profile and the cleaner epoch_0004 parent

## Contract Read
- `true I2SB` is already implemented as exact-Brownian endpoint transport with `solver_i2sb`.
- `true tokenizer` is already implemented as `pure_latent_spatial` with a null legacy tokenizer shell and structured runtime path.
- The current formal lane remains on `velocity + pure_latent_spatial` because the exact-I2SB line has not returned to the documented `LPIPS < 0.40` band.

## Remote Host Read
- SSH ok: `True`
- WSL exec ok: `True`
- HCS failure: `False`
- Hypervisor launch type: `Auto`
