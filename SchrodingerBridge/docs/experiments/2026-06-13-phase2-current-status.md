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
- Remote GPU: 9934 / 12288 MiB
- Current read: epoch_0004 settled in-band at transfer 0.672377/0.369065 and all-pairs 0.700490/0.367229; new low-LPIPS Pareto point, but style still sits just below the old safe shelf, so the line remains alive under regular Pareto patience rather than the original epoch_0003 short-screen

### Latest Settled Point
- Epoch: `epoch_0004`
- Transfer `CLIP-S / LPIPS`: `0.672377 / 0.369065`
- All-pairs `CLIP-S / LPIPS`: `0.700490 / 0.367229`
- Identity `CLIP-S / LPIPS`: `0.812943 / 0.359885`
- Eval timing: wall `222.26s`, eval `33.49s`, generation `118.32s`, decode `58.00s`

### Recovery Gate
- Min settled epoch: `3`
- All-pairs target: style `>= 0.701666`, LPIPS `<= 0.381724`
- Transfer target: style `>= 0.673934`, LPIPS `<= 0.384340`
- Latest all-pairs read: style short by 0.001176, LPIPS margin +0.014495
- Latest transfer read: style short by 0.001557, LPIPS margin +0.015275

### Best Settled Points In This Run
- Best transfer epoch: `epoch_0002` with `0.675645 / 0.395898`
- Best transfer gate read: not eligible before settled epoch 3
- Best all-pairs epoch: `epoch_0002` with `0.702225 / 0.393204`
- Best all-pairs gate read: not eligible before settled epoch 3

## Next Packets
- Structure-side preferred packet: `vel_tok32_semantic_topogate_k085`
- Structure config: [structure config](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_semantic_topogate_k085_seed42_b20a1.json)
- Structure note: [structure note](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-semantic-topogate-k085.md)
- Structure read: preferred structure-side velocity candidate; smoke proves topology gate and tokenizer observability are active
- I2SB diagnostic preferred packet: `i2sb_tok32_semantic_topogate_sigma0p02_residual`
- I2SB config: [I2SB config](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_tok32_semantic_topogate_sigma0p02_residual_seed42_b20a1.json)
- I2SB note: [I2SB note](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-i2sb-tok32-semantic-topogate-sigma0p02-residual.md)
- I2SB read: preferred current exact-Brownian theory-check packet; combines residual I2SB, topology gate, and refreshed tok32 tokenizer

## Contract Read
- `true I2SB` is already implemented as exact-Brownian endpoint transport with `solver_i2sb`.
- `true tokenizer` is already implemented as `pure_latent_spatial` with a null legacy tokenizer shell and structured runtime path.
- The current formal lane remains on `velocity + pure_latent_spatial` because the exact-I2SB line has not returned to the documented `LPIPS < 0.40` band.

## Remote Host Read
- SSH ok: `True`
- WSL exec ok: `True`
- HCS failure: `False`
- Hypervisor launch type: `Auto`
