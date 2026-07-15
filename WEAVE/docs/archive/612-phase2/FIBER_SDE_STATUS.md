# Fiber-SDE Experiment Status & Next Steps

## Current Status (June 15, 2026 02:43)

**Running Experiment**: Fiber-SDE sigma=0.02 evaluation on topogate e2 checkpoint
- Started: 02:33
- Runtime: ~10 minutes so far
- GPU: 100% utilized, 12GB VRAM
- Process PID: 432
- Config: `/mnt/i/Github/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_fiber_sde_complete_0p02.json`
- Checkpoint: topogate e2 (baseline 0.671/0.314)

**Theory Being Tested**: Fiber-aligned SDE (noise × TopoGate) breaks ODE mean collapse
- Expected: style 0.69-0.72, LPIPS 0.32-0.35
- Control: isotropic noise should degrade LPIPS

## Historical Context

### Baseline Performance
- **topogate e2**: transfer 0.671/0.314, all-pairs 0.703/0.312 (best ODE result)
- **SMoE translator e8**: 0.670/0.318 (almost identical - translation doesn't fix mean collapse)
- **Diagnosis**: Both deterministic ODE methods converge to fiber mean → smooth "plastic" outputs

### Theory Foundation
From `docs/612-phase2/FIBER_BUNDLE_DESIGN.md`:
1. **ODE Mean Collapse Theorem**: deterministic ODE on fiber → E[X|c] (average of all possible styles)
2. **Fiber-SDE Solution**: noise × TopoGate injects randomness only in texture (fiber directions), not structure (base manifold)
3. **Ehresmann Connection**: TopoGate = connection that locks base manifold, allows fiber movement

### Code Implementation (Already Done by KiritoFD)
- `src/model.py` (+67 lines): Fiber-SDE solver with `solver_fiber_aligned` flag
- `src/semantic_tokenizer.py` (+174 lines): SMoE Translator with identity initialization
- `src/losses.py` (+81 lines): Fiberwise SWD (per-cluster distribution matching)
- 10 scan configs created: 5 fiber-aligned + 5 isotropic control

## Next Steps

### 1. Monitor Current Run (~5-10 more minutes expected)
```bash
# Check if finished
ssh -p 2222 administrator@100.115.18.62 "wsl -d Ubuntu-26.04 ps aux | grep python.*run.py"

# Once done, find experiment dir
ssh -p 2222 administrator@100.115.18.62 "wsl -d Ubuntu-26.04 find /mnt/i/Github/Latent_Style/exp -maxdepth 1 -name '*fiber*' -o -name '*0p02*' -type d"

# Read results
ssh -p 2222 administrator@100.115.18.62 "wsl -d Ubuntu-26.04 cat /mnt/i/Github/Latent_Style/exp/[EXP_DIR]/full_eval/clip_lpips_curve.csv"
```

### 2. Interpret Results

**If style > 0.71 and LPIPS < 0.35**: ✅ Theory validated!
- Fiber-SDE broke the mean collapse
- Next: Full Fiber-SDE scan (sigma 0.005, 0.01, 0.02, 0.03, 0.05)
- Find optimal sigma, then train SMoE+Fiber-SDE from scratch

**If style still ~0.67**: ❌ Fiber-SDE alone insufficient
- May need: SMoE + Fiber-SDE + Fiberwise SWD combined training
- Or: I2SB (endpoint prediction) with fiber alignment
- Or: Deeper issue - tokenizer capacity, TopoGate too restrictive

### 3. Full Scan (If Theory Validates)
Launch all 10 configs sequentially:
```bash
# Script already prepared: tools/launch_fiber_sde_scan.sh
# Or launch individually via phase2_fiber_sde_fiber_sigma0p*.json
```

### 4. Training Phase (If Scan Shows Promise)
If any sigma achieves style>0.72:
1. **Retrain SMoE from scratch** with best fiber-SDE settings (config: `phase2_smoe_fiberwise_swd_seed42_b12a1.template.json`)
2. Train for 12-16 epochs (SMoE showed convergence at e8-e10)
3. Goal: **style > 0.73, LPIPS < 0.30**

## Backup Plans

### Plan B: I2SB + Fiber Alignment
If SDE doesn't break ceiling, try I2SB (endpoint prediction) which has stronger stochasticity:
- Config exists: `phase2_i2sb_topo_anchor_sigma0p10_warm_vel2_seed42_b30a1.json`
- Modify to add `solver_fiber_aligned: true`

### Plan C: Adaptive Kinetic Schedule
Current w_kinetic=0.85 fixed. Try schedule:
- Epochs 1-4: w_kinetic=1.0 (learn structure)
- Epochs 5-8: w_kinetic=0.6
- Epochs 9+: w_kinetic=0.3 (release style)

### Plan D: Multi-scale Topogate
Different blend per resolution scale:
- 8×8: blend=1.0 (lock global structure)
- 16×16: blend=0.8
- 32×32: blend=0.5
- 64×64: blend=0.3 (release local textures)

## Key Files
- **Theory doc**: `docs/612-phase2/FIBER_BUNDLE_DESIGN.md`
- **Experiment log**: `docs/612-lookback/all_experiments.csv`
- **Current running config**: `configs/aaai2027/phase2_fiber_sde_complete_0p02.json`
- **Scan configs**: `configs/aaai2027/phase2_fiber_sde_{fiber,iso}_sigma0p*.json`

## Critical Insight
The mathematical essence: **style transfer is transport on a fiber bundle where TopoGate defines the connection**. ODE follows mean fiber trajectory → blur. SDE samples fiber distribution → sharp strokes. The question is whether our SDE formulation (noise × gate) is strong enough to escape the mean attractor.
