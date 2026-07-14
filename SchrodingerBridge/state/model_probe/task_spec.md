# Task: 713 HF-route architecture probe

## Goal
Push the usable DINO-S frontier beyond the current best `target_hf_subband_ft6` (DINO-S≈0.4886) by improving the image-specific target-HF condition route, while keeping content metrics healthy (DINO-C≥0.79, LPIPS≤0.30).

## Background
- The baseline `brk_a_ll03_10ep` already reaches DINO-S≈0.4859.
- The per-subband pooled target-HF residual (`target_hf_subband_ft6`) is the current best usable architecture: DINO-S 0.4886, DINO-C 0.798, LPIPS 0.297.
- Diagnosis: the target image is strong as supervision but weak as a condition. The subband residual is live and diagonal, yet its image-specific component is small and mostly orthogonal to the desired local correction.
- Previous failed directions: raw spatial maps (content collapse), affine scale+shift, WCT-stat direction residual, direct direction auxiliary, residual amplification, time-window gating, cross-orientation mixing, target-current delta, memory dropout, low-rank content basis, deep energy-normalized residual, pure HF-head FiLM.

## Milestones
1. M1: Low-risk capacity extensions of the subband residual (TASM, LDB, ISST, MRSC).
2. M2: Structural re-parameterizations that change how target-HF conditions the residual while preserving the pretrained transport field.
3. M3: Full D5/P2A/R5 rerun for any probe that beats `target_hf_subband_ft6`.
4. M4: Write method note and update main-table evidence if a new best is found.

## Success Criteria
- At least one new architecture exceeds `target_hf_subband_ft6` on all-pairs DINO-S without content collapse (DINO-C<0.78 or LPIPS>0.31).
- Each probe trained to 6-epoch convergence from `brk_a_ll03_10ep`, batch=96, eval AdaIN 1.5, canonical DINOv2-small.
- All findings logged to `state/model_probe/findings.jsonl`.

## Constraints
- Remote RTX 3060 12GB; training VRAM 9–11GB, eval ≤7GB.
- Use `dataset_index.json` path decoupling; configs run on both local and remote via `$index:KEY`.
- No raw target spatial leakage; any spatial mechanism must be coordinate-free or content-gated.
- Preserve LL protection; do not add target-image shortcuts into LL.
