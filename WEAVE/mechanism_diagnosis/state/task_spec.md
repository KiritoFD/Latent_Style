# Mechanism Diagnosis Task Spec

## Goal
Provide坚实实验论据 for the paper's core theoretical claim:
"Identity shortcut is **caused by** frequency dominance (not merely correlated)."

## Background
Current evidence is correlational only (LL energy 69%, ablations).
Reviewer risk: "plausible but largely empirical" → rejection.
Need causal/mechanistic experiments, not more performance tables.

## Milestones (sequential)

### M1 (P0): Gradient Spectrum Analysis
- **Claim tested**: Low-frequency gradients dominate training.
- **Method**: Hook velocity-head gradients during training; apply Haar DWT to gradient tensors; measure LL/LH/HL/HH energy ratios at epochs 0/1/3/5.
- **Success criterion**: LL gradient share >60% across all stages; HH <15%. If style-conditioned gradient << content gradient, direct causal evidence.
- **Deliverable**: `fig_gradient_spectrum.pdf` + data table.

### M2 (P1): Per-step AdaIN-only vs Flow-only vs Full
- **Claim tested**: Flow Matching and Endpoint AdaIN are complementary, not redundant.
- **Method**: Inference-time ablation: (A) per-step AdaIN with z_t=z_0 (no transport); (B) Flow-only (wo_endpoint_adain, already have); (C) Full.
- **Success criterion**: A << C on CLIP-S (proves transport needed); A > Latent-WCT (proves per-step > single-shot).
- **Deliverable**: 3-row comparison table + interpretation.

### M3 (P2): HH-variance / CLIP-S Co-evolution Curve
- **Claim tested**: High-frequency subbands are the style carrier.
- **Method**: For each epoch checkpoint, measure output latent's HH subband variance on fixed test set; correlate with CLIP-S.
- **Success criterion**: Pearson r > 0.85 between HH_var and CLIP-S across epochs.
- **Deliverable**: `fig_hh_style_sync.pdf` dual-axis curve.

### M4 (P3): Why AdaIN? (Style Injection Module Comparison)
- **Claim tested**: AdaIN is the optimal style injection for this architecture, not just a legacy default.
- **Method**: Train 3 variants (5 epochs each): FiLM (learned γ,β), per-step WCT (full covariance), Cross-Attention-only. Compare to AdaIN baseline.
- **Success criterion**: AdaIN is Pareto-optimal (or within 0.005). If another module wins, honestly report and discuss.
- **Deliverable**: 4-row injection-comparison table.

## Constraints
- All experiments on RTX 3060 (12GB) remote, VRAM ≤ 11.2GB training / ≤ 7GB eval.
- D5-512 dataset, 5 epochs, batch_size=24 for training variants.
- Reuse existing checkpoints where possible (no redundant training).
- Remote: ssh -p 2222 administrator@100.115.18.62, code at I:\Github\Latent_Style\SchrodingerBridge
- Zero interaction with user during run; log decisions to logs/work.jsonl.

## Stopping Criteria
- All 4 milestones completed with deliverables, OR
- M1 fails to show LL dominance → theory may be wrong, stop and report honestly.
