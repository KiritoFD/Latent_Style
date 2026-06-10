Page-1 figure evidence bundle for the AAAI-27 draft.

Contents:

- `page1_panel_points.csv`
  - unified paper-facing table for Figure 1
  - columns:
    - `clip_style_tr`
    - `one_minus_lpips`
    - `delta_idt_tr`
    - `train_time_label`
    - `tw_artfid_all`
    - `tw_artfid_transfer`
- `artfid_rerun_local_20260609/`
  - current trusted local target-pooled ArtFID reruns for all page-1 methods
  - all reruns use `tools/compute_targetwise_artfid_fast.py`
- `artfid_rerun_20260609/`
  - older seed rerun directory kept only as a path manifest source
  - do not use it as the current paper-facing numeric source
- `samam_2250_full/`
- `latent_samam_step1500_full/`
- `latent_samst_batch1050_full/`
  - full local copies of remote eval directories pulled back for the page-1 rerun

Current page-1 methods:

- `IDT`
- `SaMAM-2250`
- `Lat SaMAM`
- `LBM-K`
- `LBM-Knee`
- `SaMST e15`
- `Lat SaMST`
- `Seedream-4.5`
- `LBM-PS-v2`

Current Figure 1 policy:

- left panel:
  - use `delta_idt_tr` vs `one_minus_lpips`
- right panel:
  - use `tw_artfid_all`
  - show `train_time_label` inside each bar
- do not mix old `comparison_20260602` numbers with the rerun numbers in this bundle

Notes:

- `Lat SaMST` training time is currently carried as `~35m`
  - this is a bounded approximation from the retained convergence packet timing
  - it should be replaced if a more direct authoritative train log is recovered
- `Lat SaMAM` training time is taken from the convergence closure note as `140.6m`
- all current ArtFID values in this bundle are taken from `artfid_rerun_local_20260609/` and supersede the older homepage bar values
