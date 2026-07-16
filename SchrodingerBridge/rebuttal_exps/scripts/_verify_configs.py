"""Verify D3/D4/D5 configs load correctly."""
import sys
sys.path.insert(0, ".")
from config_schema import load_experiment_config

for name, path in [
    ("D3", "configs/rebuttal_D3_wll_1p0.json"),
    ("D4", "configs/rebuttal_D4_direct_target.json"),
    ("D5", "configs/rebuttal_D5_hh_head.json"),
]:
    try:
        c = load_experiment_config(path)
        print(f"{name}: OK")
        print(f"  spectral_w_ll={c.bridge.spectral_w_ll}")
        print(f"  spectral_w_hh={getattr(c.bridge, 'spectral_w_hh', 'N/A')}")
        print(f"  structure_aligned_target={c.bridge.structure_aligned_target}")
        print(f"  ll_partial_style_enabled={c.bridge.ll_partial_style_enabled}")
        print(f"  ll_partial_alpha={c.bridge.ll_partial_alpha}")
        print(f"  enable_hh_head={getattr(c.model, 'enable_hh_head', 'N/A')}")
        print(f"  seed={c.training.seed}")
        print(f"  num_epochs={c.training.num_epochs}")
        print(f"  batch_size={c.training.batch_size}")
        print(f"  internal_early_stop_enabled={c.training.internal_early_stop_enabled}")
        print(f"  save_dir={c.checkpoint.save_dir}")
        print(f"  data_root={c.data.data_root}")
        print()
    except Exception as e:
        print(f"{name}: FAILED - {e}")
        import traceback
        traceback.print_exc()
