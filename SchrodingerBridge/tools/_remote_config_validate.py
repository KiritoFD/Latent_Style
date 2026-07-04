"""Quick config validation - load each config and print key params."""
import sys
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge\src")
from config_schema import load_experiment_config

CONFIGS = [
    "630_remote_base_5style.json",
    "630_remote_a1_dwt_strong_style.json",
    "630_remote_a2_cosine_heun_dwt_balanced.json",
    "630_remote_p1_spectral_rebalance.json",
    "630_remote_t1_lowfreq_style.json",
    "630_remote_p2_swd_flow_balance.json",
]

BASE = r"I:\Github\Latent_Style\SchrodingerBridge\configs"

for name in CONFIGS:
    path = f"{BASE}\\{name}"
    try:
        cfg = load_experiment_config(path)
        m = cfg.model
        b = cfg.bridge
        t = cfg.training
        d = cfg.data
        print(f"OK  {name}")
        print(f"    num_styles={m.num_styles}  dwt_route={m.cross_attn_dwt_route}  solver={m.solver_type}  schedule={m.time_schedule}")
        print(f"    alpha={m.style_extrap_alpha}  adain_ll/lh/hl/hh={m.endpoint_adain_scale_ll}/{m.endpoint_adain_scale_lh}/{m.endpoint_adain_scale_hl}/{m.endpoint_adain_scale_hh}")
        print(f"    spectral_w_ll/lh/hl/hh={b.spectral_w_ll}/{b.spectral_w_lh}/{b.spectral_w_hl}/{b.spectral_w_hh}")
        print(f"    w_endpoint_style={b.w_endpoint_style}  terminal_swd={b.terminal_swd_weight}  w_flow={b.w_flow}")
        print(f"    ep_style_lh/hl={b.spectral_w_endpoint_style_lh}/{b.spectral_w_endpoint_style_hh}")
        print(f"    freeze={t.freeze_mode}  epochs={t.num_epochs}  patience={t.patience}  bs={t.batch_size}")
        print(f"    data_root={d.data_root}")
        print(f"    save_dir={cfg.checkpoint.save_dir}")
        print()
    except Exception as e:
        print(f"FAIL  {name}: {e}")
        import traceback
        traceback.print_exc()
        print()

print("=== VALIDATION DONE ===")
