"""Generate ablation_v2 configs — only parameters consumed by clean code.

Verified consumed parameters:
- flow.py: spectral_w_ll, spectral_w_lh, spectral_w_hl, spectral_w_hh, bridge_sigma, loss_type
- model.py: style_cross_attn_gate_init, endpoint_adain_scale, style_extrap_alpha, endpoint_adain_mode
- trainer.py: learning_rate

Dead parameters (NOT consumed, excluded from ablation):
- w_endpoint_style, w_endpoint_content, single_step_swd_weight, w_flow_scale
"""
import json, os

BASE_DIR = r"g:\GitHub\Latent_Style\SchrodingerBridge\configs\ablation_v2"
os.makedirs(BASE_DIR, exist_ok=True)

# Baseline values (from refactor_clean_baseline.json):
# spectral_w_ll=0.3, spectral_w_lh=1.0, spectral_w_hl=1.0, spectral_w_hh=2.0
# bridge_sigma=0.02, loss_type=mse
# style_cross_attn_gate_init=0.05
# endpoint_adain_scale=1.0, style_extrap_alpha=0.1
# learning_rate=2e-4

TRAIN_CONFIGS = [
    # === A. Destructive: component removal ===
    ("a01_wo_endpoint_adain",
     {"model": {"endpoint_adain_scale": 0.0}},
     "Destructive: endpoint_adain_scale=0 (no AdaIN at train+inference)"),
    ("a02_wo_cross_attn",
     {"model": {"style_cross_attn_gate_init": 0.0}},
     "Destructive: gate=0 (cross-attention effectively off)"),
    ("a03_wo_flow",
     {"bridge": {"spectral_w_ll": 0.0, "spectral_w_lh": 0.0, "spectral_w_hl": 0.0, "spectral_w_hh": 0.0}},
     "Destructive: all subband weights=0 (FM loss=0, model does not train)"),

    # === B. Parameter extreme sweeps (training-time) ===
    # spectral_w_ll: baseline=0.3
    ("b01_wll_0",
     {"bridge": {"spectral_w_ll": 0.0}},
     "Extreme: w_ll=0 (no LL de-weighting)"),
    ("b02_wll_20",
     {"bridge": {"spectral_w_ll": 2.0}},
     "Extreme: w_ll=2.0 (6.7x baseline)"),

    # bridge_sigma: baseline=0.02
    ("b03_sigma_0",
     {"bridge": {"bridge_sigma": 0.0}},
     "Extreme: sigma=0 (no noise injection)"),
    ("b04_sigma_02",
     {"bridge": {"bridge_sigma": 0.2}},
     "Extreme: sigma=0.2 (10x baseline noise)"),

    # style_cross_attn_gate_init: baseline=0.05
    ("b05_gate_001",
     {"model": {"style_cross_attn_gate_init": 0.01}},
     "Extreme: gate=0.01 (5x weaker)"),
    ("b06_gate_10",
     {"model": {"style_cross_attn_gate_init": 1.0}},
     "Extreme: gate=1.0 (20x stronger, fully open)"),

    # spectral_w_hh: baseline=2.0
    ("b07_whh_0",
     {"bridge": {"spectral_w_hh": 0.0}},
     "Extreme: w_hh=0 (no HH subband loss)"),
    ("b08_whh_4",
     {"bridge": {"spectral_w_hh": 4.0}},
     "Extreme: w_hh=4.0 (2x baseline)"),

    # learning_rate: baseline=2e-4
    ("b09_lr_5e5",
     {"training": {"learning_rate": 5e-5}},
     "Extreme: lr=5e-5 (4x lower)"),
    ("b10_lr_5e4",
     {"training": {"learning_rate": 5e-4}},
     "Extreme: lr=5e-4 (2.5x higher)"),

    # loss_type: baseline=mse
    ("b11_loss_huber",
     {"bridge": {"loss_type": "huber"}},
     "Alternative: Huber loss instead of MSE"),
]

INFER_CONFIGS = [
    # endpoint_adain_scale: baseline=1.0
    ("d01_adain_0",
     {"model": {"endpoint_adain_scale": 0.0}},
     "Extreme: adain_scale=0 (no AdaIN at inference)"),
    ("d02_adain_05",
     {"model": {"endpoint_adain_scale": 0.5}},
     "adain_scale=0.5 (half strength)"),
    ("d03_adain_20",
     {"model": {"endpoint_adain_scale": 2.0}},
     "Extreme: adain_scale=2.0 (double strength)"),

    # style_extrap_alpha: baseline=0.1
    ("d04_extrap_00",
     {"model": {"style_extrap_alpha": 0.0}},
     "Extreme: extrap_alpha=0 (no extrapolation)"),
    ("d05_extrap_10",
     {"model": {"style_extrap_alpha": 1.0}},
     "Extreme: extrap_alpha=1.0 (10x extrapolation)"),

    # num_steps: baseline=8
    ("d06_steps_1",
     {"full_eval": {"num_steps": 1}},
     "Extreme: 1-step ODE (fastest)"),
    ("d07_steps_32",
     {"full_eval": {"num_steps": 32}},
     "Extreme: 32-step ODE (most accurate)"),
]

for name, override, notes in TRAIN_CONFIGS + INFER_CONFIGS:
    config = {
        "_base": "refactor_clean_baseline.json",
        "checkpoint": {"save_dir": f"I:/Github/Latent_Style/SchrodingerBridge/exp/ablation_v2/{name}"},
        "ablation": {"name": name, "axis": "ablation_v2", "stage": "comprehensive_extreme", "notes": notes}
    }
    config.update(override)
    path = os.path.join(BASE_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Created: {name}.json")

total = len(TRAIN_CONFIGS) + len(INFER_CONFIGS)
print(f"\nTotal: {len(TRAIN_CONFIGS)} training + {len(INFER_CONFIGS)} inference = {total} configs")
