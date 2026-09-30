import json

# 0.74/0.29 的配置 (20 styles)
with open(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\630_random20_heun_5ep\config.json", "r", encoding="utf-8") as f:
    cfg_target = json.load(f)

# 当前 T11 配置 (5 styles, samam dataset)
with open(r"G:\GitHub\Latent_Style\SchrodingerBridge\configs\t11e_l_samam_repro.json", "r", encoding="utf-8") as f:
    cfg_t11 = json.load(f)

# 加载 t11 基础配置链
def load_chain(cfg, base_dir):
    result = {}
    while "_base" in cfg:
        base_path = cfg["_base"]
        if not base_path.endswith(".json"):
            base_path += ".json"
        import os
        full = os.path.join(base_dir, base_path)
        with open(full, "r", encoding="utf-8") as f:
            base_cfg = json.load(f)
        for k, v in cfg.items():
            if k != "_base":
                result[k] = v
        cfg = base_cfg
    for k, v in cfg.items():
        if k not in result:
            result[k] = v
    return result

# t11 完整配置
cfg_t11_full = load_chain(cfg_t11, r"G:\GitHub\Latent_Style\SchrodingerBridge\configs")

# 关键模型参数差异
keys = [
    "num_styles", "style_dim", "style_extrap_alpha", "endpoint_adain_scale",
    "dwt_route_train_prob", "cross_attn_dwt_route", "scale_ll", "scale_lh",
    "scale_hl", "scale_hh", "spectral_ode_enabled", "spectral_ode_levels",
    "endpoint_adain_mode", "style_mask_ratio", "style_mask_mode",
    "endpoint_lowpass_levels", "solver_rk_order", "solver_corrector_steps",
    "solver_corrector_mode", "solver_stochastic_noise_scale",
    "i2sb_fiber_aligned_noise", "i2sb_fiber_noise_rms_normalize",
    "bridge_path_mode", "lowpass_mode",
    "single_step_swd_weight", "terminal_swd_weight", "single_step_edge_weight",
    "semantic_supervision_family",
]

print("=" * 100)
print(f"{'key':40s} {'target(20s)':25s} {'t11(5s)':25s}")
print("=" * 100)
for k in keys:
    tv = cfg_target.get("model", {}).get(k, "MISSING")
    t11v = cfg_t11_full.get("model", {}).get(k, "MISSING")
    flag = "  <-- DIFF" if str(tv) != str(t11v) else ""
    print(f"{k:40s} {str(tv):25s} {str(t11v):25s}{flag}")

# training
print("\nTraining:")
t_train = cfg_target.get("training", {})
t11_train = cfg_t11_full.get("training", {})
for k in ["batch_size", "num_epochs", "patience", "cudnn_benchmark", "full_eval_each_epoch"]:
    tv = t_train.get(k, "MISSING")
    t11v = t11_train.get(k, "MISSING")
    flag = "  <-- DIFF" if str(tv) != str(t11v) else ""
    print(f"{k:40s} {str(tv):25s} {str(t11v):25s}{flag}")

# data
print("\nData:")
t_data = cfg_target.get("data", {})
t11_data = cfg_t11_full.get("data", {})
for k in ["data_root", "latent_cache_dir", "pairing_cache_path", "pairing_cache_active_topk", "style_subdirs"]:
    tv = t_data.get(k, "MISSING")
    t11v = t11_data.get(k, "MISSING")
    flag = "  <-- DIFF" if str(tv) != str(t11v) else ""
    print(f"{k:40s} {str(tv)[:60]}")
    print(f"{'':40s} {'':25s} {str(t11v)[:60]}{flag}")
