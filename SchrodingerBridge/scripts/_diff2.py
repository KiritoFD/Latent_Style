import json
import os

# 直接读取保存的config（实验目录里的config.json是完整的）
with open(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\630_random20_heun_5ep\config.json", "r", encoding="utf-8") as f:
    cfg_target = json.load(f)

with open(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\t11evo\t11e_l_samam_repro\config.json", "r", encoding="utf-8") as f:
    cfg_t11 = json.load(f)

# t11 完整配置 = config.json (保存的已经展开过 _base)
keys = [
    "num_styles", "style_dim", "style_extrap_alpha", "endpoint_adain_scale",
    "dwt_route_train_prob", "cross_attn_dwt_route",
    "spectral_ode_enabled", "spectral_ode_levels",
    "endpoint_adain_mode", "style_mask_ratio", "style_mask_mode",
    "endpoint_lowpass_levels", "solver_rk_order", "solver_corrector_steps",
    "solver_corrector_mode", "solver_stochastic_noise_scale",
    "bridge_path_mode", "lowpass_mode",
    "style_condition_source",
    "affine_connection_gamma_scale", "affine_connection_beta_scale",
    "affine_connection_fiber_mode", "affine_connection_lowpass_kernel",
    "endpoint_head_mode", "transport_prediction_mode",
    "transport_endpoint_scale", "endpoint_parameterization",
]

print("=" * 100)
print(f"{'key':40s} {'target(20s,0.7434)':25s} {'t11(5s,0.7182)':25s}")
print("=" * 100)
for k in keys:
    tv = cfg_target.get("model", {}).get(k, "MISSING")
    t11v = cfg_t11.get("model", {}).get(k, "MISSING")
    flag = "  <-- DIFF" if str(tv) != str(t11v) else ""
    print(f"{k:40s} {str(tv):25s} {str(t11v):25s}{flag}")

print("\nTraining:")
t_train = cfg_target.get("training", {})
t11_train = cfg_t11.get("training", {})
for k in ["batch_size", "num_epochs", "patience", "cudnn_benchmark", "full_eval_each_epoch"]:
    tv = t_train.get(k, "MISSING")
    t11v = t11_train.get(k, "MISSING")
    flag = "  <-- DIFF" if str(tv) != str(t11v) else ""
    print(f"{k:40s} {str(tv):25s} {str(t11v):25s}{flag}")

print("\nData:")
t_data = cfg_target.get("data", {})
t11_data = cfg_t11.get("data", {})
for k in ["data_root", "latent_cache_dir", "pairing_cache_path", "pairing_cache_active_topk", "style_subdirs"]:
    tv = str(t_data.get(k, "MISSING"))[:60]
    t11v = str(t11_data.get(k, "MISSING"))[:60]
    flag = "  <-- DIFF" if tv != t11v else ""
    print(f"{k:40s} target={tv}")
    print(f"{'':40s} t11   ={t11v}{flag}")
