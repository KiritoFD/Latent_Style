import json, os

ROOT = "/mnt/i/Github/Latent_Style/SchrodingerBridge"
BATCH = "exp/20250618_lite_ot_vertical"  # 用固定的名字, 不用时间戳
BASE_CFG = "/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/config.json"

os.makedirs(f"{ROOT}/{BATCH}", exist_ok=True)

base = json.load(open(BASE_CFG))

exps = {
    "h0_vertical_fm": {
        "bridge.bridge_path_mode": "vertical",
        "bridge.coupling_cost_composition": "structure_only",
        "bridge.coupling_structure_cost_mode": "self_affinity_gw",
        "bridge.bridge_sigma": 0.0,
    },
    "h1_linear_fm": {
        "bridge.bridge_path_mode": "linear",
        "bridge.coupling_cost_composition": "structure_only",
        "bridge.coupling_structure_cost_mode": "self_affinity_gw",
        "bridge.bridge_sigma": 0.0,
    },
    "h2_euclidean_ot": {
        "bridge.bridge_path_mode": "vertical",
        "bridge.coupling_cost_composition": "appearance_only",
        "bridge.bridge_sigma": 0.0,
    },
    "h3_sde_noise": {
        "bridge.bridge_path_mode": "vertical",
        "bridge.coupling_cost_composition": "structure_only",
        "bridge.coupling_structure_cost_mode": "self_affinity_gw",
        "bridge.bridge_sigma": 0.02,
        "bridge.bridge_noise_schedule": "exact_brownian",
    },
    "h4_unbalanced_ot": {
        "bridge.bridge_path_mode": "vertical",
        "bridge.coupling_cost_composition": "structure_only",
        "bridge.coupling_structure_cost_mode": "self_affinity_gw",
        "bridge.coupling_solver": "sinkhorn_unbalanced",
        "bridge.sinkhorn_unbalanced_tau_src": 0.5,
        "bridge.bridge_sigma": 0.0,
    },
    "h5_topogate_attention": {
        "bridge.bridge_path_mode": "vertical",
        "bridge.coupling_cost_composition": "appearance_plus_structure",
        "bridge.coupling_structure_cost_mode": "topogate_attention_gw",
        "bridge.coupling_structure_cost_weight": 0.4,
        "bridge.bridge_sigma": 0.0,
    },
    "h6_combined_topogate": {
        "bridge.bridge_path_mode": "vertical",
        "bridge.coupling_solver": "sinkhorn_unbalanced",
        "bridge.sinkhorn_unbalanced_tau_src": 0.5,
        "bridge.coupling_cost_composition": "appearance_plus_structure",
        "bridge.coupling_structure_cost_mode": "topogate_attention_gw",
        "bridge.coupling_structure_cost_weight": 0.4,
        "bridge.bridge_sigma": 0.02,
        "bridge.bridge_noise_schedule": "exact_brownian",
    },
}

for name, overrides in exps.items():
    d = f"{ROOT}/{BATCH}/{name}"
    os.makedirs(d, exist_ok=True)

    c = json.loads(json.dumps(base))  # deep copy

    # --- 核心改动: 回退 tokenizer, 其他不变 ---
    c["model"]["tokenizer_family"] = "legacy_factorized"
    c["model"]["style_tokenizer"] = "factorized"
    c["model"]["semantic_self_topology_gate"] = True
    c["model"]["semantic_self_topology_blend"] = 1.0
    c["data"]["pairing_cache_path"] = ""
    c["data"]["virtual_length_multiplier"] = 0.1
    c["training"]["resume_checkpoint"] = ""
    c["training"]["resume_optimizer"] = False
    c["training"]["resume_training_state"] = False
    c["training"]["resume_prefer_local_checkpoint"] = False

    # --- 训练参数 ---
    c["training"]["num_epochs"] = 60
    c["training"]["save_interval"] = 1
    c["training"]["batch_size"] = 20
    c["training"]["virtual_length_multiplier"] = 1.0    # b40 够大了,不用 vl
    c["training"]["full_eval_each_epoch"] = True
    c["training"]["full_eval_defer_until_training_end"] = False
    c["training"]["full_eval_only_lpips_clip_style"] = True
    c["training"]["full_eval_transfer_only"] = True
    c["training"]["full_eval_stop_on_convergence"] = True
    c["training"]["full_eval_convergence_patience"] = 4
    c["training"]["full_eval_convergence_min_epochs"] = 4
    c["training"]["full_eval_output_subdir"] = "full_eval_transfer"
    c["checkpoint"]["save_dir"] = d

    # --- bridge 覆盖 ---
    for k, v in overrides.items():
        parts = k.split(".")
        target = c
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = v

    json.dump(c, open(f"{d}/config.json", "w"), indent=2)
    print(f"  {name}")

print(f"\nDone. {len(exps)} experiments in {BATCH}")
print("tokenizer=legacy_factorized, batch=20, topogate=on, train_from_scratch")
