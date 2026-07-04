#!/usr/bin/env python3
"""Generate 256-experiment ablation sweep for 620 Spatial Bridge whitening diagnosis.

Outputs:
  - configs/   One JSON per experiment
  - launch_all.sh   Sequential launcher (runs one at a time, 1 epoch each)
  - matrix.csv   Human-readable experiment matrix
"""
import json
import csv
import os
from pathlib import Path
from itertools import product

# ─── Base config (current best: endpoint_film_hd512) ───
BASE = {
    "model": {
        "contract_family": "620_spatial_bridge",
        "style_condition_source": "target_dino_patches",
        "solver_family": "solver_i2sb",
        "transport_prediction_mode": "velocity",
        "latent_channels": 4,
        "num_styles": 5,
        "base_dim": 64,
        "time_dim": 256,
        "num_res_blocks": 4,
        "style_attn_num_heads": 4,
        "style_attn_num_tokens": 256,
        "style_cross_attn_gate_init": 0.3,
        "tokenizer_dino_dim": 384,
        "style_text_enabled": False,
        "style_film_enabled": True,
        "style_attn_mode": "gated",
        "style_attn_temperature": 1.0,
        "endpoint_head_mode": "endpoint_lowhigh",
        "endpoint_film_enabled": True,
        "endpoint_style_hidden_dim": 512,
        "endpoint_film_init_std": 0.0,
        "endpoint_lowpass_kernel": 5,
        "endpoint_high_scale": 1.0,
        "endpoint_velocity_floor": 0.05,
        "velocity_hf_residual_enabled": False,
        "style_dino_adapter_enabled": False,
        "style_moe_enabled": False,
        "style_query_source": "concat",
        "style_cross_attn_skip_coarse": False,
        "style_attn_topk": 0,
        "style_shortcut_alpha": 1.0,
        "style_local_cnn_enabled": False,
        "input_anchor_noise_std": 0.0,
    },
    "bridge": {
        "bridge_path_mode": "vertical",
        "coupling_solver": "independent",
        "objective_mode": "flow_matching",
        "loss_type": "mse",
        "w_flow": 1.0,
        "single_step_swd_weight": 8.0,
        "single_step_edge_weight": 0.1,
        "bridge_sigma": 0.02,
        "bridge_noise_schedule": "delayed",
        "semantic_swd_num_projections": 64,
        "training_target_projection_kernel": 5,
        "training_target_projection_mode": "source_low_target_high",
        "training_target_projection_low_mode": "target_linear",
        "training_target_projection_low_anchor": 1.0,
        "w_content_lowpass_anchor": 0.0,
        "swd_noise_sigma": 0.02,
        "t_min": 0.0,
        "t_max": 1.0,
        "t_sampling_power": 1.0,
        "source_endpoint_aux_weight": 0.0,
        "endpoint_energy_band_weight": 0.0,
        "w_attn_entropy_reg": 0.0,
    },
    "training": {
        "seed": 42,
        "batch_size": 48,
        "learning_rate": 2e-4,
        "num_epochs": 1,
        "save_interval": 1,
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "pin_memory": True,
        "use_amp": True,
        "amp_dtype": "bf16",
        "channels_last": False,
        "use_gradient_checkpointing": False,
        "full_eval_each_epoch": True,
        "full_eval_defer_until_training_end": False,
        "full_eval_force_regen": True,
        "full_eval_profile_timing": True,
    },
    "data": {
        "data_root": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train",
        "style_subdirs": [
            "Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"
        ],
        "allow_hflip": True,
        "identity_ratio": None,
        "balance_target_styles_per_batch": True,
        "preload_to_gpu": False,
        "virtual_length_multiplier": 0.04,
        "latent_cache_mode": "packed",
        "latent_cache_dir": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed",
        "dino_cache_path": "/mnt/i/Github/Latent_Style/eval_cache/offline_pairing/dinov2_wikiart_distinct5_samam_512_train_cache.pt",
        "dino_cache_required": True,
        "dino_bank_limit_per_style": 8,
        "pairing_cache_path": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/dino_pairing_top8.pt",
        "pairing_cache_topk": 8,
        "pairing_cache_active_topk": 8,
        "pairing_cache_sample_mode": "rank_stratified",
        "pairing_cache_cross_only": True,
    },
    "full_eval": {
        "vae_model": "ema",
        "num_steps": 16,
        "step_size": 1.0,
        "style_strength": 1.0,
        "batch_size": 16,
        "target_chunk_size": 1,
        "vae_decode_batch_size": 16,
        "max_src_samples": 30,
        "max_ref_compare": 30,
        "max_ref_cache": 30,
        "ref_feature_batch_size": 16,
        "image_save_workers": 4,
        "image_save_backend": "pil_png",
        "save_summary_grid": True,
        "only_lpips_clip_style": True,
    },
}

# ─── Ablation dimensions ───
# Each axis: (display_name, config_path, values_list)
# We will do a FACTORIAL sweep of select groups, plus individual sweeps.

# Group A: Core attention & injection (6 modes × 4 gate inits = 24)
AXIS_ATTN_MODE = ("attn_mode", "model.style_attn_mode",
    ["softmax", "gated", "gated_raw", "relu2", "style_select", "sparsemax"])
AXIS_GATE_INIT = ("gate_init", "model.style_cross_attn_gate_init",
    [0.05, 0.1, 0.3, 0.5])

# Group B: Endpoint head (4 modes × 4 hidden_dims = 16)
AXIS_ENDPOINT_MODE = ("ep_mode", "model.endpoint_head_mode",
    ["velocity", "endpoint_lowhigh"])
AXIS_ENDPOINT_FILM = ("ep_film", "model.endpoint_film_enabled",
    [True, False])
AXIS_ENDPOINT_HD = ("ep_hd", "model.endpoint_style_hidden_dim",
    [64, 128, 256, 512])
AXIS_ENDPOINT_FILM_INIT = ("ep_film_init", "model.endpoint_film_init_std",
    [0.0, 0.01, 0.02, 0.05])

# Group C: Style injection in blocks (2×2 = 4)
AXIS_BLOCK_FILM = ("block_film", "model.style_film_enabled",
    [True, False])
AXIS_BLOCK_SHORTCUT = ("block_shortcut", "model.style_shortcut_alpha",
    [0.5, 1.0])

# Group D: Loss weights (3×4×3 = 36)
AXIS_SWD_W = ("swd_w", "bridge.single_step_swd_weight",
    [0.0, 4.0, 8.0, 16.0])
AXIS_EDGE_W = ("edge_w", "bridge.single_step_edge_weight",
    [0.0, 0.1, 0.5])
AXIS_SWD_SIGMA = ("swd_sigma", "bridge.swd_noise_sigma",
    [0.0, 0.02, 0.05])

# Group E: Target projection (3×3×2 = 18)
AXIS_PROJ_MODE = ("proj_mode", "bridge.training_target_projection_mode",
    ["source_low_target_high", "pure_vertical_flow", "pure_vertical_flow_wavelet"])
AXIS_LOW_MODE = ("low_mode", "bridge.training_target_projection_low_mode",
    ["all", "channel_mean", "target_linear"])
AXIS_LOW_ANCHOR = ("low_anchor", "bridge.training_target_projection_low_anchor",
    [0.5, 1.0])

# Group F: DINO / conditioning (3×2×2 = 12)
AXIS_DINO_ADAPTER = ("dino_adapter", "model.style_dino_adapter_enabled",
    [True, False])
AXIS_DINO_MOE = ("dino_moe", "model.style_moe_enabled",
    [True, False])
AXIS_QUERY_SRC = ("query_src", "model.style_query_source",
    ["concat", "sa_out_only"])

# Group G: Network architecture (3×2 = 6)
AXIS_NUM_BLOCKS = ("n_blocks", "model.num_res_blocks",
    [2, 4, 6])
AXIS_BASE_DIM = ("base_dim", "model.base_dim",
    [64, 128])

# Group H: Training hyperparams (3×3×2 = 18)
AXIS_LR = ("lr", "training.learning_rate",
    [1e-4, 2e-4, 5e-4])
AXIS_BATCH = ("batch", "training.batch_size",
    [24, 48])
AXIS_VLEN = ("vlen", "data.virtual_length_multiplier",
    [0.04, 0.2, 1.0])

# Group I: Regularization (3×2×2 = 12)
AXIS_ENTROPY_REG = ("entropy_reg", "bridge.w_attn_entropy_reg",
    [0.0, 0.01, 0.1])
AXIS_ENERGY_BAND = ("energy_band", "bridge.endpoint_energy_band_weight",
    [0.0, 0.1])
AXIS_ANCHOR_NOISE = ("anchor_noise", "model.input_anchor_noise_std",
    [0.0, 0.01])

# Group J: Endpoint details (4×3×2 = 24)
AXIS_EP_HIGH_SCALE = ("ep_high", "model.endpoint_high_scale",
    [0.5, 1.0, 2.0])
AXIS_EP_VEL_FLOOR = ("ep_vfloor", "model.endpoint_velocity_floor",
    [0.0, 0.05, 0.1])
AXIS_EP_LOWPASS = ("ep_lp", "model.endpoint_lowpass_kernel",
    [3, 5, 7])

# Group K: Attention extras (3×3×2 = 18)
AXIS_ATTN_TEMP = ("attn_temp", "model.style_attn_temperature",
    [0.5, 1.0, 2.0])
AXIS_ATTN_TOPK = ("attn_topk", "model.style_attn_topk",
    [0, 4, 16])
AXIS_SKIP_COARSE = ("skip_coarse", "model.style_cross_attn_skip_coarse",
    [True, False])

# Group L: Pairing & data (3×2 = 6)
AXIS_PAIR_TOPK = ("pair_topk", "data.pairing_cache_topk",
    [1, 4, 8])
AXIS_PAIR_CROSS = ("pair_cross", "data.pairing_cache_cross_only",
    [True, False])

# Group M: Bridge solver (3×2×2 = 12)
AXIS_BRIDGE_SIGMA = ("b_sigma", "bridge.bridge_sigma",
    [0.0, 0.02, 0.05])
AXIS_T_POWER = ("t_power", "bridge.t_sampling_power",
    [1.0, 2.0])
AXIS_T_MIN = ("t_min", "bridge.t_min",
    [0.0, 0.1])


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively."""
    result = json.loads(json.dumps(base))
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _set_path(config: dict, path: str, value) -> None:
    """Set a dot-separated path like 'model.style_attn_mode' in config dict."""
    parts = path.split(".")
    obj = config
    for part in parts[:-1]:
        obj = obj.setdefault(part, {})
    obj[parts[-1]] = value


def _make_name(overrides: list[tuple[str, str, any]]) -> str:
    """Build experiment name from overrides."""
    parts = []
    for display, _, val in overrides:
        v = str(val).replace(".", "p").replace(" ", "")
        if isinstance(val, bool):
            v = "T" if val else "F"
        parts.append(f"{display}_{v}")
    return "abl_" + "_".join(parts)


def generate_sweep(axes: list, base: dict, prefix: str = "") -> list[tuple[str, dict, list]]:
    """Generate factorial sweep over given axes."""
    names = [a[0] for a in axes]
    paths = [a[1] for a in axes]
    values = [a[2] for a in axes]
    experiments = []
    for combo in product(*values):
        overrides = list(zip(names, paths, combo))
        cfg = json.loads(json.dumps(base))
        for display, path, val in overrides:
            _set_path(cfg, path, val)
        name = prefix + _make_name(overrides)
        cfg["checkpoint"] = {"save_dir": f"./exp/620_spatial_bridge/{name}"}
        cfg["ablation"] = {
            "name": name,
            "axis": prefix.rstrip("_"),
            "stage": "ablation_1ep",
            "notes": f"1-epoch ablation: {', '.join(f'{n}={v}' for n,_,v in overrides)}"
        }
        experiments.append((name, cfg, overrides))
    return experiments


def main():
    out_dir = Path(__file__).parent / "ablation256"
    out_dir.mkdir(exist_ok=True)
    configs_dir = out_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    all_experiments = []
    seen_names = set()

    # ── Sweep A: Attention × Gate (6×4 = 24) ──
    all_experiments += generate_sweep([AXIS_ATTN_MODE, AXIS_GATE_INIT], BASE, "A_")

    # ── Sweep B: Endpoint structure (2×2×4×4 = 64, but restrict sensible combos) ──
    # ep_mode × ep_film × ep_hd: 2×2×4 = 32 (ep_film only relevant for endpoint_lowhigh)
    # + ep_film_init for film=true cases
    for ep_mode in AXIS_ENDPOINT_MODE[2]:
        for ep_film in AXIS_ENDPOINT_FILM[2]:
            for ep_hd in AXIS_ENDPOINT_HD[2]:
                overrides = [
                    (AXIS_ENDPOINT_MODE[0], AXIS_ENDPOINT_MODE[1], ep_mode),
                    (AXIS_ENDPOINT_FILM[0], AXIS_ENDPOINT_FILM[1], ep_film),
                    (AXIS_ENDPOINT_HD[0], AXIS_ENDPOINT_HD[1], ep_hd),
                ]
                # If film enabled and lowhigh, also sweep init_std
                if ep_film and ep_mode == "endpoint_lowhigh":
                    for init_std in AXIS_ENDPOINT_FILM_INIT[2]:
                        ov2 = overrides + [(AXIS_ENDPOINT_FILM_INIT[0], AXIS_ENDPOINT_FILM_INIT[1], init_std)]
                        cfg = json.loads(json.dumps(BASE))
                        for d, p, v in ov2:
                            _set_path(cfg, p, v)
                        name = "B_" + _make_name(ov2)
                        cfg["checkpoint"] = {"save_dir": f"./exp/620_spatial_bridge/{name}"}
                        cfg["ablation"] = {"name": name, "axis": "B_endpoint", "stage": "ablation_1ep",
                            "notes": f"1-epoch: {', '.join(f'{n}={v}' for n,_,v in ov2)}"}
                        all_experiments.append((name, cfg, ov2))
                else:
                    cfg = json.loads(json.dumps(BASE))
                    for d, p, v in overrides:
                        _set_path(cfg, p, v)
                    name = "B_" + _make_name(overrides)
                    cfg["checkpoint"] = {"save_dir": f"./exp/620_spatial_bridge/{name}"}
                    cfg["ablation"] = {"name": name, "axis": "B_endpoint", "stage": "ablation_1ep",
                        "notes": f"1-epoch: {', '.join(f'{n}={v}' for n,_,v in overrides)}"}
                    all_experiments.append((name, cfg, overrides))

    # ── Sweep C: Block injection (2×2 = 4) ──
    all_experiments += generate_sweep([AXIS_BLOCK_FILM, AXIS_BLOCK_SHORTCUT], BASE, "C_")

    # ── Sweep D: Loss weights (4×3×3 = 36) ──
    all_experiments += generate_sweep([AXIS_SWD_W, AXIS_EDGE_W, AXIS_SWD_SIGMA], BASE, "D_")

    # ── Sweep E: Target projection (3×3×2 = 18) ──
    all_experiments += generate_sweep([AXIS_PROJ_MODE, AXIS_LOW_MODE, AXIS_LOW_ANCHOR], BASE, "E_")

    # ── Sweep F: DINO/conditioning (2×2×2 = 8) ──
    all_experiments += generate_sweep([AXIS_DINO_ADAPTER, AXIS_DINO_MOE, AXIS_QUERY_SRC], BASE, "F_")

    # ── Sweep G: Architecture (3×2 = 6) ──
    # For base_dim=128, reduce batch to fit 12GB VRAM
    for n_blocks in AXIS_NUM_BLOCKS[2]:
        for base_dim in AXIS_BASE_DIM[2]:
            overrides = [
                (AXIS_NUM_BLOCKS[0], AXIS_NUM_BLOCKS[1], n_blocks),
                (AXIS_BASE_DIM[0], AXIS_BASE_DIM[1], base_dim),
            ]
            cfg = json.loads(json.dumps(BASE))
            for d, p, v in overrides:
                _set_path(cfg, p, v)
            if base_dim == 128:
                _set_path(cfg, "training.batch_size", 24)
                _set_path(cfg, "model.style_attn_num_heads", 8)
            name = "G_" + _make_name(overrides)
            cfg["checkpoint"] = {"save_dir": f"./exp/620_spatial_bridge/{name}"}
            cfg["ablation"] = {"name": name, "axis": "G_arch", "stage": "ablation_1ep",
                "notes": f"1-epoch: {', '.join(f'{n}={v}' for n,_,v in overrides)}"}
            all_experiments.append((name, cfg, overrides))

    # ── Sweep H: Training hyperparams (3×2×3 = 18) ──
    all_experiments += generate_sweep([AXIS_LR, AXIS_BATCH, AXIS_VLEN], BASE, "H_")

    # ── Sweep I: Regularization (3×2×2 = 12) ──
    all_experiments += generate_sweep([AXIS_ENTROPY_REG, AXIS_ENERGY_BAND, AXIS_ANCHOR_NOISE], BASE, "I_")

    # ── Sweep J: Endpoint details (3×3×3 = 27) ──
    all_experiments += generate_sweep([AXIS_EP_HIGH_SCALE, AXIS_EP_VEL_FLOOR, AXIS_EP_LOWPASS], BASE, "J_")

    # ── Sweep K: Attention extras (3×3×2 = 18) ──
    all_experiments += generate_sweep([AXIS_ATTN_TEMP, AXIS_ATTN_TOPK, AXIS_SKIP_COARSE], BASE, "K_")

    # ── Sweep L: Pairing & data (3×2 = 6) ──
    all_experiments += generate_sweep([AXIS_PAIR_TOPK, AXIS_PAIR_CROSS], BASE, "L_")

    # ── Sweep M: Bridge solver (3×2×2 = 12) ──
    all_experiments += generate_sweep([AXIS_BRIDGE_SIGMA, AXIS_T_POWER, AXIS_T_MIN], BASE, "M_")

    # ── Deduplicate ──
    final = []
    for name, cfg, overrides in all_experiments:
        if name in seen_names:
            # Add numeric suffix
            i = 2
            while f"{name}_{i}" in seen_names:
                i += 1
            name = f"{name}_{i}"
            cfg["checkpoint"]["save_dir"] = f"./exp/620_spatial_bridge/{name}"
            cfg["ablation"]["name"] = name
        seen_names.add(name)
        final.append((name, cfg, overrides))

    print(f"Total experiments: {len(final)}")

    # ── Write configs ──
    for name, cfg, _ in final:
        path = configs_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)

    # ── Write CSV ──
    with open(out_dir / "matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "name", "group", "overrides"])
        for i, (name, _, overrides) in enumerate(final):
            group = name.split("_")[1] if "_" in name else "?"
            ov_str = "; ".join(f"{n}={v}" for n, _, v in overrides)
            writer.writerow([i, name, group, ov_str])

    # ── Write launch script ──
    SRC_DIR = "/mnt/i/Github/Latent_Style/SchrodingerBridge/src"
    EXP_BASE = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
    CONFIG_BASE = "/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/ablation256/configs"

    with open(out_dir / "launch_all.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# 620 Ablation Sweep: 1-epoch experiments, run sequentially\n")
        f.write(f"# Generated: {len(final)} experiments\n")
        f.write("set -euo pipefail\n\n")
        f.write(f"SRC_DIR=\"{SRC_DIR}\"\n")
        f.write(f"EXP_BASE=\"{EXP_BASE}\"\n")
        f.write(f"CONFIG_BASE=\"{CONFIG_BASE}\"\n\n")
        f.write("cd \"$SRC_DIR\"\n")
        f.write("export PYTHONPATH=\"$SRC_DIR\"\n\n")
        f.write(f"TOTAL={len(final)}\n")
        f.write("COUNT=0\n")
        f.write("FAILED=0\n")
        f.write("FAILED_LIST=\"\"\n\n")
        f.write("run_one() {\n")
        f.write("  local NAME=$1\n")
        f.write("  local CFG=\"$CONFIG_BASE/${NAME}.json\"\n")
        f.write("  local OUTDIR=\"$EXP_BASE/$NAME\"\n")
        f.write("  mkdir -p \"$OUTDIR\"\n")
        f.write("  # Copy config into experiment dir\n")
        f.write("  cp \"$CFG\" \"$OUTDIR/config.json\"\n")
        f.write("  echo \"[$COUNT/$TOTAL] Starting $NAME ...\"\n")
        f.write("  python3 run.py --config \"$OUTDIR/config.json\" > \"$OUTDIR/train.log\" 2>&1\n")
        f.write("  local RC=$?\n")
        f.write("  if [ $RC -ne 0 ]; then\n")
        f.write("    echo \"  FAILED (rc=$RC)\"\n")
        f.write("    FAILED=$((FAILED+1))\n")
        f.write("    FAILED_LIST=\"$FAILED_LIST $NAME\"\n")
        f.write("  else\n")
        f.write("    echo \"  OK\"\n")
        f.write("  fi\n")
        f.write("  COUNT=$((COUNT+1))\n")
        f.write("}\n\n")

        for i, (name, _, _) in enumerate(final):
            f.write(f"run_one \"{name}\"\n")
            # Every 24 experiments, print progress
            if (i + 1) % 24 == 0:
                f.write(f"echo \"=== Progress: {i+1}/{len(final)} ===\"\n")

        f.write("\necho \"\"\n")
        f.write("echo \"==========================================\"\n")
        f.write("echo \"ALL DONE. Total=$TOTAL OK=$((TOTAL-FAILED)) Failed=$FAILED\"\n")
        f.write("if [ $FAILED -gt 0 ]; then\n")
        f.write("  echo \"Failed experiments:$FAILED_LIST\"\n")
        f.write("fi\n")

    # Make script executable
    os.chmod(out_dir / "launch_all.sh", 0o755)

    # ── Write a parallel version (4 at a time, limited by 12GB VRAM = ~2 concurrent) ──
    with open(out_dir / "launch_parallel2.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# 620 Ablation Sweep: run 2 experiments concurrently\n")
        f.write(f"# Generated: {len(final)} experiments, 2 at a time\n")
        f.write("set -euo pipefail\n\n")
        f.write(f"SRC_DIR=\"{SRC_DIR}\"\n")
        f.write(f"EXP_BASE=\"{EXP_BASE}\"\n")
        f.write(f"CONFIG_BASE=\"{CONFIG_BASE}\"\n\n")
        f.write("cd \"$SRC_DIR\"\n")
        f.write("export PYTHONPATH=\"$SRC_DIR\"\n\n")

        # Group by VRAM estimate: base_dim=128 needs ~8GB, base_dim=64 needs ~4GB
        # So 2x base_dim=64 can run concurrently, 1x base_dim=128 at a time
        light = [(n, c) for n, c, _ in final if c.get("model", {}).get("base_dim", 64) == 64]
        heavy = [(n, c) for n, c, _ in final if c.get("model", {}).get("base_dim", 64) == 128]

        f.write("echo \"=== Phase 1: Light experiments (base_dim=64), 2 at a time ===\"\n")
        f.write(f"echo \"Count: {len(light)}\"\n\n")

        # Run light experiments 2 at a time
        for i in range(0, len(light), 2):
            batch = light[i:i+2]
            pids = []
            for name, cfg in batch:
                f.write(f"mkdir -p \"$EXP_BASE/{name}\"\n")
                f.write(f"cp \"$CONFIG_BASE/{name}.json\" \"$EXP_BASE/{name}/config.json\"\n")
                f.write(f"python3 run.py --config \"$EXP_BASE/{name}/config.json\" > \"$EXP_BASE/{name}/train.log\" 2>&1 &\n")
                pids.append("$!")
            f.write(f"# Wait for batch\n")
            for pid in pids:
                f.write(f"wait {pid} || echo 'BATCH_JOB_FAILED'\n")
            f.write(f"echo 'Batch {i//2+1} done'\n\n")

        f.write("echo \"=== Phase 2: Heavy experiments (base_dim=128), 1 at a time ===\"\n")
        f.write(f"echo \"Count: {len(heavy)}\"\n\n")
        for name, cfg in heavy:
            f.write(f"mkdir -p \"$EXP_BASE/{name}\"\n")
            f.write(f"cp \"$CONFIG_BASE/{name}.json\" \"$EXP_BASE/{name}/config.json\"\n")
            f.write(f"python3 run.py --config \"$EXP_BASE/{name}/config.json\" > \"$EXP_BASE/{name}/train.log\" 2>&1\n")
            f.write(f"echo '{name} done'\n\n")

        f.write("echo 'ALL DONE'\n")

    os.chmod(out_dir / "launch_parallel2.sh", 0o755)

    # ── Write a collect results script ──
    with open(out_dir / "collect_results.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Collect all eval results from ablation experiments\n")
        f.write("set -euo pipefail\n\n")
        f.write(f"EXP_BASE=\"{EXP_BASE}\"\n")
        f.write("OUTFILE=\"/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/ablation256/results.csv\"\n\n")
        f.write("echo \"name,epoch,clip_style,content_lpips,clip_dir,loss_fm,loss_swd,loss_edge,style_gate_value,cross_attn_entropy,velocity_abs,endpoint_abs\" > \"$OUTFILE\"\n\n")
        f.write("for DIR in \"$EXP_BASE\"/abl_*; do\n")
        f.write("  NAME=$(basename \"$DIR\")\n")
        f.write("  # Find latest epoch eval\n")
        f.write("  EVAL_DIR=$(ls -d \"$DIR\"/full_eval/epoch_* 2>/dev/null | sort -V | tail -1)\n")
        f.write("  if [ -z \"$EVAL_DIR\" ]; then\n")
        f.write("    # Try numeric_debug\n")
        f.write("    continue\n")
        f.write("  fi\n")
        f.write("  CSV=\"$EVAL_DIR/metrics.csv\"\n")
        f.write("  if [ ! -f \"$CSV\" ]; then continue; fi\n")
        f.write("  EPOCH=$(basename \"$EVAL_DIR\" | sed 's/epoch_//')\n")
        f.write("  # Extract mean values\n")
        f.write("  python3 -c \"\n")
        f.write("import csv, sys\n")
        f.write("with open('$CSV') as f:\n")
        f.write("    r = list(csv.DictReader(f))\n")
        f.write("if not r: sys.exit(0)\n")
        f.write("clip = sum(float(x.get('clip_style',0)) for x in r)/len(r)\n")
        f.write("lpips = sum(float(x.get('content_lpips',0)) for x in r)/len(r)\n")
        f.write("clipd = sum(float(x.get('clip_dir',0)) for x in r)/len(r)\n")
        f.write("print(f'$NAME,$EPOCH,{clip:.4f},{lpips:.4f},{clipd:.4f}')\n")
        f.write("  \" >> \"$OUTFILE\" 2>/dev/null || true\n")
        f.write("done\n\n")
        f.write("echo \"Results written to $OUTFILE\"\n")
        f.write("echo \"Total experiments with results: $(wc -l < $OUTFILE)\"\n")

    os.chmod(out_dir / "collect_results.sh", 0o755)

    # Print summary
    groups = {}
    for name, _, overrides in final:
        g = name.split("_")[1] if "_" in name else "?"
        groups[g] = groups.get(g, 0) + 1
    print("\n=== Experiment groups ===")
    for g, count in sorted(groups.items()):
        print(f"  {g}: {count} experiments")
    print(f"  TOTAL: {len(final)}")
    print(f"\nOutput: {out_dir}")
    print(f"  configs/     - {len(final)} JSON config files")
    print(f"  launch_all.sh     - Sequential launcher")
    print(f"  launch_parallel2.sh - 2-at-a-time launcher")
    print(f"  collect_results.sh - Result collector")
    print(f"  matrix.csv        - Experiment matrix")


if __name__ == "__main__":
    main()
