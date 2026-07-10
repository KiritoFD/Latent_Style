import json
import copy
import os
import platform

def load_base_config():
    with open('Cycle-NCE/src/config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def create_experiment(base, name, overrides):
    cfg = copy.deepcopy(base)

    # Apply overrides
    if 'model' in overrides:
        cfg.setdefault('model', {}).update(overrides['model'])
    if 'loss' in overrides:
        cfg.setdefault('loss', {}).update(overrides['loss'])

    # Standardize training params for this sweep
    cfg.setdefault('training', {})
    cfg['training']['num_epochs'] = 60
    cfg['training']['full_eval_interval'] = 60
    cfg['training']['save_interval'] = 20
    cfg['training']['batch_size'] = 128  # Safe margin for TextureDict

    # Set unique save dir
    cfg.setdefault('checkpoint', {})
    cfg['checkpoint']['save_dir'] = f"../exp_{name}"

    filename = f"config_exp_{name}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    return filename

if __name__ == "__main__":
    base_cfg = load_base_config()

    experiments = {
        "A_baseline": {
            "model": {"ada_mix_rank": 32},
            "loss": {"w_identity": 2.5, "w_swd": 25.0, "swd_patch_sizes": [3, 5, 7], "w_delta_tv": 0.0}
        },
        "B_capacity": {
            "model": {"ada_mix_rank": 64},
            "loss": {"w_identity": 2.5, "w_swd": 25.0, "swd_patch_sizes": [3, 5, 7], "w_delta_tv": 0.0}
        },
        "C_macro_texture": {
            "model": {"ada_mix_rank": 32},
            "loss": {"w_identity": 1.5, "w_swd": 40.0, "swd_patch_sizes": [5, 7], "w_delta_tv": 0.0}
        },
        "D_impasto_tv": {
            "model": {"ada_mix_rank": 32},
            "loss": {"w_identity": 2.0, "w_swd": 30.0, "swd_patch_sizes": [3, 5], "w_delta_tv": 0.1}
        },
        "E_safe_anchor": {
            "model": {"ada_mix_rank": 32},
            "loss": {"w_identity": 4.0, "w_swd": 20.0, "swd_patch_sizes": [1, 3, 5], "w_delta_tv": 0.0}
        }
    }

    generated_configs = []
    for name, overrides in experiments.items():
        cfg_file = create_experiment(base_cfg, name, overrides)
        generated_configs.append((name, cfg_file))
        print(f"Generated {cfg_file}")

    # Generate run script
    is_windows = platform.system() == "Windows"
    script_name = "run_5_exps.bat" if is_windows else "run_5_exps.sh"

    with open(script_name, 'w', encoding='utf-8') as f:
        if not is_windows:
            f.write("#!/bin/bash\n\n")

        for name, cfg_file in generated_configs:
            f.write("echo \"=========================================\"\n")
            f.write(f"echo \"Starting Experiment: {name}\"\n")
            f.write("echo \"=========================================\"\n")
            cmd = f"uv run run.py --config {cfg_file}"
            f.write(f"{cmd}\n\n")

    if not is_windows:
        os.chmod(script_name, 0o755)

    print(f"\nSuccessfully generated {len(experiments)} configs and execution script '{script_name}'.")
    print(f"Run it using: ./{script_name}")
