import json
import os

experiments = [
    ("Baseline (1ep)", "exp/task3_baseline_1ep/full_eval/epoch_0001/summary.json"),
    ("Combo A (1ep)", "exp/task3_combo_a_1ep/full_eval/epoch_0001/summary.json"),
    ("Combo B (1ep)", "exp/task3_combo_b_3ep/full_eval/epoch_0001/summary.json"),
    ("Combo B (2ep)", "exp/task3_combo_b_3ep/full_eval/epoch_0002/summary.json"),
    ("Combo B (3ep)", "exp/task3_combo_b_3ep/full_eval/epoch_0003/summary.json"),
]

metric_keys = [
    ("analysis.style_transfer_ability.clip_style", "clip_style"),
    ("analysis.style_transfer_ability.content_lpips", "LPIPS"),
    ("runtime_observability.style_transfer_ability.model_velocity_std", "velocity_std"),
    ("runtime_observability.style_transfer_ability.model_velocity_channel_std", "velocity_channel_std"),
    ("runtime_observability.style_transfer_ability.model_endpoint_alpha", "endpoint_alpha"),
    ("runtime_observability.style_transfer_ability.model_endpoint_high_alpha", "endpoint_high_alpha"),
    ("runtime_observability.style_transfer_ability.model_endpoint_low_delta_std", "endpoint_low_delta_std"),
    ("runtime_observability.style_transfer_ability.model_endpoint_high_delta_std", "endpoint_high_delta_std"),
    ("runtime_observability.style_transfer_ability.model_style_gate_value", "style_gate_value"),
    ("runtime_observability.style_transfer_ability.model_film_gamma_abs", "film_gamma_abs"),
    ("runtime_observability.style_transfer_ability.model_film_beta_abs", "film_beta_abs"),
    ("runtime_observability.style_transfer_ability.model_cross_attn_entropy", "cross_attn_entropy"),
    ("runtime_observability.style_transfer_ability.model_cross_attn_delta_abs", "cross_attn_delta_abs"),
    ("runtime_observability.style_transfer_ability.model_endpoint_film_enabled", "endpoint_film_enabled"),
    ("runtime_observability.style_transfer_ability.model_film_enabled", "film_enabled"),
    ("runtime_observability.style_transfer_ability.model_block0_output_channel_std", "block0_ch_std"),
    ("runtime_observability.style_transfer_ability.model_block3_output_channel_std", "block3_ch_std"),
    ("runtime_observability.style_transfer_ability.model_latent_input_channel_std", "latent_ch_std"),
    ("runtime_observability.style_transfer_ability.model_endpoint_output_channel_std", "endpoint_ch_std"),
    ("runtime_observability.style_transfer_ability.model_velocity_abs", "velocity_abs"),
]

def get_nested(obj, path):
    keys = path.split('.')
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        else:
            return None
    return obj

print("=" * 120)
print(f"{'Metric':<35} {'Baseline 1ep':>15} {'Combo A 1ep':>15} {'Combo B 1ep':>15} {'Combo B 2ep':>15} {'Combo B 3ep':>15}")
print("=" * 120)

results = {}
for exp_name, path in experiments:
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        results[exp_name] = data
    else:
        print(f"WARNING: {path} not found!")

for full_key, display_name in metric_keys:
    row = f"{display_name:<35}"
    for exp_name, _ in experiments:
        if exp_name in results:
            val = get_nested(results[exp_name], full_key)
            if val is not None:
                row += f"{val:>15.6f}"
            else:
                row += f"{'N/A':>15}"
        else:
            row += f"{'N/A':>15}"
    print(row)

print("=" * 120)

print("\n\n=== WFI-like Analysis ===")
print("Note: WFI = channel_std / total_std (whitening: lower = more whitened)")
print()

for exp_name, _ in experiments:
    if exp_name not in results:
        continue
    
    vel_ch_std = get_nested(results[exp_name], "runtime_observability.style_transfer_ability.model_velocity_channel_std")
    vel_std = get_nested(results[exp_name], "runtime_observability.style_transfer_ability.model_velocity_std")
    
    if vel_ch_std and vel_std:
        wfi = vel_ch_std / vel_std
        print(f"{exp_name}: velocity_channel_std/velocity_std = {vel_ch_std:.6f} / {vel_std:.6f} = {wfi:.6f}")

print()
print("Note: Lower WFI = more whitened (better)")
