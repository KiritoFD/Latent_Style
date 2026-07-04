#!/usr/bin/env python3
"""Read film_v4_gated eval results and compare with previous experiments."""
import json, os, glob

base = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"

experiments = [
    "620_intrinsic_v2",       # Baseline (no FiLM)
    "620_film_formal",        # Early FiLM
    "620_film_gate03_5ep",    # Post-only FiLM + gate=0.3
    "620_film_v2_5ep",        # Pre+post FiLM
    "620_film_v4_gated_5ep",  # Gated attention + FiLM
]

print(f"{'Experiment':<25} {'Epoch':>8} {'Clip-S':>8} {'LPIPS':>8} {'Clip-dIDT':>10} {'Clip-T':>8}")
print("-" * 70)

for exp in experiments:
    fe_dir = os.path.join(base, exp, "full_eval")
    if not os.path.exists(fe_dir):
        print(f"{exp:<25} {'N/A':>8}")
        continue

    # Find all eval epochs
    epochs = sorted([d for d in os.listdir(fe_dir) if d.startswith("epoch_")])
    if not epochs:
        print(f"{exp:<25} {'N/A':>8}")
        continue

    # Show last 3 epochs
    for ep in epochs[-3:]:
        sj_path = os.path.join(fe_dir, ep, "summary.json")
        if not os.path.exists(sj_path):
            continue
        s = json.load(open(sj_path))
        ap = s.get("analysis", {}).get("all_pairs_overview", {})

        # Also get runtime_observability for cross_attn_entropy
        ro = s.get("runtime_observability", {})
        model_keys = [k for k in ro.keys() if k.startswith("model_")]
        entropy = None
        for k in model_keys:
            if "cross_attn_entropy" in k or "xent" in k:
                entropy = ro[k]
                break

        cs = ap.get("clip_style", 0)
        lp = ap.get("content_lpips", 0)
        cdi = ap.get("clip_s_delta_idt", 0)
        ct = ap.get("clip_t", 0)

        ent_str = f" xent={entropy:.3f}" if entropy is not None else ""
        print(f"{exp:<25} {ep:>8} {cs:>8.4f} {lp:>8.4f} {cdi:>10.4f} {ct:>8.4f}{ent_str}")
    print()

# Also check runtime_observability for film_v4_gated
print("\n=== film_v4_gated runtime_observability (epoch_0005) ===")
sj = os.path.join(base, "620_film_v4_gated_5ep/full_eval/epoch_0005/summary.json")
if os.path.exists(sj):
    s = json.load(open(sj))
    ro = s.get("runtime_observability", {})
    print("All runtime_observability keys:")
    for k, v in sorted(ro.items()):
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")
        elif isinstance(v, dict):
            print(f"  {k}: {json.dumps(v)[:200]}")
        else:
            print(f"  {k}: {str(v)[:200]}")
