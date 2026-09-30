"""Extract inference benchmark timings and metrics for 4 configs."""
import json
import os

import platform
if platform.system() == 'Windows':
    BASE = r'I:\Github\Latent_Style\SchrodingerBridge\exp\infra_infer_bench'
    BASELINE_PATH = r'I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\full_eval\epoch_0005\summary.json'
else:
    BASE = '/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/infra_infer_bench'
    BASELINE_PATH = '/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/t1_asg_5ep/full_eval/epoch_0005/summary.json'
CONFIGS = ['b4_save', 'b8_save', 'b16_save', 'b4_nosave', 'b8_nosave', 'b16_nosave']

print("=" * 100)
print(f"{'Config':<15} {'wall_total':>12} {'gen':>10} {'vae_decode':>12} {'uint8_copy':>12} {'metrics':>10} {'grid':>8}")
print("=" * 100)

def print_row(name, d):
    t = d.get('timings_sec', {})
    wt = t.get('wall_total', 0)
    gen = t.get('lancet_generation', t.get('generation', t.get('bridge_forward', 0)))
    vae = t.get('vae_decode', 0)
    u8 = t.get('uint8_cpu_copy', 0)
    met = t.get('eval_metrics_loop', 0)
    grid = t.get('summary_grid', 0)
    print(f"{name:<15} {wt:>12.2f} {gen:>10.2f} {vae:>12.2f} {u8:>12.2f} {met:>10.2f} {grid:>8.2f}")

# Baseline first
if os.path.exists(BASELINE_PATH):
    with open(BASELINE_PATH) as f:
        print_row("baseline_b2", json.load(f))
else:
    print(f"baseline not found at {BASELINE_PATH}")

for cfg in CONFIGS:
    p = os.path.join(BASE, cfg, 'full_eval', 'epoch_0005', 'summary.json')
    if os.path.exists(p):
        with open(p) as f:
            print_row(cfg, json.load(f))
    else:
        print(f"{cfg}: MISSING -> {p}")

print()
print(f"{'Config':<15} {'CLIP-S':>10} {'DINO-S':>10} {'DINO-C':>10} {'LPIPS':>10}")
print("-" * 60)

def print_metrics(name, d):
    a = d.get('analysis', {})
    ap = a.get('all_pairs_overview', {})
    st = a.get('style_transfer_ability', {})
    cs = ap.get('clip_style', st.get('clip_style'))
    lpips = ap.get('content_lpips', st.get('content_lpips'))
    # DINO-S / DINO-C
    ds = ap.get('dino_style', st.get('dino_style'))
    dc = ap.get('dino_content', st.get('dino_content'))
    print(f"{name:<15} {cs if cs else 0:>10.4f} {ds if ds else 0:>10.4f} {dc if dc else 0:>10.4f} {lpips if lpips else 0:>10.4f}")

if os.path.exists(BASELINE_PATH):
    with open(BASELINE_PATH) as f:
        print_metrics("baseline_b2", json.load(f))

for cfg in CONFIGS:
    p = os.path.join(BASE, cfg, 'full_eval', 'epoch_0005', 'summary.json')
    if os.path.exists(p):
        with open(p) as f:
            print_metrics(cfg, json.load(f))

# Print full timings_sec for each config (for debugging)
print()
print("=" * 100)
print("FULL timings_sec for each config:")
print("=" * 100)
for cfg in CONFIGS:
    p = os.path.join(BASE, cfg, 'full_eval', 'epoch_0005', 'summary.json')
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        print(f"\n--- {cfg} ---")
        print(json.dumps(d.get('timings_sec', {}), indent=2, default=str))
