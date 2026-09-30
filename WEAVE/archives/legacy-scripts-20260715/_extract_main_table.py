"""Extract all main table metrics: CLIP-S, LPIPS, DINO-C, DINO-S, and inference time for D5/P2A/R5."""
import json
import os
import platform

if platform.system() == 'Windows':
    BASE = r'I:\Github\Latent_Style\SchrodingerBridge\exp'
else:
    BASE = '/mnt/i/Github/Latent_Style/SchrodingerBridge/exp'

# D5: existing t1_asg_5ep eval
D5_SUMMARY = os.path.join(BASE, 't1_asg_5ep', 'full_eval', 'epoch_0005', 'summary.json')
D5_DINO = os.path.join(BASE, '_dino_results', 't1_asg_5ep.json')

# P2A-256: new eval
P2A_SUMMARY = os.path.join(BASE, 'main_table', 'p2a_256', 'full_eval', 'epoch_0005', 'summary.json')
P2A_DINO = os.path.join(BASE, '_dino_results', 't1_asg_5ep_p2a.json')

# R5: new eval
R5_SUMMARY = os.path.join(BASE, 'main_table', 'r5', 'full_eval', 'epoch_0005', 'summary.json')
R5_DINO = os.path.join(BASE, '_dino_results', 't1_asg_5ep_r5.json')

def extract_metrics(name, summary_path, dino_path):
    with open(summary_path) as f:
        s = json.load(f)
    with open(dino_path) as f:
        d = json.load(f)

    a = s.get('analysis', {})
    ap = a.get('all_pairs_overview', {})
    st = a.get('style_transfer_ability', {})

    clip_s = ap.get('clip_style', st.get('clip_style', 0))
    lpips = ap.get('content_lpips', st.get('content_lpips', 0))

    dino_c = d.get('dino_content', 0)
    dino_s = d.get('dino_style', 0)

    t = s.get('timings_sec', {})
    wall = t.get('wall_total', 0)
    gen = t.get('lancet_generation', 0)
    uint8 = t.get('uint8_cpu_copy', 0)

    return {
        'name': name,
        'clip_s': clip_s,
        'lpips': lpips,
        'dino_c': dino_c,
        'dino_s': dino_s,
        'wall_total': wall,
        'gen': gen,
        'uint8': uint8,
        'n_images': d.get('n_images', 0),
    }

print("=" * 120)
print(f"{'Dataset':<12} {'CLIP-S':>8} {'LPIPS':>8} {'DINO-C':>8} {'DINO-S':>8} | {'wall_total':>12} {'gen':>10} {'uint8':>10} {'n_img':>6}")
print("=" * 120)

for name, sp, dp in [('D5-512', D5_SUMMARY, D5_DINO),
                      ('P2A-256', P2A_SUMMARY, P2A_DINO),
                      ('R5-512', R5_SUMMARY, R5_DINO)]:
    if os.path.exists(sp) and os.path.exists(dp):
        m = extract_metrics(name, sp, dp)
        print(f"{m['name']:<12} {m['clip_s']:>8.4f} {m['lpips']:>8.4f} {m['dino_c']:>8.4f} {m['dino_s']:>8.4f} | {m['wall_total']:>12.2f} {m['gen']:>10.2f} {m['uint8']:>10.2f} {m['n_images']:>6}")
    else:
        print(f"{name:<12} MISSING: {sp} or {dp}")

print()
print("Training time (T1 ASG, 5 epochs, batch=24):")
print("  baseline: 23.0s/epoch x 5 = 115.0s total")
print("  optimized (batch=96+channels_last): 19.4s/epoch x 5 = 97.0s total")
print("  Params: 903,508")
