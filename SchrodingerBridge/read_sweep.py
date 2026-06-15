import json
import glob
import os

os.chdir('/mnt/i/Github/Latent_Style/SchrodingerBridge')
files = glob.glob('exp/aaai2027_phase2_smoe_fiber_sde_fiberwise_swd_k070/eval_sweep_*/summary.json')
print("-----------------------------------------------------------------")
print(" SDE Noise Scale Sweep Results:")
print("-----------------------------------------------------------------")
for f in sorted(files):
    try:
        data = json.load(open(f))
        overview = data['analysis']['all_pairs_overview']
        print(f"  {f.split('/')[-2]:<25} | Style: {overview.get('clip_style', 0):.4f} | LPIPS: {overview.get('content_lpips', 1.0):.4f}")
    except Exception as e:
        pass
print("-----------------------------------------------------------------")
