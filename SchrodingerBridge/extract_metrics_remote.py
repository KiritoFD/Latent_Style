import json, os

exps = ["fc_sb_kernel7", "fc_sb_floor0", "fc_sb_curriculum", "fc_sb_fiber_ep", "fc_sb_wavelet"]

print(f"{'Experiment':<20} {'CLIP_Style':>12} {'LPIPS':>10} {'clip_dir':>10} {'clip_content':>12}")
print("-" * 70)

for exp in exps:
    found = False
    for json_path in [
        f"exp/p3_remote_10h/{exp}/full_eval/summary.json",
        f"exp/p3_remote_10h/{exp}/full_eval/round2_convergence.json",
        f"{exp}/full_eval/summary.json",
    ]:
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                cs = data.get('mean_clip_style_score',
                     data.get('clip_style_score',
                      data.get('clip_style',
                       data.get('mean_clip_style', '?'))))
                lp = data.get('mean_lpips',
                     data.get('lpips',
                      data.get('content_lpips', '?')))
                cd = data.get('mean_clip_dir_score', '?')
                cc = data.get('mean_clip_content_score', '?')

                print(f"{exp:<20} {str(cs):>12} {str(lp):>10} {str(cd):>10} {str(cc):>12}")
                found = True
                break
            except Exception as e:
                print(f"{exp:<20} ERROR: {e}")
                found = True
                break
    if not found:
        print(f"{exp:<20} NO_JSON_FOUND")

print()
print("=== E2 BASELINE (for comparison) ===")
print("fc_sb_sigma04:     clip_style=0.708  LPIPS=0.540")
print()
print("=== DONE ===")
