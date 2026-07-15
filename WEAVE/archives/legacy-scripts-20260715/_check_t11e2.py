"""Quick check of t11e2 eval progress."""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Check t11e2 eval log
log_path = r"C:\Users\Administrator\logs\t11e2_fulleval.out"
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print("=== t11e2_fulleval.out (last 15 lines) ===")
    for l in lines[-15:]:
        print(l.rstrip())
else:
    print("t11e2_fulleval.out not found")

# Check t11e2 summary
p = r"I:\Github\Latent_Style\SchrodingerBridge\exp\t11e2_extrap05_15ep\full_eval\epoch_0015\summary.json"
if os.path.exists(p):
    with open(p, "r", encoding="utf-8") as f:
        s = json.load(f)
    ap = s.get("analysis", {}).get("all_pairs_overview", {})
    st = s.get("analysis", {}).get("style_transfer_ability", {})
    print(f"\n=== t11e2 ep15 RESULTS ===")
    print(f"  clip_s (all) = {ap.get('clip_style', '?')}")
    print(f"  lpips  (all) = {ap.get('content_lpips', '?')}")
    print(f"  clip_s (sty) = {st.get('clip_style', '?')}")
    print(f"  lpips  (sty) = {st.get('content_lpips', '?')}")
else:
    print(f"\nt11e2 summary not found yet (eval still running)")

# Also print t11_repro ep5 results for comparison
p2 = r"I:\Github\Latent_Style\SchrodingerBridge\exp\t11_repro_15ep\full_eval\epoch_0005\summary.json"
if os.path.exists(p2):
    with open(p2, "r", encoding="utf-8") as f:
        s2 = json.load(f)
    ap2 = s2.get("analysis", {}).get("all_pairs_overview", {})
    print(f"\n=== t11_repro ep5 (for comparison) ===")
    print(f"  clip_s (all) = {ap2.get('clip_style', '?')}")
    print(f"  lpips  (all) = {ap2.get('content_lpips', '?')}")
