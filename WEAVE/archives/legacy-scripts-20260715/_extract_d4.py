"""Extract D4-FiLM metrics from summary.json + dino_results."""
import json, os

ROOT = r"I:\Github\Latent_Style\SchrodingerBridge"
exp = "d4_film_hf1_15ep"

# CLIP-S and LPIPS from summary.json
summary_path = os.path.join(ROOT, "exp", exp, "full_eval", "epoch_0015", "summary.json")
with open(summary_path) as f:
    d = json.load(f)
ov = d.get("analysis", {}).get("all_pairs_overview", {})
clip_s = ov.get("clip_style", "?")
lpips = ov.get("content_lpips", "?")
one_minus_lpips = 1.0 - lpips if isinstance(lpips, (int, float)) else "?"

# DINO from dino_results
dino_path = os.path.join(ROOT, "exp", "_dino_results", f"{exp}.json")
with open(dino_path) as f:
    dino = json.load(f)
dino_sty = dino.get("dino_style", dino.get("dino_sty", "?"))
dino_con = dino.get("dino_content", dino.get("dino_con", "?"))
dino_str = dino.get("dino_structure", dino.get("dino_str", "?"))

print(f"D4-FiLM Results:")
print(f"  CLIP-S:   {clip_s}")
print(f"  1-LPIPS:  {one_minus_lpips}")
print(f"  DINO-sty: {dino_sty}")
print(f"  DINO-con: {dino_con}")
print(f"  DINO-str: {dino_str}")

# Comparison table
print(f"\n{'config':<30} {'clip_s':<8} {'1-LPIPS':<8} {'dino_sty':<8} {'dino_con':<8} {'dino_str':<8}")
print("-" * 80)
print(f"{'hp_baseline':<30} {0.7167:<8} {0.7010:<8} {0.4762:<8} {0.8052:<8} {0.0243:<8}")
print(f"{'d1_gram_hf1':<30} {0.7190:<8} {0.7035:<8} {0.4780:<8} {0.8030:<8} {0.0242:<8}")
print(f"{'d2_moment_hf1':<30} {0.7083:<8} {0.7138:<8} {0.4732:<8} {0.8163:<8} {0.0241:<8}")
print(f"{'d4_film_hf1':<30} {clip_s:<8} {one_minus_lpips:<8} {dino_sty:<8} {dino_con:<8} {dino_str:<8}")
