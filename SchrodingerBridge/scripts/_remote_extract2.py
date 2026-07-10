
import json
summary_path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\d7b_deep_backbone_lr5e5\full_eval\epoch_0015\summary.json"

with open(summary_path, "r") as f:
    summary = json.load(f)

# Print matrix_breakdown which likely contains the actual metrics
mb = summary.get("matrix_breakdown", {})
print("MATRIX_BREAKDOWN_KEYS=" + str(list(mb.keys())))
print("MATRIX_BREAKDOWN=" + json.dumps(mb, indent=2)[:3000])

# Also check analysis
an = summary.get("analysis", {})
print("ANALYSIS_KEYS=" + str(list(an.keys())))
print("ANALYSIS=" + json.dumps(an, indent=2)[:2000])

# Check appearance_deltas
ad = summary.get("appearance_deltas", {})
print("APPEARANCE_DELTAS_KEYS=" + str(list(ad.keys()) if isinstance(ad, dict) else "list"))
print("APPEARANCE_DELTAS=" + json.dumps(ad, indent=2)[:1500])
