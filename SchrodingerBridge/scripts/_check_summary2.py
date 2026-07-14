"""Check analysis section of summary.json."""
import json
path = "exp/ablation_v2/a01_wo_endpoint_adain/eval/summary.json"
with open(path) as f:
    d = json.load(f)
analysis = d.get("analysis", {})
overview = analysis.get("all_pairs_overview", {})
print("Overview keys:", list(overview.keys())[:20])
for k, v in overview.items():
    print(f"  {k}: {v}")
# Also check idt_baselines
idt = d.get("idt_baselines", {})
print("\nIDT baselines:")
for k, v in idt.items():
    print(f"  {k}: {v}")
