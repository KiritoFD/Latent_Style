"""Check SaMam summary.json matrix_breakdown fields."""
import json
from pathlib import Path

p = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\samam\summary.json")
data = json.loads(p.read_text(encoding="utf-8"))

mb = data["matrix_breakdown"]
# Get first non-identity pair
for src in mb:
    for tgt in mb[src]:
        if src != tgt:
            print(f"Sample pair: {src} -> {tgt}")
            print(json.dumps(mb[src][tgt], indent=2))
            break
    break

print("\n=== analysis keys ===")
print(json.dumps(data.get("analysis", {}), indent=2))
