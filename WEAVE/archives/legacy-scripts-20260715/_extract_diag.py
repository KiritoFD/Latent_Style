"""Extract timings from profile_timing diagnostic run."""
import json
from pathlib import Path

path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\infra_diag\b16_profile\full_eval\epoch_0005\summary.json"
d = json.loads(Path(path).read_text(encoding="utf-8"))
timings = d.get("timings_sec", d.get("timings", {}))
print("=== b16 + profile_timing ===")
for k in sorted(timings.keys()):
    print(f"  {k:30s}: {timings[k]:.3f}s")
print(f"\n  wall_total: {timings.get('wall_total', 'N/A')}")
