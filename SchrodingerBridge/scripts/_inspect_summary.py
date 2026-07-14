"""Inspect summary.json structure."""
import json
import os

base = r"I:\Github\Latent_Style\SchrodingerBridge"
sp = os.path.join(base, "exp", "seed3", "seed42_d5_eval", "full_eval", "epoch_0005", "summary.json")
with open(sp) as f:
    d = json.load(f)

print("Top keys:", list(d.keys()))
a = d.get("analysis", {})
print("Analysis keys:", list(a.keys()))
s = a.get("style_transfer_ability", {})
print("Style transfer keys:", list(s.keys()))
print("Style transfer data:", json.dumps(s, indent=2))
p = a.get("all_pairs_overview", {})
print("\nAll pairs keys:", list(p.keys()))
print("All pairs data:", json.dumps(p, indent=2))
t = d.get("timings_sec", {})
print("\nTimings keys:", list(t.keys()))
print("Timings data:", json.dumps(t, indent=2))
