import json, numpy as np

P = r"g:/GitHub/Latent_Style/SchrodingerBridge/results/D5-512/_weave_d5_musiq.json"
recs = json.load(open(P))
v = np.array([r["musiq"] for r in recs])
print("global mean :", round(v.mean(), 2))
print("min / max   :", round(v.min(), 2), "/", round(v.max(), 2))
print("median      :", round(np.median(v), 2))
for thr in (50, 55, 60):
    print(f"count > {thr}: {int((v > thr).sum())}")
print()
order = sorted(recs, key=lambda r: -r["musiq"])
print("Top 12 single-image candidates (MUSIQ desc):")
for r in order[:12]:
    print(f"  {r['musiq']:6.2f}  {r['src']} -> {r['tgt']}  {r['work']}")
