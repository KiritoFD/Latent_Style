import os, json, glob, time

ROOT = r"g:/GitHub/Latent_Style/SchrodingerBridge"
now = time.time()
print("=== global MUSIQ summaries ===")
rows = []
for pat in ["exp/**/musiq_result.json", "results/*musiq*.json"]:
    for p in sorted(glob.glob(os.path.join(ROOT, pat), recursive=True)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for k, v in d.items():
            if isinstance(v, dict) and "musiq" in v:
                age = (now - os.path.getmtime(p)) / 3600
                rows.append((v["musiq"], v.get("n_images", "?"), age, p, k))
            elif isinstance(v, (int, float)):
                age = (now - os.path.getmtime(p)) / 3600
                rows.append((v, "?", age, p, k))
rows.sort(reverse=True)
for musiq, n, age, p, k in rows:
    print(f"  {musiq:6.2f}  n={n:>4}  {age:5.1f}h  {p}  [{k}]")

print()
print("=== per-image MUSIQ json (list with 'musiq' key) ===")
for p in glob.glob(os.path.join(ROOT, "results/**/*.json"), recursive=True) + glob.glob(os.path.join(ROOT, "exp/**/*.json"), recursive=True):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if isinstance(d, list) and d and isinstance(d[0], dict) and "musiq" in d[0]:
        print(f"  n={len(d):4d}  {p}")
