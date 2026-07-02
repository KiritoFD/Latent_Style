import json, os, glob
base = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
results = {}
count = 0
for sj in sorted(glob.glob(os.path.join(base, "*/full_eval/*/summary.json"))):
    count += 1
    rel = os.path.relpath(sj, base)
    parts = rel.split(os.sep)
    if len(parts) < 3: continue
    exp = parts[0]
    epoch = parts[2]
    try:
        with open(sj) as f:
            d = json.load(f)
        key = exp + "/" + epoch
        results[key] = d
    except: pass
for sj in sorted(glob.glob(os.path.join(base, "*/full_eval_wfi/*/summary.json"))):
    count += 1
    rel = os.path.relpath(sj, base)
    parts = rel.split(os.sep)
    if len(parts) < 3: continue
    exp = parts[0]
    epoch = parts[2]
    try:
        with open(sj) as f:
            d = json.load(f)
        key = exp + "/wfi_" + epoch
        results[key] = d
    except: pass
for wb in sorted(glob.glob(os.path.join(base, "*/full_eval_wfi/*/wfi_benchmark.json"))):
    count += 1
    rel = os.path.relpath(wb, base)
    parts = rel.split(os.sep)
    if len(parts) < 3: continue
    exp = parts[0]
    epoch = parts[2]
    try:
        with open(wb) as f:
            d = json.load(f)
        key = exp + "/wfb_" + epoch
        if key not in results: results[key] = {}
        results[key]["wfi_benchmark"] = d
    except: pass
outpath = "/mnt/i/Github/Latent_Style/exp/all620.json"
with open(outpath, "w") as f:
    json.dump(results, f)
print("OK: " + str(len(results)) + " from " + str(count) + " files")