import os, json, glob

ROOT = r"g:/GitHub/Latent_Style/SchrodingerBridge"
R5 = os.path.join(ROOT, "results/R5-WikiArt")
D5 = os.path.join(ROOT, "results/D5-512")
OUR_R5 = os.path.join(ROOT, "exp/swd_cm_sem_r8/eval_r5_ht008/images")

recs = json.load(open(os.path.join(ROOT, "exp/swd_cm_sem_r8/eval_r5_ht008/_musiq_per_image.json")))
def parse(fname):
    base = fname[:-4]
    prefix, tgt = base.split("_to_")
    dup = prefix.split("__")[0]
    src = dup[:(len(dup) - 1) // 2]
    return src, tgt

# 8 candidate (src,artist_work,tgt) pairs from before
cross = [r for r in recs if parse(r["file"])[0] != parse(r["file"])[1]]
cross.sort(key=lambda r: -r["musiq"])
seen = set(); sel = []
for r in cross:
    s, t = parse(r["file"])
    if (s, t) in seen: continue
    seen.add((s, t)); sel.append(r["file"])
    if len(sel) == 8: break

# D5 method roots; Ours = weave in D5
method_dirs = {"Ours(weave)": os.path.join(D5, "weave")}
for m in ["seedream", "adain", "wct", "styleid", "cut", "sdturbo",
          "samam", "samst", "stylealigned", "zstar", "identity"]:
    p = os.path.join(D5, m)
    if os.path.isdir(p):
        method_dirs[m] = p

print("D5 method roots found:")
for k, v in method_dirs.items():
    print("  ", k, "->", v)

for fname in sel:
    print("\n=== candidate:", fname)
    for mname, mdir in method_dirs.items():
        hits = glob.glob(os.path.join(mdir, "**", fname), recursive=True)
        print(f"   {mname:12s} {'FOUND' if hits else '----':6s}")
