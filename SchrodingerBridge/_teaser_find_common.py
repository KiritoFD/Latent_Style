import os, re, json
from collections import defaultdict

ROOT = r"g:/GitHub/Latent_Style/SchrodingerBridge/results/D5-512"
STYLES = ["Early_Renaissance","Impressionism","Minimalism","Rococo","Ukiyo_e"]

def parse_weave(fn):
    # {src}_{src}__{artist}_{work}_to_{tgt}.png
    base = fn[:-4]
    if "_to_" not in base:
        return None
    prefix, tgt = base.rsplit("_to_",1)
    for s in STYLES:
        pre = s + "_" + s + "__"
        if prefix.startswith(pre):
            work = prefix[len(pre):]
            return (s, work, tgt)
    return None

def parse_double(fn):
    # {src}__{src}__{artist}_{work}__to__{tgt}.png
    base = fn[:-4]
    if "__to__" not in base:
        return None
    prefix, tgt = base.rsplit("__to__",1)
    for s in STYLES:
        pre = s + "__" + s + "__"
        if prefix.startswith(pre):
            work = prefix[len(pre):]
            return (s, work, tgt)
    return None

def collect(method, parser):
    d = os.path.join(ROOT, method)
    out = defaultdict(set)  # (src,work) -> set(tgt)
    for fn in os.listdir(d):
        if not fn.endswith(".png"):
            continue
        p = parser(fn)
        if p is None:
            continue
        src, work, tgt = p
        out[(src,work)].add(tgt)
    return out

weave = collect("weave", parse_weave)
zstar = collect("zstar", parse_double)
stylealigned = collect("stylealigned", parse_double)

print("counts: weave=%d zstar=%d stylealigned=%d" % (len(weave), len(zstar), len(stylealigned)))
common = set(weave) & set(zstar) & set(stylealigned)
print("common source (src,work) across all 3:", len(common))
# per source-style, how many targets available in all 3
full5 = [w for w in common if len(weave[w])==5 and len(zstar[w])==5 and len(stylealigned[w])==5]
print("common with all 5 targets in all 3:", len(full5))
# group by src style
by_src = defaultdict(list)
for (s,w) in full5:
    by_src[s].append(w)
for s in STYLES:
    print("  %s: %d common works" % (s, len(by_src[s])))
    for w in by_src[s][:8]:
        print("      ", w)

json.dump({"common_count": len(common), "full5": [list(x) for x in full5]},
          open(os.path.join(ROOT,"_teaser_common.json"),"w"), indent=1)
