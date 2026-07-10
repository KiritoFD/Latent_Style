"""Find original T11 evaluation results and check t11e2 eval progress."""
import sys, os, json, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXP_DIR = r"I:\Github\Latent_Style\SchrodingerBridge\exp"

# 1. Find all T11-related directories
print("=== T11-related experiment directories ===")
for d in sorted(os.listdir(EXP_DIR)):
    dl = d.lower()
    if "t11" in dl or "t10" in dl or "4j" in dl or "p08" in dl or "p08" in dl or "stochastic_dwt" in dl or "dwt_p" in dl:
        full = os.path.join(EXP_DIR, d)
        if os.path.isdir(full):
            # List contents
            contents = os.listdir(full)
            ckpts = [c for c in contents if c.endswith(".pt")]
            eval_dirs = [c for c in contents if os.path.isdir(os.path.join(full, c)) and "eval" in c.lower()]
            print(f"  {d}/")
            print(f"    ckpts: {ckpts[:5]}")
            if eval_dirs:
                for ed in eval_dirs:
                    ed_full = os.path.join(full, ed)
                    for sub in os.listdir(ed_full):
                        sub_full = os.path.join(ed_full, sub)
                        summary = os.path.join(sub_full, "summary.json")
                        if os.path.exists(summary):
                            try:
                                with open(summary, "r", encoding="utf-8") as f:
                                    s = json.load(f)
                                # Extract key metrics
                                ap = s.get("analysis", {}).get("all_pairs_overview", {})
                                st = s.get("analysis", {}).get("style_transfer_ability", {})
                                print(f"    {ed}/{sub}: clip_s={ap.get('clip_style', '?'):.4f}, lpips={ap.get('content_lpips', '?'):.4f} (all) | clip_s={st.get('clip_style', '?'):.4f}, lpips={st.get('content_lpips', '?'):.4f} (sty)")
                            except Exception as e:
                                print(f"    {ed}/{sub}: ERR {e}")

# 2. Search for any summary.json with clip_s around 0.72-0.73
print("\n=== Searching for historical clip_s ~0.72-0.73 ===")
for root, dirs, files in os.walk(EXP_DIR):
    if "summary.json" in files:
        p = os.path.join(root, "summary.json")
        try:
            with open(p, "r", encoding="utf-8") as f:
                s = json.load(f)
            ap = s.get("analysis", {}).get("all_pairs_overview", {})
            clip_s = ap.get("clip_style", 0)
            lpips = ap.get("content_lpips", 0)
            if clip_s > 0.70 and lpips < 0.35:
                rel = os.path.relpath(p, EXP_DIR)
                print(f"  {rel}: clip_s={clip_s:.4f}, lpips={lpips:.4f}")
        except:
            pass

# 3. Check t11e2 eval progress
print("\n=== t11e2 eval progress ===")
t11e2_log = r"C:\Users\Administrator\logs\t11e2_fulleval.out"
if os.path.exists(t11e2_log):
    with open(t11e2_log, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for l in lines[-20:]:
        print(l.rstrip())
else:
    # Check if eval started via monitor
    print("(no t11e2_fulleval.out yet)")

# Check t11e2 summary
for ep in ["epoch_0015"]:
    p = os.path.join(EXP_DIR, "t11e2_extrap05_15ep", "full_eval", ep, "summary.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)
        ap = s.get("analysis", {}).get("all_pairs_overview", {})
        print(f"\nt11e2 {ep}: clip_s={ap.get('clip_style', '?')}, lpips={ap.get('content_lpips', '?')}")
    else:
        print(f"\nt11e2 {ep}: summary not found yet")

# 4. Check progress.json
print("\n=== progress.json (T11 section) ===")
prog_p = os.path.join(EXP_DIR, "..", "docs", "630", "state", "progress.json")
if os.path.exists(prog_p):
    with open(prog_p, "r", encoding="utf-8") as f:
        prog = json.load(f)
    # Find T11-related entries
    def search_t11(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if "t11" in str(k).lower() or "t11" in str(v).lower() if isinstance(v, str) else False:
                    print(f"  {path}.{k}: {v}")
                search_t11(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                search_t11(v, f"{path}[{i}]")
    search_t11(prog)
else:
    print(f"progress.json not found at {prog_p}")

# 5. Check findings.jsonl for T11
print("\n=== findings.jsonl (T11 entries, last 5) ===")
find_p = os.path.join(EXP_DIR, "..", "docs", "630", "state", "findings.jsonl")
if os.path.exists(find_p):
    with open(find_p, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    t11_lines = [l for l in lines if "t11" in l.lower() or "T11" in l]
    for l in t11_lines[-5:]:
        print(l.rstrip()[:300])
else:
    print("findings.jsonl not found")
