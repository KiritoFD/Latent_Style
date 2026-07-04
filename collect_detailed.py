import json
from pathlib import Path

EXP_BASE = Path(r"G:\GitHub\Latent_Style\exp\72_fewshot")
BASE_STYLES = {"Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"}

results = []
for exp_dir in sorted(EXP_BASE.iterdir()):
    if not exp_dir.name.startswith("5p"):
        continue
    summaries = list(exp_dir.glob("full_eval/*/summary.json"))
    if not summaries:
        continue
    s = json.loads(summaries[-1].read_text(encoding="utf-8"))
    mb = s["matrix_breakdown"]
    
    parts = exp_dir.name.split("_")
    n_new = int(parts[0].replace("5p", ""))
    n_shots = int(parts[1].replace("shot", ""))
    
    styles = sorted(set(k for d in mb.values() for k in d.keys()))
    new_styles = sorted(st for st in styles if st not in BASE_STYLES)
    
    b2b_cs, b2b_lp, n2b_cs, n2b_lp, b2n_cs, b2n_lp, n2n_cs, n2n_lp = [], [], [], [], [], [], [], []
    for src in styles:
        for tgt in styles:
            if src == tgt:
                continue
            d = mb.get(src, {}).get(tgt, {})
            if d.get("count", 0) == 0:
                continue
            cs = d["clip_style"]
            lp = d["content_lpips"]
            src_new = src not in BASE_STYLES
            tgt_new = tgt not in BASE_STYLES
            if not src_new and not tgt_new:
                b2b_cs.append(cs); b2b_lp.append(lp)
            elif src_new and not tgt_new:
                n2b_cs.append(cs); n2b_lp.append(lp)
            elif not src_new and tgt_new:
                b2n_cs.append(cs); b2n_lp.append(lp)
            else:
                n2n_cs.append(cs); n2n_lp.append(lp)
    
    results.append({
        "exp": exp_dir.name, "n_new": n_new, "shots": n_shots,
        "b2b_cs": sum(b2b_cs)/len(b2b_cs) if b2b_cs else 0,
        "b2b_lp": sum(b2b_lp)/len(b2b_lp) if b2b_lp else 0,
        "n2b_cs": sum(n2b_cs)/len(n2b_cs) if n2b_cs else 0,
        "n2b_lp": sum(n2b_lp)/len(n2b_lp) if n2b_lp else 0,
        "b2n_cs": sum(b2n_cs)/len(b2n_cs) if b2n_cs else 0,
        "b2n_lp": sum(b2n_lp)/len(b2n_lp) if b2n_lp else 0,
        "n2n_cs": sum(n2n_cs)/len(n2n_cs) if n2n_cs else 0,
        "n2n_lp": sum(n2n_lp)/len(n2n_lp) if n2n_lp else 0,
    })

results.sort(key=lambda x: (x["n_new"], x["shots"]))
print(f"{'Exp':<16} {'New':>3} {'Sh':>3} | {'b2b_cs':>7} {'b2b_lp':>7} | {'b2n_cs':>7} {'b2n_lp':>7} | {'n2b_cs':>7} {'n2b_lp':>7} | {'n2n_cs':>7} {'n2n_lp':>7}")
print("-" * 105)
for r in results:
    print(f"{r['exp']:<16} {r['n_new']:>3} {r['shots']:>3} | {r['b2b_cs']:>7.4f} {r['b2b_lp']:>7.4f} | {r['b2n_cs']:>7.4f} {r['b2n_lp']:>7.4f} | {r['n2b_cs']:>7.4f} {r['n2b_lp']:>7.4f} | {r['n2n_cs']:>7.4f} {r['n2n_lp']:>7.4f}")
