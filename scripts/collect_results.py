import json, os
from pathlib import Path

EXP_BASE = Path(r"G:\GitHub\Latent_Style\exp\72_fewshot")
results = []

for exp_dir in sorted(EXP_BASE.iterdir()):
    if not exp_dir.name.startswith("5p"):
        continue
    # find summary
    summaries = list(exp_dir.glob("full_eval/*/summary.json"))
    if not summaries:
        continue
    s = json.load(open(summaries[-1], "r", encoding="utf-8"))
    a = s["analysis"]
    ov = a["all_pairs_overview"]
    st = a.get("style_transfer_ability", {})
    
    # extract params from name: 5p{N}_shot{M}
    parts = exp_dir.name.split("_")
    n_persons = parts[0].replace("5p", "")
    n_shots = parts[1].replace("shot", "")
    
    results.append({
        "exp": exp_dir.name,
        "new_styles": int(n_persons),
        "shots": int(n_shots),
        "all_clip_s": ov["clip_style"],
        "all_lpips": ov["content_lpips"],
        "all_clip_t": ov.get("clip_t", 0),
        "st_clip_s": st.get("clip_style", 0),
        "st_lpips": st.get("content_lpips", 0),
        "st_clip_t": st.get("clip_t", 0),
    })

# sort by new_styles, then shots
results.sort(key=lambda x: (x["new_styles"], x["shots"]))

print(f"{'Exp':<20} {'N_new':>5} {'Shots':>5} | {'all_clip_s':>10} {'all_lpips':>10} {'all_clip_t':>10} | {'st_clip_s':>10} {'st_lpips':>10} {'st_clip_t':>10}")
print("-" * 110)
for r in results:
    print(f"{r['exp']:<20} {r['new_styles']:>5} {r['shots']:>5} | {r['all_clip_s']:>10.4f} {r['all_lpips']:>10.4f} {r['all_clip_t']:>10.4f} | {r['st_clip_s']:>10.4f} {r['st_lpips']:>10.4f} {r['st_clip_t']:>10.4f}")
