import json, os, sys, glob

base = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
results = []

for cs in sorted(glob.glob(os.path.join(base, "*", "full_eval", "curve_summary.json"))):
    try:
        d = json.load(open(cs))
        exp_name = os.path.basename(os.path.dirname(os.path.dirname(cs)))
        latest = d.get("latest", {})
        best_t = d.get("best_transfer", {})
        best_ap = d.get("best_all_pairs", {})
        src = best_t if best_t else latest
        if not src:
            continue
        row_count = d.get("row_count", "?")
        results.append({
            "experiment": exp_name,
            "epochs": row_count,
            "best_epoch": src.get("epoch", "?"),
            "clip_style": src.get("transfer_clip_style"),
            "clip_s_delta_idt": src.get("transfer_clip_s_delta_idt"),
            "clip_t": src.get("transfer_clip_t"),
            "lpips": src.get("transfer_content_lpips"),
            "ap_clip_style": best_ap.get("all_pairs_clip_style") if best_ap else None,
            "ap_lpips": best_ap.get("all_pairs_content_lpips") if best_ap else None,
            "idt_clip_style": src.get("identity_clip_style"),
            "idt_lpips": src.get("identity_content_lpips"),
            "timestamp": latest.get("timestamp", ""),
        })
    except Exception as e:
        print(f"ERROR:{cs}:{e}", file=sys.stderr)

print("experiment|epochs|best_epoch|clip_style|clip_s_delta_idt|clip_t|lpips|ap_clip_style|ap_lpips|idt_clip_style|idt_lpips|timestamp")
for r in results:
    vals = [str(r.get(k, "")) for k in ["experiment","epochs","best_epoch","clip_style","clip_s_delta_idt","clip_t","lpips","ap_clip_style","ap_lpips","idt_clip_style","idt_lpips","timestamp"]]
    print("|".join(vals))
