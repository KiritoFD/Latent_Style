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

        wfi_val = None
        wfi_dir = os.path.join(os.path.dirname(os.path.dirname(cs)), "full_eval_wfi")
        if os.path.isdir(wfi_dir):
            wfi_epochs = sorted(os.listdir(wfi_dir))
            if wfi_epochs:
                wfi_csv = os.path.join(wfi_dir, wfi_epochs[-1], "metrics.csv")
                if os.path.isfile(wfi_csv):
                    import csv
                    with open(wfi_csv) as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            wfi_vals = [float(r.get("clip_style_wfi", 0)) for r in rows if r.get("clip_style_wfi")]
                            content_wfi_vals = [float(r.get("content_wfi", 0)) for r in rows if r.get("content_wfi")]
                            if wfi_vals:
                                wfi_val = sum(wfi_vals) / len(wfi_vals)

        results.append({
            "experiment": exp_name,
            "epochs": row_count,
            "best_epoch": src.get("epoch", "?"),
            "clip_style": round(src.get("transfer_clip_style", 0), 4),
            "clip_s_delta_idt": round(src.get("transfer_clip_s_delta_idt", 0), 4),
            "clip_t": round(src.get("transfer_clip_t", 0), 4),
            "lpips": round(src.get("transfer_content_lpips", 0), 4),
            "ap_clip_style": round(best_ap.get("all_pairs_clip_style", 0), 4) if best_ap else None,
            "ap_lpips": round(best_ap.get("all_pairs_content_lpips", 0), 4) if best_ap else None,
            "idt_clip_style": round(src.get("identity_clip_style", 0), 4),
            "idt_lpips": round(src.get("identity_content_lpips", 0), 4),
            "wfi_avg": round(wfi_val, 4) if wfi_val else None,
            "timestamp": latest.get("timestamp", ""),
        })
    except Exception as e:
        print(f"ERROR:{cs}:{e}", file=sys.stderr)

print(json.dumps(results, indent=2))
