import json, os, sys, glob

all_results = {}

for family_prefix, eval_subdir in [
    ("620_spatial_bridge", "full_eval"),
    ("aaai", "full_eval"),
    ("aaai", "full_eval_fast10"),
]:
    base = "/mnt/i/Github/Latent_Style/exp"
    pattern = os.path.join(base, f"{family_prefix}*", eval_subdir, "curve_summary.json")
    for cs in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(cs))
            exp_name = os.path.basename(os.path.dirname(os.path.dirname(cs)))
            latest = d.get("latest", {})
            best_t = d.get("best_transfer", {})
            src = best_t if best_t else latest
            if not src:
                continue

            wfi_score = None
            wfi_dir = os.path.join(os.path.dirname(os.path.dirname(cs)), "full_eval_wfi")
            if os.path.isdir(wfi_dir):
                wfi_epochs = sorted(os.listdir(wfi_dir))
                if wfi_epochs:
                    wbj = os.path.join(wfi_dir, wfi_epochs[-1], "wfi_benchmark.json")
                    if os.path.isfile(wbj):
                        wd = json.load(open(wbj))
                        tw = wd.get("transfer_wfi", {}).get("wfi_score", {})
                        if tw:
                            wfi_score = round(tw.get("mean", 0), 4)

            all_results[exp_name] = {
                "family": family_prefix,
                "eval_type": eval_subdir,
                "epochs": d.get("row_count", "?"),
                "best_epoch": src.get("epoch", "?"),
                "clip_style": round(src.get("transfer_clip_style", 0), 4),
                "clip_s_delta_idt": round(src.get("transfer_clip_s_delta_idt", 0), 4),
                "clip_t": round(src.get("transfer_clip_t", 0), 4),
                "lpips": round(src.get("transfer_content_lpips", 0), 4),
                "ap_clip_style": round(best_t.get("all_pairs_clip_style", 0), 4) if best_t else None,
                "ap_lpips": round(best_t.get("all_pairs_content_lpips", 0), 4) if best_t else None,
                "idt_clip_style": round(src.get("identity_clip_style", 0), 4),
                "idt_lpips": round(src.get("identity_content_lpips", 0), 4),
                "wfi_score": wfi_score,
                "timestamp": latest.get("timestamp", ""),
            }
        except Exception as e:
            print(f"ERROR:{cs}:{e}", file=sys.stderr)

for exp, r in sorted(all_results.items()):
    parts = [
        exp,
        str(r["epochs"]),
        str(r["best_epoch"]),
        f"{r['clip_style']:.4f}",
        f"{r['clip_s_delta_idt']:.4f}",
        f"{r['clip_t']:.4f}",
        f"{r['lpips']:.4f}",
        f"{r['ap_clip_style']:.4f}" if r['ap_clip_style'] else "N/A",
        f"{r['ap_lpips']:.4f}" if r['ap_lpips'] else "N/A",
        f"{r['idt_clip_style']:.4f}",
        f"{r['idt_lpips']:.4f}",
        f"{r['wfi_score']:.4f}" if r['wfi_score'] else "N/A",
        r["timestamp"],
    ]
    print("|".join(parts))
