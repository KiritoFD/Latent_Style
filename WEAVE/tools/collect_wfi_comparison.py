#!/usr/bin/env python3
"""Collect WFI/CLIP/LPIPS from all experiments and create comparison table."""
import json, os, glob

base = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"

experiments = [
    ("620_intrinsic_v2", "epoch_0008", "Baseline (no FiLM)"),
    ("620_film_formal", "epoch_0008", "Early FiLM v1"),
    ("620_film_gate03_5ep", "epoch_0005", "Post-FiLM + gate=0.3"),
    ("620_film_v2_5ep", "epoch_0005", "Pre+Post FiLM (softmax)"),
    ("620_film_v4_gated_5ep", "epoch_0005", "Gated attn + FiLM"),
]

results = []
for exp_name, epoch, desc in experiments:
    wfi_dir = os.path.join(base, exp_name, "full_eval_wfi", epoch)
    # Also check regular full_eval
    fe_dir = os.path.join(base, exp_name, "full_eval", epoch)

    result = {"exp": exp_name, "epoch": epoch, "desc": desc}

    # Try WFI eval dir first, then regular eval dir
    for d in [wfi_dir, fe_dir]:
        wfi_json = os.path.join(d, "wfi_benchmark.json")
        sj = os.path.join(d, "summary.json")

        if os.path.exists(wfi_json):
            w = json.load(open(wfi_json))
            gen = w.get("generated_wfi", {})
            result["wfi_score"] = gen.get("wfi_score", {}).get("mean")
            result["contrast_ratio"] = gen.get("contrast_ratio", {}).get("mean")
            result["dynamic_range"] = gen.get("dynamic_range", {}).get("mean")
            result["saturation"] = gen.get("saturation_mean", {}).get("mean")
            result["brightness"] = gen.get("brightness_mean", {}).get("mean")
            result["hist_entropy"] = gen.get("hist_entropy", {}).get("mean")
            # Transfer vs identity
            tw = w.get("transfer_wfi", {})
            iw = w.get("identity_wfi", {})
            result["wfi_transfer"] = tw.get("wfi_score", {}).get("mean") if tw else None
            result["wfi_idt"] = iw.get("wfi_score", {}).get("mean") if iw else None

        if os.path.exists(sj):
            s = json.load(open(sj))
            ap = s.get("analysis", {}).get("all_pairs_overview", {})
            result["clip_style"] = ap.get("clip_style")
            result["content_lpips"] = ap.get("content_lpips")
            result["clip_s_delta_idt"] = ap.get("clip_s_delta_idt")
            transfer = s.get("analysis", {}).get("style_transfer_ability", {})
            result["transfer_clip_style"] = transfer.get("clip_style")
            result["transfer_lpips"] = transfer.get("content_lpips")

            # Runtime observability
            ro = s.get("runtime_observability", {})
            ap_ro = ro.get("all_pairs_overview", {})
            result["cross_attn_entropy"] = ap_ro.get("model_cross_attn_entropy")
            result["film_gamma"] = ap_ro.get("model_film_gamma_abs")
            result["pre_film_gamma"] = ap_ro.get("model_pre_film_gamma_abs")
            result["style_bias"] = ap_ro.get("model_style_bias_abs")
            result["attn_delta"] = ap_ro.get("model_cross_attn_delta_abs")
            result["velocity_abs"] = ap_ro.get("model_velocity_abs")

    results.append(result)

# Print comparison table
print(f"{'Experiment':<25} {'Desc':<25} {'WFI':>7} {'Cont':>7} {'Sat':>7} {'ClipS':>7} {'LPIPS':>7} {'xEnt':>7} {'gamma':>7}")
print("-" * 110)
for r in results:
    wfi = r.get("wfi_score")
    cr = r.get("contrast_ratio")
    sat = r.get("saturation")
    cs = r.get("clip_style")
    lp = r.get("content_lpips")
    xe = r.get("cross_attn_entropy")
    g = r.get("film_gamma")
    wfi_s = f"{wfi:.4f}" if wfi is not None else "N/A"
    cr_s = f"{cr:.3f}" if cr is not None else "N/A"
    sat_s = f"{sat:.4f}" if sat is not None else "N/A"
    cs_s = f"{cs:.4f}" if cs is not None else "N/A"
    lp_s = f"{lp:.4f}" if lp is not None else "N/A"
    xe_s = f"{xe:.3f}" if xe is not None else "N/A"
    g_s = f"{g:.4f}" if g is not None else "N/A"
    print(f"{r['exp']:<25} {r['desc']:<25} {wfi_s:>7} {cr_s:>7} {sat_s:>7} {cs_s:>7} {lp_s:>7} {xe_s:>7} {g_s:>7}")

# Print transfer vs identity WFI
print(f"\n{'Experiment':<25} {'WFI(all)':>8} {'WFI(xfer)':>8} {'WFI(idt)':>8} {'xfer-LPIPS':>10}")
print("-" * 65)
for r in results:
    wfi = r.get("wfi_score")
    tw = r.get("wfi_transfer")
    iw = r.get("wfi_idt")
    tlp = r.get("transfer_lpips")
    wfi_s = f"{wfi:.4f}" if wfi is not None else "N/A"
    tw_s = f"{tw:.4f}" if tw is not None else "N/A"
    iw_s = f"{iw:.4f}" if iw is not None else "N/A"
    tlp_s = f"{tlp:.4f}" if tlp is not None else "N/A"
    print(f"{r['exp']:<25} {wfi_s:>8} {tw_s:>8} {iw_s:>8} {tlp_s:>10}")

# Write JSON
out = os.path.join(base, "wfi_comparison.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWritten to {out}")
