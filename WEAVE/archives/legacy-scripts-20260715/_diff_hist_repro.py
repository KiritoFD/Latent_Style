import json

hist = r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\FCSB\local_t\630_local_t11_stochastic_dwt_p08\full_eval\epoch_0005\summary.json"
repro = r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\FCSB\local_t\630_local_t11_stochastic_dwt_p08\repro_local\epoch_0005\summary.json"

for label, path in [("HIST", hist), ("REPRO", repro)]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    print(f"\n=== {label} ===")
    apo = d.get("analysis", {}).get("all_pairs_overview", {})
    print(f"  clip_style: {apo.get('clip_style')}")
    print(f"  content_lpips: {apo.get('content_lpips')}")
    idt = d.get("idt_baselines", {})
    print(f"  idt clip_style_global: {idt.get('clip_style_global')}")
    s = d.get("settings", {})
    for k in ["batch_size", "generation_batch_size", "metric_batch_size", "target_chunk_size", "vae_decode_batch_size", "num_steps", "step_size", "style_strength", "only_lpips_clip_style", "lpips_chunk_size", "vae_model", "max_src_samples"]:
        v = s.get(k, "N/A")
        print(f"  {k}: {v}")
    # Check timings
    timings = d.get("timings_sec", {})
    for k in ["eval_total", "lancet_generation", "metric_clip", "metric_lpips"]:
        v = timings.get(k, "N/A")
        print(f"  timing_{k}: {v}")