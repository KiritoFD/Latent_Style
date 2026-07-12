"""M4: Why AdaIN? Inference-time Style Injection Module Comparison

Compare 4 endpoint_adain modes using the SAME trained checkpoint:
1. spatial_fiber (AdaIN, baseline): global mean+std on high-freq fiber
2. spatial_fiber_wct (WCT): global full-covariance matching on high-freq fiber
3. per_subband: per-subband AdaIN (mean+std per LH/HL/HH)
4. per_subband_wct: per-subband WCT (full covariance per LH/HL/HH)

Also test:
5. no_adain (endpoint_adain_scale=0): pure Flow, no injection

For each mode, measure:
- Per-subband style_transfer_ratio (LH/HL/HH)
- Content preservation: L2(output, content)
- Style matching: L2(output_stats, style_stats)
- Inference time
"""
import sys, os, json, time, torch
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge\src")
os.chdir(r"I:\Github\Latent_Style\SchrodingerBridge")

from config_schema import ExperimentConfig
from model import build_model_from_config
from utils.dataset import AdaCUTLatentDataset
from wavelet import dwt2_haar, idwt2_haar
from torch.utils.data import DataLoader

CONFIG_PATH = r"I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline\config.json"
CKPT_PATH = r"I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline\epoch_0005.pt"
OUTPUT_PATH = r"I:\Github\Latent_Style\SchrodingerBridge\mechanism_diagnosis\state\m4_why_adain.json"
NUM_SAMPLES = 50

MODES = [
    ("no_adain", {"endpoint_adain_scale": 0.0}),
    ("spatial_fiber_adain", {"endpoint_adain_mode": "spatial_fiber", "endpoint_adain_scale": 1.0}),
    ("spatial_fiber_wct", {"endpoint_adain_mode": "spatial_fiber_wct", "endpoint_adain_scale": 1.0}),
    ("per_subband_adain", {"endpoint_adain_mode": "per_subband", "endpoint_adain_scale": 1.0}),
    ("per_subband_wct", {"endpoint_adain_mode": "per_subband_wct", "endpoint_adain_scale": 1.0}),
]


def load_config(path):
    with open(path) as f:
        raw = json.load(f)
    if "model" in raw and raw["model"].get("solver_family") == "solver_i2sb":
        raw["model"]["solver_family"] = "euler_legacy"
    return ExperimentConfig.from_mapping(raw)


def _stats(x):
    return x.float().mean().item(), x.float().std().item()


def evaluate_mode(model, dataloader, device, num_samples):
    """Measure per-subband transfer ratio, content preservation, style matching."""
    model.eval()
    subbands = ["ll", "lh", "hl", "hh"]
    transfer_ratios = {s: [] for s in subbands}
    content_l2 = []
    style_l2 = []
    times = []

    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples:
                break
            content = batch["content"].to(device).float()
            target_style_id = batch["target_style_id"].to(device)
            style_latent = batch.get("target_style", None)
            if style_latent is not None:
                style_latent = style_latent.to(device).float()

            t0 = time.time()
            z_out = model.integrate_transport(
                content, style_id=target_style_id,
                num_steps=8, step_size=1.0,
                style_latent=style_latent,
            )
            times.append(time.time() - t0)

            # Per-subband analysis
            o_ll, o_lh, o_hl, o_hh = dwt2_haar(z_out.float())
            c_ll, c_lh, c_hl, c_hh = dwt2_haar(content.float())
            if style_latent is not None:
                s_ll, s_lh, s_hl, s_hh = dwt2_haar(style_latent.float())

            out_subs = {"ll": o_ll, "lh": o_lh, "hl": o_hl, "hh": o_hh}
            content_subs = {"ll": c_ll, "lh": c_lh, "hl": c_hl, "hh": c_hh}
            style_subs = {"ll": s_ll, "lh": s_lh, "hl": s_hl, "hh": s_hh} if style_latent is not None else None

            for s in subbands:
                o_m, o_sd = _stats(out_subs[s])
                c_m, c_sd = _stats(content_subs[s])
                if style_subs is not None:
                    s_m, s_sd = _stats(style_subs[s])
                    out_style_d = abs(o_m - s_m) + abs(o_sd - s_sd)
                    style_content_d = abs(s_m - c_m) + abs(s_sd - c_sd)
                    if style_content_d > 1e-8:
                        transfer_ratios[s].append(max(0.0, 1.0 - out_style_d / style_content_d))
                    else:
                        transfer_ratios[s].append(0.0)

            # Global metrics
            content_l2.append(torch.norm(z_out - content).item() / content.numel())
            if style_latent is not None:
                style_l2.append(torch.norm(z_out - style_latent).item() / style_latent.numel())

            count += 1

    results = {}
    for s in subbands:
        results[s] = sum(transfer_ratios[s]) / max(len(transfer_ratios[s]), 1)
    results["content_l2"] = sum(content_l2) / max(len(content_l2), 1)
    results["style_l2"] = sum(style_l2) / max(len(style_l2), 1) if style_l2 else None
    results["avg_time"] = sum(times) / max(len(times), 1)
    results["num_samples"] = count
    return results


def main():
    print("Loading config...")
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda")

    data_cfg = config.data
    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.data_root,
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=False,
        identity_ratio=None,
        batch_size_hint=4,
        balance_target_styles_per_batch=False,
        preload_to_gpu=False,
        preload_max_vram_gb=0.0,
        preload_reserve_ratio=0.35,
        virtual_length_multiplier=1.0,
        content_style_sampling_weights=None,
        target_style_sampling_weights=None,
        pairing_cache_path=data_cfg.pairing_cache_path,
        pairing_cache_topk=int(data_cfg.pairing_cache_topk),
        pairing_cache_active_topk=int(data_cfg.pairing_cache_active_topk),
        pairing_cache_sample_mode=str(data_cfg.pairing_cache_sample_mode),
        pairing_cache_rank_schedule=str(data_cfg.pairing_cache_rank_schedule),
        pairing_cache_min_topk=int(data_cfg.pairing_cache_min_topk),
        pairing_cache_curriculum_epochs=0,
        pairing_cache_rank_power=float(data_cfg.pairing_cache_rank_power),
        pairing_cache_explore_prob=0.0,
        pairing_cache_explore_topk=0,
        pairing_cache_dual_target_mix=0.0,
        pairing_cache_dual_target_topk=0,
        pairing_cache_aux_target_topk=0,
        pairing_cache_cross_only=bool(data_cfg.pairing_cache_cross_only),
        latent_cache_mode=str(data_cfg.latent_cache_mode),
        latent_cache_dir=str(data_cfg.latent_cache_dir),
        style_caption_path="",
        device="cpu",
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    all_results = []

    for mode_name, overrides in MODES:
        print(f"\n=== Mode: {mode_name} ===")
        # Rebuild config (overrides applied AFTER build to bypass validated() resetting extra fields)
        mode_config = load_config(CONFIG_PATH)
        model = build_model_from_config(mode_config.model, bridge_cfg=mode_config.bridge).to(device)
        # Override model_cfg attributes directly on the built model
        for k, v in overrides.items():
            setattr(model.model_cfg, k, v)
            # Also update extra dict so _cfg_get fallback works
            if hasattr(model.model_cfg, 'extra') and isinstance(model.model_cfg.extra, dict):
                model.model_cfg.extra[k] = v

        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded (missing={len(missing)}, unexpected={len(unexpected)})")
        # Verify override took effect
        _check_scale = float(getattr(model.model_cfg, 'endpoint_adain_scale', -1))
        _check_mode = str(getattr(model.model_cfg, 'endpoint_adain_mode', 'NOT_SET'))
        print(f"  Verified: endpoint_adain_scale={_check_scale}, endpoint_adain_mode={_check_mode}")

        stats = evaluate_mode(model, dataloader, device, NUM_SAMPLES)
        stats["mode"] = mode_name
        stats["overrides"] = overrides
        all_results.append(stats)

        print(f"  LH ratio={stats['lh']:.4f}  HL ratio={stats['hl']:.4f}  "
              f"HH ratio={stats['hh']:.4f}  LL ratio={stats['ll']:.4f}")
        print(f"  content_l2={stats['content_l2']:.6f}  style_l2={stats['style_l2']:.6f}  "
              f"time={stats['avg_time']:.3f}s")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    print("\n=== SUMMARY: Style Injection Module Comparison ===")
    print(f"{'Mode':<25} {'LL':>7} {'LH':>7} {'HL':>7} {'HH':>7} {'C-L2':>10} {'S-L2':>10} {'Time':>7}")
    for r in all_results:
        print(f"  {r['mode']:<23} {r['ll']:>7.4f} {r['lh']:>7.4f} {r['hl']:>7.4f} {r['hh']:>7.4f} "
              f"{r['content_l2']:>10.6f} {r['style_l2'] or 0:>10.6f} {r['avg_time']:>7.3f}")


if __name__ == "__main__":
    main()
