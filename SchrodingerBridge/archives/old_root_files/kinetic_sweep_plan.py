"""Kinetic Sweep Plan - 11 theoretically-grounded configs"""
import json, shutil, subprocess, sys, time, csv
from pathlib import Path

ROOT = Path("G:/GitHub/Latent_Style/SchrodingerBridge")
SRC = ROOT / "src"
SWEEP_DIR = ROOT / "kinetic_sweep"
OVF50 = "G:/GitHub/Latent_Style/style_data/overfit50"
CACHE = "G:/GitHub/Latent_Style/Cycle-NCE/eval_cache"

# Base config loaded from PK1
BASE_CONFIG = {
    "model": {
        "latent_channels": 4, "base_dim": 64, "style_dim": 160, "time_dim": 256,
        "num_styles": 5, "lift_channels": 128, "num_hires_blocks": 2, "num_res_blocks": 4,
        "num_decoder_blocks": 2, "num_groups": 4, "latent_scale_factor": 0.18215,
        "residual_gain": 1.0, "style_spatial_pre_gain_16": 0.35, "style_strength_default": 1.0,
        "style_strength_step_curve": "linear", "upsample_mode": "nearest",
        "style_id_spatial_jitter_px": 0, "upsample_blur": True, "upsample_blur_kernel": "box3",
        "style_attn_num_tokens": 128, "style_attn_num_heads": 4, "style_attn_sharpen_scale": 2.5,
        "style_attn_temperature": 0.08, "hires_block_type": "conv", "body_block_type": "global_attn",
        "decoder_block_type": "conv", "semantic_attn_temperature": 0.12, "feature_attn_num_heads": 4,
        "window_attn_window_size": 8, "skip_fusion_mode": "add_proj", "skip_routing_mode": "add_proj",
        "skip_naive_gain": 1.0, "style_skip_content_retention_boost": 0.0,
        "input_anchor_noise_std": 0.0, "input_anchor_noise_eval": False,
        "ablation_skip_clean": True, "ablation_skip_blur": True,
        "ablation_no_residual": False, "ablation_no_residual_gain": 1.0,
        "ablation_disable_spatial_prior": False, "output_moment_match": False,
        "output_moment_match_eps": 1e-06, "output_moment_match_train_only": False,
        "use_style_blender": False
    },
    "bridge": {
        "objective_mode": "flow_matching", "kinetic_mode": "path",
        "ot_cost_mode": "swd", "coupling_solver": "sinkhorn",
        "sinkhorn_epsilon": 0.05, "sinkhorn_iters": 60, "sinkhorn_stabilize": True,
        "bridge_sigma": 0.05, "t_min": 0.05, "t_max": 0.95, "terminal_num_steps": 4,
        "terminal_swd_on_identity": False, "w_kinetic": 1.0, "w_curvature": 0.0,
        "curvature_dt": 0.15, "w_low_freq": 0.0, "w_cycle": 0.0, "terminal_swd_weight": 20.0,
        "w_color": 0.0, "w_repulsive": 0.0, "w_nce": 0.0, "w_flow": 0.0,
        "low_freq_kernel_size": 5, "semantic_swd_num_projections": 64, "swd_distance_mode": "cdf",
        "swd_use_high_freq": False, "swd_num_projections": 64, "swd_patch_sizes": [3, 5, 7, 15],
        "loss_type": "mse"
    },
    "training": {
        "seed": 42, "batch_size": 32, "accumulation_steps": 2, "num_workers": 2,
        "shuffle": False, "persistent_workers": True, "prefetch_factor": 4, "pin_memory": True,
        "cpu_threads": 4, "cpu_interop_threads": 2, "learning_rate": 0.0002, "min_learning_rate": 1e-05,
        "weight_decay": 0.0001, "scheduler": "cosine", "grad_clip_norm": 1.0, "num_epochs": 7,
        "save_interval": 1, "log_interval": 20, "use_tqdm": True, "use_amp": True,
        "amp_dtype": "bf16", "allow_tf32": True, "cudnn_benchmark": True, "channels_last": True,
        "use_gradient_checkpointing": True, "fused_adamw": True, "resume_checkpoint": "",
        "full_eval_batch_size": 6,
        "test_image_dir": "../style_data/overfit50",
        "full_eval_cache_dir": "../Cycle-NCE/eval_cache",
        "full_eval_image_classifier_path": "../Cycle-NCE/eval_cache/eval_style_image_classifier.pt",
        "full_eval_clip_hf_cache_dir": "../Cycle-NCE/eval_cache/hf",
        "full_eval_clip_backend": "hf", "full_eval_classifier_only": False,
        "full_eval_disable_lpips": False, "full_eval_enable_art_fid": False,
        "full_eval_enable_kid": False
    },
    "data": {
        "data_root": "G:/GitHub/Latent_Style/latent-256",
        "style_subdirs": ["photo", "Hayao", "monet", "vangogh", "cezanne"],
        "allow_hflip": True, "balance_target_styles_per_batch": True,
        "preload_to_gpu": False, "preload_max_vram_gb": 6.0, "preload_reserve_ratio": 0.4,
        "virtual_length_multiplier": 1
    },
    "checkpoint": {
        "save_dir": "./kinetic_sweep/<NAME>/checkpoints"
    }
}

SWEEP_CONFIGS = [
    ("PK_null",     {"w_kinetic": 0, "w_curvature": 0, "kinetic_mode": "path", "note": "No path constraint at all"}),
    ("CK_05",       {"w_kinetic": 0, "w_curvature": 0.5, "kinetic_mode": "path", "note": "Curvature-only, no length penalty"}),
    ("CK_10",       {"w_kinetic": 0, "w_curvature": 1.0, "kinetic_mode": "path", "note": "Curvature-only strong"}),
    ("TGK_05",      {"w_kinetic": 0.5, "w_curvature": 0, "kinetic_mode": "time_gated", "kinetic_gate_exponent": 1.0, "note": "Time-gated: early free, late penalized"}),
    ("TGK_10",      {"w_kinetic": 1.0, "w_curvature": 0, "kinetic_mode": "time_gated", "kinetic_gate_exponent": 1.0, "note": "Time-gated strong"}),
    ("TGK_05e2",    {"w_kinetic": 0.5, "w_curvature": 0, "kinetic_mode": "time_gated", "kinetic_gate_exponent": 2.0, "note": "Time-gated quadratic (aggressive early freedom)"}),
    ("TGK_CK",      {"w_kinetic": 0.3, "w_curvature": 0.3, "kinetic_mode": "time_gated", "kinetic_gate_exponent": 1.0, "note": "Time-gated + curvature combined"}),
    ("TGK_strongW",  {"w_kinetic": 0.3, "w_curvature": 0, "kinetic_mode": "time_gated", "kinetic_gate_exponent": 1.0, "terminal_swd_weight": 40.0, "note": "Time-gated + stronger SWD style pull"}),
    ("CK_loose",    {"w_kinetic": 0, "w_curvature": 0.5, "kinetic_mode": "path", "sinkhorn_epsilon": 0.1, "note": "Curvature + loose OT coupling"}),
    ("PK_null_hiSWD", {"w_kinetic": 0, "w_curvature": 0, "kinetic_mode": "path", "terminal_swd_weight": 40.0, "note": "No path constr + stronger terminal SWD"}),
    ("TGK_light",   {"w_kinetic": 0.1, "w_curvature": 0.2, "kinetic_mode": "time_gated", "kinetic_gate_exponent": 1.0, "terminal_swd_weight": 30.0, "note": "Light touch everything"}),
]

def setup_run(name, overrides):
    run_dir = SWEEP_DIR / name
    src_dir = run_dir / "src"
    if run_dir.exists():
        print(f"  {name}: already exists, skipping setup")
        return True
    run_dir.mkdir(parents=True, exist_ok=True)
    # Copy src
    shutil.copytree(str(SRC), str(src_dir), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Write config
    cfg = json.loads(json.dumps(BASE_CONFIG))  # deep copy
    cfg["bridge"].update(overrides)
    cfg["checkpoint"]["save_dir"] = f"./kinetic_sweep/{name}/checkpoints"
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"  {name}: configured OK")
    return True

def train(name):
    run_dir = SWEEP_DIR / name
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists() and list(ckpt_dir.glob("epoch_0007.pt")):
        print(f"  {name}: checkpoint exists, skipping training")
        return True
    print(f"  {name}: training...")
    result = subprocess.run(
        ["python", "src/run.py", "--config", "config.json"],
        cwd=str(run_dir),
        capture_output=True, text=True, timeout=7200,
        env={"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PATH": str(ROOT.parent / "latent-256")}
    )
    if result.returncode != 0:
        print(f"  {name}: TRAINING FAILED")
        print(result.stderr[-2000:])
        return False
    print(f"  {name}: training OK")
    return True

def evaluate(name):
    ckpt_dir = SWEEP_DIR / name / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        print(f"  {name}: no checkpoints found")
        return None
    best_ckpt = ckpts[-1]  # epoch_0007
    out_dir = SWEEP_DIR / "eval_results" / name
    if (out_dir / "summary.json").exists():
        print(f"  {name}: eval exists, skipping")
        return out_dir / "summary.json"
    print(f"  {name}: evaluating {best_ckpt.name}...")
    result = subprocess.run(
        ["python", "run_evaluation.py",
         "--checkpoint", str(best_ckpt),
         "--output", str(out_dir),
         "--batch_size", "6",
         "--num_steps", "4",
         "--force_regen"],
        cwd=str(ROOT),
        capture_output=True, text=True, timeout=3600,
        env={"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    )
    if result.returncode != 0:
        print(f"  {name}: EVAL FAILED")
        print(result.stderr[-2000:])
        # Check if summary.json was created despite error
        if (out_dir / "summary.json").exists():
            return out_dir / "summary.json"
        return None
    return out_dir / "summary.json"

def extract_metrics(summary_path):
    d = json.loads(summary_path.read_text(encoding="utf-8"))
    a = d.get("analysis", d)
    ap = a.get("style_transfer_ability", a.get("all_pairs_overview", {}))
    p2a = a.get("photo_to_art_performance", {})
    return {
        "clip_style": ap.get("clip_style", 0),
        "clip_content": ap.get("clip_content", 0),
        "content_lpips": ap.get("content_lpips", 0),
        "clip_dir": ap.get("clip_dir", 0),
        "p2a_clip_style": p2a.get("clip_style", 0),
        "p2a_clip_content": p2a.get("clip_content", 0),
        "p2a_lpips": p2a.get("content_lpips", 0),
    }

def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Setup all runs
    print("=== Phase 1: Setup ===")
    for name, overrides in SWEEP_CONFIGS:
        setup_run(name, overrides)

    # Phase 2: Train all
    print("\n=== Phase 2: Training ===")
    for name, overrides in SWEEP_CONFIGS:
        train(name)

    # Phase 3: Evaluate all
    print("\n=== Phase 3: Evaluation ===")
    results = {}
    for name, overrides in SWEEP_CONFIGS:
        summary = evaluate(name)
        if summary and summary.exists():
            results[name] = extract_metrics(summary)
            print(f"  {name}: clip_style={results[name]['clip_style']:.4f} lpips={results[name]['content_lpips']:.4f}")
        else:
            print(f"  {name}: NO RESULTS")

    # Phase 4: Comparison table
    print("\n=== Phase 4: Comparison ===")
    # SaMST numbers (overfit50)
    samst = {"clip_style": 0.7035, "clip_content": 0.7621, "content_lpips": 0.5571, "clip_dir": 0.5302}
    # Baseline D0 (OMF)
    # We'll get these from the eval results we already have
    
    header = f"{'Config':18s} {'clip_style':10s} {'clip_content':12s} {'LPIPS':8s} {'clip_dir':8s} {'p2a_style':10s} {'p2a_lpips':8s}"
    print(header)
    print("-" * len(header))
    print(f"{'SaMST':18s} {samst['clip_style']:<10.4f} {samst['clip_content']:<12.4f} {samst['content_lpips']:<8.4f} {samst['clip_dir']:<8.4f} {'-':>10s} {'-':>8s}")
    for name, overrides in SWEEP_CONFIGS:
        if name in results:
            r = results[name]
            note = overrides.get("note", "")
            mark = " <<<" if r["clip_style"] >= samst["clip_style"] else ""
            print(f"{name:18s} {r['clip_style']:<10.4f} {r['clip_content']:<12.4f} {r['content_lpips']:<8.4f} {r['clip_dir']:<8.4f} {r['p2a_clip_style']:<10.4f} {r['p2a_lpips']:<8.4f}{mark}")
        else:
            print(f"{name:18s} {'N/A':>10s}")
    print()

    # Save results to JSON
    (SWEEP_DIR / "sweep_results.json").write_text(
        json.dumps({"samst": samst, "ours": {n: results[n] for n in results}}, indent=2),
        encoding="utf-8"
    )
    print("Results saved to", SWEEP_DIR / "sweep_results.json")

if __name__ == "__main__":
    main()
