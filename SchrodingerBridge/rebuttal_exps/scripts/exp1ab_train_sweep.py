"""Exp1a/1b: Training hyperparameter sensitivity sweep.

Sweeps:
  Exp1a: lambda_LL (bridge.spectral_w_ll) - missing 0.1, 0.4, 0.5
         (already have 0.0=b01, 0.3=baseline, 2.0=b02 in ablation_v2)
  Exp1b: alpha (bridge.ll_partial_alpha) - missing 0.1, 0.2, 0.4, 0.5
         (already have 0.3=baseline)

For each missing point:
  1. Create a derived config that inherits from canonical config.json
  2. Override the swept parameter + checkpoint.save_dir
  3. Disable internal_early_stop to ensure epoch_0004.pt is saved
  4. Train 5 epochs (~2 min each on RTX 3060)
  5. Evaluate epoch_0004.pt with paper-canonical compute_dino_metrics.py
     using inference.json override (endpoint_adain_scale=2.0)

Run on remote RTX 3060.
"""
import json, os, sys, subprocess, time, csv, shutil
from pathlib import Path

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

# Use canonical config as base (relative path from experiments/rebuttal/)
BASE_CONFIG_REF = "../../config.json"
# Inference override (endpoint_adain_scale=2.0)
INFERENCE_OVERRIDE = "inference.json"
TEST_DIR = "data/test"
HF_CACHE = "exp/eval_cache/hf"
MAX_EPOCHS = 5
EVAL_EPOCH = 4

# Existing ablation_v2 checkpoints to re-eval with paper-canonical DINO
ABLATION_V2_DIR = WEAVE_ROOT / "exp" / "ablation_v2"
EXISTING_CKPTS = {
    "lambda_ll_0.0": ABLATION_V2_DIR / "b01_wll_0" / "epoch_0005.pt",
    "lambda_ll_2.0": ABLATION_V2_DIR / "b02_wll_20" / "epoch_0005.pt",
}

# Production baseline (already evaluated; will read from existing dino_summary.json)
PRODUCTION_BASELINE = {
    "lambda_ll_0.3": WEAVE_ROOT / "exp" / "repro_weave_d5" / "dino_summary.json",
    "alpha_0.3": WEAVE_ROOT / "exp" / "repro_weave_d5" / "dino_summary.json",
}

# Sweep definitions: only the MISSING points need training
SWEEPS = {
    "lambda_ll": {
        "config_key": "bridge.spectral_w_ll",
        "train_values": [0.1, 0.4, 0.5],
        # Existing: 0.0 (b01), 0.3 (baseline), 2.0 (b02)
    },
    "alpha": {
        "config_key": "bridge.ll_partial_alpha",
        "train_values": [0.1, 0.2, 0.4, 0.5],
        # Existing: 0.3 (baseline)
    },
}

OUTPUT = WEAVE_ROOT / "exp" / "rebuttal" / "exp1ab_train_sweep.json"
CONFIG_DIR = WEAVE_ROOT / "experiments" / "rebuttal"
SAVE_ROOT = WEAVE_ROOT / "runs" / "rebuttal_sweep"


def fmt_tag(sweep_name, value):
    v = str(value).replace(".", "p")
    return f"{sweep_name}_{v}"


def create_config(sweep_name, value, config_key):
    """Create a derived config that inherits from canonical config.json."""
    tag = fmt_tag(sweep_name, value)
    save_dir = str(SAVE_ROOT / tag).replace("\\", "/")
    # Use forward slashes for cross-platform consistency
    save_dir = save_dir.replace("I:/Github/Latent_Style/WEAVE/", "")

    config = {
        "_base": BASE_CONFIG_REF,
        "bridge": {},
        "training": {
            "num_epochs": MAX_EPOCHS,
            "save_interval": 1,
            "internal_early_stop_enabled": False,  # ensure epoch_0004 is saved
            "full_eval_each_epoch": False,
        },
        "checkpoint": {
            "save_dir": f"runs/rebuttal_sweep/{tag}",
        },
    }
    # Set the swept parameter
    keys = config_key.split(".")
    cur = config
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / f"{tag}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(config_path), tag


def run_training(config_path, tag):
    """Run training with given config. Stream output to stdout."""
    print(f"  Training {tag}...", flush=True)
    cmd = [sys.executable, "-u", "run.py", "--config", config_path]
    start = time.time()
    # Stream output live
    proc = subprocess.Popen(
        cmd, cwd=str(WEAVE_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True,
    )
    for line in proc.stdout:
        sys.stdout.write(f"    {line}")
        sys.stdout.flush()
    proc.wait()
    elapsed = time.time() - start
    print(f"  Training done in {elapsed:.1f}s, returncode={proc.returncode}", flush=True)
    return proc.returncode == 0, elapsed


def run_eval(ckpt_path, tag):
    """Evaluate a checkpoint with paper-canonical DINO metrics."""
    out_dir = WEAVE_ROOT / "exp" / "rebuttal" / f"eval_{tag}"
    out_dir_str = str(out_dir).replace("\\", "/")
    out_dir_str = out_dir_str.replace("I:/Github/Latent_Style/WEAVE/", "")

    if out_dir.exists():
        dino_summary = out_dir / "dino_summary.json"
        if dino_summary.exists():
            print(f"  Eval already done for {tag}, skipping", flush=True)
            return read_dino_summary(dino_summary)

    # Step 1: Generate images + LPIPS/CLIP via run_evaluation.py
    print(f"  Generating images for {tag}...", flush=True)
    gen_cmd = [
        sys.executable, "-u", "utils/run_evaluation.py",
        "--checkpoint", str(ckpt_path),
        "--config_override", INFERENCE_OVERRIDE,
        "--output", out_dir_str,
        "--test_dir", TEST_DIR,
        "--batch_size", "2",
        "--ref_feature_batch_size", "2",
        "--vae_decode_batch_size", "16",
        "--force_regen",
    ]
    proc = subprocess.run(gen_cmd, cwd=str(WEAVE_ROOT))
    if proc.returncode != 0:
        print(f"  ERROR: image generation failed for {tag}", flush=True)
        return None

    # Step 2: Paper-canonical DINO metrics
    print(f"  Computing DINO metrics for {tag}...", flush=True)
    dino_cmd = [
        sys.executable, "-u", "utils/compute_dino_metrics.py",
        "--eval_dir", out_dir_str,
        "--test_dir", TEST_DIR,
        "--cache_dir", HF_CACHE,
    ]
    subprocess.run(dino_cmd, cwd=str(WEAVE_ROOT))

    dino_summary = out_dir / "dino_summary.json"
    if not dino_summary.exists():
        print(f"  ERROR: dino_summary.json not found for {tag}", flush=True)
        return None
    return read_dino_summary(dino_summary)


def read_dino_summary(path):
    """Read paper-canonical DINO summary."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "clip_s": d.get("all_clip_s"),
        "lpips": d.get("all_lpips"),
        "dino_s": d.get("all_dino_s"),
        "dino_c": d.get("all_dino_c"),
        "dino_structure": d.get("all_dino_structure"),
        "off_dino_s": d.get("off_dino_s"),
        "off_dino_c": d.get("off_dino_c"),
    }


def eval_existing_ablation_v2(name, ckpt_path):
    """Re-evaluate existing ablation_v2 checkpoint with paper-canonical DINO."""
    print(f"\n--- Re-evaluating existing {name} ---", flush=True)
    if not ckpt_path.exists():
        print(f"  ERROR: checkpoint not found: {ckpt_path}", flush=True)
        return None
    return run_eval(ckpt_path, name)


def read_production_baseline():
    """Read production baseline metrics from existing dino_summary.json."""
    path = WEAVE_ROOT / "exp" / "repro_weave_d5" / "dino_summary.json"
    if not path.exists():
        print(f"  WARNING: production baseline not found at {path}", flush=True)
        return None
    return read_dino_summary(path)


def run_sweep(sweep_name, sweep_def):
    """Run a full sweep for one parameter."""
    print(f"\n{'='*60}")
    print(f"Sweep: {sweep_name} ({sweep_def['config_key']})")
    print(f"{'='*60}")

    results = {}

    # 1. Production baseline (value=0.3 for both sweeps)
    if sweep_name == "lambda_ll":
        baseline_value = 0.3
    else:  # alpha
        baseline_value = 0.3
    baseline_metrics = read_production_baseline()
    if baseline_metrics:
        results[str(baseline_value)] = {**baseline_metrics, "source": "production_baseline"}
        print(f"  Baseline ({sweep_name}={baseline_value}): DINO-S={baseline_metrics['dino_s']:.6f}", flush=True)

    # 2. Existing ablation_v2 checkpoints (re-eval with paper-canonical DINO)
    for name, ckpt in EXISTING_CKPTS.items():
        if not name.startswith(sweep_name):
            continue
        value_str = name.split("_")[-1]
        metrics = eval_existing_ablation_v2(name, ckpt)
        if metrics:
            results[value_str] = {**metrics, "source": "ablation_v2_reeval"}
            print(f"  {name}: DINO-S={metrics['dino_s']:.6f}", flush=True)

    # 3. Train missing points
    for value in sweep_def["train_values"]:
        tag = fmt_tag(sweep_name, value)
        print(f"\n--- {sweep_name} = {value} (training required) ---", flush=True)

        # Skip if already trained + evaluated
        ckpt_path = SAVE_ROOT / tag / f"epoch_{EVAL_EPOCH:04d}.pt"
        eval_done = (WEAVE_ROOT / "exp" / "rebuttal" / f"eval_{tag}" / "dino_summary.json").exists()

        if eval_done:
            print(f"  Already trained + evaluated, reading existing results", flush=True)
            metrics = read_dino_summary(WEAVE_ROOT / "exp" / "rebuttal" / f"eval_{tag}" / "dino_summary.json")
            if metrics:
                results[str(value)] = {**metrics, "source": "trained"}
                continue

        if not ckpt_path.exists():
            # Create config and train
            config_path, tag = create_config(sweep_name, value, sweep_def["config_key"])
            success, train_time = run_training(config_path, tag)
            if not success:
                results[str(value)] = {"error": "training_failed"}
                continue
        else:
            print(f"  Checkpoint already exists: {ckpt_path}", flush=True)

        # Evaluate
        metrics = run_eval(ckpt_path, tag)
        if metrics:
            results[str(value)] = {**metrics, "source": "trained"}
            print(f"  {sweep_name}={value}: DINO-S={metrics['dino_s']:.6f}, CLIP-S={metrics['clip_s']:.4f}, LPIPS={metrics['lpips']:.4f}", flush=True)
        else:
            results[str(value)] = {"error": "eval_failed"}

    return results


def main():
    print("=" * 60)
    print("Exp1a/1b: Training Hyperparameter Sensitivity Sweep")
    print("=" * 60)
    print(f"WEAVE_ROOT: {WEAVE_ROOT}")
    print(f"Config dir: {CONFIG_DIR}")
    print(f"Save root:  {SAVE_ROOT}")
    print(f"Eval epoch: {EVAL_EPOCH}")
    print()

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for sweep_name, sweep_def in SWEEPS.items():
        all_results[sweep_name] = run_sweep(sweep_name, sweep_def)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for sweep_name, results in all_results.items():
        print(f"\n{sweep_name} ({SWEEPS[sweep_name]['config_key']}):")
        print(f"  {'Value':<8} {'DINO-S':<12} {'DINO-C':<12} {'CLIP-S':<10} {'LPIPS':<10} {'Source':<25}")
        for value in sorted(results.keys(), key=lambda x: float(x)):
            r = results[value]
            if "dino_s" in r and r["dino_s"] is not None:
                print(f"  {value:<8} {r['dino_s']:<12.6f} {r['dino_c']:<12.6f} {r['clip_s']:<10.4f} {r['lpips']:<10.4f} {r.get('source','?'):<25}")
            else:
                print(f"  {value:<8} FAILED: {r.get('error', 'unknown')}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()
