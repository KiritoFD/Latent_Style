"""Exp1a-fine: lambda_LL fine-grained sweep around 0.3.

已有数据点 (粗扫): 0.0, 0.1, 0.3, 0.4, 0.5, 2.0
本脚本补充 0.3 附近的细致点: 0.15, 0.2, 0.25, 0.35, 0.45

输出包含 DINO-S, DINO-C, CLIP-S, LPIPS。
"""
import json, os, sys, subprocess, time
from pathlib import Path

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

BASE_CONFIG_REF = "../../config.json"
INFERENCE_OVERRIDE = "inference.json"
TEST_DIR = "data/test"
HF_CACHE = "exp/eval_cache/hf"
MAX_EPOCHS = 5
EVAL_EPOCH = 4

# 细致扫描点 (0.3 周围)
FINE_VALUES = [0.15, 0.2, 0.25, 0.35, 0.45]
CONFIG_KEY = "bridge.spectral_w_ll"

OUTPUT = WEAVE_ROOT / "exp" / "rebuttal" / "exp1a_fine_sweep.json"
CONFIG_DIR = WEAVE_ROOT / "experiments" / "rebuttal"
SAVE_ROOT = WEAVE_ROOT / "runs" / "rebuttal_sweep"


def fmt_tag(value):
    v = str(value).replace(".", "p")
    return f"lambda_ll_{v}"


def create_config(value):
    tag = fmt_tag(value)
    config = {
        "_base": BASE_CONFIG_REF,
        "bridge": {},
        "training": {
            "num_epochs": MAX_EPOCHS,
            "save_interval": 1,
            "internal_early_stop_enabled": False,
            "full_eval_each_epoch": False,
        },
        "checkpoint": {
            "save_dir": f"runs/rebuttal_sweep/{tag}",
        },
    }
    keys = CONFIG_KEY.split(".")
    cur = config
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / f"{tag}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(config_path), tag


def run_training(config_path, tag):
    print(f"  Training {tag}...", flush=True)
    cmd = [sys.executable, "-u", "run.py", "--config", config_path]
    start = time.time()
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


def read_dino_summary(path):
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


def run_eval(ckpt_path, tag):
    out_dir = WEAVE_ROOT / "exp" / "rebuttal" / f"eval_{tag}"
    out_dir_str = str(out_dir).replace("\\", "/")
    out_dir_str = out_dir_str.replace("I:/Github/Latent_Style/WEAVE/", "")

    if out_dir.exists():
        dino_summary = out_dir / "dino_summary.json"
        if dino_summary.exists():
            print(f"  Eval already done for {tag}, skipping", flush=True)
            return read_dino_summary(dino_summary)

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


def main():
    print("=" * 60)
    print("Exp1a-fine: lambda_LL fine sweep around 0.3")
    print(f"Values: {FINE_VALUES}")
    print("=" * 60)
    print(f"WEAVE_ROOT: {WEAVE_ROOT}")
    print(f"Save root:  {SAVE_ROOT}")
    print(f"Eval epoch: {EVAL_EPOCH}")
    print()

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for value in FINE_VALUES:
        tag = fmt_tag(value)
        print(f"\n--- lambda_ll = {value} ---", flush=True)

        ckpt_path = SAVE_ROOT / tag / f"epoch_{EVAL_EPOCH:04d}.pt"
        eval_done = (WEAVE_ROOT / "exp" / "rebuttal" / f"eval_{tag}" / "dino_summary.json").exists()

        if eval_done:
            print(f"  Already trained + evaluated, reading existing results", flush=True)
            metrics = read_dino_summary(WEAVE_ROOT / "exp" / "rebuttal" / f"eval_{tag}" / "dino_summary.json")
            if metrics:
                results[str(value)] = {**metrics, "source": "trained"}
                continue

        if not ckpt_path.exists():
            config_path, tag = create_config(value)
            success, train_time = run_training(config_path, tag)
            if not success:
                results[str(value)] = {"error": "training_failed"}
                continue
        else:
            print(f"  Checkpoint already exists: {ckpt_path}", flush=True)

        metrics = run_eval(ckpt_path, tag)
        if metrics:
            results[str(value)] = {**metrics, "source": "trained"}
            print(f"  lambda_ll={value}: DINO-S={metrics['dino_s']:.6f}, DINO-C={metrics['dino_c']:.6f}, CLIP-S={metrics['clip_s']:.4f}, LPIPS={metrics['lpips']:.4f}", flush=True)
        else:
            results[str(value)] = {"error": "eval_failed"}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (fine sweep around 0.3)")
    print("=" * 60)
    print(f"  {'Value':<8} {'DINO-S':<12} {'DINO-C':<12} {'CLIP-S':<10} {'LPIPS':<10} {'Source':<15}")
    for value in sorted(results.keys(), key=lambda x: float(x)):
        r = results[value]
        if "dino_s" in r and r["dino_s"] is not None:
            print(f"  {value:<8} {r['dino_s']:<12.6f} {r['dino_c']:<12.6f} {r['clip_s']:<10.4f} {r['lpips']:<10.4f} {r.get('source','?'):<15}")
        else:
            print(f"  {value:<8} FAILED: {r.get('error', 'unknown')}")

    # Merge with existing coarse sweep results
    coarse_path = WEAVE_ROOT / "exp" / "rebuttal" / "exp1ab_train_sweep.json"
    if coarse_path.exists():
        coarse = json.loads(coarse_path.read_text(encoding="utf-8"))
        if "lambda_ll" in coarse:
            print("\n" + "=" * 60)
            print("MERGED (coarse + fine)")
            print("=" * 60)
            merged = dict(coarse["lambda_ll"])
            merged.update(results)
            print(f"  {'Value':<8} {'DINO-S':<12} {'DINO-C':<12} {'CLIP-S':<10} {'LPIPS':<10} {'Source':<25}")
            for value in sorted(merged.keys(), key=lambda x: float(x)):
                r = merged[value]
                if "dino_s" in r and r["dino_s"] is not None:
                    print(f"  {value:<8} {r['dino_s']:<12.6f} {r['dino_c']:<12.6f} {r['clip_s']:<10.4f} {r['lpips']:<10.4f} {r.get('source','?'):<25}")
                else:
                    print(f"  {value:<8} FAILED: {r.get('error', 'unknown')}")
            results["_merged"] = merged

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to: {OUTPUT}")
    print("EXP1A_FINE_EXIT=0")


if __name__ == "__main__":
    main()
