"""Exp A: Per-epoch external evaluation for internal early-stop validation.

For each seed's checkpoint sequence, evaluate DINO-S/CLIP-S/LPIPS/DINO-C at every
available epoch. Then compute oracle regret = DINO-S(e_oracle) - DINO-S(e_internal).

Usage:
    python scripts/expA_per_epoch_eval.py --run_dir <path> --seed <n> [--epochs 1,2,3,...]
"""
import argparse, json, os, sys, subprocess, time, csv
from pathlib import Path

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

INFERENCE_OVERRIDE = "inference.json"
TEST_DIR = "data/test"
HF_CACHE = "exp/eval_cache/hf"


def find_available_epochs(run_dir):
    """Find all epoch_XXXX.pt checkpoints in run_dir."""
    epochs = []
    for f in sorted(Path(run_dir).glob("epoch_*.pt")):
        try:
            epoch_num = int(f.stem.split("_")[1])
            epochs.append((epoch_num, f))
        except (ValueError, IndexError):
            continue
    return epochs


def read_dino_summary(eval_dir):
    """Read DINO metrics from eval_dir/dino_summary.json."""
    dino_path = Path(eval_dir) / "dino_summary.json"
    if not dino_path.exists():
        return None
    d = json.loads(dino_path.read_text(encoding="utf-8"))
    return {
        "dino_s": d.get("all_dino_s"),
        "dino_c": d.get("all_dino_c"),
        "dino_structure": d.get("all_dino_structure"),
        "clip_s": d.get("all_clip_s"),
        "lpips": d.get("all_lpips"),
    }


def read_run_metrics_csv(run_dir):
    """Read existing robustness_eval_epoch_metrics.csv if it exists."""
    csv_path = Path(run_dir) / "robustness_eval_epoch_metrics.csv"
    if not csv_path.exists():
        return {}
    results = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            results[epoch] = {
                "dino_s": float(row.get("dino_s", 0) or 0),
                "dino_c": float(row.get("dino_c", 0) or 0),
                "clip_s": float(row.get("clip_s", 0) or 0),
                "lpips": float(row.get("lpips", 0) or 0),
                "checkpoint": row.get("checkpoint", ""),
            }
    return results


def eval_single_epoch(run_dir, epoch_num, ckpt_path, seed, force=False, tag=""):
    """Evaluate a single epoch checkpoint. Returns metrics dict or None."""
    tag_prefix = f"expA_{tag}_" if tag else "expA_"
    eval_dir = WEAVE_ROOT / "exp" / "rebuttal" / f"{tag_prefix}seed{seed}" / f"epoch_{epoch_num:04d}"
    eval_dir_str = str(eval_dir).replace("\\", "/").replace("I:/Github/Latent_Style/WEAVE/", "")

    # Check if already done
    dino_summary = eval_dir / "dino_summary.json"
    if dino_summary.exists() and not force:
        print(f"  Epoch {epoch_num}: already evaluated, reading cache", flush=True)
        metrics = read_dino_summary(eval_dir)
        if metrics and metrics.get("dino_s", 0) > 0:
            return metrics

    print(f"  Epoch {epoch_num}: generating images...", flush=True)
    gen_cmd = [
        sys.executable, "-u", "utils/run_evaluation.py",
        "--checkpoint", str(ckpt_path),
        "--config_override", INFERENCE_OVERRIDE,
        "--output", eval_dir_str,
        "--test_dir", TEST_DIR,
        "--batch_size", "2",
        "--ref_feature_batch_size", "2",
        "--vae_decode_batch_size", "16",
        "--force_regen",
    ]
    t0 = time.time()
    proc = subprocess.run(gen_cmd, cwd=str(WEAVE_ROOT))
    if proc.returncode != 0:
        print(f"  Epoch {epoch_num}: ERROR image generation failed", flush=True)
        return None
    t_gen = time.time() - t0
    print(f"  Epoch {epoch_num}: generation done in {t_gen:.0f}s", flush=True)

    # Compute DINO metrics
    print(f"  Epoch {epoch_num}: computing DINO...", flush=True)
    dino_cmd = [
        sys.executable, "-u", "utils/compute_dino_metrics.py",
        "--eval_dir", eval_dir_str,
        "--test_dir", TEST_DIR,
        "--cache_dir", HF_CACHE,
    ]
    subprocess.run(dino_cmd, cwd=str(WEAVE_ROOT))

    metrics = read_dino_summary(eval_dir)
    if metrics:
        print(f"  Epoch {epoch_num}: DINO-S={metrics['dino_s']:.4f}, DINO-C={metrics['dino_c']:.4f}, CLIP-S={metrics['clip_s']:.4f}, LPIPS={metrics['lpips']:.4f}", flush=True)
    return metrics


def parse_internal_dynamics(run_dir):
    """Parse internal_dynamics.jsonl to find when gate fired."""
    dyn_path = Path(run_dir) / "internal_dynamics.jsonl"
    if not dyn_path.exists():
        return None
    records = []
    for line in dyn_path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))
    # Find first epoch where transition is True
    for r in records:
        if float(r.get("internal_probe_transition", 0)) >= 1.0:
            return int(r["epoch"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=str, default="", help="Comma-separated epoch list; empty=all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tag", type=str, default="", help="Tag prefix for output dir (e.g. D3, D4, D5)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    seed = args.seed

    print("=" * 60)
    print(f"Exp A: Per-epoch evaluation for seed={seed}")
    print(f"Run dir: {run_dir}")
    print("=" * 60)

    # Find available epochs
    all_epochs = find_available_epochs(run_dir)
    if not all_epochs:
        print("ERROR: No epoch checkpoints found")
        sys.exit(1)

    print(f"Available epochs: {[e for e, _ in all_epochs]}")

    # Filter to requested epochs
    if args.epochs:
        requested = [int(x) for x in args.epochs.split(",")]
        epochs_to_eval = [(e, p) for e, p in all_epochs if e in requested]
    else:
        epochs_to_eval = all_epochs

    print(f"Epochs to evaluate: {[e for e, _ in epochs_to_eval]}")
    print()

    # Read existing metrics
    existing = read_run_metrics_csv(run_dir)
    print(f"Existing metrics from CSV: {list(existing.keys())}")

    # Evaluate each epoch
    results = {}
    for epoch_num, ckpt_path in epochs_to_eval:
        print(f"\n--- Seed {seed}, Epoch {epoch_num} ---", flush=True)

        # Check existing CSV first (non-zero DINO-S)
        if epoch_num in existing and existing[epoch_num].get("dino_s", 0) > 0 and not args.force:
            print(f"  Using existing CSV metrics: DINO-S={existing[epoch_num]['dino_s']:.4f}")
            results[epoch_num] = existing[epoch_num]
            continue

        metrics = eval_single_epoch(run_dir, epoch_num, ckpt_path, seed, force=args.force, tag=args.tag)
        if metrics:
            results[epoch_num] = {
                **metrics,
                "checkpoint": ckpt_path.name,
            }

    # Find internal early-stop epoch
    e_internal = parse_internal_dynamics(run_dir)
    print(f"\nInternal early-stop epoch: {e_internal}")

    # Find oracle epoch (max DINO-S)
    if results:
        e_oracle = max(results.keys(), key=lambda e: results[e].get("dino_s", 0))
        dino_s_oracle = results[e_oracle].get("dino_s", 0)
        dino_s_internal = results.get(e_internal, {}).get("dino_s", 0) if e_internal else 0
        regret = dino_s_oracle - dino_s_internal

        print(f"\nOracle epoch (max DINO-S): {e_oracle} (DINO-S={dino_s_oracle:.4f})")
        if e_internal:
            print(f"Internal stop epoch: {e_internal} (DINO-S={dino_s_internal:.4f})")
            print(f"Regret: {regret:.6f}")
            print(f"Epoch offset: {e_internal - e_oracle}")

    # Save results
    tag_prefix = f"expA_{args.tag}_" if args.tag else "expA_"
    out_dir = WEAVE_ROOT / "exp" / "rebuttal" / f"{tag_prefix}seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "per_epoch_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "checkpoint", "dino_s", "dino_c", "clip_s", "lpips"])
        for epoch in sorted(results.keys()):
            r = results[epoch]
            writer.writerow([
                epoch,
                r.get("checkpoint", ""),
                r.get("dino_s", 0),
                r.get("dino_c", 0),
                r.get("clip_s", 0),
                r.get("lpips", 0),
            ])
    print(f"\nCSV saved: {csv_path}")

    # JSON summary
    summary = {
        "seed": seed,
        "run_dir": str(run_dir),
        "e_internal": e_internal,
        "e_oracle": e_oracle if results else None,
        "regret": regret if results and e_internal else None,
        "epoch_offset": (e_internal - e_oracle) if results and e_internal else None,
        "per_epoch": {str(e): results[e] for e in sorted(results.keys())},
    }
    json_path = out_dir / "oracle_regret.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"JSON saved: {json_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY (seed={seed})")
    print(f"{'='*60}")
    print(f"  {'Epoch':<8} {'DINO-S':<12} {'DINO-C':<12} {'CLIP-S':<10} {'LPIPS':<10} {'Note':<15}")
    for epoch in sorted(results.keys()):
        r = results[epoch]
        note = ""
        if e_internal == epoch:
            note = "<-- internal"
        if e_oracle == epoch:
            note = "<-- oracle"
        print(f"  {epoch:<8} {r.get('dino_s',0):<12.4f} {r.get('dino_c',0):<12.4f} {r.get('clip_s',0):<10.4f} {r.get('lpips',0):<10.4f} {note}")

    print(f"\nEXPA_SEED{seed}_EXIT=0")


if __name__ == "__main__":
    main()
