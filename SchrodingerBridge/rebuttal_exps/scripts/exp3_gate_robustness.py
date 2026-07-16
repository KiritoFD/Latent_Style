"""Exp3: Gradient Gate Robustness Analysis.

Parses internal_dynamics.jsonl from existing robustness runs to determine
when (or if) the internal gradient gate fires across 3 seeds x 3 batch sizes.

Existing runs (all trained 2026-07-15):
  - seed=42,  probe_batch=4  (production baseline) -> runs/submission/hf_oriented_internal_early_stop
  - seed=7,   probe_batch=4  -> runs/submission/robustness/early_stop_seed7
  - seed=123, probe_batch=4  -> runs/submission/robustness/early_stop_seed123
  - seed=42,  probe_batch=2  -> runs/submission/robustness/early_stop_probe_b2
  - seed=42,  probe_batch=8  -> runs/submission/robustness/early_stop_probe_b8

Also collects eval metrics from robustness_eval_epoch_metrics.json where available.
"""
import json, os, sys
from pathlib import Path

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

OUTPUT = WEAVE_ROOT / "exp" / "rebuttal" / "exp3_gate_robustness.json"

RUNS = [
    {"label": "seed42_b4", "seed": 42, "probe_b": 4, "dyn_path": WEAVE_ROOT / "runs/submission/hf_oriented_internal_early_stop/internal_dynamics.jsonl",
     "eval_path": WEAVE_ROOT / "exp/repro_weave_d5/dino_summary.json"},
    {"label": "seed7_b4",  "seed": 7,  "probe_b": 4, "dyn_path": WEAVE_ROOT / "runs/submission/robustness/early_stop_seed7/internal_dynamics.jsonl",
     "eval_path": WEAVE_ROOT / "runs/submission/robustness/early_stop_seed7/robustness_eval_epoch_metrics.json"},
    {"label": "seed123_b4","seed": 123,"probe_b": 4, "dyn_path": WEAVE_ROOT / "runs/submission/robustness/early_stop_seed123/internal_dynamics.jsonl",
     "eval_path": WEAVE_ROOT / "runs/submission/robustness/early_stop_seed123/robustness_eval_epoch_metrics.json"},
    {"label": "seed42_b2", "seed": 42, "probe_b": 2, "dyn_path": WEAVE_ROOT / "runs/submission/robustness/early_stop_probe_b2/internal_dynamics.jsonl",
     "eval_path": None},
    {"label": "seed42_b8", "seed": 42, "probe_b": 8, "dyn_path": WEAVE_ROOT / "runs/submission/robustness/early_stop_probe_b8/internal_dynamics.jsonl",
     "eval_path": None},
]


def parse_dynamics(path):
    """Parse internal_dynamics.jsonl, return list of epoch records."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def find_gate_fire(records):
    """Find the epoch where the gate first fires (stop_requested=1)."""
    for r in records:
        if float(r.get("internal_probe_stop_requested", 0)) >= 1.0:
            return int(r["epoch"])
    return None


def extract_dynamics_summary(records, fire_epoch):
    """Extract key dynamics at the fire epoch (or last epoch if no fire)."""
    if not records:
        return None
    target_epoch = fire_epoch if fire_epoch else records[-1]["epoch"]
    for r in records:
        if int(r["epoch"]) == target_epoch:
            return {
                "epoch": target_epoch,
                "gate_mean": r.get("internal_probe_gate_mean"),
                "gate_delta": r.get("internal_probe_gate_delta"),
                "shared_ll_hf_ratio": r.get("internal_probe_shared_ll_hf_grad_ratio"),
                "transition": r.get("internal_probe_transition"),
                "stop_requested": r.get("internal_probe_stop_requested"),
            }
    return None


def read_eval_metrics(path):
    """Read eval metrics from robustness_eval_epoch_metrics.json or dino_summary.json."""
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    # robustness_eval_epoch_metrics.json is a list of epoch records
    if isinstance(data, list):
        if not data:
            return None
        # Take the first (usually only one) entry
        r = data[0]
        return {
            "epoch": r.get("epoch"),
            "dino_s": r.get("dino_s"),
            "dino_c": r.get("dino_c"),
            "clip_s": r.get("clip_s"),
            "lpips": r.get("lpips"),
        }
    # dino_summary.json (production baseline) has all_dino_s etc.
    return {
        "epoch": 4,  # production uses epoch_0004.pt
        "dino_s": data.get("all_dino_s"),
        "dino_c": data.get("all_dino_c"),
        "clip_s": data.get("all_clip_s"),
        "lpips": data.get("all_lpips"),
    }


def main():
    print("=" * 60)
    print("Exp3: Gradient Gate Robustness Analysis")
    print("=" * 60)

    results = []
    for run in RUNS:
        print(f"\n--- {run['label']} (seed={run['seed']}, probe_b={run['probe_b']}) ---")
        records = parse_dynamics(run["dyn_path"])
        if not records:
            print(f"  ERROR: dynamics not found at {run['dyn_path']}")
            results.append({**run, "status": "no_dynamics"})
            continue

        fire_epoch = find_gate_fire(records)
        n_epochs = len(records)
        print(f"  Total epochs: {n_epochs}")
        print(f"  Gate fires at: {fire_epoch if fire_epoch else 'NEVER'}")

        dynamics = extract_dynamics_summary(records, fire_epoch)
        if dynamics:
            print(f"  At fire/last epoch {dynamics['epoch']}: gate={dynamics['gate_mean']:.6f}, "
                  f"delta={dynamics['gate_delta']:.6f}, ratio={dynamics['shared_ll_hf_ratio']:.6f}")

        eval_metrics = read_eval_metrics(run["eval_path"])
        if eval_metrics:
            print(f"  Eval: DINO-S={eval_metrics['dino_s']:.6f}, DINO-C={eval_metrics['dino_c']:.6f}, "
                  f"CLIP-S={eval_metrics['clip_s']:.4f}, LPIPS={eval_metrics['lpips']:.4f}")
        else:
            print(f"  Eval: not available")

        results.append({
            "label": run["label"],
            "seed": run["seed"],
            "probe_batch_size": run["probe_b"],
            "total_epochs": n_epochs,
            "gate_fire_epoch": fire_epoch,
            "gate_fired": fire_epoch is not None,
            "dynamics_at_fire": dynamics,
            "eval_metrics": eval_metrics,
            "status": "complete",
        })

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: 3x3 Grid (3 seeds x 3 probe batch sizes)")
    print("=" * 60)
    print(f"  {'Label':<14} {'Seed':<6} {'ProbeB':<8} {'Fire?':<8} {'Epoch':<6} {'DINO-S':<10} {'CLIP-S':<10}")
    for r in results:
        if r.get("status") != "complete":
            print(f"  {r['label']:<14} {r.get('seed','?'):<6} {r.get('probe_b','?'):<8} ERROR")
            continue
        eval = r.get("eval_metrics") or {}
        fire_str = "YES" if r["gate_fired"] else "NO"
        epoch_str = str(r["gate_fire_epoch"]) if r["gate_fired"] else "-"
        dino_s = f"{eval.get('dino_s',0):.4f}" if eval else "N/A"
        clip_s = f"{eval.get('clip_s',0):.4f}" if eval else "N/A"
        print(f"  {r['label']:<14} {r['seed']:<6} {r['probe_batch_size']:<8} {fire_str:<8} {epoch_str:<6} {dino_s:<10} {clip_s:<10}")

    # Statistics
    fired = [r for r in results if r.get("gate_fired")]
    not_fired = [r for r in results if r.get("status") == "complete" and not r.get("gate_fired")]
    print(f"\n  Gate fired: {len(fired)}/{len(results)} configurations")
    if fired:
        fire_epochs = [r["gate_fire_epoch"] for r in fired]
        print(f"  Fire epochs: {fire_epochs}")
        print(f"  All fire epochs in [3, 4]: {all(3 <= e <= 4 for e in fire_epochs)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()
