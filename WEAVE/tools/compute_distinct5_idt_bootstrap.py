from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


DEFAULT_METHODS = {
    "LBM-F e1": "SchrodingerBridge/exp/distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3_b44_remote/full_eval/epoch_0001/metrics.csv",
    "LBM-H e1": "SchrodingerBridge/exp/distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote/full_eval/epoch_0001/metrics.csv",
    "LBM-H e2": "SchrodingerBridge/exp/distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote/full_eval/epoch_0002/metrics.csv",
    "LBM-K e1": "SchrodingerBridge/exp/distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote/full_eval/epoch_0001/metrics.csv",
    "SaMST e5": "Related_Works/baseline_pipeline/results/samst_distinct5_512_real_b1_e5_20260603/eval_bundle/eval_epoch5/epoch_0005/metrics.csv",
    "SaMST e15": "Related_Works/baseline_pipeline/results/samst_distinct5_512_real_b2_e15_20260602/eval_epoch15/epoch_0015/metrics.csv",
}


def _load_metrics(path: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["clip_style"] = float(row["clip_style"])
            row["content_lpips"] = float(row["content_lpips"])
            rows.append(row)
    return rows


def _row_key(row: dict[str, str | float]) -> tuple[str, str, str]:
    src_style = str(row["src_style"])
    src_image = str(row["src_image"])
    prefix = f"{src_style}__"
    if src_image.startswith(prefix):
        src_image = src_image[len(prefix) :]
    return src_style, str(row["tgt_style"]), src_image


def _bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> tuple[float, float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    draws: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _idx in range(n):
            total += values[rng.randrange(n)]
        draws.append(total / n)
    draws.sort()
    mean = sum(values) / n
    lo = draws[int(0.025 * samples)]
    hi = draws[int(0.975 * samples)]
    prob_positive = sum(v > 0.0 for v in draws) / samples
    return mean, lo, hi, prob_positive


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap for Distinct5 CLIP-S deltas over IDT.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--idt",
        type=Path,
        default=Path("SchrodingerBridge/docs/experiments/idt_eval_20260602/distinct5_512/idt_5x5/metrics.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("SchrodingerBridge/docs/experiments/distinct5_512_20260602/bootstrap/paired_idt_transfer_bootstrap.csv"),
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Optional extra method in the form 'Label=relative/or/absolute/path/to/metrics.csv'.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    idt_path = args.idt if args.idt.is_absolute() else repo_root / args.idt
    out_path = args.out if args.out.is_absolute() else repo_root / args.out

    idt_rows = {
        _row_key(row): row
        for row in _load_metrics(idt_path)
        if row["src_style"] != row["tgt_style"]
    }

    methods = dict(DEFAULT_METHODS)
    for raw in args.method:
        if "=" not in str(raw):
            raise ValueError(f"Invalid --method value: {raw}")
        label, rel_path = str(raw).split("=", 1)
        methods[label.strip()] = rel_path.strip()

    results: list[dict[str, str | int | float]] = []
    for index, (method, rel_path) in enumerate(methods.items()):
        path = repo_root / rel_path
        method_rows = {
            _row_key(row): row
            for row in _load_metrics(path)
            if row["src_style"] != row["tgt_style"]
        }
        common_keys = sorted(set(idt_rows) & set(method_rows))
        if not common_keys:
            raise RuntimeError(f"No paired rows for {method}: {path}")

        diffs = [
            float(method_rows[key]["clip_style"]) - float(idt_rows[key]["clip_style"])
            for key in common_keys
        ]
        mean, lo, hi, prob_positive = _bootstrap_mean_ci(
            diffs,
            samples=args.samples,
            seed=args.seed + index,
        )
        method_clip = sum(float(method_rows[key]["clip_style"]) for key in common_keys) / len(common_keys)
        idt_clip = sum(float(idt_rows[key]["clip_style"]) for key in common_keys) / len(common_keys)
        method_lpips = sum(float(method_rows[key]["content_lpips"]) for key in common_keys) / len(common_keys)

        results.append(
            {
                "method": method,
                "scope": "transfer",
                "n": len(common_keys),
                "missing_from_method": len(set(idt_rows) - set(method_rows)),
                "extra_in_method": len(set(method_rows) - set(idt_rows)),
                "clip_style": method_clip,
                "idt_clip_style": idt_clip,
                "delta_idt_clip_style": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "bootstrap_prob_delta_gt_0": prob_positive,
                "content_lpips": method_lpips,
                "metrics_path": str(path),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    for row in results:
        print(
            f"{row['method']}: delta={float(row['delta_idt_clip_style']):.6f} "
            f"CI=[{float(row['ci95_low']):.6f},{float(row['ci95_high']):.6f}] "
            f"P>0={float(row['bootstrap_prob_delta_gt_0']):.4f}"
        )
    print(out_path)


if __name__ == "__main__":
    main()
