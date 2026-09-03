from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def build_runs() -> list[dict[str, float | int | bool | str]]:
    runs: list[dict[str, float | int | bool | str]] = []
    decode_scales = [0.191, 0.193, 0.195, 0.197]
    residual_scales = [1.00, 1.03, 1.06]
    for decode_scale in decode_scales:
        scale_tag = f"{decode_scale:.3f}".replace(".", "p")
        for residual_scale in residual_scales:
            residual_tag = f"{residual_scale:.2f}".replace(".", "p")
            runs.append(
                {
                    "name": f"endpoint_scale{scale_tag}_res{residual_tag}",
                    "force_integrate": False,
                    "num_steps": 12,
                    "step_size": 1.00,
                    "vae_decode_scale": decode_scale,
                    "residual_scale": residual_scale,
                }
            )
    for step_size in (1.01, 1.02, 1.03):
        runs.append(
            {
                "name": f"endpoint_scale0p191_res1p00_step{step_size:.2f}".replace(".", "p"),
                "force_integrate": False,
                "num_steps": 12,
                "step_size": step_size,
                "vae_decode_scale": 0.191,
                "residual_scale": 1.00,
            }
        )
    return runs


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Small high-style chase around the best t01 endpoint inference point.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "SchrodingerBridge" / "exp" / "diffeomorphic_tangent_sweep" / "t01_ws0p03_g6_nl0p05" / "epoch_0008.pt",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=repo_root / "SchrodingerBridge" / "exp" / "inference_param_sweep_t01_style_chase",
    )
    parser.add_argument("--test_dir", default="style_data/overfit50")
    parser.add_argument("--style_subdirs", default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_src_samples", type=int, default=30)
    parser.add_argument("--max_ref_compare", type=int, default=20)
    parser.add_argument("--max_ref_cache", type=int, default=64)
    parser.add_argument("--ref_feature_batch_size", type=int, default=8)
    parser.add_argument("--eval_lpips_chunk_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default="../Cycle-NCE/eval_cache")
    parser.add_argument("--clip_hf_cache_dir", default="../Cycle-NCE/eval_cache/hf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    src_root = repo_root / "SchrodingerBridge" / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")
    py = sys.executable

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runs = build_runs()
    print(json.dumps({"total_runs": len(runs), "output_root": str(output_root)}, ensure_ascii=False))

    rows: list[dict[str, object]] = []
    for run in runs:
        name = str(run["name"])
        out_dir = output_root / name
        cmd = [
            py,
            "-m",
            "utils.run_evaluation",
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--output",
            str(out_dir),
            "--force_regen",
            "--test_dir",
            str(args.test_dir),
            "--style_subdirs",
            str(args.style_subdirs),
            "--batch_size",
            str(args.batch_size),
            "--max_src_samples",
            str(args.max_src_samples),
            "--max_ref_compare",
            str(args.max_ref_compare),
            "--max_ref_cache",
            str(args.max_ref_cache),
            "--ref_feature_batch_size",
            str(args.ref_feature_batch_size),
            "--eval_lpips_chunk_size",
            str(args.eval_lpips_chunk_size),
            "--seed",
            str(args.seed),
            "--cache_dir",
            str(args.cache_dir),
            "--clip_hf_cache_dir",
            str(args.clip_hf_cache_dir),
            "--num_steps",
            str(run["num_steps"]),
            "--step_size",
            str(run["step_size"]),
            "--vae_decode_scale",
            str(run["vae_decode_scale"]),
            "--residual_scale",
            str(run["residual_scale"]),
        ]
        subprocess.run(cmd, cwd=repo_root, env=env, check=True)

        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        overview = summary.get("analysis", {}).get("all_pairs_overview", {})
        row = {
            "name": name,
            "force_integrate": bool(run["force_integrate"]),
            "num_steps": int(run["num_steps"]),
            "step_size": float(run["step_size"]),
            "vae_decode_scale": float(run["vae_decode_scale"]),
            "residual_scale": float(run["residual_scale"]),
            "clip_style": overview.get("clip_style"),
            "clip_content": overview.get("clip_content"),
            "content_lpips": overview.get("content_lpips"),
            "clip_dir": overview.get("clip_dir"),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    with (output_root / "sweep_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "force_integrate",
                "num_steps",
                "step_size",
                "vae_decode_scale",
                "residual_scale",
                "clip_style",
                "clip_content",
                "content_lpips",
                "clip_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
