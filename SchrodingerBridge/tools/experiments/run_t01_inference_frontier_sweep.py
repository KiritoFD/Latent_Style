from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def _tag_float(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def build_default_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []

    decode_scales = [0.18800, 0.19100]
    endpoint_steps = [1.00, 0.95, 0.90, 0.85]
    integrate_num_steps = [2, 4, 6, 8, 12]
    integrate_step_sizes = [0.70, 0.80, 0.90, 1.00]

    for scale in decode_scales:
        scale_tag = str(scale).replace(".", "p")
        for step_size in endpoint_steps:
            runs.append(
                {
                    "name": f"endpoint_s{_tag_float(step_size)}_scale{scale_tag}",
                    "force_integrate": False,
                    "num_steps": 12,
                    "step_size": step_size,
                    "vae_decode_scale": scale,
                }
            )
        for num_steps in integrate_num_steps:
            for step_size in integrate_step_sizes:
                runs.append(
                    {
                        "name": f"integrate_n{num_steps}_s{_tag_float(step_size)}_scale{scale_tag}",
                        "force_integrate": True,
                        "num_steps": num_steps,
                        "step_size": step_size,
                        "vae_decode_scale": scale,
                    }
                )
    return runs


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="t01-first inference frontier sweep on strict-750.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "SchrodingerBridge" / "exp" / "diffeomorphic_tangent_sweep" / "t01_ws0p03_g6_nl0p05" / "epoch_0008.pt",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=repo_root / "SchrodingerBridge" / "exp" / "inference_param_sweep_t01_comprehensive",
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

    runs = build_default_runs()
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
        ]
        if bool(run["force_integrate"]):
            cmd.append("--force_integrate")
        subprocess.run(cmd, cwd=repo_root, env=env, check=True)

        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        overview = summary.get("analysis", {}).get("all_pairs_overview", {})
        row = {
            "name": name,
            "force_integrate": bool(run["force_integrate"]),
            "num_steps": int(run["num_steps"]),
            "step_size": float(run["step_size"]),
            "vae_decode_scale": float(run["vae_decode_scale"]),
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
