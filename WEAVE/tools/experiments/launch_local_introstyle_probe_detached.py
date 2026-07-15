from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch eval_introstyle_probe.py as a detached local background job with PYTHONPATH set.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--style-bank-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing

    cmd = [
        "python",
        str(repo_root / "tools" / "eval_introstyle_probe.py"),
        "--manifest",
        str(args.manifest),
        "--style-bank-root",
        str(args.style_bank_root),
        "--output_csv",
        str(args.output_csv),
        "--output_json",
        str(args.output_json),
        "--model-id",
        str(args.model_id),
        "--device",
        str(args.device),
        "--batch_size",
        str(int(args.batch_size)),
        "--ensemble_size",
        str(int(args.ensemble_size)),
    ]

    stdout_path = Path(args.stdout_log)
    stderr_path = Path(args.stderr_log)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    print(stdout_path)
    print(stderr_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
