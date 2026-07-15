from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the local round-1 fast-eval watcher as a detached background job.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--test-dir", default=r"F:\wikiart_distinct5_samam_512_classview_real\test")
    parser.add_argument("--cache-dir", default=r"G:\GitHub\Latent_Style\eval_cache")
    parser.add_argument("--clip-hf-cache-dir", default=r"G:\GitHub\Latent_Style\eval_cache\hf")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vae-decode-batch-size", type=int, default=16)
    parser.add_argument("--target-chunk-size", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    cmd = [
        "python",
        str(repo_root / "tools" / "experiments" / "watch_local_round1_family_fast_eval.py"),
        "--config",
        str(args.config),
        "--local-root",
        str(args.local_root),
        "--test-dir",
        str(args.test_dir),
        "--cache-dir",
        str(args.cache_dir),
        "--clip-hf-cache-dir",
        str(args.clip_hf_cache_dir),
        "--batch-size",
        str(int(args.batch_size)),
        "--vae-decode-batch-size",
        str(int(args.vae_decode_batch_size)),
        "--target-chunk-size",
        str(int(args.target_chunk_size)),
        "--poll-seconds",
        str(int(args.poll_seconds)),
        "--patience",
        str(int(args.patience)),
    ]
    stdout_path = Path(args.stdout_log)
    stderr_path = Path(args.stderr_log)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        subprocess.Popen(
            cmd,
            cwd=str(repo_root.parent),
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
