from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent.parent

DEFAULT_FILES = [
    Path("Related_Works/baseline_pipeline/scripts/run_samst_distinct5_local.py"),
    Path("Related_Works/baseline_pipeline/scripts/generate_samst_distinct5_eval.py"),
    Path("Related_Works/baseline_pipeline/scripts/run_samst_distinct5_eval_bundle.py"),
    Path("Related_Works/repos/SaMST-main/train_model/train2/train.py"),
]


def _build_archive_bytes(files: list[Path]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for rel_path in files:
            abs_path = WORKSPACE_ROOT / rel_path
            if not abs_path.is_file():
                raise FileNotFoundError(abs_path)
            tar.add(abs_path, arcname=rel_path.as_posix())
    return buffer.getvalue()


def _run(cmd: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Push the local SaMST step-checkpoint/eval packet to the remote 3060 WSL workspace."
    )
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--remote-root", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip remote py_compile verification after the sync.",
    )
    args = parser.parse_args()

    files = DEFAULT_FILES
    if args.dry_run:
        for rel_path in files:
            print(rel_path.as_posix())
        return 0

    archive_bytes = _build_archive_bytes(files)
    remote = f"{args.user}@{args.host}"
    remote_extract = (
        f"wsl -d {args.wsl_distro} --cd {args.remote_root} --exec tar -xf -"
    )
    extract_cmd = ["ssh", "-p", str(args.port), "-T", "-o", "LogLevel=ERROR", remote, remote_extract]
    extract = _run(extract_cmd, input_bytes=archive_bytes)
    sys.stdout.buffer.write(extract.stdout)
    if extract.returncode != 0:
        return extract.returncode

    if args.no_verify:
        return 0

    remote_files = [path.as_posix() for path in files]
    verify_cmd = [
        "ssh",
        "-p",
        str(args.port),
        "-T",
        "-o",
        "LogLevel=ERROR",
        remote,
        " ".join(
            [
                f"wsl -d {args.wsl_distro} --cd {args.remote_root} --exec python3 -m py_compile",
                *remote_files,
            ]
        ),
    ]
    verify = _run(verify_cmd)
    sys.stdout.buffer.write(verify.stdout)
    return verify.returncode


if __name__ == "__main__":
    raise SystemExit(main())
