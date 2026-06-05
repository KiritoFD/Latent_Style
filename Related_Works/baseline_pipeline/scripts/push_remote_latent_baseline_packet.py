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

DEFAULT_PATHS = [
    Path("Related_Works/baseline_pipeline/scripts/run_samam_latent_baseline.py"),
    Path("Related_Works/baseline_pipeline/scripts/generate_samam_latent_eval.py"),
    Path("Related_Works/baseline_pipeline/scripts/run_samam_latent_eval_bundle.py"),
    Path("Related_Works/baseline_pipeline/scripts/run_samst_latent_baseline.py"),
    Path("Related_Works/baseline_pipeline/scripts/generate_samst_latent_eval.py"),
    Path("Related_Works/baseline_pipeline/scripts/run_samst_latent_eval_bundle.py"),
    Path("Related_Works/repos/SaMam/ARCHI/Decoder.py"),
    Path("Related_Works/repos/SaMam/MODEL/SaMam_model.py"),
    Path("Related_Works/repos/SaMam/TRAIN/lightning_module/latent_dataset.py"),
    Path("Related_Works/repos/SaMam/TRAIN/lightning_module/latent_datamodule.py"),
    Path("Related_Works/repos/SaMam/TRAIN/lightning_module/latent_lightningmodel.py"),
    Path("Related_Works/repos/SaMam/TRAIN/train_SaMam_latent.py"),
    Path("Related_Works/repos/SaMST-main/networks/transfer_net.py"),
    Path("Related_Works/repos/SaMST-main/networks/latent_transfer_net.py"),
    Path("Related_Works/repos/SaMST-main/train_model/train2/train_latent.py"),
    Path("SchrodingerBridge/src/utils/inference.py"),
    Path("SchrodingerBridge/src/utils/run_evaluation.py"),
    Path("SchrodingerBridge/src/utils/targetwise_artfid_summary.py"),
]


def _iter_files(rel_path: Path):
    abs_path = WORKSPACE_ROOT / rel_path
    if abs_path.is_file():
        yield rel_path, abs_path
        return
    if abs_path.is_dir():
        for file in abs_path.rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts:
                yield file.relative_to(WORKSPACE_ROOT), file
        return
    raise FileNotFoundError(abs_path)


def _build_archive_bytes(paths: list[Path]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for rel_path in paths:
            for archive_rel, abs_path in _iter_files(rel_path):
                tar.add(abs_path, arcname=archive_rel.as_posix())
    return buffer.getvalue()


def _run(cmd: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Push the reviewed latent baseline packet to the remote 3060 WSL workspace.")
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--remote-root", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        for rel_path in DEFAULT_PATHS:
            print(rel_path.as_posix())
        return 0

    archive_bytes = _build_archive_bytes(DEFAULT_PATHS)
    remote = f"{args.user}@{args.host}"
    remote_extract = f"wsl -d {args.wsl_distro} --cd {args.remote_root} --exec tar -xf -"
    extract_cmd = ["ssh", "-p", str(args.port), "-T", "-o", "LogLevel=ERROR", remote, remote_extract]
    extract = _run(extract_cmd, input_bytes=archive_bytes)
    sys.stdout.buffer.write(extract.stdout)
    if extract.returncode != 0:
        return extract.returncode

    if args.no_verify:
        return 0

    remote_files = []
    for rel_path in DEFAULT_PATHS:
        for archive_rel, _ in _iter_files(rel_path):
            if archive_rel.suffix == ".py":
                remote_files.append(archive_rel.as_posix())
    verify_cmd = [
        "ssh",
        "-p",
        str(args.port),
        "-T",
        "-o",
        "LogLevel=ERROR",
        remote,
        " ".join(
            [f"wsl -d {args.wsl_distro} --cd {args.remote_root} --exec python3 -m py_compile", *remote_files]
        ),
    ]
    verify = _run(verify_cmd)
    sys.stdout.buffer.write(verify.stdout)
    return verify.returncode


if __name__ == "__main__":
    raise SystemExit(main())
