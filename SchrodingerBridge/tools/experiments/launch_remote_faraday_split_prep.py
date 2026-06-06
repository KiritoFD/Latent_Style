from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = SB_ROOT.parent
GENERIC_LAUNCHER = SCRIPT_DIR / "launch_remote_wsl_command.py"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _run_bytes(cmd: list[str], *, input_bytes: bytes) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        cmd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _load_packet_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _archive_split_packet(*, local_root: Path, remote_parent_rel: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for file in sorted(local_root.rglob("*")):
            if not file.is_file():
                continue
            rel = file.relative_to(local_root.parent).as_posix()
            arcname = f"{remote_parent_rel.rstrip('/')}/{rel}"
            tar.add(file, arcname=arcname)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync a fixed-rule WikiArt stress split packet to the remote owner surface and launch latent/pairing preparation through the host-owned WSL launcher."
    )
    parser.add_argument("--split-slug", required=True)
    parser.add_argument("--packet-summary", default="F:/wikiart_faraday_splits/packet_summary.json")
    parser.add_argument("--local-splits-root", default="F:/wikiart_faraday_splits")
    parser.add_argument("--remote-splits-root", default="/mnt/i/wikiart_faraday_splits")
    parser.add_argument("--remote-workspace-root", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--python-bin", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--health-wait-seconds", type=int, default=30)
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=1500)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=11000)
    parser.add_argument("--overwrite-latents", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-packet-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = _load_packet_summary(Path(args.packet_summary).resolve())
    packet = None
    for item in summary.get("packets", []):
        if str(item.get("slug", "")).strip() == str(args.split_slug).strip():
            packet = item
            break
    if packet is None:
        raise KeyError(f"split slug not found in packet summary: {args.split_slug}")

    local_root = Path(packet["local_root"]).resolve()
    if not local_root.is_dir():
        raise FileNotFoundError(local_root)
    styles = list(packet["styles"])
    remote_split_root = f"{args.remote_splits_root.rstrip('/')}/{args.split_slug}"
    remote_log = f"{args.remote_workspace_root.rstrip('/')}/SchrodingerBridge/_codex_tmp/{args.split_slug}_prep.log"
    task_name = f"faraday-prep-{args.split_slug}"

    archive_bytes = _archive_split_packet(
        local_root=local_root,
        remote_parent_rel=Path(args.remote_splits_root).name,
    )

    print(f"split_slug={args.split_slug}")
    print(f"local_root={local_root}")
    print(f"remote_split_root={remote_split_root}")
    print(f"packet_gib={len(archive_bytes) / (1024 ** 3):.3f}")
    print(f"styles={','.join(styles)}")

    if not args.dry_run and (not args.skip_packet_sync):
        remote = f"{args.user}@{args.host}"
        extract = _run_bytes(
            [
                "ssh",
                "-p",
                str(int(args.port)),
                "-T",
                "-o",
                "LogLevel=ERROR",
                remote,
                f"wsl -d {args.wsl_distro} --cd /mnt/i --exec tar -xf -",
            ],
            input_bytes=archive_bytes,
        )
        sys.stdout.buffer.write(extract.stdout)
        if extract.returncode != 0:
            return extract.returncode
    elif args.skip_packet_sync:
        print("packet_sync=skipped")

    launcher_cmd = [
        sys.executable,
        str(GENERIC_LAUNCHER),
        "--task-name",
        task_name,
        "--remote-log-path",
        remote_log,
        "--remote-wsl-cwd",
        args.remote_workspace_root,
        "--remote-workspace-root",
        args.remote_workspace_root,
        "--python-bin",
        args.python_bin,
        "--host",
        args.host,
        "--port",
        str(int(args.port)),
        "--user",
        args.user,
        "--wsl-distro",
        args.wsl_distro,
        "--max-prelaunch-memory-mib",
        str(int(args.max_prelaunch_memory_mib)),
        "--health-wait-seconds",
        str(int(args.health_wait_seconds)),
        "--max-runtime-memory-mib",
        str(int(args.max_runtime_memory_mib)),
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/encode_image_folder_latents.py",
        "--sync-path",
        "SchrodingerBridge/tools/build_latent_packed_cache.py",
        "--sync-path",
        "SchrodingerBridge/tools/build_latent_prototype_pairing_cache.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/run_faraday_split_prep.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/encode_image_folder_latents.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/build_latent_packed_cache.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/build_latent_prototype_pairing_cache.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/experiments/run_faraday_split_prep.py",
    ]
    if args.dry_run:
        launcher_cmd.append("--dry-run")
    launcher_cmd.extend(
        [
            "--",
            args.python_bin,
            "SchrodingerBridge/tools/experiments/run_faraday_split_prep.py",
            "--split-root",
            remote_split_root,
            "--styles",
            ",".join(styles),
            "--python-bin",
            args.python_bin,
            "--vae-model",
            "ema",
            "--vae-cache-dir",
            f"{args.remote_workspace_root.rstrip('/')}/eval_cache/hf",
            "--image-size",
            "512",
            "--batch-size",
            str(int(args.batch_size)),
            "--seed",
            "20260603",
        ]
    )
    if args.overwrite_latents:
        launcher_cmd.append("--overwrite-latents")
    if args.rebuild_cache:
        launcher_cmd.append("--rebuild-cache")

    result = _run(launcher_cmd)
    sys.stdout.write(result.stdout)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
