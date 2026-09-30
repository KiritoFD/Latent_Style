from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from config_schema import load_config
from dino_cache_utils import default_dino_cache_output, infer_image_root_for_latent_root
from local_gpu_lock import run_with_local_gpu_lock


def _run(cmd: list[str], *, cwd: Path) -> int:
    print("[run_local_round1_dino_cache_build] " + " ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    proc = subprocess.run(cmd, check=False, cwd=str(cwd), env=env)
    return int(proc.returncode)


def _run_locked(cmd: list[str], *, owner: str, cwd: Path) -> int:
    print("[run_local_round1_dino_cache_build] " + " ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    return run_with_local_gpu_lock(cmd, owner=owner, cwd=str(cwd), env=env)


def _local_path_from_wsl_mount(text: str) -> Path:
    raw = str(text).strip()
    if raw.startswith("/mnt/") and len(raw) > 6:
        drive = raw[5].upper()
        remainder = raw[7:].replace("/", "\\")
        candidate = Path(f"{drive}:\\{remainder}") if remainder else Path(f"{drive}:\\")
        if candidate.exists():
            return candidate
        for alt_drive in ("F", "G", "I"):
            alt = Path(f"{alt_drive}:\\{remainder}") if remainder else Path(f"{alt_drive}:\\")
            if alt.exists():
                return alt
        return candidate
    return Path(raw)


def _remote_scp_target_from_local_path(path: Path) -> str:
    text = str(path.resolve())
    if len(text) >= 2 and text[1] == ":":
        drive = text[0].upper()
        remainder = text[2:].replace("\\", "/").lstrip("/")
        return f"/{drive}:/{remainder}" if remainder else f"/{drive}:/"
    raise ValueError(f"Cannot derive remote scp target from non-drive path: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a matching DINO cache locally under the shared GPU lock and optionally sync it to the remote host.")
    parser.add_argument("--config", required=True, help="Workspace-relative config path.")
    parser.add_argument("--image-root-override", type=Path, default=None)
    parser.add_argument("--output-override", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sync-remote", action="store_true")
    parser.add_argument("--remote-host", default="administrator@100.115.18.62")
    parser.add_argument("--remote-port", type=int, default=2222)
    args = parser.parse_args()

    config_path = (WORKSPACE / Path(args.config)).resolve()
    cfg = load_config(config_path)
    run_name = str((cfg.get("ablation") or {}).get("name", config_path.stem)).strip() or config_path.stem
    data_cfg = cfg.get("data") or {}
    latent_root = _local_path_from_wsl_mount(str(data_cfg.get("data_root", "")).strip())
    if not latent_root.exists():
        raise FileNotFoundError(f"Local latent root not found: {latent_root}")
    image_root = Path(args.image_root_override).resolve() if args.image_root_override is not None else infer_image_root_for_latent_root(latent_root)
    output_path = Path(args.output_override).resolve() if args.output_override is not None else default_dino_cache_output(latent_root, workspace_root=WORKSPACE)
    styles = [str(x).strip() for x in data_cfg.get("style_subdirs", []) if str(x).strip()]

    build_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_offline_dino_pairing_cache.py"),
        "--image-root",
        str(image_root),
        "--latent-root",
        str(latent_root),
        "--output",
        str(output_path),
        "--styles",
        ",".join(styles),
        "--batch-size",
        str(max(1, int(args.batch_size))),
        "--device",
        str(args.device),
    ]
    rc = _run_locked(build_cmd, owner=f"run_local_round1_dino_cache_build:{run_name}", cwd=WORKSPACE)
    if rc != 0:
        return rc
    if not output_path.exists():
        raise FileNotFoundError(f"DINO cache build reported success but output is missing: {output_path}")

    if bool(args.sync_remote):
        remote_target = f"{args.remote_host}:{_remote_scp_target_from_local_path(output_path)}"
        rc = _run(
            [
                "scp",
                "-P",
                str(int(args.remote_port)),
                str(output_path),
                remote_target,
            ],
            cwd=WORKSPACE,
        )
        if rc != 0:
            return rc

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
