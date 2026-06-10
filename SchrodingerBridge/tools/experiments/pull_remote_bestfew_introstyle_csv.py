from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull a remote best-few IntroStyle CSV via scp.")
    parser.add_argument("--host", default="administrator@100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--remote-file", required=True)
    parser.add_argument("--local-file", type=Path, required=True)
    args = parser.parse_args()

    local_file = Path(args.local_file)
    local_file.parent.mkdir(parents=True, exist_ok=True)
    remote_path = str(args.remote_file).replace("\\", "/")
    if remote_path.startswith("/mnt/c/"):
        remote_path = "/C:/" + remote_path[len("/mnt/c/") :]
    elif remote_path.startswith("/mnt/") and len(remote_path) > 6:
        drive = remote_path[5].upper()
        rest = remote_path[7:]
        remote_path = f"/{drive}:/{rest}"
    cmd = [
        "scp",
        "-P",
        str(args.port),
        f"{args.host}:{remote_path}",
        str(local_file),
    ]
    print("[pull_remote_bestfew_introstyle_csv] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(local_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
