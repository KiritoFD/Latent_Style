from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REMOTE_ROOT = "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_hold4mid_e8_carriergate_injection_seed42_b8a2/full_eval_fresh_localreview"
LOCAL_ROOT = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\carriergate_fresh_localreview_20260609")


def _run(cmd: list[str]) -> int:
    print("[pull_remote_carriergate_fresh_eval] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    remote_script = f"set -euo pipefail\ncd {REMOTE_ROOT}\ntar -cf - .\n".encode("utf-8")
    ssh_cmd = [
        "ssh",
        "-p",
        "2222",
        "administrator@100.115.18.62",
        "wsl -d Ubuntu-26.04 -- bash -s",
    ]
    tar_path = LOCAL_ROOT / "carriergate_fresh_localreview.tar"
    with tar_path.open("wb") as f:
        proc = subprocess.run(ssh_cmd, input=remote_script, stdout=f, stderr=subprocess.PIPE, check=False)
    sys.stdout.write(proc.stderr.decode("utf-8", errors="replace"))
    if proc.returncode != 0:
        return int(proc.returncode)
    extract_cmd = [
        "tar",
        "-xf",
        str(tar_path),
        "-C",
        str(LOCAL_ROOT),
    ]
    rc = _run(extract_cmd)
    if rc != 0:
        return rc
    print(LOCAL_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
