#!/usr/bin/env python3
"""B2 POC 训练诊断脚本."""
import os
import subprocess
import sys
from pathlib import Path

REMOTE_ROOT = "/mnt/i/Github/Latent_Style/SchrodingerBridge"
LOG_PATH = f"{REMOTE_ROOT}/exp/620_spectral_poc/train.log"
EXP_DIR = f"{REMOTE_ROOT}/exp/620_spectral_poc"


def main():
    print("=== Full train.log ===")
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        print(content)
    else:
        print(f"LOG not found: {LOG_PATH}")

    print("\n=== logs dir ===")
    logs_dir = os.path.join(EXP_DIR, "logs")
    if os.path.exists(logs_dir):
        for item in os.listdir(logs_dir):
            full = os.path.join(logs_dir, item)
            size = os.path.getsize(full) if os.path.isfile(full) else "DIR"
            print(f"  {item}: {size}")
            if os.path.isfile(full) and size < 10000:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    print(f"    --- content ---")
                    print(f.read())
    else:
        print("  logs dir not found")

    print("\n=== GPU status ===")
    r = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    print(r.stdout[-1500:] if r.stdout else r.stderr)

    print("\n=== dmesg tail (OOM check) ===")
    r = subprocess.run("dmesg 2>/dev/null | tail -20", shell=True, capture_output=True, text=True)
    print(r.stdout or "  dmesg not available")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
