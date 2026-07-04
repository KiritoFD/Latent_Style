#!/usr/bin/env python3
"""检查 B2 POC 训练状态 (在远程 WSL 中执行)."""
import os
import subprocess
import sys
from pathlib import Path

REMOTE_ROOT = "/mnt/i/Github/Latent_Style/SchrodingerBridge"
LOG_PATH = f"{REMOTE_ROOT}/exp/620_spectral_poc/train.log"
EXP_DIR = f"{REMOTE_ROOT}/exp/620_spectral_poc"


def main():
    print("=== tmux sessions ===")
    r = subprocess.run("tmux list-sessions 2>/dev/null", shell=True, capture_output=True, text=True)
    print(r.stdout or r.stderr or "No sessions")

    print("\n=== exp dir contents ===")
    if os.path.exists(EXP_DIR):
        for item in sorted(os.listdir(EXP_DIR)):
            full = os.path.join(EXP_DIR, item)
            size = os.path.getsize(full) if os.path.isfile(full) else "DIR"
            print(f"  {item}: {size}")
    else:
        print(f"  EXP_DIR not found: {EXP_DIR}")

    print("\n=== train.log tail (last 60 lines) ===")
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        for line in lines[-60:]:
            print(line.rstrip())
    else:
        print(f"  LOG not found: {LOG_PATH}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
