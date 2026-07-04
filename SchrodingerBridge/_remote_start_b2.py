#!/usr/bin/env python3
"""远程启动 B2 POC 训练 (在远程 WSL 中用 python3 执行).

用法:
  wsl python3 /mnt/c/Users/administrator/b2_sync_tmp/_remote_start_b2.py
"""
import json
import subprocess
import sys
from pathlib import Path

REMOTE_ROOT = "/mnt/i/Github/Latent_Style/SchrodingerBridge"
CONFIG_PATH = f"{REMOTE_ROOT}/configs/620_spectral_poc.json"
LOG_PATH = f"{REMOTE_ROOT}/exp/620_spectral_poc/train.log"
SESSION_NAME = "b2_poc"


def run(cmd, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"FAIL: {cmd}", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        sys.exit(1)
    return r


def main():
    print("=== Config data section ===")
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    d = cfg["data"]
    t = cfg["training"]
    print(f"  dino_cache_path: {d.get('dino_cache_path')!r}")
    print(f"  dino_cache_required: {d.get('dino_cache_required')}")
    print(f"  pairing_cache_path: {d.get('pairing_cache_path')}")
    print(f"  latent_cache_dir: {d.get('latent_cache_dir')}")
    print(f"  num_workers: {t.get('num_workers')}")
    print(f"  persistent_workers: {t.get('persistent_workers')}")
    print(f"  num_epochs: {t.get('num_epochs')}")
    print(f"  batch_size: {t.get('batch_size')}")

    print("\n=== Kill old tmux session ===")
    r = run(f"tmux kill-session -t {SESSION_NAME} 2>/dev/null")
    print(f"  killed (rc={r.returncode})")

    print("\n=== Start new tmux session ===")
    run(f"mkdir -p {REMOTE_ROOT}/exp/620_spectral_poc", check=True)
    train_cmd = (
        f"cd {REMOTE_ROOT} && "
        f"PYTHONUNBUFFERED=1 python3 run.py --config configs/620_spectral_poc.json "
        f"2>&1 | tee {LOG_PATH}"
    )
    tmux_cmd = ["tmux", "new-session", "-d", "-s", SESSION_NAME, train_cmd]
    r = subprocess.run(tmux_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL to start tmux: {r.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"  TMUX_STARTED (session={SESSION_NAME})")

    print("\n=== Sessions ===")
    r = run("tmux list-sessions")
    print(r.stdout)

    print("=== DONE ===")


if __name__ == "__main__":
    main()
