#!/usr/bin/env python3
"""Windows native Python 启动 B2 POC 训练 (在远程 Windows 上直接执行).

用法 (远程 cmd.exe):
  cd I:\Github\Latent_Style\SchrodingerBridge
  python _win_start_b2.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJ_ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge")
CONFIG_PATH = PROJ_ROOT / "configs" / "620_spectral_poc.json"
EXP_DIR = PROJ_ROOT / "exp" / "620_spectral_poc"
LOG_PATH = EXP_DIR / "train.log"


def main():
    print("=== Config verification ===")
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    d = cfg["data"]
    t = cfg["training"]
    print(f"  data_root: {d.get('data_root')}")
    print(f"  dino_cache_path: {d.get('dino_cache_path')!r}")
    print(f"  dino_cache_required: {d.get('dino_cache_required')}")
    print(f"  pairing_cache_path: {d.get('pairing_cache_path')}")
    print(f"  latent_cache_dir: {d.get('latent_cache_dir')}")
    print(f"  test_image_dir: {t.get('test_image_dir')}")
    print(f"  num_workers: {t.get('num_workers')}")
    print(f"  persistent_workers: {t.get('persistent_workers')}")
    print(f"  num_epochs: {t.get('num_epochs')}")
    print(f"  batch_size: {t.get('batch_size')}")

    print("\n=== Prepare exp dir ===")
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  EXP_DIR: {EXP_DIR}")

    print("\n=== Launch training (detached) ===")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # 不要设置 PYTHONPATH=src, 否则 run.py 递归导入自身

    log_file = open(LOG_PATH, "w", encoding="utf-8")

    proc = subprocess.Popen(
        ["python", "-u", "run.py", "--config", str(CONFIG_PATH)],
        cwd=str(PROJ_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )
    print(f"  PID: {proc.pid}")
    print(f"  Log: {LOG_PATH}")

    # 等待几秒确认进程存活
    time.sleep(5)
    if proc.poll() is None:
        print(f"  STATUS: RUNNING (pid={proc.pid})")
    else:
        print(f"  STATUS: EXITED (rc={proc.returncode})")
        print("  --- early log ---")
        log_file.close()
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            print(f.read()[-2000:])

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
