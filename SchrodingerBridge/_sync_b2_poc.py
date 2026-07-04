#!/usr/bin/env python3
"""FC-SB Phase 4: 同步 B2/B4 代码到远程 (SCP方式) + 启动 B2 POC 训练

用 SCP 上传到远程 Windows 临时目录, 再用 WSL cp 移动到 /mnt/i/ 目标路径。
避免 base64 命令行长度限制。

用法 (本地 Windows):
  cd g:\GitHub\Latent_Style\SchrodingerBridge
  python _sync_b2_poc.py
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

SSH_HOST = "administrator@100.115.18.62"
SSH_PORT = "2222"
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15"]
REMOTE_ROOT = "/mnt/i/Github/Latent_Style/SchrodingerBridge"
REMOTE_TMP = "C:/Users/administrator/b2_sync_tmp"  # Windows 路径, SCP 目标

FILES_TO_SYNC = [
    "src/config_schema.py",
    "src/model620.py",
    "src/utils/inference.py",
    "src/utils/run_evaluation.py",
    "src/fiber_moe620.py",
    "src/losses620.py",
    "src/spectral620.py",
    "src/spectral_bridge620.py",
    "src/spectral_losses620.py",
    "src/model.py",
    "src/trainer.py",
    "configs/620_spectral_poc.json",
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ssh_run(cmd: str, timeout: int = 60) -> tuple[int, str]:
    full_cmd = ["ssh", "-p", SSH_PORT] + SSH_OPTS + [SSH_HOST, cmd]
    try:
        proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -2, str(e)


def scp_upload(local_path: Path, remote_win_path: str) -> bool:
    """用 SCP 上传单个文件到远程 Windows 路径."""
    scp_cmd = ["scp", "-P", SSH_PORT] + SSH_OPTS + [str(local_path).replace("\\", "/"), f"{SSH_HOST}:{remote_win_path}"]
    try:
        proc = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace")
        return proc.returncode == 0
    except Exception as e:
        log(f"  scp exception: {e}")
        return False


def upload_file(local_path: Path, remote_linux_path: str) -> bool:
    """SCP 上传到 Windows 临时目录, 再用 WSL cp 移动到 Linux 路径."""
    if not local_path.exists():
        log(f"[ERROR] local file missing: {local_path}")
        return False

    # 用扁平化文件名避免路径问题
    flat_name = local_path.name
    win_tmp = f"{REMOTE_TMP}/{flat_name}"

    # Step 1: SCP 上传到 Windows 临时目录
    if not scp_upload(local_path, win_tmp):
        log(f"[FAIL-SCP] {local_path.name}")
        return False

    # Step 2: WSL 创建目标目录并移动文件
    remote_dir = str(Path(remote_linux_path).parent)
    # WSL 中 /mnt/c/ 对应 Windows C:/
    win_tmp_wsl = win_tmp.replace("C:/", "/mnt/c/").replace("\\", "/")
    wsl_cmd = f'wsl bash -c "mkdir -p {remote_dir} && cp {win_tmp_wsl} {remote_linux_path} && echo MV_OK"'
    rc, out = ssh_run(wsl_cmd, timeout=30)
    if "MV_OK" in out:
        log(f"[OK] {local_path.name} -> {remote_linux_path}")
        return True
    log(f"[FAIL-MV] {local_path.name}: {out[:200]}")
    return False


def verify_remote() -> bool:
    log("=== Verifying remote files ===")
    all_ok = True
    for f in FILES_TO_SYNC:
        remote_path = f"{REMOTE_ROOT}/{f}"
        cmd = f'wsl bash -c "test -f {remote_path} && echo EXISTS || echo MISSING"'
        rc, out = ssh_run(cmd, timeout=30)
        exists = "EXISTS" in out
        if not exists:
            all_ok = False
        log(f"  {f}: {'EXISTS' if exists else 'MISSING'}")
    return all_ok


def launch_training() -> bool:
    log("=== Launching B2 POC training in remote tmux ===")
    ssh_run('wsl bash -c "tmux kill-session -t b2_poc 2>/dev/null; echo KILLED"', timeout=15)

    # 创建日志目录
    ssh_run(f'wsl bash -c "mkdir -p {REMOTE_ROOT}/exp/620_spectral_poc"', timeout=15)

    train_cmd = (
        f'cd {REMOTE_ROOT} && '
        f'PYTHONUNBUFFERED=1 python3 run.py --config configs/620_spectral_poc.json '
        f'2>&1 | tee exp/620_spectral_poc/train.log'
    )
    tmux_cmd = f'wsl bash -c "tmux new-session -d -s b2_poc \\"{train_cmd}\\" && echo TMUX_STARTED"'
    rc, out = ssh_run(tmux_cmd, timeout=30)
    if "TMUX_STARTED" in out:
        log("[OK] tmux session 'b2_poc' started")
        return True
    log(f"[FAIL] tmux start: rc={rc}, out={out[:300]}")
    return False


def main() -> int:
    local_root = Path(__file__).resolve().parent
    log("=== FC-SB Phase 4: Sync B2 POC to remote (SCP) ===")
    log(f"Files to sync: {len(FILES_TO_SYNC)}")

    # 创建远程临时目录
    ssh_run(f'wsl bash -c "mkdir -p /mnt/c/Users/administrator/b2_sync_tmp"', timeout=15)

    # Step 1: 上传文件
    log("=== Step 1: Uploading files (SCP) ===")
    success_count = 0
    for rel_path in FILES_TO_SYNC:
        local_path = local_root / rel_path
        remote_path = f"{REMOTE_ROOT}/{rel_path}"
        if upload_file(local_path, remote_path):
            success_count += 1
    log(f"Uploaded: {success_count}/{len(FILES_TO_SYNC)}")
    if success_count != len(FILES_TO_SYNC):
        log("[ABORT] Upload incomplete")
        return 1

    # Step 2: 验证 + 语法检查
    log("")
    if not verify_remote():
        log("[ABORT] Verification failed")
        return 1

    log("")
    log("=== Remote syntax check (key files) ===")
    key_files = ["src/spectral620.py", "src/spectral_bridge620.py", "src/spectral_losses620.py", "src/losses620.py", "src/config_schema.py", "src/model.py", "src/trainer.py"]
    syntax_ok = True
    for f in key_files:
        remote_path = f"{REMOTE_ROOT}/{f}"
        cmd = f'wsl bash -c "cd {REMOTE_ROOT} && python3 -c \\"import ast; ast.parse(open(\x27{remote_path}\x27).read()); print(\x27SYNTAX_OK\x27)\\""'
        rc, out = ssh_run(cmd, timeout=30)
        ok = "SYNTAX_OK" in out
        if not ok:
            syntax_ok = False
        log(f"  {f}: {'OK' if ok else 'FAIL ' + out[-100:]}")
    if not syntax_ok:
        log("[ABORT] Syntax check failed")
        return 1

    # Step 3: Haar 测试
    log("")
    log("=== Remote Haar reconstruction test ===")
    haar_cmd = (
        f'wsl bash -c "cd {REMOTE_ROOT} && python3 -c \\"'
        f'import sys; sys.path.insert(0, \x27src\x27); import torch; '
        f'from spectral620 import dwt2_haar, idwt2_haar; '
        f'x=torch.randn(2,4,32,32); ll,lh,hl,hh=dwt2_haar(x); '
        f'x_rec=idwt2_haar(ll,lh,hl,hh); err=(x-x_rec).abs().max().item(); '
        f'print(\x27RECON_ERR\x27, err); print(\x27HAAR_OK\x27 if err<1e-6 else \x27HAAR_FAIL\x27)'
        f'\\""'
    )
    rc, out = ssh_run(haar_cmd, timeout=60)
    log(f"  {out.strip()[-200:]}")
    if "HAAR_OK" not in out:
        log("[ABORT] Haar test failed")
        return 1

    # Step 4: 启动训练
    log("")
    log("=== Launch training ===")
    if not launch_training():
        return 1

    log("")
    log("=== SYNC COMPLETE ===")
    log("Monitor: ssh -p 2222 administrator@100.115.18.62 \"wsl tail -30 /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/620_spectral_poc/train.log\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
