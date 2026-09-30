from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


DEFAULT_LOCK_PATH = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\.local_gpu_eval.lock")
_WINDOWS_STILL_ACTIVE = 259
_WINDOWS_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _pid_alive(pid: int) -> bool:
    pid = int(pid)
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(_WINDOWS_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return int(exit_code.value) == _WINDOWS_STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_lock(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_lock(path: Path, *, pid: int, owner: str) -> None:
    payload = {"pid": int(pid), "owner": str(owner)}
    tmp_path = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def _acquire(path: Path, *, owner: str, wait: bool = True, poll_seconds: float = 5.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "owner": str(owner)}, f, ensure_ascii=False)
            return
        except FileExistsError:
            existing = _read_lock(path)
            stale = False
            if isinstance(existing, dict):
                pid = int(existing.get("pid", -1))
                stale = pid <= 0 or (not _pid_alive(pid))
            else:
                stale = True
            if stale:
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            if not wait:
                raise RuntimeError(f"Local GPU lock is already held: {path}")
            time.sleep(max(0.5, float(poll_seconds)))


def _release(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def update_lock_owner(*, lock_path: Path = DEFAULT_LOCK_PATH, pid: int, owner: str) -> None:
    _write_lock(lock_path, pid=int(pid), owner=str(owner))


def run_with_local_gpu_lock(
    cmd: list[str],
    *,
    owner: str,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> int:
    with local_gpu_lock(owner=owner, lock_path=lock_path):
        proc = subprocess.Popen(cmd, cwd=None if cwd is None else str(cwd), env=env)
        update_lock_owner(lock_path=lock_path, pid=int(proc.pid), owner=str(owner))
        return int(proc.wait())


@contextmanager
def local_gpu_lock(*, owner: str, lock_path: Path = DEFAULT_LOCK_PATH, wait: bool = True, poll_seconds: float = 5.0):
    _acquire(lock_path, owner=owner, wait=wait, poll_seconds=poll_seconds)
    try:
        yield
    finally:
        _release(lock_path)
