from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def _resolve_nvidia_smi() -> str | None:
    candidates = [
        shutil.which("nvidia-smi"),
        "/usr/lib/wsl/lib/nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _sample_gpu(nvidia_smi: str, gpu_index: int) -> dict[str, float] | None:
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return None
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return None
    row = rows[min(max(0, gpu_index), len(rows) - 1)]
    parts = [part.strip() for part in row.split(",")]
    if len(parts) < 4:
        return None
    try:
        return {
            "timestamp": float(time.time()),
            "memory_used_mib": float(parts[0]),
            "memory_total_mib": float(parts[1]),
            "util_gpu": float(parts[2]),
            "power_draw_w": float(parts[3]),
        }
    except ValueError:
        return None


def _write_summary(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        payload = {
            "samples": 0,
            "gpu_memory_total_gb": 0.0,
            "gpu_vram_used_gb_mean": 0.0,
            "gpu_vram_used_gb_min": 0.0,
            "gpu_vram_used_gb_peak": 0.0,
            "gpu_util_mean": 0.0,
            "gpu_util_min": 0.0,
            "gpu_util_peak": 0.0,
            "gpu_power_w_mean": 0.0,
            "gpu_power_w_min": 0.0,
            "gpu_power_w_peak": 0.0,
        }
    else:
        mem_used = [row["memory_used_mib"] for row in rows]
        mem_total = [row["memory_total_mib"] for row in rows]
        util = [row["util_gpu"] for row in rows]
        power = [row["power_draw_w"] for row in rows]
        mib_to_gb = 1.0 / 1024.0
        payload = {
            "samples": len(rows),
            "gpu_memory_total_gb": max(mem_total) * mib_to_gb,
            "gpu_vram_used_gb_mean": sum(mem_used) / len(mem_used) * mib_to_gb,
            "gpu_vram_used_gb_min": min(mem_used) * mib_to_gb,
            "gpu_vram_used_gb_peak": max(mem_used) * mib_to_gb,
            "gpu_util_mean": sum(util) / len(util),
            "gpu_util_min": min(util),
            "gpu_util_peak": max(util),
            "gpu_power_w_mean": sum(power) / len(power),
            "gpu_power_w_min": min(power),
            "gpu_power_w_peak": max(power),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample GPU VRAM/util/power while a PID is alive.")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--interval-sec", type=float, default=2.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--linger-sec", type=float, default=3.0)
    args = parser.parse_args()

    nvidia_smi = _resolve_nvidia_smi()
    if not nvidia_smi:
        raise FileNotFoundError("nvidia-smi not found for GPU monitor")

    rows: list[dict[str, float]] = []
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "elapsed_sec",
                "memory_used_mib",
                "memory_total_mib",
                "util_gpu",
                "power_draw_w",
            ],
        )
        writer.writeheader()
        start = time.time()
        last_alive = start
        while True:
            sample = _sample_gpu(nvidia_smi, int(args.gpu_index))
            if sample is not None:
                sample["elapsed_sec"] = float(sample["timestamp"] - start)
                rows.append(sample)
                writer.writerow(sample)
                f.flush()
            alive = _pid_alive(int(args.pid))
            now = time.time()
            if alive:
                last_alive = now
            if (not alive) and (now - last_alive) >= max(0.0, float(args.linger_sec)):
                break
            time.sleep(max(0.25, float(args.interval_sec)))

    _write_summary(args.summary_out, rows)
    print(
        f"[monitor_pid_gpu] pid={args.pid} samples={len(rows)} "
        f"csv={args.csv_out} summary={args.summary_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
