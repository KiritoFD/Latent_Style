#!/usr/bin/env python3
"""Launch batch eval+WFI in background, properly detached."""
import subprocess, sys, os, time

repo = "/mnt/i/Github/Latent_Style/SchrodingerBridge"
base = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
log_file = os.path.join(base, "batch_eval_wfi.log")

cmd = [
    sys.executable,
    os.path.join(repo, "tools/batch_eval_wfi.py"),
]

print(f"Launching batch eval+WFI...")
print(f"Log: {log_file}")

# Open log file and launch process
with open(log_file, "w") as log:
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=repo,
        start_new_session=True,
        close_fds=True,
        env={**os.environ, "PYTHONPATH": os.path.join(repo, "src")},
    )

print(f"Started PID: {proc.pid}")
print(f"Log: {log_file}")

# Wait a moment and check if it's still running
time.sleep(3)
if proc.poll() is None:
    print("Process is running (good)")
else:
    print(f"Process exited immediately with code {proc.returncode}")
    with open(log_file) as f:
        print("Log output:")
        print(f.read()[-2000:])
