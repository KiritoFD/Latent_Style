"""Launch D7b eval + DINO on remote. Runs synchronously (blocking) — eval ~3-5min, DINO ~1-2min."""
import subprocess
import sys
import os

REMOTE = "administrator@100.115.18.62"
PORT = "2222"

# Build the command - use cmd.exe with & separator since PowerShell has escape issues
# Eval script path: scripts/_run_eval_dino.ps1 (already on remote)
cmd_parts = [
    "powershell", "-ExecutionPolicy", "Bypass", "-File",
    r"I:\Github\Latent_Style\SchrodingerBridge\scripts\_run_eval_dino.ps1",
    "d7b_deep_backbone_lr5e5",
]

remote_cmd = " ".join(cmd_parts)
# Wrap in cd to correct directory
full_remote = f'cd /d I:\\Github\\Latent_Style\\SchrodingerBridge && powershell -ExecutionPolicy Bypass -File scripts\\_run_eval_dino.ps1 d7b_deep_backbone_lr5e5'

ssh_cmd = [
    "ssh", "-p", PORT, "-o", "LogLevel=ERROR", REMOTE,
    full_remote
]

print(f"LAUNCHING: {' '.join(ssh_cmd)}", flush=True)
print("=" * 80, flush=True)

# Run with long timeout - eval can take 5-10 minutes
result = subprocess.run(ssh_cmd, capture_output=False, timeout=1200)

print("=" * 80, flush=True)
print(f"EXIT_CODE={result.returncode}", flush=True)
