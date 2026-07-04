import subprocess
import os

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"

print("=" * 70)
print("Remote Server Environment Check")
print("=" * 70)

def ssh_cmd(command, desc):
    print(f"\n[CHECK] {desc}")
    cmd = f'ssh -p {ssh_port} {ssh_host} {command}'
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    if result.stdout.strip():
        print(f"Output: {result.stdout.strip()[:500]}")
    if result.returncode != 0:
        print(f"Error (code {result.returncode}): {result.stderr.strip()[:300]}")
    return result.returncode == 0

# Basic checks
ssh_cmd("echo 'SSH_OK'", "Basic SSH connectivity")
ssh_cmd("whoami", "Current user")
ssh_cmd("pwd", "Working directory")
ssh_cmd("uname -a", "System info")

# Directory checks
print("\n--- Directory Checks ---")
ssh_cmd("ls -la /home/xy/ 2>&1 | head -10", "Home directory contents")
ssh_cmd("test -d /home/xy/Latent_Style && echo 'EXISTS' || echo 'NOT_FOUND'", "Latent_Style directory")
ssh_cmd("test -d /home/xy/Latent_Style/SchrodingerBridge && echo 'EXISTS' || echo 'NOT_FOUND'", "Project directory")

# Experiment directories
print("\n--- Experiment Directories ---")
ssh_cmd("ls -la /home/xy/Latent_Style/SchrodingerBridge/exp/ 2>&1 | head -20", "exp directory")
ssh_cmd("test -d /home/xy/Latent_Style/SchrodingerBridge/exp/p3_remote_10h && echo 'EXISTS' || echo 'NOT_FOUND'", "p3_remote_10h directory")

# Python check
print("\n--- Python Check ---")
ssh_cmd("which python3 || which python", "Python location")
ssh_cmd("python3 --version 2>/dev/null || python --version 2>/dev/null", "Python version")

print("\n" + "=" * 70)
