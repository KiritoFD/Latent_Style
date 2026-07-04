import subprocess
import os

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"

print("=" * 70)
print("Deep Dive: Experiment Directory Structure")
print("=" * 70)

def wsl(command, desc):
    print(f"\n[CHECK] {desc}")
    cmd = f'ssh -p {ssh_port} {ssh_host} wsl {command}'
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore',
        timeout=30
    )
    out = result.stdout.strip()
    if out:
        print(f"[OUT]\n{out[:2500]}")
    return out

# Check p3_remote_10h contents in detail
wsl("ls -la /home/xy/exp/p3_remote_10h/", "p3_remote_10h full listing")

# Check if experiments are subdirectories
wsl("find /home/xy/exp/p3_remote_10h -maxdepth 2 -type d 2>/dev/null | head -30", "Directory tree (depth 2)")

# Check SchrodingerBridge/exp structure
wsl("ls -la /home/xy/Latent_Style/SchrodingerBridge/exp/ 2>/dev/null | head -30", "SchrodingerBridge/exp contents")

# Search more broadly for experiment directories
wsl("find /home/xy -maxdepth 4 -type d -name 'fc_sb_*' 2>/dev/null", "Find any fc_sb dirs (depth 4)")
wsl("find /home/xy -maxdepth 5 -type d -name 'full_eval' 2>/dev/null | head -20", "Find full_eval dirs")

# Check for summary.json anywhere under the project
wsl("find /home/xy/Latent_Style/SchrodingerBridge -name 'summary.json' 2>/dev/null | head -20", "Summary.json in project")
wsl("find /home/xy/exp -name 'summary.json' 2>/dev/null | head -20", "Summary.json in exp")

# List what's actually in these directories to understand structure
print("\n--- Understanding directory structure ---")
wsl("ls -la /home/xy/exp/p3_remote_10h/*/ 2>/dev/null | head -50 || echo 'NO_SUBDIRS'", "Contents of p3_remote_10h subdirs")

# Maybe experiments are named differently?
wsl("ls /home/xy/exp/p3_remote_10h/ 2>/dev/null", "List all items in p3_remote_10h")

print("\n" + "=" * 70)
