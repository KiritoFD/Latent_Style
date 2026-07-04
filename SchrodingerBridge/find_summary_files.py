import subprocess
import os

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"

print("=" * 70)
print("Searching for summary.json files on remote server")
print("=" * 70)

def wsl_cmd(command, desc, timeout=60):
    print(f"\n[SEARCH] {desc}")
    cmd = f'ssh -p {ssh_port} {ssh_host} wsl {command}'
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=timeout
        )
        if result.stdout.strip():
            print(f"[FOUND]\n{result.stdout.strip()[:2000]}")
        if result.returncode != 0 and result.stderr.strip():
            print(f"[ERR] {result.stderr.strip()[:300]}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("[TIMEOUT]")
        return ""

# Search for experiment directories
print("\n--- Searching for experiment directories ---")
out1 = wsl_cmd("find /home/xy -type d -name 'fc_sb_*' 2>/dev/null | head -20", "Find fc_sb experiment directories")

# Search for any summary.json files in the project
print("\n--- Searching for summary.json files ---")
out2 = wsl_cmd("find /home/xy/Latent_Style/SchrodingerBridge -name 'summary.json' 2>/dev/null | head -30", "Find all summary.json files")

# Check specific experiment paths
exps = ["fc_sb_kernel7", "fc_sb_floor0", "fc_sb_curriculum", "fc_sb_fiber_ep", "fc_sb_wavelet"]
print("\n--- Checking specific experiment paths ---")
for exp in exps:
    paths_to_check = [
        f"/home/xy/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/{exp}",
        f"/home/xy/Latent_Style/SchrodingerBridge/{exp}",
        f"/home/xy/exp/p3_remote_10h/{exp}",
    ]
    for path in paths_to_check:
        out = wsl_cmd(f"test -d '{path}' && ls -la '{path}/' 2>/dev/null | head -15 || echo 'NOT_FOUND'", f"Check {path}")
        if "NOT_FOUND" not in out and out.strip():
            break

# Also search for round2_convergence.json
print("\n--- Searching for round2_convergence.json ---")
out3 = wsl_cmd("find /home/xy/Latent_Style/SchrodingerBridge -name 'round2_convergence.json' 2>/dev/null | head -20", "Find convergence files")

# List what's in p3_remote_10h if it exists
print("\n--- Contents of p3_remote_10h (if exists) ---")
out4 = wsl_cmd("ls -la /home/xy/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/ 2>/dev/null || echo 'DIR_NOT_FOUND'", "List p3_remote_10h contents")

print("\n" + "=" * 70)
print("Search Complete")
print("=" * 70)
