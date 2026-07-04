import subprocess
import os

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"

print("=" * 70)
print("Exploring Actual Project Location")
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
        print(f"[OUT]\n{out[:2000]}")
    return out

# Check the actual Latent_Style directory structure
wsl("ls -la /home/xy/Latent_Style/", "Latent_Style directory contents")

# Check exp directory
wsl("ls -la /home/xy/exp/", "exp directory contents")

# Look for p3_remote_10h
wsl("find /home/xy -type d -name 'p3_remote_10h' 2>/dev/null", "Find p3_remote_10h directory")

# Look for experiment directories directly under exp
wsl("ls -la /home/xy/exp/ 2>/dev/null | head -30", "List all in exp/")

# Search for fc_sb experiments
wsl("find /home/xy/exp -type d -name 'fc_sb_*' 2>/dev/null | head -20", "Find fc_sb experiment dirs")

# If found, check their contents
print("\n--- Checking fc_sb experiment structures ---")
exps = ["fc_sb_kernel7", "fc_sb_floor0", "fc_sb_curriculum", "fc_sb_fiber_ep", "fc_sb_wavelet"]
for exp in exps:
    # Try multiple possible locations
    for base_path in ["/home/xy/exp", "/home/xy/Latent_Style"]:
        full_path = f"{base_path}/{exp}"
        out = wsl(f"test -d '{full_path}' && find '{full_path}' -name 'summary.json' -o -name 'round2_convergence.json' 2>/dev/null | head -5 || echo 'NO_FILES'", f"Check {full_path}")
        if "NO_FILES" not in out and out.strip():
            break

# Also check if there's a SchrodingerBridge inside Latent_Style
wsl("ls -la /home/xy/Latent_Style/SchrodingerBridge/ 2>/dev/null | head -20 || echo 'NOT_FOUND'", "Check Latent_Style/SchrodingerBridge")

print("\n" + "=" * 70)
